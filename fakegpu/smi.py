from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import platform
import signal
import socket
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cli import (
    add_json_flag_argument,
    command_prog,
)
from ._smi_environment import (
    _modeled_device_topology,
    _modeled_fault_model,
    _modeled_mig_layout,
    _nonnegative_int,
)
from ._smi_query import (
    GPU_QUERY_FIELDS,
    PROCESS_QUERY_FIELDS,
    QueryField,
    RUNTIME_QUERY_FIELDS,
)
from ._smi_render import (
    render_detail,
    render_gpu_list,
    render_health_events,
    render_mig_view,
    render_nvlink_status,
    render_query,
    render_table,
    render_topology_matrix,
)
from ._smi_state import (
    DEFAULT_STALE_AFTER_SECONDS,
    SCHEMA_VERSION,
    _catalog_metadata,
    _discover_state_paths,
    _integer_mapping,
    _load_states,
    _mapping_list,
    _profile_metadata,
    _state_sources,
    _synthetic_pci_bus_id,
    _synthetic_uuid,
    build_inventory,
)
from ._version import __version__


REPORT_SCHEMA_VERSION = "fakegpu.smi_report.v1"
DEFAULT_PUBLISHER_DETAIL_LIMIT = 64
MAXIMUM_PUBLISHER_DETAIL_LIMIT = 1024
DEFAULT_MAX_STATE_BYTES = 1024 * 1024
MINIMUM_MAX_STATE_BYTES = 64 * 1024
MAXIMUM_MAX_STATE_BYTES = 64 * 1024 * 1024
# (query kind, query argument, help argument, fields, inventory key), in the
# order --help-query-* options take precedence.
_QUERIES = (
    ("gpu", "query_gpu", "help_query_gpu", GPU_QUERY_FIELDS, "devices"),
    (
        "compute-apps",
        "query_compute_apps",
        "help_query_compute_apps",
        PROCESS_QUERY_FIELDS,
        "processes",
    ),
    (
        "runtime",
        "query_runtime",
        "help_query_runtime",
        RUNTIME_QUERY_FIELDS,
        "runtimes",
    ),
)


def configured_state_path() -> Path | None:
    explicit = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    directory = os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if directory:
        return Path(directory).expanduser().resolve() / f"{os.getpid()}.json"
    return None


def _size_limit(
    value: int | None,
    env_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        try:
            value = int(os.environ.get(env_name, default))
        except ValueError:
            value = default
    return min(maximum, max(minimum, int(value)))


class SmiStatePublisher:
    """Publish FakeCUDA process and device diagnostics for an external viewer."""

    def __init__(
        self,
        path: str | Path,
        snapshot: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float = 0.25,
        runtime_overhead_bytes: int = 0,
        detail_limit: int | None = None,
        max_state_bytes: int | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.snapshot = snapshot
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.runtime_overhead_bytes = max(0, int(runtime_overhead_bytes))
        self.detail_limit = _size_limit(
            detail_limit,
            "FAKEGPU_SMI_DETAIL_LIMIT",
            default=DEFAULT_PUBLISHER_DETAIL_LIMIT,
            minimum=0,
            maximum=MAXIMUM_PUBLISHER_DETAIL_LIMIT,
        )
        self.max_state_bytes = _size_limit(
            max_state_bytes,
            "FAKEGPU_SMI_MAX_STATE_BYTES",
            default=DEFAULT_MAX_STATE_BYTES,
            minimum=MINIMUM_MAX_STATE_BYTES,
            maximum=MAXIMUM_MAX_STATE_BYTES,
        )
        self._software = _software_metadata()
        self._catalogs = _catalog_metadata()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._registered = False
        self._sigterm_handler: Callable[[int, Any], None] | None = None
        self._attempted_writes = 0
        self._successful_writes = 0
        self._failed_writes = 0
        self._last_duration_us = 0
        self._max_duration_us = 0
        self._last_serialized_bytes = 0
        self._last_error = ""
        self._warned_publish_error: tuple[str, str] | None = None
        self._write_lock = threading.Lock()
        self._last_state: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.publish_once(running=True)
        self._thread = threading.Thread(
            target=self._run,
            name="fakegpu-smi-publisher",
            daemon=True,
        )
        self._thread.start()
        if not self._registered:
            atexit.register(self.stop)
            self._record_termination()
            self._registered = True

    def _record_termination(self) -> None:
        """Publish SIGTERM as the exit reason, then terminate as before.

        ``kill``, nvitop and process managers stop processes with SIGTERM,
        whose default action skips ``atexit``. Only the default action is
        replaced; a handler installed by the program is left alone.
        """

        if (
            threading.current_thread() is not threading.main_thread()
            or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        ):
            return

        def terminate(number: int, _frame: Any) -> None:
            self._stop.set()
            # The main thread may hold this lock if the signal arrived during
            # its own publish; write anyway rather than deadlock. Keep the
            # lock until exit so the publisher thread cannot write afterwards.
            self._write_lock.acquire(timeout=0.5)
            if self._last_state is not None:
                final = {
                    **self._last_state,
                    "running": False,
                    "timestamp_ns": time.time_ns(),
                    "exit_signal": signal.Signals(number).name,
                }
                try:
                    _atomic_write_json(self.path, final, max_bytes=self.max_state_bytes)
                except Exception:
                    pass
            signal.signal(number, signal.SIG_DFL)
            os.kill(os.getpid(), number)

        signal.signal(signal.SIGTERM, terminate)
        self._sigterm_handler = terminate

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, 2 * self.interval_seconds))
        self._thread = None
        try:
            self.publish_once(running=False)
        except Exception as exc:
            self._warn_publish_failure(exc, context="final state publish")
        if self._registered:
            atexit.unregister(self.stop)
            if (
                self._sigterm_handler is not None
                and threading.current_thread() is threading.main_thread()
                and signal.getsignal(signal.SIGTERM) is self._sigterm_handler
            ):
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self._sigterm_handler = None
            self._registered = False

    def publish_once(self, *, running: bool) -> dict[str, Any]:
        with self._write_lock:
            return self._publish_locked(running=running)

    def _publish_locked(self, *, running: bool) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        self._attempted_writes += 1
        # Count this write optimistically so the state being published can
        # report it; roll back if the write fails.
        self._successful_writes += 1
        try:
            state, serialized_bytes = self._publish_once(
                running=running
            )
        except Exception as exc:
            self._successful_writes -= 1
            self._failed_writes += 1
            self._last_error = type(exc).__name__
            raise
        else:
            self._last_serialized_bytes = serialized_bytes
            self._last_error = ""
            self._warned_publish_error = None
            return state
        finally:
            self._last_duration_us = (
                time.perf_counter_ns() - started_ns
            ) // 1000
            self._max_duration_us = max(
                self._max_duration_us,
                self._last_duration_us,
            )

    def _warn_publish_failure(
        self,
        exc: Exception,
        *,
        context: str,
    ) -> None:
        signature = (type(exc).__name__, str(exc))
        if signature == self._warned_publish_error:
            return
        self._warned_publish_error = signature
        warnings.warn(
            f"FakeGPU SMI {context} failed for {self.path}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _publish_once(
        self,
        *,
        running: bool,
    ) -> tuple[dict[str, Any], int]:
        raw = self.snapshot()
        hostname = socket.gethostname()
        devices: list[dict[str, Any]] = []
        for item in raw.get("devices") or []:
            devices.append(
                self._device_state(item, raw, hostname, len(devices))
            )
        dispatch_tracking = raw.get("dispatch_tracking")
        dispatch_tracking = (
            dict(dispatch_tracking)
            if isinstance(dispatch_tracking, Mapping)
            else {}
        )
        topology = _modeled_device_topology(devices)
        faults = _modeled_fault_model(
            devices,
            detail_limit=self.detail_limit,
        )
        mig = _modeled_mig_layout(devices)
        state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": time.time_ns(),
            "hostname": hostname,
            "pid": os.getpid(),
            "process_name": _process_name(),
            "runtime": "fakecuda",
            "fakegpu": {
                "version": __version__,
                "runtime": "fakecuda",
                "backend": str(
                    raw.get("runtime_backend") or "unknown"
                ),
                "mode": os.environ.get("FAKEGPU_MODE", "simulate"),
                "oom_policy": os.environ.get(
                    "FAKEGPU_OOM_POLICY",
                    "default",
                ),
                "unsupported_api_policy": os.environ.get(
                    "FAKEGPU_UNSUPPORTED_API",
                    "default",
                ),
                "distributed_mode": os.environ.get(
                    "FAKEGPU_DIST_MODE",
                    "disabled",
                ),
                "memory_tracking_enabled": bool(
                    raw.get("memory_tracking_enabled", True)
                ),
                "dispatch_memory_tracking_enabled": bool(
                    dispatch_tracking.get("enabled", False)
                ),
                **self._catalogs,
            },
            "software": dict(self._software),
            "publisher": {
                "interval_seconds": self.interval_seconds,
                "runtime_overhead_bytes": (
                    self.runtime_overhead_bytes
                ),
                "source": "python_runtime",
                "health": {
                    "attempted_writes": self._attempted_writes,
                    "successful_writes": self._successful_writes,
                    "failed_writes": self._failed_writes,
                    "last_duration_us": self._last_duration_us,
                    "max_duration_us": self._max_duration_us,
                    "last_serialized_bytes": (
                        self._last_serialized_bytes
                    ),
                    "last_error": self._last_error,
                },
                "limits": {
                    "detail_entries": self.detail_limit,
                    "max_state_bytes": self.max_state_bytes,
                },
            },
            "running": running,
            "tracking_confidence": raw.get(
                "tracking_confidence", "C2_torch_tensor_lifetime"
            ),
            "stage": str(
                raw.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            ),
            "allocator_model": str(
                raw.get("allocator_model") or "unknown"
            ),
            "dispatch_tracking": dispatch_tracking,
            "topology": topology,
            "faults": faults,
            "mig": mig,
            "devices": devices,
        }
        serialized_bytes = _atomic_write_json(
            self.path,
            state,
            max_bytes=self.max_state_bytes,
        )
        self._last_state = state
        return state, serialized_bytes

    def _device_state(
        self,
        item: Mapping[str, Any],
        raw: Mapping[str, Any],
        hostname: str,
        index_default: int,
    ) -> dict[str, Any]:
        current = _nonnegative_int(item.get("current_memory"))
        peak = max(
            current,
            _nonnegative_int(
                item.get("peak_memory"),
                default=current,
            ),
        )
        reserved = max(
            current,
            _nonnegative_int(
                item.get("current_reserved_memory"),
                default=current,
            ),
        )
        reserved_peak = max(
            reserved,
            peak,
            _nonnegative_int(
                item.get("peak_reserved_memory"),
                default=reserved,
            ),
        )
        total = _nonnegative_int(item.get("total_memory"))
        reported = reserved + self.runtime_overhead_bytes
        reported_peak = reserved_peak + self.runtime_overhead_bytes
        if total:
            reported = min(total, reported)
            reported_peak = min(total, reported_peak)
        index = _nonnegative_int(
            item.get("index"),
            default=index_default,
        )
        profile_id = str(item.get("profile_id", ""))
        profile = _profile_metadata(item)
        free = max(0, total - reported) if total else None
        headroom = (
            max(0, total - reported_peak) if total else None
        )
        largest_allocations = _mapping_list(
            item.get("largest_allocations")
        )
        largest_allocations_total = len(largest_allocations)
        largest_allocations = largest_allocations[
            : self.detail_limit
        ]
        return {
            "index": index,
            "name": str(item.get("name", "Fake NVIDIA GPU")),
            "profile_id": profile_id,
            "profile": profile,
            "uuid": _synthetic_uuid(hostname, index, profile_id),
            "pci_bus_id": _synthetic_pci_bus_id(index),
            "identity_source": "synthetic",
            "architecture": profile.get("architecture"),
            "compute_capability": profile.get(
                "compute_capability"
            ),
            "compiler_target": profile.get("compiler_target"),
            "total_memory": total,
            "free_memory": free,
            "activity": dict(item.get("activity") or {}),
            **(
                {"transfer": item["transfer"]}
                if isinstance(item.get("transfer"), Mapping)
                else {}
            ),
            "tracked_memory": current,
            "peak_tracked_memory": peak,
            "reserved_memory": reserved,
            "peak_reserved_memory": reserved_peak,
            "inactive_split_bytes": _nonnegative_int(
                item.get("inactive_split_bytes")
            ),
            "segment_count": _nonnegative_int(
                item.get("segment_count")
            ),
            "reported_memory_source": "reserved",
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "reported_memory": reported,
            "reported_peak_memory": reported_peak,
            "headroom_bytes": headroom,
            "headroom_percent": (
                round(headroom / total * 100, 3)
                if total
                else None
            ),
            "allocation_count": _nonnegative_int(
                item.get("allocation_count")
            ),
            "free_count": _nonnegative_int(
                item.get("free_count")
            ),
            "allocator_model": str(
                item.get("allocator_model")
                or raw.get("allocator_model")
                or "unknown"
            ),
            "current_bytes_by_category": _integer_mapping(
                item.get("current_bytes_by_category")
            ),
            "peak_by_stage": _integer_mapping(
                item.get("peak_by_stage")
            ),
            "reserved_peak_by_stage": _integer_mapping(
                item.get("reserved_peak_by_stage")
            ),
            "largest_allocations": largest_allocations,
            "largest_allocations_total": (
                largest_allocations_total
            ),
            "largest_allocations_retained": len(
                largest_allocations
            ),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish_once(running=True)
            except Exception as exc:
                self._warn_publish_failure(
                    exc,
                    context="background state publish",
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Inspect FakeCUDA devices, processes, profiles, memory, runtime "
            "configuration, and modeled topology."
        ),
    )
    parser.add_argument(
        "view",
        nargs="?",
        choices=("topo", "nvlink", "events", "mig"),
        help=(
            "Show an NVIDIA-style modeled topology matrix, NVLink "
            "status, health events, or MIG instances."
        ),
    )
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--state-dir")
    parser.add_argument("--include-exited", action="store_true")
    output_group = parser.add_mutually_exclusive_group()
    add_json_flag_argument(
        output_group,
        help="Emit the complete state and normalized inventory as JSON.",
    )
    output_group.add_argument(
        "-L",
        "--list-gpus",
        action="store_true",
        help="List simulated GPUs with UUID, profile, and compute capability.",
    )
    output_group.add_argument(
        "-q",
        "--detail",
        action="store_true",
        help="Show detailed FakeGPU runtime, profile, allocator, and process data.",
    )
    output_group.add_argument(
        "--query-gpu",
        metavar="FIELDS",
        help="Query comma-separated GPU fields.",
    )
    output_group.add_argument(
        "--query-compute-apps",
        metavar="FIELDS",
        help="Query comma-separated process fields.",
    )
    output_group.add_argument(
        "--query-runtime",
        metavar="FIELDS",
        help="Query comma-separated FakeGPU runtime and publisher fields.",
    )
    parser.add_argument(
        "-m",
        "--matrix",
        action="store_true",
        help="Show the topology matrix; valid with the topo view.",
    )
    parser.add_argument(
        "-s",
        "--status",
        action="store_true",
        help="Show link status; valid with the nvlink view.",
    )
    mig_group = parser.add_mutually_exclusive_group()
    mig_group.add_argument(
        "-lgi",
        "--list-gpu-instances",
        action="store_true",
        help="List modeled GPU instances; valid with the mig view.",
    )
    mig_group.add_argument(
        "-lci",
        "--list-compute-instances",
        action="store_true",
        help="List modeled compute instances; valid with the mig view.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help=(
            "Query output format: csv, csv,noheader, csv,nounits, or json."
        ),
    )
    parser.add_argument(
        "-i",
        "--id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Select GPU index, UUID, PCI bus ID, or profile ID; "
            "comma-separated values are accepted."
        ),
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=_positive_float,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help="Mark running state files older than this threshold as stale.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"FakeGPU-SMI {__version__}",
    )
    parser.add_argument(
        "--help-query-gpu",
        action="store_true",
        help="List supported --query-gpu fields and exit.",
    )
    parser.add_argument(
        "--help-query-compute-apps",
        action="store_true",
        help="List supported --query-compute-apps fields and exit.",
    )
    parser.add_argument(
        "--help-query-runtime",
        action="store_true",
        help="List supported --query-runtime fields and exit.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=_positive_float,
        metavar="SECONDS",
        help="Refresh repeatedly at the given interval; JSON output becomes NDJSON.",
    )
    parser.add_argument(
        "--count",
        type=_positive_int,
        help="Stop after this many refreshes; requires --loop.",
    )
    return parser


def _check_view_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.count is not None and args.loop is None:
        parser.error("--count requires --loop")
    if args.matrix and args.view != "topo":
        parser.error("-m/--matrix requires the topo view")
    if args.status and args.view != "nvlink":
        parser.error("-s/--status requires the nvlink view")
    if (
        args.list_gpu_instances
        or args.list_compute_instances
    ) and args.view != "mig":
        parser.error("-lgi/-lci require the mig view")
    if args.view and (
        args.json
        or args.list_gpus
        or args.detail
        or args.query_gpu
        or args.query_compute_apps
        or args.query_runtime
    ):
        parser.error(
            "topo, nvlink, events, and mig views cannot be combined "
            "with another output mode"
        )
    if args.format is not None and not (
        args.query_gpu or args.query_compute_apps or args.query_runtime
    ):
        parser.error(
            "--format requires --query-gpu, --query-compute-apps, "
            "or --query-runtime"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    for query_kind, _attr, help_attr, available, _key in _QUERIES:
        if getattr(args, help_attr):
            print(_render_query_help(query_kind, available))
            return 0
    _check_view_arguments(args, parser)
    query_format = _parse_query_format(
        args.format or "csv",
        parser=parser,
    )
    query = None
    for query_kind, attr, _help_attr, available, key in _QUERIES:
        value = getattr(args, attr)
        if value is not None:
            fields = _parse_query_fields(value, available=available, parser=parser)
            query = (query_kind, available, key, fields)
            break
    selectors = _device_selectors(args.id)

    explicit_paths, state_dir, fallback_state = _state_sources(
        args.state,
        args.state_dir,
    )
    from .machine.layout import is_machine_state_dir, layout_state, machine_state_dir

    if not (explicit_paths or state_dir or fallback_state):
        # Like nvidia-smi on a real host: show the background machine.
        state_dir = machine_state_dir()
        if state_dir is None:
            parser.error(
                "provide --state, --state-dir, or FAKEGPU_SMI_STATE_PATH, "
                "or start the background machine with 'fakegpu daemon start'"
            )
    # The machine lists its GPUs even when no process uses them.
    show_layout = is_machine_state_dir(state_dir)

    refresh = 0
    saw_states = False
    try:
        while True:
            paths = _discover_state_paths(
                explicit_paths=explicit_paths,
                state_dir=state_dir,
                fallback_state=fallback_state,
            )
            states, errors = _load_states(
                paths,
                include_exited=bool(args.include_exited),
            )
            machine = layout_state() if show_layout else None
            if machine is not None:
                states = [machine, *states]
            saw_states = saw_states or bool(states)
            inventory = build_inventory(
                states,
                device_selectors=selectors,
                stale_after_seconds=args.stale_after_seconds,
            )

            if refresh and not args.json and query_format["kind"] != "json":
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[H")
                else:
                    print()
            if args.json:
                payload = {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "generated_at_ns": time.time_ns(),
                    "inventory": inventory,
                    "states": states,
                    "errors": errors,
                }
                if args.loop is None:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(json.dumps(payload, sort_keys=True))
            elif args.list_gpus:
                print(render_gpu_list(inventory, errors=errors))
            elif args.detail:
                print(render_detail(inventory, errors=errors))
            elif args.view == "topo":
                print(render_topology_matrix(inventory, errors=errors))
            elif args.view == "nvlink":
                print(render_nvlink_status(inventory, errors=errors))
            elif args.view == "events":
                print(render_health_events(inventory, errors=errors))
            elif args.view == "mig":
                print(
                    render_mig_view(
                        inventory,
                        errors=errors,
                        instance_kind=(
                            "gpu"
                            if args.list_gpu_instances
                            else "compute"
                            if args.list_compute_instances
                            else "all"
                        ),
                    )
                )
            elif query is not None:
                query_kind, available, key, fields = query
                print(
                    render_query(
                        inventory[key],
                        fields=fields,
                        available=available,
                        query_kind=query_kind,
                        output_format=query_format,
                    )
                )
            else:
                print(
                    render_table(
                        states,
                        errors=errors,
                        device_selectors=selectors,
                        stale_after_seconds=args.stale_after_seconds,
                        inventory=inventory,
                    )
                )
            sys.stdout.flush()

            refresh += 1
            if args.loop is None:
                break
            if args.count is not None and refresh >= args.count:
                break
            time.sleep(args.loop)
    except KeyboardInterrupt:
        pass
    return 0 if saw_states else 1


def _device_selectors(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    )


def _parse_query_fields(
    value: str | None,
    *,
    available: Mapping[str, QueryField],
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    if value is None:
        return ()
    fields = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    if not fields:
        parser.error("query field list must not be empty")
    unknown = [field for field in fields if field not in available]
    if unknown:
        parser.error(
            "unsupported query field(s): "
            + ", ".join(unknown)
            + "; use the matching --help-query option"
        )
    return fields


def _parse_query_format(
    value: str,
    *,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    parts = tuple(
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    )
    if parts == ("json",):
        return {"kind": "json", "noheader": True, "nounits": True}
    if not parts or parts[0] != "csv":
        parser.error("--format must start with csv or equal json")
    options = set(parts[1:])
    unknown = options - {"noheader", "nounits"}
    if unknown:
        parser.error(
            "unsupported --format option(s): "
            + ", ".join(sorted(unknown))
        )
    return {
        "kind": "csv",
        "noheader": "noheader" in options,
        "nounits": "nounits" in options,
    }


def _render_query_help(
    query_kind: str,
    fields: Mapping[str, QueryField],
) -> str:
    lines = [f"Supported {query_kind} query fields:"]
    for name, spec in fields.items():
        suffix = f" [{spec.unit}]" if spec.unit else ""
        lines.append(f"  {name}{suffix}")
    return "\n".join(lines)


def _software_metadata() -> dict[str, Any]:
    torch_module = sys.modules.get("torch")
    torch_version = (
        str(torch_module.__version__) if torch_module is not None else None
    )
    torch_cuda = (
        torch_module.version.cuda if torch_module is not None else None
    )
    configured_cuda = os.environ.get("FAKEGPU_CUDA_VERSION")
    cuda_version = str(configured_cuda or torch_cuda or "12.1")
    cuda_source = (
        "FAKEGPU_CUDA_VERSION"
        if configured_cuda
        else "torch.version.cuda"
        if torch_cuda
        else "fakecuda_default"
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_build": (
            str(torch_cuda) if torch_cuda is not None else None
        ),
        "cuda_version": cuda_version,
        "cuda_version_source": cuda_source,
        "driver_version": os.environ.get(
            "FAKEGPU_DRIVER_VERSION",
            "simulated",
        ),
    }


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int,
) -> int:
    serialized = (
        json.dumps(payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(serialized) > max_bytes:
        raise ValueError(
            f"serialized state is {len(serialized)} bytes; "
            f"limit is {max_bytes} bytes"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return len(serialized)


def _process_name() -> str:
    # Only the entry script's basename, never the rest of argv: later
    # arguments may carry filesystem paths or secrets, and joining them
    # into a published state file (and a Prometheus label value) would
    # leak them and create unbounded label cardinality.
    if not sys.argv or not sys.argv[0]:
        return "python"
    return os.path.basename(sys.argv[0])


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "expected a finite number greater than zero"
        )
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed
