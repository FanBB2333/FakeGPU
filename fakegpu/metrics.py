from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._cli import (
    add_json_flag_argument,
    bounded_argument,
    command_prog,
)
from ._smi_environment import FAULT_SEVERITY_RANK
from ._smi_state import (
    DEFAULT_STALE_AFTER_SECONDS,
    NATIVE_ACTIVITY_COUNTERS,
    _discover_state_paths,
    _load_states,
    _state_sources,
    build_inventory,
)


SNAPSHOT_SCHEMA_VERSION = "fakegpu.metrics_snapshot.v1"
HISTORY_SCHEMA_VERSION = "fakegpu.metrics_history.v1"
DEFAULT_HISTORY_SIZE = 300
MAXIMUM_HISTORY_SIZE = 1440
DEFAULT_MAX_PROCESS_SERIES = 128
MAXIMUM_PROCESS_SERIES = 256
MAXIMUM_RETAINED_ERRORS = 64
MAXIMUM_ERROR_LENGTH = 512
MAXIMUM_LABEL_LENGTH = 256
DEFAULT_PORT = 9400
DEFAULT_COLLECTION_INTERVAL_SECONDS = 1.0
PROMETHEUS_CONTENT_TYPE = (
    "text/plain; version=0.0.4; charset=utf-8"
)
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
DEVICE_MEMORY_KINDS = (
    "total",
    "used",
    "free",
    "reported_peak",
    "tracked",
    "tracked_peak",
    "reserved",
    "reserved_peak",
    "inactive_split",
    "headroom",
)
PROCESS_MEMORY_KINDS = (
    "used",
    "reported_peak",
    "tracked",
    "tracked_peak",
    "reserved",
    "reserved_peak",
)

# Each family is (name, help, samples); samples(record) returns the
# (extra labels, value) pairs of one device, process, or runtime record.
DEVICE_FAMILIES = (
    (
        "fakegpu_device_memory_bytes",
        "Current and peak device memory values from FakeGPU state.",
        lambda device: [
            ({"kind": kind}, device["memory"][f"{kind}_bytes"])
            for kind in DEVICE_MEMORY_KINDS
        ],
    ),
    (
        "fakegpu_device_processes",
        "Published process count by device and state.",
        lambda device: [
            ({"status": status}, device[key])
            for status, key in (
                ("all", "process_count"),
                ("stale", "stale_process_count"),
                ("exited", "exited_process_count"),
            )
        ],
    ),
    (
        "fakegpu_device_state_age_seconds",
        "Maximum age of contributing process state for a device.",
        lambda device: [({}, device["age_seconds"])],
    ),
    (
        "fakegpu_device_health_severity",
        "Modeled health severity rank from zero to four.",
        lambda device: [
            ({}, _severity_rank(device["health"]["max_severity"]))
        ],
    ),
    (
        "fakegpu_device_health_events",
        "Modeled health event occurrences and distinct event types.",
        lambda device: [
            ({"kind": "occurrences"}, device["health"]["event_count"]),
            ({"kind": "types"}, device["health"]["event_types_total"]),
        ],
    ),
    (
        "fakegpu_device_mig_instances",
        "Number of modeled MIG compute instances.",
        lambda device: [({}, device["mig"]["instance_count"])],
    ),
    (
        "fakegpu_device_mig_memory_bytes",
        "Modeled MIG memory capacity allocation.",
        lambda device: [
            ({"kind": "allocated"}, device["mig"]["allocated_memory_bytes"]),
            (
                {"kind": "unallocated"},
                device["mig"]["unallocated_memory_bytes"],
            ),
        ],
    ),
    (
        "fakegpu_device_nvlink",
        "Modeled NVLink relationships and configured bandwidth.",
        lambda device: [
            ({"kind": kind}, device["topology"]["nvlink"][kind])
            for kind in (
                "active_links",
                "peer_count",
                "aggregate_bandwidth_gbps",
            )
        ],
    ),
    (
        "fakegpu_device_native_activity",
        "Aggregated native runtime activity; values may reset.",
        lambda device: [
            ({"kind": kind}, device["native_activity"][kind])
            for kind in NATIVE_ACTIVITY_COUNTERS
        ],
    ),
)
PROCESS_FAMILIES = (
    (
        "fakegpu_process_memory_bytes",
        "Memory values for bounded retained process series.",
        lambda process: [
            ({"kind": kind}, process["memory"][f"{kind}_bytes"])
            for kind in PROCESS_MEMORY_KINDS
        ],
    ),
    (
        "fakegpu_process_state_age_seconds",
        "Age of the state file backing a retained process series.",
        lambda process: [({}, process["age_seconds"])],
    ),
)
RUNTIME_FAMILIES = (
    (
        "fakegpu_publisher_writes",
        "Publisher write attempts, successes, and failures.",
        lambda runtime: [
            ({"result": result}, runtime["publisher"]["health"][key])
            for result, key in (
                ("attempted", "attempted_writes"),
                ("successful", "successful_writes"),
                ("failed", "failed_writes"),
            )
        ],
    ),
    (
        "fakegpu_publisher_duration_microseconds",
        "Last and maximum state publisher duration.",
        lambda runtime: [
            ({"kind": kind}, runtime["publisher"]["health"][key])
            for kind, key in (
                ("last", "last_duration_us"),
                ("maximum", "max_duration_us"),
            )
        ],
    ),
    (
        "fakegpu_publisher_state_bytes",
        "Serialized byte size reported by the state publisher.",
        lambda runtime: [
            ({}, runtime["publisher"]["health"]["last_serialized_bytes"])
        ],
    ),
    (
        "fakegpu_runtime_state_age_seconds",
        "Age of a FakeGPU publisher runtime state.",
        lambda runtime: [({}, runtime["age_seconds"])],
    ),
)


def build_metrics_snapshot(
    inventory: Mapping[str, Any],
    *,
    errors: Sequence[str] = (),
    generated_at_ns: int | None = None,
    source_state_count: int | None = None,
    max_process_series: int = DEFAULT_MAX_PROCESS_SERIES,
) -> dict[str, Any]:
    """Build a bounded monitoring snapshot from normalized SMI inventory."""

    timestamp_ns = int(
        generated_at_ns
        if generated_at_ns is not None
        else time.time_ns()
    )
    process_limit = min(
        MAXIMUM_PROCESS_SERIES,
        max(0, int(max_process_series)),
    )
    raw_processes = [
        item
        for item in inventory.get("processes") or []
        if isinstance(item, Mapping)
    ]
    raw_processes.sort(
        key=lambda item: (
            -_number(
                _mapping(item.get("memory")).get("used_bytes")
            ),
            _text(item.get("host")),
            _number(item.get("gpu_index")),
            _number(item.get("pid")),
        )
    )
    retained_processes = raw_processes[:process_limit]

    devices = [
        _device_record(item)
        for item in inventory.get("devices") or []
        if isinstance(item, Mapping)
    ]
    processes = [
        _process_record(item) for item in retained_processes
    ]
    runtimes = [
        _runtime_record(item)
        for item in inventory.get("runtimes") or []
        if isinstance(item, Mapping)
    ]
    retained_errors = [
        _text(error, limit=MAXIMUM_ERROR_LENGTH)
        for error in errors[:MAXIMUM_RETAINED_ERRORS]
    ]
    error_count = len(errors)
    state_count = (
        _number(source_state_count)
        if source_state_count is not None
        else _number(inventory.get("runtime_count"))
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at_ns": timestamp_ns,
        "scrape_success": error_count == 0,
        "error_count": error_count,
        "errors_retained": len(retained_errors),
        "errors": retained_errors,
        "source_state_count": state_count,
        "device_count": len(devices),
        "process_series_total": len(raw_processes),
        "process_series_retained": len(processes),
        "process_series_dropped": (
            len(raw_processes) - len(processes)
        ),
        "runtime_count": len(runtimes),
        "devices": devices,
        "processes": processes,
        "runtimes": runtimes,
    }


def render_prometheus(
    snapshot: Mapping[str, Any],
    *,
    history_sample_count: int = 1,
) -> str:
    """Render a metrics snapshot in Prometheus text exposition format."""

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported metrics snapshot schema")
    output: list[str] = []

    _metric_family(
        output,
        "fakegpu_exporter_scrape_success",
        "Whether the latest FakeGPU state scrape completed without errors.",
        [({}, int(bool(snapshot.get("scrape_success"))))],
    )
    _metric_family(
        output,
        "fakegpu_exporter_scrape_errors",
        "Number of state files that failed during the latest scrape.",
        [({}, _number(snapshot.get("error_count")))],
    )
    _metric_family(
        output,
        "fakegpu_exporter_source_states",
        "Number of valid FakeGPU process state files in the latest scrape.",
        [({}, _number(snapshot.get("source_state_count")))],
    )
    _metric_family(
        output,
        "fakegpu_exporter_history_samples",
        "Number of bounded in-memory history samples retained.",
        [({}, max(0, int(history_sample_count)))],
    )
    _metric_family(
        output,
        "fakegpu_exporter_snapshot_timestamp_seconds",
        "Unix timestamp of the latest metrics snapshot.",
        [
            (
                {},
                _number(snapshot.get("generated_at_ns"))
                / 1_000_000_000,
            )
        ],
    )
    _metric_family(
        output,
        "fakegpu_exporter_process_series",
        "Process series selected or dropped by the cardinality limit.",
        [
            (
                {"kind": "total"},
                _number(snapshot.get("process_series_total")),
            ),
            (
                {"kind": "retained"},
                _number(snapshot.get("process_series_retained")),
            ),
            (
                {"kind": "dropped"},
                _number(snapshot.get("process_series_dropped")),
            ),
        ],
    )

    devices = snapshot["devices"]
    _metric_family(
        output,
        "fakegpu_device_info",
        "Static identity and current modeled state of a FakeGPU device.",
        [
            (
                {
                    **_device_labels(device),
                    "name": device["name"],
                    "architecture": device["architecture"],
                    "compute_capability": device["compute_capability"],
                    "state": device["status"],
                    "health": device["health"]["status"],
                    "mig_mode": device["mig"]["mode"],
                },
                1,
            )
            for device in devices
        ],
    )
    _entity_families(
        output,
        DEVICE_FAMILIES,
        devices,
        [_device_labels(device) for device in devices],
    )

    processes = snapshot["processes"]
    _metric_family(
        output,
        "fakegpu_process_info",
        "Identity and state of a retained FakeGPU process series.",
        [
            (
                {
                    **_process_labels(process),
                    "stage": process["stage"],
                    "state": process["status"],
                    "runtime": process["runtime"],
                    "backend": process["backend"],
                },
                1,
            )
            for process in processes
        ],
    )
    _entity_families(
        output,
        PROCESS_FAMILIES,
        processes,
        [_process_labels(process) for process in processes],
    )

    runtimes = snapshot["runtimes"]
    _metric_family(
        output,
        "fakegpu_runtime_info",
        "Identity and state of a FakeGPU publisher runtime.",
        [
            (
                {
                    **_runtime_labels(runtime),
                    "state": runtime["status"],
                },
                1,
            )
            for runtime in runtimes
        ],
    )
    _entity_families(
        output,
        RUNTIME_FAMILIES,
        runtimes,
        [_runtime_labels(runtime) for runtime in runtimes],
    )
    return "\n".join(output) + "\n"


class MetricsHistory:
    """Thread-safe bounded in-memory metrics snapshot history."""

    def __init__(self, max_samples: int = DEFAULT_HISTORY_SIZE):
        self.max_samples = min(
            MAXIMUM_HISTORY_SIZE,
            max(1, int(max_samples)),
        )
        self._samples: deque[dict[str, Any]] = deque(
            maxlen=self.max_samples
        )
        self._lock = threading.Lock()

    def append(self, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            self._samples.append(copy.deepcopy(dict(snapshot)))

    def _append_owned(self, snapshot: dict[str, Any]) -> None:
        """Retain a snapshot whose ownership has transferred to history."""

        with self._lock:
            self._samples.append(snapshot)

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            samples = list(self._samples)
        # Owned snapshots are never mutated after publication. Copying outside
        # the lock lets collection continue while a history response is built.
        return copy.deepcopy(samples)

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class MetricsCollector:
    """Read SMI state files and retain bounded normalized history."""

    def __init__(
        self,
        *,
        explicit_paths: Sequence[Path] = (),
        state_dir: Path | None = None,
        fallback_state: Path | None = None,
        include_exited: bool = False,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        max_process_series: int = DEFAULT_MAX_PROCESS_SERIES,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ):
        self.explicit_paths = tuple(
            Path(path).expanduser().resolve()
            for path in explicit_paths
        )
        self.state_dir = (
            Path(state_dir).expanduser().resolve()
            if state_dir is not None
            else None
        )
        self.fallback_state = (
            Path(fallback_state).expanduser().resolve()
            if fallback_state is not None
            else None
        )
        self.include_exited = bool(include_exited)
        self.stale_after_seconds = max(
            0.001,
            float(stale_after_seconds),
        )
        self.max_process_series = max_process_series
        self.history = MetricsHistory(history_size)
        self._lock = threading.Lock()

    def collect_once(self) -> dict[str, Any]:
        return copy.deepcopy(self._collect_once_owned())

    def _collect_once_owned(self) -> dict[str, Any]:
        with self._lock:
            generated_at_ns = time.time_ns()
            paths = _discover_state_paths(
                explicit_paths=self.explicit_paths,
                state_dir=self.state_dir,
                fallback_state=self.fallback_state,
            )
            states, errors = _load_states(
                paths,
                include_exited=self.include_exited,
            )
            try:
                inventory = build_inventory(
                    states,
                    stale_after_seconds=self.stale_after_seconds,
                    now_ns=generated_at_ns,
                )
            except (KeyError, TypeError, ValueError) as exc:
                inventory = build_inventory([])
                errors.append(
                    "inventory: "
                    f"{type(exc).__name__}: "
                    f"{_text(exc, limit=MAXIMUM_ERROR_LENGTH)}"
                )
            snapshot = build_metrics_snapshot(
                inventory,
                errors=errors,
                generated_at_ns=generated_at_ns,
                source_state_count=len(states),
                max_process_series=self.max_process_series,
            )
            self.history._append_owned(snapshot)
            return snapshot


class MetricsApplication:
    """Own the latest snapshot and its periodic collection thread."""

    def __init__(
        self,
        collector: MetricsCollector,
        *,
        interval_seconds: float = (
            DEFAULT_COLLECTION_INTERVAL_SECONDS
        ),
    ):
        self.collector = collector
        self.interval_seconds = max(
            0.05,
            float(interval_seconds),
        )
        self._latest: dict[str, Any] | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> dict[str, Any]:
        return copy.deepcopy(self._refresh_owned())

    def _refresh_owned(self) -> dict[str, Any]:
        snapshot = self.collector._collect_once_owned()
        with self._latest_lock:
            self._latest = snapshot
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._snapshot_owned())

    def _snapshot_owned(self) -> dict[str, Any]:
        """Borrow the published snapshot for internal read-only rendering."""
        with self._latest_lock:
            latest = self._latest
        return latest if latest is not None else self._refresh_owned()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._refresh_owned()
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="fakegpu-metrics-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, 2 * self.interval_seconds))
        self._thread = None

    def history_payload(self) -> dict[str, Any]:
        samples = self.collector.history.snapshots()
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "max_samples": self.collector.history.max_samples,
            "sample_count": len(samples),
            "samples": samples,
        }

    def health_payload(self) -> dict[str, Any]:
        snapshot = self._snapshot_owned()
        status = (
            "ok"
            if snapshot.get("scrape_success")
            and _number(snapshot.get("source_state_count")) > 0
            else "empty"
            if snapshot.get("scrape_success")
            else "degraded"
        )
        return {
            "status": status,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "generated_at_ns": snapshot.get("generated_at_ns"),
            "scrape_success": snapshot.get("scrape_success"),
            "source_state_count": snapshot.get(
                "source_state_count"
            ),
            "device_count": snapshot.get("device_count"),
            "process_series_retained": snapshot.get(
                "process_series_retained"
            ),
            "error_count": snapshot.get("error_count"),
            "history_sample_count": len(self.collector.history),
        }

    def _collect_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._refresh_owned()


class MetricsHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: MetricsApplication,
    ):
        self.application = application
        super().__init__(server_address, MetricsRequestHandler)


class MetricsRequestHandler(BaseHTTPRequestHandler):
    server: MetricsHttpServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/metrics":
            snapshot = self.server.application._snapshot_owned()
            payload = render_prometheus(
                snapshot,
                history_sample_count=len(
                    self.server.application.collector.history
                ),
            ).encode("utf-8")
            self._send(200, PROMETHEUS_CONTENT_TYPE, payload)
            return
        if path == "/healthz":
            self._send_json(
                200,
                self.server.application.health_payload(),
            )
            return
        if path == "/api/v1/history":
            self._send_json(
                200,
                self.server.application.history_payload(),
            )
            return
        self._send_json(
            404,
            {"error": "not_found", "path": path},
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, Any],
    ) -> None:
        body = (
            json.dumps(payload, sort_keys=True) + "\n"
        ).encode("utf-8")
        self._send(status, JSON_CONTENT_TYPE, body)

    def _send(
        self,
        status: int,
        content_type: str,
        payload: bytes,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args, parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Export bounded FakeGPU device, process, runtime, MIG, "
            "topology, and health metrics."
        ),
    )
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--state-dir")
    parser.add_argument("--include-exited", action="store_true")
    parser.add_argument(
        "--stale-after-seconds",
        type=_positive_float,
        default=DEFAULT_STALE_AFTER_SECONDS,
    )
    parser.add_argument(
        "--max-process-series",
        type=_process_series_limit,
        default=DEFAULT_MAX_PROCESS_SERIES,
        help=(
            "Retain at most this many highest-memory process series "
            f"(default: {DEFAULT_MAX_PROCESS_SERIES}, maximum: "
            f"{MAXIMUM_PROCESS_SERIES}, 0 disables process series)."
        ),
    )
    parser.add_argument(
        "--history-size",
        type=_history_size,
        default=DEFAULT_HISTORY_SIZE,
        help=(
            "Retain this many in-memory snapshots "
            f"(default: {DEFAULT_HISTORY_SIZE}, maximum: "
            f"{MAXIMUM_HISTORY_SIZE})."
        ),
    )
    add_json_flag_argument(parser, help="Emit one normalized snapshot as JSON.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve /metrics, /healthz, and /api/v1/history.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host for --serve (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"HTTP bind port for --serve (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=DEFAULT_COLLECTION_INTERVAL_SECONDS,
        help="Collection interval in seconds for --serve.",
    )

    return parser


def _run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.serve and args.json:
        parser.error("--json cannot be combined with --serve")

    explicit_paths, state_dir, fallback_state = _state_sources(
        args.state,
        args.state_dir,
    )
    if (
        not explicit_paths
        and state_dir is None
        and fallback_state is None
    ):
        parser.error(
            "provide --state, --state-dir, "
            "FAKEGPU_SMI_STATE_PATH, or FAKEGPU_SMI_STATE_DIR"
        )

    collector = MetricsCollector(
        explicit_paths=explicit_paths,
        state_dir=state_dir,
        fallback_state=fallback_state,
        include_exited=args.include_exited,
        stale_after_seconds=args.stale_after_seconds,
        max_process_series=args.max_process_series,
        history_size=args.history_size,
    )
    if not args.serve:
        snapshot = collector.collect_once()
        if args.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            print(
                render_prometheus(
                    snapshot,
                    history_sample_count=len(collector.history),
                ),
                end="",
            )
        return (
            0
            if snapshot["scrape_success"]
            and snapshot["source_state_count"] > 0
            else 1
        )

    application = MetricsApplication(
        collector,
        interval_seconds=args.interval,
    )
    try:
        server = MetricsHttpServer(
            (str(args.host), int(args.port)),
            application,
        )
    except OSError as exc:
        print(f"error: could not start metrics server: {exc}", file=sys.stderr)
        return 2
    application.start()
    host, port = server.server_address[:2]
    print(
        f"FakeGPU metrics listening on http://{host}:{port}",
        file=sys.stderr,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.stop()
    return 0

def _device_record(item: Mapping[str, Any]) -> dict[str, Any]:
    memory = _mapping(item.get("memory"))
    health = _mapping(item.get("health"))
    mig = _mapping(item.get("mig"))
    topology = _mapping(item.get("topology"))
    nvlink = _mapping(topology.get("nvlink"))
    native = _mapping(item.get("native_activity"))
    state = _mapping(item.get("state"))
    return {
        "host": _text(item.get("host")),
        "index": _number(item.get("index")),
        "name": _text(item.get("name")),
        "uuid": _text(item.get("uuid")),
        "pci_bus_id": _text(item.get("pci_bus_id")),
        "profile_id": _text(item.get("profile_id")),
        "architecture": _text(item.get("architecture")),
        "compute_capability": _text(
            item.get("compute_capability")
        ),
        "status": _text(state.get("status"), default="unknown"),
        "age_seconds": _optional_number(
            state.get("max_age_seconds")
        ),
        "process_count": _number(item.get("process_count")),
        "stale_process_count": _number(
            item.get("stale_process_count")
        ),
        "exited_process_count": _number(
            item.get("exited_process_count")
        ),
        "memory": {
            f"{kind}_bytes": _number(memory.get(f"{kind}_bytes"))
            for kind in DEVICE_MEMORY_KINDS
        },
        "health": {
            "status": _text(
                health.get("status"),
                default="unknown",
            ),
            "hardware_health": _text(
                health.get("hardware_health"),
                default="unobserved",
            ),
            "max_severity": _text(
                health.get("max_severity"),
                default="none",
            ),
            "event_count": _number(health.get("event_count")),
            "event_types_total": _number(
                health.get("event_types_total")
            ),
        },
        "mig": {
            "mode": _text(mig.get("mode"), default="disabled"),
            "instance_count": _number(
                mig.get("instance_count")
            ),
            "allocated_memory_bytes": _number(
                mig.get("allocated_memory_bytes")
            ),
            "unallocated_memory_bytes": _number(
                mig.get("unallocated_memory_bytes")
            ),
        },
        "topology": {
            "source": _text(topology.get("source")),
            "nvlink": {
                "active_links": _number(
                    nvlink.get("active_links")
                ),
                "peer_count": _number(
                    nvlink.get("peer_count")
                ),
                "aggregate_bandwidth_gbps": _number(
                    nvlink.get("aggregate_bandwidth_gbps")
                ),
            },
        },
        "native_activity": {
            key: _number(native.get(key))
            for key in NATIVE_ACTIVITY_COUNTERS
        },
    }


def _process_record(item: Mapping[str, Any]) -> dict[str, Any]:
    memory = _mapping(item.get("memory"))
    fakegpu = _mapping(item.get("fakegpu"))
    return {
        "host": _text(item.get("host")),
        "gpu_index": _number(item.get("gpu_index")),
        "gpu_uuid": _text(item.get("gpu_uuid")),
        "profile_id": _text(item.get("profile_id")),
        "pid": _number(item.get("pid")),
        "process_name": _text(item.get("process_name")),
        "stage": _text(item.get("stage"), default="unknown"),
        "status": _text(item.get("status"), default="unknown"),
        "age_seconds": _optional_number(item.get("age_seconds")),
        "runtime": _text(fakegpu.get("runtime")),
        "backend": _text(fakegpu.get("backend")),
        "memory": {
            f"{kind}_bytes": _number(memory.get(f"{kind}_bytes"))
            for kind in PROCESS_MEMORY_KINDS
        },
    }


def _runtime_record(item: Mapping[str, Any]) -> dict[str, Any]:
    fakegpu = _mapping(item.get("fakegpu"))
    publisher = _mapping(item.get("publisher"))
    health = _mapping(publisher.get("health"))
    return {
        "host": _text(item.get("host")),
        "pid": _number(item.get("pid")),
        "process_name": _text(item.get("process_name")),
        "runtime": _text(fakegpu.get("runtime")),
        "backend": _text(fakegpu.get("backend")),
        "status": _text(item.get("status"), default="unknown"),
        "age_seconds": _optional_number(item.get("age_seconds")),
        "publisher": {
            "source": _text(publisher.get("source")),
            "health": {
                key: _number(health.get(key))
                for key in (
                    "attempted_writes",
                    "successful_writes",
                    "failed_writes",
                    "last_duration_us",
                    "max_duration_us",
                    "last_serialized_bytes",
                )
            },
        },
    }


def _metric_family(
    output: list[str],
    name: str,
    help_text: str,
    samples: Sequence[
        tuple[Mapping[str, Any], int | float]
    ],
) -> None:
    output.append(f"# HELP {name} {help_text}")
    output.append(f"# TYPE {name} gauge")
    for labels, value in samples:
        if value is None:
            # An unknown value is left out, not exported as 0: an age of 0
            # would read as fresh state.
            continue
        suffix = _prometheus_labels(labels)
        output.append(f"{name}{suffix} {_format_number(value)}")


def _entity_families(
    output: list[str],
    families: Sequence[tuple[str, str, Any]],
    items: Sequence[Mapping[str, Any]],
    label_sets: Sequence[Mapping[str, str]],
) -> None:
    for name, help_text, samples in families:
        _metric_family(
            output,
            name,
            help_text,
            [
                ({**labels, **extra}, value)
                for item, labels in zip(items, label_sets)
                for extra, value in samples(item)
            ],
        )


def _device_labels(
    device: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": device["host"],
        "gpu_index": str(device["index"]),
        "gpu_uuid": device["uuid"],
        "profile_id": device["profile_id"],
    }


def _process_labels(
    process: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": process["host"],
        "gpu_index": str(process["gpu_index"]),
        "gpu_uuid": process["gpu_uuid"],
        "pid": str(process["pid"]),
        "process_name": process["process_name"],
    }


def _runtime_labels(
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": runtime["host"],
        "pid": str(runtime["pid"]),
        "process_name": runtime["process_name"],
        "runtime": runtime["runtime"],
        "backend": runtime["backend"],
    }


def _prometheus_labels(labels: Mapping[str, Any]) -> str:
    if not labels:
        return ""
    values = [
        f'{key}="{_escape_label(_text(value))}"'
        for key, value in sorted(labels.items())
    ]
    return "{" + ",".join(values) + "}"


def _escape_label(value: str) -> str:
    # Label values are UTF-8. A lone surrogate, such as an undecodable byte in
    # a published sys.argv[0], would otherwise fail encoding the whole page.
    value = value.encode("utf-8", "replace").decode("utf-8")
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _format_number(value: Any) -> str:
    number = _number(value)
    if isinstance(number, int):
        return str(number)
    return format(number, ".15g")


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number < 0:
        return 0
    return number


def _optional_number(value: Any) -> int | float | None:
    return None if value is None else _number(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(
    value: Any,
    *,
    default: str = "",
    limit: int = MAXIMUM_LABEL_LENGTH,
) -> str:
    text = str(value) if value is not None else default
    return text[:limit]


def _severity_rank(value: Any) -> int:
    return FAULT_SEVERITY_RANK.get(_text(value).lower(), 0)


_positive_float = bounded_argument(
    float, lambda v: math.isfinite(v) and v > 0, "expected a positive number"
)
_port = bounded_argument(
    int, lambda v: 1 <= v <= 65535, "port must be an integer from 1 to 65535"
)
_history_size = bounded_argument(
    int,
    lambda v: 1 <= v <= MAXIMUM_HISTORY_SIZE,
    f"history size must be between 1 and {MAXIMUM_HISTORY_SIZE}",
)
_process_series_limit = bounded_argument(
    int,
    lambda v: 0 <= v <= MAXIMUM_PROCESS_SERIES,
    f"process series limit must be between 0 and {MAXIMUM_PROCESS_SERIES}",
)


if __name__ == "__main__":
    raise SystemExit(main())
