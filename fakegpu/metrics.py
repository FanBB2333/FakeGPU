from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .smi import (
    DEFAULT_STALE_AFTER_SECONDS,
    _discover_state_paths,
    _load_states,
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

    devices = [
        item
        for item in snapshot.get("devices") or []
        if isinstance(item, Mapping)
    ]
    _metric_family(
        output,
        "fakegpu_device_info",
        "Static identity and current modeled state of a FakeGPU device.",
        [
            (
                {
                    **_device_labels(device),
                    "name": _text(device.get("name")),
                    "architecture": _text(
                        device.get("architecture")
                    ),
                    "compute_capability": _text(
                        device.get("compute_capability")
                    ),
                    "state": _text(device.get("status")),
                    "health": _text(
                        _mapping(device.get("health")).get("status")
                    ),
                    "mig_mode": _text(
                        _mapping(device.get("mig")).get("mode")
                    ),
                },
                1,
            )
            for device in devices
        ],
    )
    memory_samples = []
    process_count_samples = []
    state_age_samples = []
    health_severity_samples = []
    health_event_samples = []
    mig_instance_samples = []
    mig_memory_samples = []
    nvlink_samples = []
    native_samples = []
    for device in devices:
        labels = _device_labels(device)
        memory = _mapping(device.get("memory"))
        for kind in (
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
        ):
            memory_samples.append(
                (
                    {**labels, "kind": kind},
                    _number(memory.get(f"{kind}_bytes")),
                )
            )
        for status, key in (
            ("all", "process_count"),
            ("stale", "stale_process_count"),
            ("exited", "exited_process_count"),
        ):
            process_count_samples.append(
                (
                    {**labels, "status": status},
                    _number(device.get(key)),
                )
            )
        state_age_samples.append(
            (labels, _number(device.get("age_seconds")))
        )

        health = _mapping(device.get("health"))
        health_severity_samples.append(
            (
                labels,
                _severity_rank(health.get("max_severity")),
            )
        )
        health_event_samples.extend(
            [
                (
                    {**labels, "kind": "occurrences"},
                    _number(health.get("event_count")),
                ),
                (
                    {**labels, "kind": "types"},
                    _number(health.get("event_types_total")),
                ),
            ]
        )

        mig = _mapping(device.get("mig"))
        mig_instance_samples.append(
            (labels, _number(mig.get("instance_count")))
        )
        mig_memory_samples.extend(
            [
                (
                    {**labels, "kind": "allocated"},
                    _number(mig.get("allocated_memory_bytes")),
                ),
                (
                    {**labels, "kind": "unallocated"},
                    _number(mig.get("unallocated_memory_bytes")),
                ),
            ]
        )

        topology = _mapping(device.get("topology"))
        nvlink = _mapping(topology.get("nvlink"))
        nvlink_samples.extend(
            [
                (
                    {**labels, "kind": "active_links"},
                    _number(nvlink.get("active_links")),
                ),
                (
                    {**labels, "kind": "peer_count"},
                    _number(nvlink.get("peer_count")),
                ),
                (
                    {
                        **labels,
                        "kind": "aggregate_bandwidth_gbps",
                    },
                    _number(
                        nvlink.get("aggregate_bandwidth_gbps")
                    ),
                ),
            ]
        )

        native = _mapping(device.get("native_activity"))
        for kind in (
            "io_calls",
            "io_bytes",
            "kernel_launches",
            "gemm_calls",
            "gemm_flops",
            "compatibility_events",
            "unsupported_api_calls",
        ):
            native_samples.append(
                (
                    {**labels, "kind": kind},
                    _number(native.get(kind)),
                )
            )

    _metric_family(
        output,
        "fakegpu_device_memory_bytes",
        "Current and peak device memory values from FakeGPU state.",
        memory_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_processes",
        "Published process count by device and state.",
        process_count_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_state_age_seconds",
        "Maximum age of contributing process state for a device.",
        state_age_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_health_severity",
        "Modeled health severity rank from zero to four.",
        health_severity_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_health_events",
        "Modeled health event occurrences and distinct event types.",
        health_event_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_mig_instances",
        "Number of modeled MIG compute instances.",
        mig_instance_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_mig_memory_bytes",
        "Modeled MIG memory capacity allocation.",
        mig_memory_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_nvlink",
        "Modeled NVLink relationships and configured bandwidth.",
        nvlink_samples,
    )
    _metric_family(
        output,
        "fakegpu_device_native_activity",
        "Aggregated native runtime activity; values may reset.",
        native_samples,
    )

    processes = [
        item
        for item in snapshot.get("processes") or []
        if isinstance(item, Mapping)
    ]
    _metric_family(
        output,
        "fakegpu_process_info",
        "Identity and state of a retained FakeGPU process series.",
        [
            (
                {
                    **_process_labels(process),
                    "stage": _text(process.get("stage")),
                    "state": _text(process.get("status")),
                    "runtime": _text(process.get("runtime")),
                    "backend": _text(process.get("backend")),
                },
                1,
            )
            for process in processes
        ],
    )
    process_memory_samples = []
    for process in processes:
        labels = _process_labels(process)
        memory = _mapping(process.get("memory"))
        for kind in (
            "used",
            "reported_peak",
            "tracked",
            "tracked_peak",
            "reserved",
            "reserved_peak",
        ):
            process_memory_samples.append(
                (
                    {**labels, "kind": kind},
                    _number(memory.get(f"{kind}_bytes")),
                )
            )
    _metric_family(
        output,
        "fakegpu_process_memory_bytes",
        "Memory values for bounded retained process series.",
        process_memory_samples,
    )
    _metric_family(
        output,
        "fakegpu_process_state_age_seconds",
        "Age of the state file backing a retained process series.",
        [
            (
                _process_labels(process),
                _number(process.get("age_seconds")),
            )
            for process in processes
        ],
    )

    runtimes = [
        item
        for item in snapshot.get("runtimes") or []
        if isinstance(item, Mapping)
    ]
    _metric_family(
        output,
        "fakegpu_runtime_info",
        "Identity and state of a FakeGPU publisher runtime.",
        [
            (
                {
                    **_runtime_labels(runtime),
                    "state": _text(runtime.get("status")),
                },
                1,
            )
            for runtime in runtimes
        ],
    )
    publisher_write_samples = []
    publisher_duration_samples = []
    publisher_size_samples = []
    runtime_age_samples = []
    for runtime in runtimes:
        labels = _runtime_labels(runtime)
        publisher = _mapping(runtime.get("publisher"))
        health = _mapping(publisher.get("health"))
        for result, key in (
            ("attempted", "attempted_writes"),
            ("successful", "successful_writes"),
            ("failed", "failed_writes"),
        ):
            publisher_write_samples.append(
                (
                    {**labels, "result": result},
                    _number(health.get(key)),
                )
            )
        for kind, key in (
            ("last", "last_duration_us"),
            ("maximum", "max_duration_us"),
        ):
            publisher_duration_samples.append(
                (
                    {**labels, "kind": kind},
                    _number(health.get(key)),
                )
            )
        publisher_size_samples.append(
            (
                labels,
                _number(health.get("last_serialized_bytes")),
            )
        )
        runtime_age_samples.append(
            (labels, _number(runtime.get("age_seconds")))
        )
    _metric_family(
        output,
        "fakegpu_publisher_writes",
        "Publisher write attempts, successes, and failures.",
        publisher_write_samples,
    )
    _metric_family(
        output,
        "fakegpu_publisher_duration_microseconds",
        "Last and maximum state publisher duration.",
        publisher_duration_samples,
    )
    _metric_family(
        output,
        "fakegpu_publisher_state_bytes",
        "Serialized byte size reported by the state publisher.",
        publisher_size_samples,
    )
    _metric_family(
        output,
        "fakegpu_runtime_state_age_seconds",
        "Age of a FakeGPU publisher runtime state.",
        runtime_age_samples,
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

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(list(self._samples))

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
        self.max_process_series = min(
            MAXIMUM_PROCESS_SERIES,
            max(0, int(max_process_series)),
        )
        self.history = MetricsHistory(history_size)
        self._lock = threading.Lock()

    def collect_once(self) -> dict[str, Any]:
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
                inventory = _empty_inventory()
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
            self.history.append(snapshot)
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
        snapshot = self.collector.collect_once()
        with self._latest_lock:
            self._latest = snapshot
        return copy.deepcopy(snapshot)

    def snapshot(self) -> dict[str, Any]:
        with self._latest_lock:
            latest = (
                copy.deepcopy(self._latest)
                if self._latest is not None
                else None
            )
        return latest if latest is not None else self.refresh()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.refresh()
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
        snapshot = self.snapshot()
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
            self.refresh()


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
            snapshot = self.server.application.snapshot()
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
    parser = argparse.ArgumentParser(
        prog="fakegpu metrics",
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one normalized snapshot as JSON.",
    )
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
    args = parser.parse_args(argv)
    if args.serve and args.json:
        parser.error("--json cannot be combined with --serve")

    explicit_paths = [
        Path(value).expanduser().resolve() for value in args.state
    ]
    state_dir_text = (
        args.state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
    )
    state_dir = (
        Path(state_dir_text).expanduser().resolve()
        if state_dir_text
        else None
    )
    fallback_state_text = None
    if not explicit_paths and state_dir is None:
        fallback_state_text = os.environ.get(
            "FAKEGPU_SMI_STATE_PATH"
        )
    fallback_state = (
        Path(fallback_state_text).expanduser().resolve()
        if fallback_state_text
        else None
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
            key: _number(memory.get(key))
            for key in (
                "total_bytes",
                "used_bytes",
                "free_bytes",
                "reported_peak_bytes",
                "tracked_bytes",
                "tracked_peak_bytes",
                "reserved_bytes",
                "reserved_peak_bytes",
                "inactive_split_bytes",
                "headroom_bytes",
            )
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
            for key in (
                "io_calls",
                "io_bytes",
                "kernel_launches",
                "gemm_calls",
                "gemm_flops",
                "compatibility_events",
                "unsupported_api_calls",
            )
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
            key: _number(memory.get(key))
            for key in (
                "used_bytes",
                "reported_peak_bytes",
                "tracked_bytes",
                "tracked_peak_bytes",
                "reserved_bytes",
                "reserved_peak_bytes",
            )
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
        suffix = _prometheus_labels(labels)
        output.append(f"{name}{suffix} {_format_number(value)}")


def _device_labels(
    device: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": _text(device.get("host")),
        "gpu_index": str(_number(device.get("index"))),
        "gpu_uuid": _text(device.get("uuid")),
        "profile_id": _text(device.get("profile_id")),
    }


def _process_labels(
    process: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": _text(process.get("host")),
        "gpu_index": str(_number(process.get("gpu_index"))),
        "gpu_uuid": _text(process.get("gpu_uuid")),
        "pid": str(_number(process.get("pid"))),
        "process_name": _text(process.get("process_name")),
    }


def _runtime_labels(
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "host": _text(runtime.get("host")),
        "pid": str(_number(runtime.get("pid"))),
        "process_name": _text(runtime.get("process_name")),
        "runtime": _text(runtime.get("runtime")),
        "backend": _text(runtime.get("backend")),
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
    return text[: max(0, int(limit))]


def _severity_rank(value: Any) -> int:
    return {
        "none": 0,
        "info": 1,
        "warning": 2,
        "error": 3,
        "critical": 4,
    }.get(_text(value).lower(), 0)


def _empty_inventory() -> dict[str, Any]:
    return {
        "device_count": 0,
        "mig_instance_count": 0,
        "process_count": 0,
        "runtime_count": 0,
        "devices": [],
        "processes": [],
        "runtimes": [],
    }


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "expected a positive number"
        )
    return parsed


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 1 to 65535"
        ) from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 1 to 65535"
        )
    return parsed


def _history_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"history size must be between 1 and {MAXIMUM_HISTORY_SIZE}"
        ) from exc
    if not 1 <= parsed <= MAXIMUM_HISTORY_SIZE:
        raise argparse.ArgumentTypeError(
            f"history size must be between 1 and {MAXIMUM_HISTORY_SIZE}"
        )
    return parsed


def _process_series_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "process series limit must be between "
            f"0 and {MAXIMUM_PROCESS_SERIES}"
        ) from exc
    if not 0 <= parsed <= MAXIMUM_PROCESS_SERIES:
        raise argparse.ArgumentTypeError(
            "process series limit must be between "
            f"0 and {MAXIMUM_PROCESS_SERIES}"
        )
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
