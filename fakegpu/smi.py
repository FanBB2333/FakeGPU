from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA_VERSION = "fakegpu.smi_state.v1"


def configured_state_path() -> Path | None:
    explicit = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    directory = os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if directory:
        return Path(directory).expanduser().resolve() / f"{os.getpid()}.json"
    return None


class SmiStatePublisher:
    """Publish lightweight FakeCUDA process memory for an external viewer."""

    def __init__(
        self,
        path: str | Path,
        snapshot: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float = 0.25,
        runtime_overhead_bytes: int = 0,
    ):
        self.path = Path(path).expanduser().resolve()
        self.snapshot = snapshot
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.runtime_overhead_bytes = max(0, int(runtime_overhead_bytes))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._registered = False

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
            self._registered = True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, 2 * self.interval_seconds))
        self._thread = None
        try:
            self.publish_once(running=False)
        except Exception:
            pass

    def publish_once(self, *, running: bool) -> dict[str, Any]:
        raw = self.snapshot()
        devices: list[dict[str, Any]] = []
        for item in raw.get("devices") or []:
            current = int(item.get("current_memory", 0) or 0)
            peak = max(current, int(item.get("peak_memory", current) or current))
            total = int(item.get("total_memory", 0) or 0)
            reported = current + self.runtime_overhead_bytes
            reported_peak = peak + self.runtime_overhead_bytes
            if total:
                reported = min(total, reported)
                reported_peak = min(total, reported_peak)
            devices.append(
                {
                    "index": int(item.get("index", len(devices))),
                    "name": str(item.get("name", "Fake NVIDIA GPU")),
                    "profile_id": str(item.get("profile_id", "")),
                    "total_memory": total,
                    "tracked_memory": current,
                    "peak_tracked_memory": peak,
                    "runtime_overhead_bytes": self.runtime_overhead_bytes,
                    "reported_memory": reported,
                    "reported_peak_memory": reported_peak,
                }
            )
        state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": time.time_ns(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "process_name": _process_name(),
            "runtime": "fakecuda",
            "running": bool(running),
            "tracking_confidence": raw.get(
                "tracking_confidence", "C2_torch_tensor_lifetime"
            ),
            "stage": str(
                raw.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            ),
            "devices": devices,
        }
        _atomic_write_json(self.path, state)
        return state

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.publish_once(running=True)
            except Exception:
                continue


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu nvidia-smi",
        description="Display FakeCUDA process memory published by the simulated runtime.",
    )
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--state-dir")
    parser.add_argument("--include-exited", action="store_true")
    parser.add_argument("--json", action="store_true")
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
    args = parser.parse_args(argv)

    if args.count is not None and args.loop is None:
        parser.error("--count requires --loop")

    explicit_paths = [Path(value).expanduser().resolve() for value in args.state]
    state_dir_text = args.state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
    state_dir = Path(state_dir_text).expanduser().resolve() if state_dir_text else None
    fallback_state_text = None
    if not explicit_paths and state_dir is None:
        fallback_state_text = os.environ.get("FAKEGPU_SMI_STATE_PATH")
    fallback_state = (
        Path(fallback_state_text).expanduser().resolve()
        if fallback_state_text
        else None
    )
    if not explicit_paths and state_dir is None and fallback_state is None:
        parser.error("provide --state, --state-dir, or FAKEGPU_SMI_STATE_PATH")

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
            saw_states = saw_states or bool(states)

            if refresh and not args.json:
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[H")
                else:
                    print()
            if args.json:
                payload = {"states": states, "errors": errors}
                if args.loop is None:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(json.dumps(payload, sort_keys=True))
            else:
                print(render_table(states, errors=errors))
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


def _discover_state_paths(
    *,
    explicit_paths: Sequence[Path],
    state_dir: Path | None,
    fallback_state: Path | None,
) -> list[Path]:
    paths = list(explicit_paths)
    if state_dir is not None:
        paths.extend(sorted(state_dir.glob("*.json")))
    if fallback_state is not None:
        paths.append(fallback_state)
    return list(dict.fromkeys(paths))


def _load_states(
    paths: Sequence[Path],
    *,
    include_exited: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("state root must be an object")
            if state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            if include_exited or bool(state.get("running")):
                states.append(state)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return states, errors


def render_table(
    states: Sequence[dict[str, Any]], *, errors: Sequence[str] = ()
) -> str:
    device_rows: dict[tuple[str, int], dict[str, Any]] = {}
    process_rows: list[dict[str, Any]] = []
    for state in states:
        host = str(state.get("hostname", "localhost"))
        stage = str(state.get("stage") or "unknown")
        confidence = str(state.get("tracking_confidence") or "unknown")
        for device in state.get("devices") or []:
            key = (host, int(device.get("index", 0)))
            aggregate = device_rows.setdefault(
                key,
                {
                    "names": set(),
                    "profiles": set(),
                    "total": 0,
                    "used": 0,
                },
            )
            aggregate["names"].add(str(device.get("name", "Fake NVIDIA GPU")))
            profile_id = str(device.get("profile_id", "") or "unknown")
            aggregate["profiles"].add(profile_id)
            total = int(device.get("total_memory", 0) or 0)
            aggregate["total"] = max(int(aggregate["total"]), total)
            used = int(
                device.get("reported_memory", device.get("tracked_memory", 0)) or 0
            )
            tracked = int(device.get("tracked_memory", 0) or 0)
            peak_tracked = max(
                tracked,
                int(device.get("peak_tracked_memory", tracked) or tracked),
            )
            reported_peak = device.get("reported_peak_memory")
            if reported_peak is None:
                reported_peak = peak_tracked + int(
                    device.get("runtime_overhead_bytes", 0) or 0
                )
                if total:
                    reported_peak = min(total, reported_peak)
            reported_peak = max(used, int(reported_peak or used))
            aggregate["used"] += used
            process_rows.append(
                {
                    "host": key[0],
                    "gpu": key[1],
                    "pid": int(state.get("pid", 0) or 0),
                    "name": str(state.get("process_name", "python")),
                    "profile": profile_id,
                    "stage": stage,
                    "confidence": confidence,
                    "used": used,
                    "peak": reported_peak,
                    "tracked": tracked,
                    "tracked_peak": peak_tracked,
                    "running": bool(state.get("running")),
                }
            )

    lines = ["FakeGPU-SMI (simulated CUDA memory)"]
    if not device_rows:
        lines.append("No published FakeCUDA processes found.")
    else:
        lines.extend(
            [
                "Devices:",
                "| Host | GPU | Profile | Name | Current / Total |",
                "|---|---:|---|---|---:|",
            ]
        )
        for (host, index), item in sorted(device_rows.items()):
            names = ", ".join(sorted(item["names"]))
            profiles = ", ".join(sorted(item["profiles"]))
            lines.append(
                f"| {_table_cell(host)} | {index} | {_table_cell(profiles)} | "
                f"{_table_cell(names)} | {_mib(item['used'])} MiB / "
                f"{_mib(item['total'])} MiB |"
            )
        lines.extend(
            [
                "Processes:",
                "| Host | GPU | Profile | PID | Process | Stage | Simulated current | Simulated peak | Tracked current | Tracked peak | Confidence |",
                "|---|---:|---|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in sorted(
            process_rows,
            key=lambda row: (row["host"], row["gpu"], row["pid"]),
        ):
            suffix = "" if item["running"] else " (exited)"
            lines.append(
                f"| {_table_cell(item['host'])} | {item['gpu']} | "
                f"{_table_cell(item['profile'])} | {item['pid']} | "
                f"{_table_cell(item['name'] + suffix)} | {_table_cell(item['stage'])} | "
                f"{_mib(item['used'])} MiB | {_mib(item['peak'])} MiB | "
                f"{_mib(item['tracked'])} MiB | {_mib(item['tracked_peak'])} MiB | "
                f"{_table_cell(item['confidence'])} |"
            )
    for error in errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_name() -> str:
    if not sys.argv:
        return "python"
    return " ".join(sys.argv[:3])


def _mib(value: int) -> int:
    return int(round(int(value) / 2**20))


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
