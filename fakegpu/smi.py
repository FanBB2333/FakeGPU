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
            total = int(item.get("total_memory", 0) or 0)
            reported = min(total, current + self.runtime_overhead_bytes) if total else current
            devices.append(
                {
                    "index": int(item.get("index", len(devices))),
                    "name": str(item.get("name", "Fake NVIDIA GPU")),
                    "profile_id": str(item.get("profile_id", "")),
                    "total_memory": total,
                    "tracked_memory": current,
                    "peak_tracked_memory": int(item.get("peak_memory", current) or current),
                    "runtime_overhead_bytes": self.runtime_overhead_bytes,
                    "reported_memory": reported,
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
            "tracking_confidence": raw.get("tracking_confidence", "C2_torch_tensor_lifetime"),
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
    args = parser.parse_args(argv)

    paths = [Path(value).expanduser().resolve() for value in args.state]
    state_dir = args.state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if state_dir:
        paths.extend(sorted(Path(state_dir).expanduser().resolve().glob("*.json")))
    if not paths:
        explicit = os.environ.get("FAKEGPU_SMI_STATE_PATH")
        if explicit:
            paths.append(Path(explicit).expanduser().resolve())
    if not paths:
        parser.error("provide --state, --state-dir, or FAKEGPU_SMI_STATE_PATH")

    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in dict.fromkeys(paths):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            if args.include_exited or bool(state.get("running")):
                states.append(state)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if args.json:
        print(json.dumps({"states": states, "errors": errors}, indent=2, sort_keys=True))
    else:
        print(render_table(states, errors=errors))
    return 0 if states else 1


def render_table(states: Sequence[dict[str, Any]], *, errors: Sequence[str] = ()) -> str:
    device_rows: dict[tuple[str, int], dict[str, Any]] = {}
    process_rows: list[dict[str, Any]] = []
    for state in states:
        for device in state.get("devices") or []:
            key = (str(state.get("hostname", "localhost")), int(device.get("index", 0)))
            aggregate = device_rows.setdefault(
                key,
                {
                    "name": str(device.get("name", "Fake NVIDIA GPU")),
                    "total": int(device.get("total_memory", 0) or 0),
                    "used": 0,
                },
            )
            used = int(device.get("reported_memory", device.get("tracked_memory", 0)) or 0)
            aggregate["used"] += used
            process_rows.append(
                {
                    "host": key[0],
                    "gpu": key[1],
                    "pid": int(state.get("pid", 0) or 0),
                    "name": str(state.get("process_name", "python")),
                    "used": used,
                    "tracked": int(device.get("tracked_memory", 0) or 0),
                    "running": bool(state.get("running")),
                }
            )

    lines = ["FakeGPU-SMI (simulated CUDA memory)"]
    if not device_rows:
        lines.append("No published FakeCUDA processes found.")
    else:
        lines.extend(
            [
                "+------+------------------------------+-------------------------+",
                "| GPU  | Name                         | Memory-Usage            |",
                "+------+------------------------------+-------------------------+",
            ]
        )
        for (_host, index), item in sorted(device_rows.items()):
            lines.append(
                f"| {index:<4} | {item['name'][:28]:<28} | "
                f"{_mib(item['used']):>8} MiB / {_mib(item['total']):>8} MiB |"
            )
        lines.extend(
            [
                "+------+------------------------------+-------------------------+",
                "Processes:",
                "| GPU | PID      | Process                    | Simulated GPU Memory | Tracked tensors |",
                "|---:|---:|---|---:|---:|",
            ]
        )
        for item in sorted(process_rows, key=lambda row: (row["gpu"], row["pid"])):
            suffix = "" if item["running"] else " (exited)"
            lines.append(
                f"| {item['gpu']} | {item['pid']} | `{item['name']}`{suffix} | "
                f"{_mib(item['used'])} MiB | {_mib(item['tracked'])} MiB |"
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
