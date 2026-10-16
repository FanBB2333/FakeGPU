"""State, events and control methods of the FakeGPU machine daemon.

Every client goes through :meth:`MachineService.dispatch`, which takes and
returns plain JSON objects. The daemon serves it over a Unix socket; another
transport (for example HTTP for a web management page) only needs to pass
requests to the same method. Protocol ``fakegpu.machine.v1``::

    request   {"v": 1, "id": 7, "method": "status", "params": {}}
    response  {"v": 1, "id": 7, "ok": true, "result": {...}}
              {"v": 1, "id": 7, "ok": false,
               "error": {"code": "invalid_params", "message": "..."}}

Methods:

``hello``
    Protocol version and daemon identity.
``status``
    Machine devices (layout plus any device a process brought), live and
    recently ended processes, telemetry, short history and a summary.
``events`` ``{"after": seq, "limit": n}``
    Events with ``seq > after``, oldest first.
``config.get`` / ``config.update`` ``{"changes": {...}}``
    Read or change the machine layout and settings. New processes pick up a
    new layout; running processes keep the devices they started with.
``process.signal`` ``{"pid": n, "signal": "SIGTERM", "source": "console"}``
    Send SIGTERM, SIGINT or SIGKILL to a process attached to this machine.
``daemon.stop``
    Stop the daemon. Running FakeCUDA processes are not affected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal as signals
import socket
import time
from typing import Any, Callable, Mapping

from .._monitor_reference import ReferenceTelemetry
from .._process import describe, process_marker
from .._smi_state import _load_states, _synthetic_uuid, build_inventory
from .._version import __version__
from ..profile_catalog import get_profile
from .config import MachineConfig, MachinePaths

PROTOCOL = "fakegpu.machine.v1"
EVENT_BUFFER = 1000
EVENT_LOG_MAX_BYTES = 5 * 1024 * 1024
HISTORY_SAMPLES = 120
# Ended processes stay in the process list this long before their state file
# is removed, so ``fakegpu nvidia-smi --include-exited`` can still show them.
ENDED_RETENTION_SECONDS = 30
SIGNALS = ("SIGTERM", "SIGINT", "SIGKILL")
LIVE = ("running", "stalled")


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _iso(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, timezone.utc).isoformat(timespec="seconds")


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds // 60 % 60:02d}m"


def _layout_text(profiles: list[str]) -> str:
    groups: list[list] = []
    for profile in profiles:
        if groups and groups[-1][0] == profile:
            groups[-1][1] += 1
        else:
            groups.append([profile, 1])
    return ",".join(f"{name}:{count}" for name, count in groups)


@dataclass
class _Process:
    pid: int
    start_ticks: str | None
    name: str
    user: str
    command: str
    attached_ns: int
    path: Path
    state: dict
    status: str = "running"
    reason: str = ""
    ended_ns: int = 0

    def device_profiles(self) -> list[str]:
        devices = sorted(self.state.get("devices") or [], key=lambda d: d.get("index", 0))
        return [str(device.get("profile_id", "")) for device in devices]

    def peak_bytes(self) -> int:
        return sum(int(d.get("reported_peak_memory") or 0) for d in self.state.get("devices") or [])

    def summary(self, now_ns: int) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "user": self.user,
            "command": self.command,
            "status": self.status,
            "reason": self.reason,
            "attached_at": _iso(self.attached_ns),
            "ended_at": _iso(self.ended_ns) if self.ended_ns else None,
            "runtime_seconds": round(((self.ended_ns or now_ns) - self.attached_ns) / 1e9, 1),
            "layout": _layout_text(self.device_profiles()),
            "peak_bytes": self.peak_bytes(),
        }


@dataclass
class _DeviceHistory:
    time: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    memory_percent: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    gpu: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    temperature: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    power_w: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    # Modeled link throughput, TX plus RX, in GB/s
    pcie_gbps: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    c2c_gbps: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))
    nvlink_gbps: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SAMPLES))

    def to_dict(self) -> dict[str, list]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}


class MachineService:
    def __init__(self, paths: MachinePaths, config: MachineConfig | None = None):
        self.paths = paths
        paths.create()
        self.config = config or MachineConfig.load(paths.config)
        self.config.save(paths.config)
        self.host = socket.gethostname()
        self.started_ns = time.time_ns()
        self.stop_requested = False
        self._processes: dict[int, _Process] = {}
        self._ended: deque[dict] = deque(maxlen=50)
        self._events: deque[dict] = deque(maxlen=EVENT_BUFFER)
        self._seq = 0
        self._invalid: dict[Path, str] = {}
        self._high_memory: set[str] = set()
        self._oversubscribed: set[str] = set()
        self._history: dict[str, _DeviceHistory] = {}
        self._telemetry = ReferenceTelemetry() if self.config.telemetry == "reference" else None
        self._platform = self.config.resolved_platform().to_dict()
        self._status: dict[str, Any] = {}
        self._methods: dict[str, Callable[..., Any]] = {
            "hello": self.hello,
            "status": self.status,
            "events": self.events,
            "config.get": self.get_config,
            "config.update": self.update_config,
            "process.signal": self.signal_process,
            "daemon.stop": self.stop,
        }
        self._restore_events()

    # ------------------------------------------------------------------ API

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response: dict[str, Any] = {"v": 1, "id": request.get("id")}
        try:
            method = self._methods.get(str(request.get("method")))
            if method is None:
                raise ServiceError("unknown_method", f"unknown method {request.get('method')!r}")
            params = request.get("params") or {}
            if not isinstance(params, Mapping):
                raise ServiceError("invalid_params", "params must be an object")
            response.update(ok=True, result=method(**params))
        except ServiceError as exc:
            response.update(ok=False, error={"code": exc.code, "message": str(exc)})
        except (TypeError, ValueError) as exc:
            response.update(ok=False, error={"code": "invalid_params", "message": str(exc)})
        return response

    def hello(self) -> dict[str, Any]:
        return {"protocol": PROTOCOL, "daemon": self._daemon_info()}

    def status(self) -> dict[str, Any]:
        if not self._status:
            self.poll()
        return self._status

    def events(self, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        selected = [event for event in self._events if event["seq"] > int(after)]
        return selected[-max(1, min(int(limit), EVENT_BUFFER)):]

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def update_config(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        new = self.config.updated(changes)
        old = self.config
        if new == old:
            return new.to_dict()
        new.save(self.paths.config)
        self.config = new
        self._platform = new.resolved_platform().to_dict()
        if new.telemetry != old.telemetry:
            self._telemetry = ReferenceTelemetry() if new.telemetry == "reference" else None
        diff = "; ".join(
            f"{name} {getattr(old, name)} -> {getattr(new, name)}"
            for name in new.to_dict()
            if getattr(old, name) != getattr(new, name)
        )
        running = sum(process.status in LIVE for process in self._processes.values())
        layout = ("devices", "platform", "pcie_generation", "nvlink_bridges")
        changed_layout = any(getattr(new, name) != getattr(old, name) for name in layout)
        note = f" ({running} running process(es) keep their devices)" if changed_layout and running else ""
        self.emit("config.changed", f"machine config changed: {diff}{note}", data={"changes": dict(changes)})
        self.poll()
        return new.to_dict()

    def signal_process(self, pid: int, signal: str = "SIGTERM", source: str = "console") -> dict[str, Any]:
        if signal not in SIGNALS:
            raise ServiceError("invalid_params", f"signal must be one of {', '.join(SIGNALS)}")
        process = self._processes.get(int(pid))
        if process is None or process.status not in LIVE or not self._same_process(process):
            raise ServiceError("not_found", f"pid {pid} is not a running process on this machine")
        os.kill(process.pid, getattr(signals, signal))
        self.emit(
            "process.signal_sent",
            f"{source} sent {signal} to {process.name} (pid {process.pid})",
            pid=process.pid,
            data={"signal": signal, "source": source},
        )
        return {"pid": process.pid, "signal": signal}

    def stop(self) -> dict[str, Any]:
        self.stop_requested = True
        return {"stopping": True}

    # -------------------------------------------------------------- polling

    def poll(self, now_ns: int | None = None) -> None:
        now = time.time_ns() if now_ns is None else now_ns
        seen: dict[int, tuple[Path, dict]] = {}
        for path in sorted(self.paths.state_dir.glob("*.json")):
            states, errors = _load_states([path], include_exited=True)
            if errors:
                if self._invalid.get(path) != errors[0]:
                    self._invalid[path] = errors[0]
                    self.emit("state.invalid", f"ignored unreadable state file {path.name}", level="warning")
                continue
            self._invalid.pop(path, None)
            state = states[0]
            pid = state.get("pid")
            if type(pid) is not int or state.get("hostname") != self.host:
                continue
            if pid not in seen or state.get("timestamp_ns", 0) > seen[pid][1].get("timestamp_ns", 0):
                seen[pid] = (path, state)
        for pid, (path, state) in seen.items():
            self._observe(pid, path, state, now)
        for process in list(self._processes.values()):
            if process.pid not in seen and process.status in LIVE:
                self._end(process, "vanished", "state file was removed", now, level="warning")
        self._prune(now)
        self._status = self._build_status(now)

    def _observe(self, pid: int, path: Path, state: dict, now: int) -> None:
        process = self._processes.get(pid)
        age = (now - int(state.get("timestamp_ns") or 0)) / 1e9
        if process is None:
            self._discover(pid, path, state, age, now)
            return
        process.path, process.state = path, state
        if process.status not in LIVE:
            return
        if not state.get("running"):
            if state.get("exit_signal"):
                self._end(process, "terminated", f"terminated by {state['exit_signal']}", now, level="warning")
            else:
                self._end(process, "exited", "exited", now)
        elif not self._same_process(process):
            self._end(
                process,
                "vanished",
                "ended without a final heartbeat (killed with SIGKILL or crashed)",
                now,
                level="error",
            )
        elif age > self.config.stale_after_seconds and process.status == "running":
            process.status = "stalled"
            self.emit(
                "process.stalled",
                f"{process.name} (pid {pid}) stopped publishing for {age:.0f}s",
                level="warning",
                pid=pid,
            )
        elif age <= self.config.stale_after_seconds and process.status == "stalled":
            process.status = "running"
            self.emit("process.resumed", f"{process.name} (pid {pid}) is publishing again", pid=pid)

    def _discover(self, pid: int, path: Path, state: dict, age: float, now: int) -> None:
        try:
            marker = process_marker(pid)
        except OSError:
            marker = None
        alive = marker is not None and marker["state"] != "Z"
        if not state.get("running") or not alive:
            # Left behind by a process that ended while no daemon was watching.
            if age > ENDED_RETENTION_SECONDS:
                path.unlink(missing_ok=True)
            return
        info = describe(pid)
        name = str(state.get("process_name") or "python")
        process = _Process(
            pid=pid,
            start_ticks=marker["start_ticks"],
            # ``python -c`` and ``python -`` report the flag as their script name.
            name=f"python {name}" if name.startswith("-") else name,
            user=info["user"],
            command=info["command"],
            attached_ns=now,
            path=path,
            state=state,
        )
        self._processes[pid] = process
        layout = _layout_text(process.device_profiles())
        note = ""
        if layout and process.device_profiles() != self.config.profiles[: len(process.device_profiles())]:
            note = f" (machine layout is {self.config.devices})"
        self.emit(
            "process.attached",
            f"{process.name} (pid {pid}) attached to {layout or 'no devices'}{note}",
            pid=pid,
            data={"layout": layout, "command": process.command},
        )

    def _same_process(self, process: _Process) -> bool:
        try:
            marker = process_marker(process.pid)
        except OSError:
            return True
        return marker is not None and marker["state"] != "Z" and marker["start_ticks"] == process.start_ticks

    def _end(self, process: _Process, status: str, reason: str, now: int, *, level: str = "info") -> None:
        process.status, process.reason, process.ended_ns = status, reason, now
        runtime = _duration((now - process.attached_ns) / 1e9)
        peak = process.peak_bytes()
        detail = f", ran {runtime}" + (f", peak {_gib(peak)}" if peak else "")
        self.emit(
            f"process.{status}",
            f"{process.name} (pid {process.pid}) {reason}{detail}",
            level=level,
            pid=process.pid,
            data={"reason": reason, "peak_bytes": peak},
        )

    def _prune(self, now: int) -> None:
        for process in list(self._processes.values()):
            if process.status in LIVE or (now - process.ended_ns) / 1e9 < ENDED_RETENTION_SECONDS:
                continue
            process.path.unlink(missing_ok=True)
            self._ended.appendleft(process.summary(now))
            del self._processes[process.pid]

    # --------------------------------------------------------------- status

    def _build_status(self, now: int) -> dict[str, Any]:
        live = [process for process in self._processes.values() if process.status in LIVE]
        inventory = build_inventory(
            [process.state for process in live],
            stale_after_seconds=self.config.stale_after_seconds,
            now_ns=now,
        )
        devices = self._devices(inventory)
        sensors, activity = ({}, {})
        if self._telemetry is not None:
            sensors, activity = self._telemetry.sample({"devices": devices, "processes": inventory["processes"]})
        for device in devices:
            device["telemetry"] = self._device_telemetry(device, sensors.get(device["uuid"]))
            self._record_history(device, now)
            self._check_memory(device)
        processes = self._process_rows(live, inventory, activity, now)
        ended = [p.summary(now) for p in self._processes.values() if p.status not in LIVE]
        total = sum(device["memory"]["total_bytes"] for device in devices if device["in_layout"])
        used = sum(device["memory"]["used_bytes"] for device in devices)
        return {
            "protocol": PROTOCOL,
            "daemon": self._daemon_info(now),
            "config": self.config.to_dict(),
            "platform": self._platform,
            "paths": {"root": str(self.paths.root), "state_dir": str(self.paths.state_dir)},
            "devices": devices,
            "processes": processes,
            "ended": ended + list(self._ended),
            "summary": {
                "device_count": sum(device["in_layout"] for device in devices),
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "process_count": len(processes),
                "stalled_count": sum(process.status == "stalled" for process in live),
                "alerts": len(self._high_memory) + len(self._oversubscribed),
            },
            "last_event_seq": self._seq,
        }

    def _devices(self, inventory: dict) -> list[dict[str, Any]]:
        observed = {device["uuid"]: device for device in inventory["devices"]}
        # Every FakeCUDA process publishes all devices it can see; only the
        # ones it holds memory on count as its devices.
        users: dict[str, int] = {}
        for row in inventory["processes"]:
            if row["memory"].get("used_bytes"):
                users[row["gpu_uuid"]] = users.get(row["gpu_uuid"], 0) + 1
        rows = []
        slots = self._platform["devices"]
        for index, profile_id in enumerate(self.config.profiles):
            uuid = _synthetic_uuid(self.host, index, profile_id)
            row = self._device_row(index, profile_id, uuid, observed.pop(uuid, None), users, True)
            row["slot"] = dict(slots[index]) if index < len(slots) else None
            rows.append(row)
        for uuid, device in observed.items():
            row = self._device_row(device["index"], device["profile_id"], uuid, device, users, False)
            row["slot"] = None
            rows.append(row)
        return rows

    def _device_row(
        self, index: int, profile_id: str, uuid: str, observed: dict | None, users: dict, in_layout: bool
    ) -> dict:
        profile = get_profile(profile_id)
        memory = (observed or {}).get("memory") or {}
        allocator = (observed or {}).get("allocator") or {}
        total = int(memory.get("total_bytes") or profile.memory_bytes)
        used = int(memory.get("used_bytes") or 0)
        return {
            "index": index,
            "uuid": uuid,
            "profile_id": profile.id,
            "name": profile.torch_name or profile.name,
            "architecture": profile.architecture.capitalize(),
            "compute_capability": profile.compute_capability_text,
            "in_layout": in_layout,
            "power_limit_w": profile.max_power_limit_mw / 1000 or None,
            "memory": {
                "total_bytes": total,
                "used_bytes": used,
                "peak_bytes": int(memory.get("reported_peak_bytes") or 0),
                "tracked_bytes": int(memory.get("tracked_bytes") or 0),
                "reserved_bytes": int(memory.get("reserved_bytes") or 0),
                "percent": round(used / total * 100, 2) if total else 0.0,
            },
            "process_count": users.get(uuid, 0),
            "transfer": (observed or {}).get("transfer"),
            "allocator": {
                "categories": dict(allocator.get("categories") or {}),
                "stage_peaks": dict(allocator.get("stage_peaks") or {}),
                "largest_allocations": list(allocator.get("largest_allocations") or []),
                "allocation_count": int(allocator.get("allocation_count") or 0),
                "segment_count": int(allocator.get("segment_count") or 0),
            },
        }

    @staticmethod
    def _device_telemetry(device: dict, values: dict | None) -> dict[str, Any] | None:
        if not values:
            return None
        return {
            "source": "a100_reference_model",
            "gpu_percent": values["gpu"],
            "memory_bandwidth_percent": values["memory"],
            "temperature_c": values["temperature"],
            "power_w": round(values["power_mw"] / 1000, 1),
            "sm_clock_mhz": values["sm_clock"],
            "memory_clock_mhz": values["memory_clock"],
            "pstate": f"P{values['pstate']}",
        }

    def _record_history(self, device: dict, now: int) -> None:
        history = self._history.setdefault(device["uuid"], _DeviceHistory())
        telemetry = device["telemetry"] or {}
        history.time.append(round(now / 1e9, 1))
        history.memory_percent.append(device["memory"]["percent"])
        history.gpu.append(telemetry.get("gpu_percent"))
        history.temperature.append(telemetry.get("temperature_c"))
        history.power_w.append(telemetry.get("power_w"))
        transfer = device["transfer"]
        for link in ("pcie", "c2c", "nvlink"):
            rate = None
            if transfer is not None:
                entry = transfer["links"][link]
                rate = round((entry["tx_bytes_per_second"] + entry["rx_bytes_per_second"]) / 1e9, 3)
            getattr(history, f"{link}_gbps").append(rate)
        device["history"] = history.to_dict()

    def _check_memory(self, device: dict) -> None:
        uuid, memory = device["uuid"], device["memory"]
        label = f"GPU {device['index']} ({device['profile_id']})"
        warning = self.config.memory_warning_percent
        if memory["percent"] >= warning and uuid not in self._high_memory:
            self._high_memory.add(uuid)
            self.emit(
                "memory.high",
                f"{label} memory at {memory['percent']:.0f}% ({_gib(memory['used_bytes'])} of {_gib(memory['total_bytes'])})",
                level="warning",
                device=device["index"],
            )
        elif memory["percent"] < warning - 5 and uuid in self._high_memory:
            self._high_memory.discard(uuid)
            self.emit("memory.normal", f"{label} memory back to {memory['percent']:.0f}%", device=device["index"])
        if memory["used_bytes"] > memory["total_bytes"] and uuid not in self._oversubscribed:
            self._oversubscribed.add(uuid)
            self.emit(
                "memory.oversubscribed",
                f"{label}: processes together use {_gib(memory['used_bytes'])}, more than its "
                f"{_gib(memory['total_bytes'])}; each process only checks its own allocations",
                level="error",
                device=device["index"],
            )
        elif memory["used_bytes"] <= memory["total_bytes"]:
            self._oversubscribed.discard(uuid)

    def _process_rows(self, live: list[_Process], inventory: dict, activity: dict, now: int) -> list[dict]:
        rows = {}
        for process in live:
            state = process.state
            fakegpu = state.get("fakegpu") or {}
            rows[process.pid] = {
                **process.summary(now),
                "stage": str(state.get("stage") or "unknown"),
                "backend": str(fakegpu.get("backend") or "unknown"),
                "tracking_confidence": str(state.get("tracking_confidence") or "unknown"),
                "heartbeat_age_seconds": round((now - int(state.get("timestamp_ns") or 0)) / 1e9, 1),
                "software": dict(state.get("software") or {}),
                "devices": [],
                "memory_bytes": 0,
                "gpu_percent": None,
            }
        for row in inventory["processes"]:
            target = rows.get(row["pid"])
            if target is None:
                continue
            used = int(row["memory"].get("used_bytes") or 0)
            target["devices"].append({"index": row["gpu_index"], "profile_id": row["profile_id"], "used_bytes": used})
            target["memory_bytes"] += used
            sample = activity.get((row["gpu_uuid"], row["pid"]))
            if sample is not None:
                target["gpu_percent"] = max(target["gpu_percent"] or 0, sample["gpu"])
        return sorted(rows.values(), key=lambda row: row["pid"])

    def _daemon_info(self, now: int | None = None) -> dict[str, Any]:
        now = time.time_ns() if now is None else now
        return {
            "pid": os.getpid(),
            "host": self.host,
            "version": __version__,
            "started_at": _iso(self.started_ns),
            "uptime_seconds": round((now - self.started_ns) / 1e9, 1),
        }

    # --------------------------------------------------------------- events

    def emit(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        pid: int | None = None,
        device: int | None = None,
        data: dict | None = None,
    ) -> dict[str, Any]:
        self._seq += 1
        now = time.time_ns()
        event = {
            "seq": self._seq,
            "time": _iso(now),
            "timestamp_ns": now,
            "level": level,
            "kind": kind,
            "message": message,
            "pid": pid,
            "device": device,
            "data": data or {},
        }
        self._events.append(event)
        log = self.paths.events
        if log.exists() and log.stat().st_size > EVENT_LOG_MAX_BYTES:
            os.replace(log, log.with_suffix(".jsonl.1"))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        return event

    def _restore_events(self) -> None:
        if not self.paths.events.exists():
            return
        lines = self.paths.events.read_text(encoding="utf-8").splitlines()[-EVENT_BUFFER:]
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._events.append(event)
            self._seq = max(self._seq, int(event.get("seq") or 0))
