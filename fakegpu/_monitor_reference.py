"""Reference-backed display model. It does not estimate GPU performance."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from .profile_catalog import get_profile


class ReferenceTelemetry:
    def __init__(self):
        self.reference = json.loads(
            (Path(__file__).parent / "data" / "a100_monitor_reference.json").read_text()
        )
        self._previous: dict[tuple, tuple] = {}
        self._temperatures: dict[str, tuple[float, float]] = {}

    def sample(self, inventory: dict, *, now: float | None = None) -> tuple[dict, dict]:
        current = time.monotonic() if now is None else now
        previous = {}
        process_values = {}
        device_values = {}
        reference = self.reference
        for row in inventory["processes"]:
            activity = row.get("activity") or {}
            if row["profile_id"] != reference["profile_id"] or activity.get("source") != "cpu_dispatch_proxy":
                continue
            key = (row["gpu_uuid"], row["pid"])
            stamp = row["timestamp_ns"]
            busy, io_bytes = activity.get("busy_ns"), activity.get("io_bytes")
            if any(type(value) is not int or value < 0 for value in (stamp, busy, io_bytes)):
                continue
            old = self._previous.get(key)
            gpu = memory = 0.0
            if old is not None:
                if stamp == old[0]:
                    gpu, memory = old[3:]
                elif stamp > old[0] and busy >= old[1] and io_bytes >= old[2]:
                    elapsed = stamp - old[0]
                    gpu = min(100.0, 100 * (busy - old[1]) / elapsed)
                    bandwidth = reference["specification"]["memory_bandwidth_gb_s"]
                    memory = min(100.0, 100 * (io_bytes - old[2]) / (elapsed * bandwidth))
            previous[key] = (stamp, busy, io_bytes, gpu, memory)
            process_values[key] = {"gpu": round(gpu), "memory": round(memory), "timestamp_us": stamp // 1000}
        self._previous = previous
        active_uuids = set()
        for device in inventory["devices"]:
            uuid = device["uuid"]
            rows = [row for row in inventory["processes"] if row["gpu_uuid"] == uuid]
            keys = [(uuid, row["pid"]) for row in rows]
            # Never apply A100 observations to another profile. A device with
            # no processes is idle and settles at the observed idle readings.
            if device["profile_id"] != reference["profile_id"] or any(key not in process_values for key in keys):
                continue
            active_uuids.add(uuid)
            index = device["index"] % len(reference["observed_idle"]["power_w"])
            idle_power = reference["observed_idle"]["power_w"][index]
            idle_temp = reference["observed_idle"]["temperature_c"][index]
            gpu = min(100, sum(process_values[key]["gpu"] for key in keys))
            memory = min(100, sum(process_values[key]["memory"] for key in keys))
            assumptions = reference["simulation_assumptions"]
            limit = reference["specification"]["power_limit_w"]
            power = idle_power + (limit - idle_power) * gpu / 100
            target = idle_temp + assumptions["temperature_rise_at_full_activity_c"] * gpu / 100
            old_time, old_temp = self._temperatures.get(uuid, (current, float(idle_temp)))
            alpha = 1 - math.exp(-max(0, current - old_time) / assumptions["thermal_time_constant_seconds"])
            temperature = old_temp + alpha * (target - old_temp)
            self._temperatures[uuid] = (current, temperature)
            profile = get_profile(device["profile_id"])
            device_values[uuid] = {
                "gpu": gpu, "memory": memory, "temperature": round(temperature),
                "power_mw": round(power * 1000), "pstate": reference["observed_idle"]["pstate"],
                "sm_clock": profile.core_clock_mhz if gpu else assumptions["idle_sm_clock_mhz"],
                "memory_clock": profile.memory_clock_mhz,
            }
        self._temperatures = {k: v for k, v in self._temperatures.items() if k in active_uuids}
        return device_values, process_values
