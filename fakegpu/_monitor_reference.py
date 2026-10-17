"""Modeled sensor readings for display. They do not estimate GPU performance.

The ``a100`` profile starts from public idle observations
(``data/a100_monitor_reference.json``). Every other catalog profile starts
from the fixed idle readings of ``DEFAULT_MODEL``. Both follow the publishers'
CPU dispatch activity with the same curves: power rises linearly to the
profile's power cap and temperature moves toward a load-dependent target with
a first-order thermal response.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from .profile_catalog import ProfileCatalogError, get_profile

# Simulation assumptions for profiles without observed readings, not
# measurements. The fallbacks apply where the catalog has no figure.
DEFAULT_MODEL = {
    "idle_temperature_c": 30,
    "idle_power_w": 1.0,
    "pstate": 0,
    "fallback_power_limit_w": 300.0,
    "fallback_memory_bandwidth_gb_s": 1000.0,
    "fallback_sm_clock_mhz": 1500,
    "fallback_memory_clock_mhz": 1500,
}
SOURCES = {"reference": "a100_reference_model", "default": "default_model"}


class ReferenceTelemetry:
    def __init__(self):
        self.reference = json.loads(
            (Path(__file__).parent / "data" / "a100_monitor_reference.json").read_text()
        )
        self._previous: dict[tuple, tuple] = {}
        self._temperatures: dict[str, tuple[float, float]] = {}
        self._models: dict[str, dict | None] = {}

    def source(self, profile_id: str) -> str:
        """Which model fills a profile's readings."""
        return SOURCES["reference" if profile_id == self.reference["profile_id"] else "default"]

    def _model(self, profile_id: str) -> dict | None:
        """Curve parameters of a profile; ``None`` for one outside the catalog."""
        if profile_id not in self._models:
            try:
                profile = get_profile(profile_id)
            except ProfileCatalogError:
                self._models[profile_id] = None
                return None
            assumptions = self.reference["simulation_assumptions"]
            model = {
                "rise": assumptions["temperature_rise_at_full_activity_c"],
                "time_constant": assumptions["thermal_time_constant_seconds"],
                "idle_sm_clock": assumptions["idle_sm_clock_mhz"],
            }
            # Never apply A100 observations to another profile.
            if profile_id == self.reference["profile_id"]:
                observed, specification = self.reference["observed_idle"], self.reference["specification"]
                model.update(
                    idle_temperature=observed["temperature_c"], idle_power=observed["power_w"],
                    pstate=observed["pstate"], limit=specification["power_limit_w"],
                    bandwidth=specification["memory_bandwidth_gb_s"],
                    sm_clock=profile.core_clock_mhz, memory_clock=profile.memory_clock_mhz,
                )
            else:
                limit_mw = profile.max_power_limit_mw or profile.typical_power_usage_mw
                model.update(
                    idle_temperature=[DEFAULT_MODEL["idle_temperature_c"]],
                    idle_power=[DEFAULT_MODEL["idle_power_w"]],
                    pstate=DEFAULT_MODEL["pstate"],
                    limit=limit_mw / 1000 if limit_mw else DEFAULT_MODEL["fallback_power_limit_w"],
                    bandwidth=profile.memory_bandwidth_gbps or DEFAULT_MODEL["fallback_memory_bandwidth_gb_s"],
                    sm_clock=profile.core_clock_mhz or profile.base_clock_mhz
                    or DEFAULT_MODEL["fallback_sm_clock_mhz"],
                    memory_clock=profile.memory_clock_mhz or DEFAULT_MODEL["fallback_memory_clock_mhz"],
                )
            self._models[profile_id] = model
        return self._models[profile_id]

    def sample(self, inventory: dict, *, now: float | None = None) -> tuple[dict, dict]:
        current = time.monotonic() if now is None else now
        previous = {}
        process_values = {}
        device_values = {}
        for row in inventory["processes"]:
            # A stale heartbeat gives no evidence that the last observed
            # activity is still running. Keep its memory, not its old load.
            if row["status"] != "running":
                continue
            activity = row.get("activity") or {}
            model = self._model(row["profile_id"])
            if model is None or activity.get("source") != "cpu_dispatch_proxy":
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
                    memory = min(100.0, 100 * (io_bytes - old[2]) / (elapsed * model["bandwidth"]))
            previous[key] = (stamp, busy, io_bytes, gpu, memory)
            process_values[key] = {"gpu": round(gpu), "memory": round(memory), "timestamp_us": stamp // 1000}
        self._previous = previous
        active_uuids = set()
        for device in inventory["devices"]:
            uuid = device["uuid"]
            rows = [row for row in inventory["processes"] if row["gpu_uuid"] == uuid]
            keys = [(uuid, row["pid"]) for row in rows]
            model = self._model(device["profile_id"])
            # A device with no processes is idle and settles at its idle
            # readings; one whose processes publish no activity stays unknown.
            if model is None or any(key not in process_values for key in keys):
                continue
            active_uuids.add(uuid)
            index = device["index"] % len(model["idle_power"])
            idle_power = model["idle_power"][index]
            idle_temp = model["idle_temperature"][index]
            gpu = min(100, sum(process_values[key]["gpu"] for key in keys))
            memory = min(100, sum(process_values[key]["memory"] for key in keys))
            power = idle_power + (model["limit"] - idle_power) * gpu / 100
            target = idle_temp + model["rise"] * gpu / 100
            old_time, old_temp = self._temperatures.get(uuid, (current, float(idle_temp)))
            alpha = 1 - math.exp(-max(0, current - old_time) / model["time_constant"])
            temperature = old_temp + alpha * (target - old_temp)
            self._temperatures[uuid] = (current, temperature)
            device_values[uuid] = {
                "gpu": gpu, "memory": memory, "temperature": round(temperature),
                "power_mw": round(power * 1000), "pstate": model["pstate"],
                "sm_clock": model["sm_clock"] if gpu else model["idle_sm_clock"],
                "memory_clock": model["memory_clock"],
            }
        self._temperatures = {k: v for k, v in self._temperatures.items() if k in active_uuids}
        return device_values, process_values
