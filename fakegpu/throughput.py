"""Derive peak throughput from per-SM, per-clock architecture rates.

NVIDIA computes published peaks as ``operations per clock per SM x enabled SMs
x boost clock``. ``data/throughput_rates.json`` holds those per-SM rates by
compute capability, with separate GeForce and professional entries where the
products differ (for example FP32-accumulate tensor rates), plus per-profile
exceptions. Profiles keep NVIDIA's published values; derived values fill only
the fields NVIDIA does not publish for a configuration.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .profile_catalog import THROUGHPUT_FIELDS, GpuProfile

RATES_PATH = Path(__file__).resolve().parent / "data" / "throughput_rates.json"
RATES_SCHEMA = "fakegpu.throughput_rates.v1"


@lru_cache(maxsize=1)
def load_rates() -> dict[str, Any]:
    payload = json.loads(RATES_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != RATES_SCHEMA:
        raise ValueError(f"{RATES_PATH}: expected schema {RATES_SCHEMA}")
    return payload


def rate_class(profile: GpuProfile) -> str:
    return "geforce" if profile.segment == "consumer" else "professional"


def per_sm_rates(profile: GpuProfile, rates: dict[str, Any] | None = None) -> dict[str, float]:
    """Return dense operations per clock per SM for ``profile``, by throughput field."""

    rates = rates if rates is not None else load_rates()
    capability = profile.compute_capability_text
    merged: dict[str, float] = {}
    for wanted in ("all", rate_class(profile)):
        for entry in rates["rates"]:
            if entry["compute_capability"] == capability and entry["class"] == wanted:
                merged.update(entry["per_sm_per_clock"])
    for override in rates.get("overrides", ()):
        if override["profile_id"] == profile.id:
            merged.update(override["per_sm_per_clock"])
    return {key: float(value) for key, value in merged.items() if key in THROUGHPUT_FIELDS}


def derive_throughput(profile: GpuProfile, rates: dict[str, Any] | None = None) -> dict[str, float]:
    """Return derived dense peaks (TFLOPS or TOPS) for every field with a known rate."""

    if profile.sm_count <= 0 or profile.core_clock_mhz <= 0:
        return {}
    cycles_per_second = profile.sm_count * profile.core_clock_mhz * 1_000_000
    return {
        key: float(f"{rate * cycles_per_second / 1e12:.4g}")
        for key, rate in per_sm_rates(profile, rates).items()
    }
