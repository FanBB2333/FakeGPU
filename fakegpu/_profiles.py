"""Resolve the GPU profile the patch layer should present, from FAKEGPU_*.

Split out of ``torch_patch`` unchanged. These read ``FAKEGPU_PROFILE``,
``FAKEGPU_PROFILES``, ``FAKEGPU_DEVICE_NAME``, and the bundled profile
catalog to answer what ``torch.cuda`` should report for a device: its name,
compute capability, and total memory. They are pure functions of the
environment; the mutable runtime state they feed lives in ``torch_patch``.
"""

from __future__ import annotations

import os
from typing import Any

from .profile_catalog import load_profiles


_PROFILE_CATALOG = load_profiles()


_PROFILE_CC: dict[str, tuple[int, int]] = {
    profile_id: profile.compute_capability
    for profile_id, profile in _PROFILE_CATALOG.items()
}


_PROFILE_NAMES: dict[str, str] = {
    profile_id: profile.torch_name for profile_id, profile in _PROFILE_CATALOG.items()
}


_PROFILE_TOTAL_MEMORY: dict[str, int] = {
    profile_id: profile.memory_bytes for profile_id, profile in _PROFILE_CATALOG.items()
}


_PROFILE_SUPPORTED_TYPES: dict[str, tuple[str, ...]] = {
    profile_id: profile.supported_types
    for profile_id, profile in _PROFILE_CATALOG.items()
}


def _resolve_profile_id() -> str | None:
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
    if profiles_env:
        first_spec = profiles_env.split(",")[0].strip()
        return first_spec.split(":")[0].strip().lower()

    profile_env = os.environ.get("FAKEGPU_PROFILE", "")
    if profile_env:
        return profile_env.strip().lower()

    device_name = os.environ.get("FAKEGPU_DEVICE_NAME", "").strip().lower()
    if device_name:
        reverse_names = {value.lower(): key for key, value in _PROFILE_NAMES.items()}
        return reverse_names.get(device_name)

    return None


def _resolve_compute_capability() -> tuple[int, int]:
    profile_id = _resolve_profile_id()
    if profile_id and profile_id in _PROFILE_CC:
        return _PROFILE_CC[profile_id]
    return (8, 0)


def _resolve_device_name() -> str:
    name = os.environ.get("FAKEGPU_DEVICE_NAME", "")
    if name:
        return name
    profile_id = _resolve_profile_id()
    if profile_id:
        return _PROFILE_NAMES.get(profile_id, "NVIDIA A100-SXM4-80GB")
    return "NVIDIA A100-SXM4-80GB"


def _resolve_total_memory() -> int:
    profile_id = _resolve_profile_id()
    if profile_id:
        return _PROFILE_TOTAL_MEMORY.get(profile_id, 80 * 1024**3)
    return 80 * 1024**3


def _resolve_per_device_profiles(
    num_devices: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve per-device profile info from FAKEGPU_PROFILES.

    Returns a list of dicts, one per device, each with keys:
      'profile_id', 'name', 'total_memory', 'compute_major', 'compute_minor'
    """
    profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
    target_count = int(
        num_devices
        if num_devices is not None
        else os.environ.get("FAKEGPU_DEVICE_COUNT", "8")
    )
    result: list[dict[str, Any]] = []

    if profiles_env:
        for spec in profiles_env.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            pid = parts[0].strip().lower()
            count = (
                int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1
            )
            for _ in range(count):
                cc = _PROFILE_CC.get(pid, (8, 0))
                result.append(
                    {
                        "profile_id": pid,
                        "name": _PROFILE_NAMES.get(pid, "NVIDIA A100-SXM4-80GB"),
                        "total_memory": _PROFILE_TOTAL_MEMORY.get(pid, 80 * 1024**3),
                        "compute_major": cc[0],
                        "compute_minor": cc[1],
                    }
                )

    if not result:
        # Uniform config: all devices share the same profile
        pid = _resolve_profile_id() or "a100"
        cc = _PROFILE_CC.get(pid, (8, 0))
        entry = {
            "profile_id": pid,
            "name": _PROFILE_NAMES.get(pid, "NVIDIA A100-SXM4-80GB"),
            "total_memory": _PROFILE_TOTAL_MEMORY.get(pid, 80 * 1024**3),
            "compute_major": cc[0],
            "compute_minor": cc[1],
        }
        for _ in range(target_count):
            result.append(dict(entry))

    if len(result) != target_count and len(result) > 0:
        while len(result) < target_count:
            result.append(dict(result[-1]))
        result = result[:target_count]

    return result
