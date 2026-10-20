"""The background machine's GPUs as an SMI state for viewers.

``fakegpu nvidia-smi`` and ``fakegpu nvitop`` read the states FakeCUDA
processes publish. A real machine lists its GPUs while nothing runs, so
while the daemon runs the viewers add one more state: every GPU of the
machine layout on its platform, idle, used by no process.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

from .._smi_environment import _modeled_device_topology
from .._smi_state import LAYOUT_ONLY_KEY, SCHEMA_VERSION, _synthetic_pci_bus_id, _synthetic_uuid
from ..profile_catalog import get_profile
from .client import daemon_pid
from .config import MachineConfig, MachinePaths


def machine_state_dir(paths: MachinePaths | None = None) -> Path | None:
    """State directory of the running daemon, or ``None`` when it is stopped."""

    paths = paths or MachinePaths.from_env()
    return paths.state_dir if daemon_pid(paths) is not None else None


def is_machine_state_dir(state_dir: Path | str | None, paths: MachinePaths | None = None) -> bool:
    if state_dir is None:
        return False
    running = machine_state_dir(paths)
    return running is not None and Path(state_dir).expanduser().resolve() == running.resolve()


def layout_state(paths: MachinePaths | None = None) -> dict[str, Any] | None:
    """The machine layout as a layout-only SMI state; ``None`` without a daemon."""

    paths = paths or MachinePaths.from_env()
    pid = daemon_pid(paths)
    if pid is None:
        return None
    try:
        config = MachineConfig.load(paths.config)
    except (OSError, ValueError):
        return None
    environment = config.resolved_platform().environment()
    profiles = [get_profile(profile_id) for profile_id in config.profiles]
    host = socket.gethostname()
    devices = [
        {
            "index": index,
            "name": profile.torch_name or profile.name,
            "profile_id": profile.id,
            "uuid": _synthetic_uuid(host, index, profile.id),
            "pci_bus_id": _synthetic_pci_bus_id(index),
            "identity_source": "synthetic",
            "total_memory": profile.memory_bytes,
            "free_memory": profile.memory_bytes,
            "reported_memory": 0,
            "reported_peak_memory": 0,
        }
        for index, profile in enumerate(profiles)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_ns": time.time_ns(),
        "hostname": host,
        "pid": pid,
        "process_name": "fakegpu-daemon",
        "running": True,
        LAYOUT_ONLY_KEY: True,
        "topology": _modeled_device_topology(devices, environment),
        "devices": devices,
    }
