"""Machine layout and the files that the daemon and its clients share.

Everything lives under ``$FAKEGPU_HOME/machine`` (default ``~/.fakegpu``)::

    config.json    machine layout and console settings
    state/         FakeCUDA processes publish their SMI state here
    events.jsonl   event log, one JSON object per line
    daemon.pid     PID of the running daemon (locked while it runs)
    daemon.sock    control socket, see ``fakegpu.machine.service`` (in the
                   temp directory instead when this path is too long)
    daemon.log     daemon stdout/stderr
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from ..platforms import AUTO, ResolvedPlatform, resolve_platform
from ..profile_catalog import get_profile

TELEMETRY_MODES = ("none", "reference")


@dataclass(frozen=True)
class MachinePaths:
    root: Path

    def __post_init__(self) -> None:
        # The background child runs from /, so resolve caller-relative paths
        # before launching it or deriving its socket path.
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> MachinePaths:
        home = Path(environ.get("FAKEGPU_HOME") or "~/.fakegpu")
        return cls(home / "machine")

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def pidfile(self) -> Path:
        return self.root / "daemon.pid"

    @property
    def socket(self) -> Path:
        path = self.root / "daemon.sock"
        # Unix socket paths are limited to about 104 bytes on macOS.
        if len(os.fsencode(path)) < 100:
            return path
        digest = hashlib.sha256(os.fsencode(self.root)).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"fakegpu-{os.getuid()}-{digest}.sock"

    @property
    def log(self) -> Path:
        return self.root / "daemon.log"

    def create(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)


def parse_devices(spec: str) -> list[str]:
    """Expand ``"a100:2,h100:2"`` into one catalog profile ID per device."""

    profiles: list[str] = []
    for item in spec.split(","):
        name, _, count = item.strip().partition(":")
        if not name:
            continue
        if count and (not count.strip().isdigit() or int(count) < 1):
            raise ValueError(f"invalid device count in {item.strip()!r}")
        profile = get_profile(name).id
        profiles.extend([profile] * (int(count) if count else 1))
    if not profiles:
        raise ValueError("the machine needs at least one device, e.g. a100:8")
    return profiles


def layout_text(profiles: Sequence[str]) -> str:
    """Collapse one profile ID per device back into ``"a100:2,h100:2"``."""

    return ",".join(f"{name}:{len(list(group))}" for name, group in groupby(profiles))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class MachineConfig:
    devices: str = "a100:8"
    # Host platform from fakegpu.platforms: "auto" picks one by form factor
    # (SXM modules -> HGX/DGX baseboard, PCIe cards -> server/workstation/
    # desktop slots, superchips -> their Grace platform).
    platform: str = AUTO
    # Overrides the PCIe generation of PCIe slots (0 keeps the template's).
    pcie_generation: int = 0
    # NVLink bridges between PCIe cards, e.g. "0-1,2-3" or "0-1-2-3".
    nvlink_bridges: str = ""
    # "reference" models temperature/power/clocks from CPU activity, starting
    # from public idle readings for a100 and a default curve (30°C, 1 W idle)
    # for every other profile. "none" leaves them N/A.
    telemetry: str = "reference"
    # Keep the daemon running after the console exits.
    background: bool = True
    stale_after_seconds: float = 2.0
    memory_warning_percent: float = 90.0

    def __post_init__(self) -> None:
        if not isinstance(self.devices, str):
            raise ValueError("devices must be a string such as a100:8")
        parse_devices(self.devices)
        if not isinstance(self.platform, str):
            raise ValueError("platform must be a string such as auto or hgx-a100-8")
        if isinstance(self.pcie_generation, bool) or not isinstance(self.pcie_generation, int):
            raise ValueError("pcie_generation must be an integer (0 keeps the platform's)")
        if not isinstance(self.nvlink_bridges, str):
            raise ValueError("nvlink_bridges must be a string such as 0-1,2-3")
        self.resolved_platform()
        if self.telemetry not in TELEMETRY_MODES:
            raise ValueError(f"telemetry must be one of {', '.join(TELEMETRY_MODES)}")
        if not _number(self.stale_after_seconds) or not 0 < self.stale_after_seconds <= 3600:
            raise ValueError("stale_after_seconds must be in (0, 3600]")
        if not _number(self.memory_warning_percent) or not 0 < self.memory_warning_percent <= 100:
            raise ValueError("memory_warning_percent must be in (0, 100]")

    @property
    def profiles(self) -> list[str]:
        return parse_devices(self.devices)

    def resolved_platform(self) -> ResolvedPlatform:
        return resolve_platform(
            self.profiles,
            self.platform,
            pcie_generation=int(self.pcie_generation),
            nvlink_bridges=self.nvlink_bridges,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def updated(self, changes: Mapping[str, Any]) -> MachineConfig:
        known = {field.name for field in fields(self)}
        unknown = sorted(set(changes) - known)
        if unknown:
            raise ValueError(f"unknown machine setting(s): {', '.join(unknown)}")
        return replace(self, **changes)

    @classmethod
    def load(cls, path: Path) -> MachineConfig:
        if not path.exists():
            return cls()
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError(f"{path} must hold a JSON object")
        return cls().updated(settings)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
