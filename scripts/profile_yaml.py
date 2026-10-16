"""Rewrite the optional specification block of a GPU profile YAML file.

Profiles are flat ``key: value`` files that both the Python catalog and the
native runtime parse line by line, so this helper edits lines instead of
round-tripping through a YAML library: comments, key order and the required
fields stay exactly as written.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fakegpu.profile_catalog import (  # noqa: E402
    THROUGHPUT_BASES,
    THROUGHPUT_FIELDS,
    _OPTIONAL_INTEGER_FIELDS,
    _OPTIONAL_NUMBER_FIELDS,
)

SPEC_KEYS = (
    "base_clock_mhz",
    "memory_type",
    "memory_bandwidth_gbps",
    "ecc",
    "form_factor",
    "pcie_generation",
    "pcie_lanes",
    "nvlink_generation",
    "nvlink_links",
    "nvlink_bandwidth_gbps",
    "c2c_bandwidth_gbps",
    "mig_max_instances",
    "min_power_limit_mw",
)
assert set(SPEC_KEYS) >= set(_OPTIONAL_INTEGER_FIELDS + _OPTIONAL_NUMBER_FIELDS)
SCALAR_KEYS = SPEC_KEYS + THROUGHPUT_FIELDS + ("throughput_source",)
LIST_KEYS = tuple(THROUGHPUT_BASES)
ANCHOR = "pci_device_id"


def format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def read_block(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return the optional scalars and provenance lists currently in ``path``."""

    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and current is not None:
            lists[current].append(stripped[2:].strip())
            continue
        current = None
        key, _, value = stripped.partition(":")
        if key in SCALAR_KEYS and value.strip():
            scalars[key] = value.strip()
        elif key in LIST_KEYS and not value.strip():
            current = key
            lists[key] = []
    return scalars, lists


def write_block(
    path: Path,
    scalars: Mapping[str, Any],
    basis: Mapping[str, str],
) -> bool:
    """Replace the optional block of ``path``; return whether the file changed.

    ``scalars`` holds every optional key to keep (absent keys are removed) and
    ``basis`` maps each throughput field to ``published`` or ``derived``.
    """

    unknown = set(scalars) - set(SCALAR_KEYS)
    if unknown:
        raise ValueError(f"{path}: unsupported keys {sorted(unknown)}")
    kept: list[str] = []
    skipping_list = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if skipping_list and stripped.startswith("- "):
            continue
        skipping_list = False
        key, _, value = stripped.partition(":")
        if key in SCALAR_KEYS:
            continue
        if key in LIST_KEYS and not value.strip():
            skipping_list = True
            continue
        kept.append(line)

    block = []
    for key in SCALAR_KEYS:
        if key in scalars and scalars[key] not in (None, ""):
            value = scalars[key]
            text = format_number(value) if isinstance(value, (int, float)) else str(value)
            block.append(f"{key}: {text}")
    anchor = next(i for i, line in enumerate(kept) if line.startswith(f"{ANCHOR}:"))
    lines = kept[: anchor + 1] + block + kept[anchor + 1 :]
    for list_key, label in THROUGHPUT_BASES.items():
        names = [name for name in THROUGHPUT_FIELDS if basis.get(name) == label and name in scalars]
        if names:
            lines.append(f"{list_key}:")
            lines.extend(f"  - {name}" for name in names)

    text = "\n".join(lines) + "\n"
    if text == path.read_text(encoding="utf-8"):
        return False
    path.write_text(text, encoding="utf-8")
    return True
