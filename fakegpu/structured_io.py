from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StructuredDataError(ValueError):
    pass


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load one JSON, TOML, or YAML mapping.

    YAML support is optional. Python 3.10 uses the ``tomli`` compatibility
    dependency for TOML; Python 3.11 and newer use the standard library.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"structured data file not found: {resolved}")
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
                import tomli as tomllib

            payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise StructuredDataError(
                    "YAML input requires PyYAML; install fakegpu[validation] "
                    "or provide JSON/TOML"
                ) from exc
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        else:
            raise StructuredDataError(
                f"unsupported structured data suffix {suffix!r}: {resolved}"
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, StructuredDataError):
            raise
        raise StructuredDataError(f"cannot parse {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StructuredDataError(
            f"structured data root must be an object: {resolved}"
        )
    return dict(payload)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


__all__ = ["StructuredDataError", "load_mapping", "write_json"]
