from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StructuredDataError(ValueError):
    pass


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load one JSON, gzip-compressed JSON, TOML, or YAML mapping.

    YAML support is optional. Python 3.10 uses the ``tomli`` compatibility
    dependency for TOML; Python 3.11 and newer use the standard library.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"structured data file not found: {resolved}")
    suffix = resolved.suffix.lower()
    try:
        if resolved.name.lower().endswith(".json.gz"):
            with gzip.open(resolved, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        elif suffix == ".json":
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
            try:
                payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise StructuredDataError(f"cannot parse {resolved}: {exc}") from exc
        else:
            raise StructuredDataError(
                f"unsupported structured data suffix {suffix!r}: {resolved}"
            )
    except (OSError, EOFError, UnicodeError, ValueError) as exc:
        if isinstance(exc, StructuredDataError):
            raise
        raise StructuredDataError(f"cannot parse {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StructuredDataError(f"structured data root must be an object: {resolved}")
    return dict(payload)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON; ``.json.gz`` uses deterministic, atomic streaming compression."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.name.lower().endswith(".json.gz"):
        _write_compressed_json(resolved, payload)
        return resolved
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def _write_compressed_json(path: Path, payload: Mapping[str, Any]) -> None:
    # A sibling temporary file keeps replacement atomic on the target filesystem.
    # Empty filename and fixed mtime make bytes independent of path and wall time.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        temporary = Path(raw.name)
        try:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def emit_json(target: str | Path, payload: Mapping[str, Any]) -> Path | None:
    """Serialize ``payload`` exactly once for CLI ``--json`` output.

    ``"-"`` prints the document to stdout; anything else writes it through
    :func:`write_json`.  Returns the resolved output path, or ``None`` when
    the document went to stdout.
    """
    if str(target) == "-":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return None
    return write_json(target, payload)


__all__ = [
    "StructuredDataError",
    "emit_json",
    "load_mapping",
    "write_json",
]
