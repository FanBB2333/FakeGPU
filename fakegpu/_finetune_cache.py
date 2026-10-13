"""Describe explicitly selected kernel-cache directories before native execution.

This records file state, not a claim that every kernel will hit a cache or that
the OS page cache, dataset cache, clocks, and system load are controlled.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path


SCHEMA_VERSION = "fakegpu.finetune_cache_regime.v1"
CACHE_VARIABLES = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "PYTORCH_KERNEL_CACHE_PATH",
)
MODES = ("uncontrolled", "empty_kernel_cache", "populated_kernel_cache")


def capture_cache_regime(mode: str = "uncontrolled", *, environ=None) -> dict:
    """Inspect caller-created directories; never create, clear, or warm caches."""
    if mode not in MODES:
        raise ValueError(f"cache regime must be one of {', '.join(MODES)}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "scope": "specified_kernel_cache_directories",
        "files": [],
    }
    if mode == "uncontrolled":
        return result
    env = os.environ if environ is None else environ
    roots = []
    for variable in CACHE_VARIABLES:
        value = env.get(variable)
        if not value:
            raise ValueError(f"controlled kernel caches require explicit {variable}")
        root = Path(value).expanduser()
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError(
                f"{variable} must name an existing absolute cache directory"
            )
        root = root.resolve()
        if any(
            root == other or root in other.parents or other in root.parents
            for other in roots
        ):
            raise ValueError(
                "controlled cache directories must be distinct and disjoint"
            )
        roots.append(root)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("controlled kernel caches cannot contain symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("controlled kernel caches require ordinary files")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result["files"].append(
                {
                    "cache": variable,
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    if mode == "empty_kernel_cache" and result["files"]:
        raise ValueError(
            "empty_kernel_cache requires all selected directories to be empty"
        )
    if mode == "populated_kernel_cache" and not result["files"]:
        raise ValueError(
            "populated_kernel_cache requires a recorded nonempty cache seed"
        )
    return result


def validate_cache_regime(value: Mapping) -> None:
    """Reject malformed regime evidence rather than silently mixing cache states."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "mode", "scope", "files"}
        or value["schema_version"] != SCHEMA_VERSION
        or value["mode"] not in MODES
        or value["scope"] != "specified_kernel_cache_directories"
        or not isinstance(value["files"], list)
    ):
        raise ValueError("invalid native kernel cache regime")
    seen = set()
    for entry in value["files"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "cache",
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError("invalid native kernel cache file")
        path, size, digest = entry["path"], entry["bytes"], entry["sha256"]
        if (
            entry["cache"] not in CACHE_VARIABLES
            or not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or any(part in {".", ".."} for part in path.split("/"))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid native kernel cache file identity")
        key = (entry["cache"], path)
        if key in seen:
            raise ValueError("duplicate native kernel cache file")
        seen.add(key)
    if bool(value["files"]) != (value["mode"] == "populated_kernel_cache"):
        raise ValueError("native kernel cache contents disagree with declared regime")
