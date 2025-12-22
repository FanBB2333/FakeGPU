from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_PRELOAD_LIBS: tuple[str, ...] = (
    "libcublas.so.12",
    "libcudart.so.12",
    "libcuda.so.1",
    "libnvidia-ml.so.1",
)

_state_lock = threading.Lock()
_initialized = False
_loaded_dir: Path | None = None
_handles: dict[str, ctypes.CDLL] = {}


def is_initialized() -> bool:
    return _initialized


def library_dir(*, build_dir: str | os.PathLike[str] | None = None, lib_dir: str | os.PathLike[str] | None = None) -> Path:
    """
    Resolve the directory that contains FakeGPU shared libraries.

    Resolution order:
    1) Explicit `lib_dir` / `build_dir`
    2) Environment: $FAKEGPU_LIB_DIR / $FAKEGPU_BUILD_DIR
    3) Installed package: `fakegpu/_native`
    4) Repo checkout: `./build` or `./build_asan` near CMakeLists.txt
    """

    candidates: list[Path] = []

    if lib_dir is not None:
        candidates.append(Path(lib_dir))
    if build_dir is not None:
        candidates.append(Path(build_dir))

    env_lib_dir = os.environ.get("FAKEGPU_LIB_DIR")
    if env_lib_dir:
        candidates.append(Path(env_lib_dir))
    env_build_dir = os.environ.get("FAKEGPU_BUILD_DIR")
    if env_build_dir:
        candidates.append(Path(env_build_dir))

    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / "_native")

    # If running from a repo checkout, discover build dirs near the repo root.
    for parent in (pkg_dir, *pkg_dir.parents):
        if (parent / "CMakeLists.txt").is_file():
            candidates.append(parent / "build")
            candidates.append(parent / "build_asan")
            break

    seen: set[Path] = set()
    ordered_unique: list[Path] = []
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        ordered_unique.append(p)

    for cand in ordered_unique:
        ok, _missing = _check_dir(cand)
        if ok:
            return cand

    details = "\n".join(
        f"- {p} (missing: {', '.join(_check_dir(p)[1])})" for p in ordered_unique if p.exists()
    )
    raise FileNotFoundError(
        "FakeGPU shared libraries not found.\n"
        "Build the project first (cmake -S . -B build && cmake --build build), or set $FAKEGPU_BUILD_DIR.\n"
        + (f"Tried:\n{details}\n" if details else "")
    )


@dataclass(frozen=True)
class InitResult:
    lib_dir: Path
    handles: Mapping[str, ctypes.CDLL]


def init(
    *,
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    update_env: bool = True,
    force: bool = False,
) -> InitResult:
    """
    Dynamically load FakeGPU shared libraries into the current Python process.

    This must be called early (before importing frameworks like torch) to ensure FakeGPU wins library resolution.
    """

    global _initialized, _loaded_dir

    with _state_lock:
        if _initialized and not force:
            return InitResult(lib_dir=_loaded_dir or library_dir(build_dir=build_dir, lib_dir=lib_dir), handles=dict(_handles))

        resolved_dir = library_dir(build_dir=build_dir, lib_dir=lib_dir)
        if update_env:
            _apply_env_inplace(resolved_dir)

        mode = ctypes.RTLD_GLOBAL
        if hasattr(os, "RTLD_NOW"):
            mode |= os.RTLD_NOW

        handles: dict[str, ctypes.CDLL] = {}
        for lib in _PRELOAD_LIBS:
            path = resolved_dir / lib
            if not path.exists():
                # Try common unversioned fallbacks (useful when packaging).
                alt_path = resolved_dir / _fallback_name(lib)
                if alt_path.exists():
                    path = alt_path
                else:
                    raise FileNotFoundError(f"Missing required FakeGPU library: {path}")
            handles[lib] = ctypes.CDLL(str(path), mode=mode)

        _handles.clear()
        _handles.update(handles)
        _loaded_dir = resolved_dir
        _initialized = True
        return InitResult(lib_dir=resolved_dir, handles=dict(handles))


def env(
    *,
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Build an environment dict that enables FakeGPU via LD_LIBRARY_PATH + LD_PRELOAD.

    Useful for subprocesses (equivalent to the repo's `./fgpu` script).
    """

    resolved_dir = library_dir(build_dir=build_dir, lib_dir=lib_dir)
    env_map = dict(os.environ) if base_env is None else dict(base_env)
    _apply_env(env_map, resolved_dir)
    return env_map


def run(
    cmd: Sequence[str],
    *,
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    check: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with FakeGPU enabled (see `env()`)."""

    completed = subprocess.run(
        list(cmd),
        check=check,
        text=True,
        env=env(build_dir=build_dir, lib_dir=lib_dir),
        **kwargs,
    )
    return completed


def _check_dir(path: Path) -> tuple[bool, list[str]]:
    if not path.is_dir():
        return False, list(_PRELOAD_LIBS)
    missing = [lib for lib in _PRELOAD_LIBS if not (path / lib).exists() and not (path / _fallback_name(lib)).exists()]
    return len(missing) == 0, missing


def _fallback_name(libname: str) -> str:
    # Best-effort mapping from versioned soname to common unversioned filename.
    if libname == "libcublas.so.12":
        return "libcublas.so"
    if libname == "libcudart.so.12":
        return "libcudart.so"
    if libname == "libcuda.so.1":
        return "libcuda.so"
    if libname == "libnvidia-ml.so.1":
        return "libnvidia-ml.so"
    return libname


def _apply_env_inplace(resolved_dir: Path) -> None:
    updated = env(lib_dir=resolved_dir, base_env=os.environ)  # type: ignore[arg-type]
    os.environ.update(updated)


def _apply_env(env_map: dict[str, str], resolved_dir: Path) -> None:
    dir_str = str(resolved_dir)

    ld_library_path = env_map.get("LD_LIBRARY_PATH", "")
    env_map["LD_LIBRARY_PATH"] = _prepend_path(dir_str, ld_library_path)

    preload_paths = [str(resolved_dir / name) for name in _PRELOAD_LIBS]
    existing_preload = env_map.get("LD_PRELOAD", "")
    env_map["LD_PRELOAD"] = ":".join(preload_paths + ([existing_preload] if existing_preload else []))


def _prepend_path(prefix: str, existing: str) -> str:
    if not existing:
        return prefix
    parts = [p for p in existing.split(":") if p]
    if parts and parts[0] == prefix:
        return existing
    return ":".join([prefix] + parts)
