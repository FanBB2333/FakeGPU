from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .profile_catalog import (
    ProfileCatalogError,
    catalog_summary,
    get_profile,
    load_profiles,
    validate_catalog,
)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _check(status: str, name: str, detail: str) -> dict[str, str]:
    return {"status": status, "name": name, "detail": detail}


def _collect(profile_id: str) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    profiles = {}
    selected = None
    summary: dict[str, Any] = {}

    try:
        profiles = load_profiles()
        summary = catalog_summary(profiles)
        validation = validate_catalog(profiles)
        if validation.errors:
            for message in validation.errors:
                checks.append(_check("fail", "profile catalog", message))
        else:
            checks.append(
                _check(
                    "pass",
                    "profile catalog",
                    (
                        f"{summary['profile_count']} profiles, "
                        f"{len(summary['architectures'])} architectures, "
                        f"{len(summary['compute_capabilities'])} compute capabilities"
                    ),
                )
            )
        for message in validation.warnings:
            checks.append(_check("warn", "profile catalog", message))

        selected = get_profile(profile_id, profiles=profiles)
        checks.append(
            _check(
                "pass",
                "selected profile",
                (
                    f"{selected.id}: {selected.name}, {selected.architecture.title()}, "
                    f"cc {selected.compute_capability_text}, "
                    f"{_format_bytes(selected.memory_bytes)} {selected.memory_kind} memory"
                ),
            )
        )
    except ProfileCatalogError as exc:
        checks.append(_check("fail", "profile catalog", str(exc)))

    try:
        from ._api import library_dir

        native_dir = library_dir()
        checks.append(_check("pass", "native libraries", str(native_dir)))
    except (FileNotFoundError, OSError) as exc:
        first_line = str(exc).splitlines()[0]
        checks.append(
            _check(
                "warn",
                "native libraries",
                f"{first_line}; fakecuda remains available without a native build",
            )
        )

    torch_info: dict[str, Any] = {"installed": False}
    if importlib.util.find_spec("torch") is None:
        checks.append(
            _check(
                "warn",
                "PyTorch",
                "not installed; fakegpu demo and the fakecuda runtime need PyTorch",
            )
        )
    else:
        try:
            import torch

            torch_info = {
                "installed": True,
                "version": str(torch.__version__),
                "cuda_build": getattr(torch.version, "cuda", None),
                "real_cuda_available": bool(torch.cuda.is_available()),
                "real_cuda_device_count": (
                    int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
                ),
            }
            checks.append(
                _check(
                    "pass",
                    "PyTorch",
                    (
                        f"{torch_info['version']}, "
                        f"CUDA build {torch_info['cuda_build'] or 'none'}"
                    ),
                )
            )
            if torch_info["real_cuda_available"]:
                checks.append(
                    _check(
                        "info",
                        "real CUDA",
                        f"{torch_info['real_cuda_device_count']} CUDA device(s) visible",
                    )
                )
            else:
                checks.append(
                    _check(
                        "info",
                        "real CUDA",
                        "no real CUDA device visible; this is valid for fakecuda",
                    )
                )
        except Exception as exc:
            checks.append(
                _check("warn", "PyTorch", f"import failed: {type(exc).__name__}: {exc}")
            )

    return {
        "ok": not any(item["status"] == "fail" for item in checks),
        "fakegpu_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package_path": str(Path(__file__).resolve().parent),
        "profile_summary": summary,
        "selected_profile": selected.to_dict() if selected is not None else None,
        "profiles": [profile.to_dict() for profile in profiles.values()],
        "torch": torch_info,
        "checks": checks,
    }


def _print_profile_table(profiles: list[dict[str, Any]]) -> None:
    if not profiles:
        return
    print("GPU profiles")
    print(f"{'ID':<33} {'ARCH':<10} {'CC':<5} {'MEMORY':>10}  STATUS")
    for profile in profiles:
        print(
            f"{profile['id']:<33} "
            f"{profile['architecture']:<10} "
            f"{profile['compute_capability']:<5} "
            f"{_format_bytes(int(profile['memory_bytes'])):>10}  "
            f"{profile['profile_status']}"
        )
    print()


def _print_plain(payload: dict[str, Any], *, list_profiles: bool) -> None:
    if list_profiles:
        _print_profile_table(payload["profiles"])
    print(
        f"FakeGPU {payload['fakegpu_version']} doctor "
        f"(Python {payload['python']['version']})"
    )
    labels = {
        "pass": "PASS",
        "warn": "WARN",
        "info": "INFO",
        "fail": "FAIL",
    }
    for item in payload["checks"]:
        print(
            f"[{labels[item['status']]:4}] "
            f"{item['name']}: {item['detail']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu doctor",
        description="Inspect FakeGPU profiles, native libraries, and PyTorch availability.",
    )
    parser.add_argument(
        "--profile",
        default="a100",
        help="Profile ID to inspect (default: a100).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List every built-in profile with architecture and compute capability.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for warnings as well as failures.",
    )
    args = parser.parse_args(argv)

    payload = _collect(args.profile)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plain(payload, list_profiles=args.list_profiles)

    has_failure = any(item["status"] == "fail" for item in payload["checks"])
    has_warning = any(item["status"] == "warn" for item in payload["checks"])
    return 1 if has_failure or (args.strict and has_warning) else 0


if __name__ == "__main__":
    raise SystemExit(main())
