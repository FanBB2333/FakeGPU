#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "fakegpu.allocator_trace.v1"
COMPARISON_SCHEMA_VERSION = "fakegpu.allocator_trace_comparison.v1"
MIB = 1024**2
TRACE = (
    ("allocate", "small_a", 256 * 1024),
    ("allocate", "small_b", 512 * 1024),
    ("free", "small_a", 0),
    ("allocate", "small_c", 128 * 1024),
    ("free", "small_b", 0),
    ("free", "small_c", 0),
    ("empty_cache", "small_pool", 0),
    ("allocate", "medium_a", 3 * MIB),
    ("allocate", "medium_b", 4 * MIB),
    ("free", "medium_a", 0),
    ("free", "medium_b", 0),
    ("empty_cache", "medium_pool", 0),
    ("allocate", "large_a", 12 * MIB),
    ("free", "large_a", 0),
    ("empty_cache", "large_pool", 0),
)


def capture(*, mode: str, profile: str) -> dict[str, Any]:
    if mode == "fakecuda":
        os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")
        import fakegpu

        fakegpu.init(runtime="fakecuda", profile=profile, device_count=1)
    import torch

    if mode == "real" and not torch.cuda.is_available():
        raise RuntimeError("real capture requires a visible CUDA device")
    torch.cuda.empty_cache()
    gc.collect()
    baseline = _record(torch, "baseline")
    torch.cuda.reset_peak_memory_stats()
    tensors: dict[str, Any] = {}
    stages = []
    for index, (operation, name, nbytes) in enumerate(TRACE):
        if operation == "allocate":
            tensors[name] = torch.empty(
                nbytes // 4,
                dtype=torch.float32,
                device="cuda",
            )
        elif operation == "free":
            tensors.pop(name, None)
            gc.collect()
        elif operation == "empty_cache":
            torch.cuda.empty_cache()
        else:  # pragma: no cover - checked-in trace is fixed
            raise AssertionError(operation)
        if mode == "real":
            torch.cuda.synchronize()
        item = _record(torch, f"{index:02d}_{operation}_{name}")
        item.update(
            {
                "operation": operation,
                "allocation_name": name,
                "requested_bytes": nbytes,
                "allocated_delta_bytes": max(
                    0,
                    int(item["allocated_bytes"]) - int(baseline["allocated_bytes"]),
                ),
                "reserved_delta_bytes": max(
                    0,
                    int(item["reserved_bytes"]) - int(baseline["reserved_bytes"]),
                ),
            }
        )
        stages.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "mode": mode,
        "profile": profile if mode == "fakecuda" else None,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "baseline": baseline,
        "stages": stages,
    }


def compare(
    real: dict[str, Any],
    fake: dict[str, Any],
    *,
    reserved_tolerance_bytes: int = 2 * MIB,
) -> dict[str, Any]:
    if real.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("real report has an unsupported schema")
    if fake.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fakecuda report has an unsupported schema")
    real_stages = list(real.get("stages") or [])
    fake_stages = list(fake.get("stages") or [])
    if [item.get("label") for item in real_stages] != [
        item.get("label") for item in fake_stages
    ]:
        raise ValueError("real and fakecuda stage labels differ")

    rows = []
    for real_item, fake_item in zip(real_stages, fake_stages):
        real_allocated = int(real_item["allocated_delta_bytes"])
        fake_allocated = int(fake_item["allocated_delta_bytes"])
        real_reserved = int(real_item["reserved_delta_bytes"])
        fake_reserved = int(fake_item["reserved_delta_bytes"])
        rows.append(
            {
                "label": real_item["label"],
                "operation": real_item["operation"],
                "requested_bytes": int(real_item["requested_bytes"]),
                "real_allocated_bytes": real_allocated,
                "fakecuda_allocated_bytes": fake_allocated,
                "allocated_error_bytes": fake_allocated - real_allocated,
                "real_reserved_bytes": real_reserved,
                "fakecuda_reserved_bytes": fake_reserved,
                "reserved_error_bytes": fake_reserved - real_reserved,
                "absolute_reserved_error_bytes": abs(fake_reserved - real_reserved),
            }
        )
    max_allocated_error = max(
        (abs(int(item["allocated_error_bytes"])) for item in rows),
        default=0,
    )
    max_reserved_error = max(
        (int(item["absolute_reserved_error_bytes"]) for item in rows),
        default=0,
    )
    checks = {
        "allocated_trace_exact": max_allocated_error == 0,
        "reserved_trace_within_tolerance": (
            max_reserved_error <= reserved_tolerance_bytes
        ),
        "empty_cache_returns_to_baseline": all(
            int(item["fakecuda_reserved_bytes"]) == 0
            for item in rows
            if item["operation"] == "empty_cache"
        ),
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "success" if all(checks.values()) else "failed",
        "real": {
            "hostname": real.get("hostname"),
            "gpu_name": real.get("gpu_name"),
            "compute_capability": real.get("compute_capability"),
            "torch_version": real.get("torch_version"),
            "torch_cuda_version": real.get("torch_cuda_version"),
        },
        "fakecuda": {
            "hostname": fake.get("hostname"),
            "profile": fake.get("profile"),
            "torch_version": fake.get("torch_version"),
        },
        "reserved_tolerance_bytes": int(reserved_tolerance_bytes),
        "max_absolute_allocated_error_bytes": max_allocated_error,
        "max_absolute_reserved_error_bytes": max_reserved_error,
        "checks": checks,
        "stages": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        (
            f"| `{item['label']}` | {item['real_allocated_bytes']:,} | "
            f"{item['fakecuda_allocated_bytes']:,} | "
            f"{item['real_reserved_bytes']:,} | "
            f"{item['fakecuda_reserved_bytes']:,} | "
            f"{item['reserved_error_bytes']:+,} |"
        )
        for item in report["stages"]
    ]
    checks = [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in report["checks"].items()
    ]
    return "\n".join(
        [
            "# FakeGPU Allocator Trace Validation",
            "",
            f"**Status:** `{report['status']}`",
            "",
            f"- Real GPU: `{report['real']['gpu_name']}`",
            f"- Compute capability: `{report['real']['compute_capability']}`",
            f"- FakeCUDA profile: `{report['fakecuda']['profile']}`",
            f"- Maximum reserved error: "
            f"{report['max_absolute_reserved_error_bytes']:,} bytes",
            "",
            "## Checks",
            "",
            *checks,
            "",
            "| Stage | Real allocated | Fake allocated | Real reserved | Fake reserved | Reserved error |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )


def _record(torch: Any, label: str) -> dict[str, Any]:
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "inactive_split_bytes": int(
            stats.get("inactive_split_bytes.all.current", 0) or 0
        ),
        "segment_count": int(stats.get("segment.all.current", 0) or 0),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0) or 0),
        "num_ooms": int(stats.get("num_ooms", 0) or 0),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--mode", choices=["real", "fakecuda"], required=True)
    capture_parser.add_argument(
        "--profile",
        default="rtx-pro-5000-blackwell",
    )
    capture_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--real", type=Path, required=True)
    compare_parser.add_argument("--fakecuda", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--markdown", type=Path)
    compare_parser.add_argument(
        "--reserved-tolerance-bytes",
        type=int,
        default=2 * MIB,
    )
    args = parser.parse_args()

    if args.action == "capture":
        report = capture(mode=args.mode, profile=args.profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0

    real = json.loads(args.real.read_text(encoding="utf-8"))
    fake = json.loads(args.fakecuda.read_text(encoding="utf-8"))
    report = compare(
        real,
        fake,
        reserved_tolerance_bytes=args.reserved_tolerance_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(args.output)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
