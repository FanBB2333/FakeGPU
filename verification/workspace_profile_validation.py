#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAPTURE_SCHEMA_VERSION = "fakegpu.workspace_capture.v1"
CATALOG_SCHEMA_VERSION = "fakegpu.workspace_profiles.v1"
MIB = 1024**2
MM_OPERATOR = "aten::mm(Tensor self, Tensor mat2) -> Tensor"
CONV_OPERATOR = (
    "aten::convolution(Tensor input, Tensor weight, Tensor? bias, "
    "SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, "
    "SymInt[] output_padding, SymInt groups) -> Tensor"
)
WORKLOADS = (
    "mm-fp32-1024",
    "mm-bf16-1024",
    "conv-fp32-n2-c64-h128-k64",
    "conv-bf16-n2-c64-h128-k64",
    "conv-fp32-n4-c128-h64-k128",
)


def capture(*, profile_id: str, warmups: int = 3) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("workspace capture requires a visible CUDA device")
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    measurements = []
    for name in WORKLOADS:
        measurements.append(_measure_workload(torch, name, warmups=warmups))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "status": "success",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "profile_id": profile_id,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "warmups": int(warmups),
        "measurements": measurements,
    }


def build_catalog(reports: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = []
    seen_ids: set[str] = set()
    for report in reports:
        if report.get("schema_version") != CAPTURE_SCHEMA_VERSION:
            raise ValueError("workspace capture report has an unsupported schema")
        for measurement in report.get("measurements") or []:
            profile_id = _catalog_profile_id(report, measurement)
            if profile_id in seen_ids:
                raise ValueError(f"duplicate generated profile id {profile_id!r}")
            seen_ids.add(profile_id)
            profiles.append(
                {
                    "id": profile_id,
                    "operator": measurement["operator"],
                    "lifetime": "operator_local",
                    "kind": "operator_workspace_storage",
                    "priority": 100,
                    "confidence": "measured_physical_gpu_exact_stack_shape",
                    "match": {
                        "device_types": ["cuda"],
                        "profile_ids": [report["profile_id"]],
                        "compute_capabilities": [report["compute_capability"]],
                        "torch_versions": [report["torch_version"]],
                        "cuda_versions": [report["torch_cuda_version"]],
                        "input_dtypes": measurement["input_dtypes"],
                        "input_shapes": measurement["input_shapes"],
                    },
                    "bytes": int(measurement["allocated_workspace_peak_bytes"]),
                    "validated_envelope": {
                        "gpu_name": report["gpu_name"],
                        "profile_id": report["profile_id"],
                        "compute_capability": report["compute_capability"],
                        "torch_version": report["torch_version"],
                        "cuda_version": report["torch_cuda_version"],
                        "cudnn_version": report.get("cudnn_version"),
                        "workload": measurement["name"],
                        "warmups": report["warmups"],
                    },
                }
            )
    profiles.sort(key=lambda item: str(item["id"]))
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "profiles": profiles,
    }


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        for measurement in report.get("measurements") or []:
            by_workload.setdefault(str(measurement["name"]), []).append(
                {
                    "profile_id": report["profile_id"],
                    "gpu_name": report["gpu_name"],
                    "compute_capability": report["compute_capability"],
                    "torch_version": report["torch_version"],
                    "torch_cuda_version": report["torch_cuda_version"],
                    "allocated_workspace_peak_bytes": int(
                        measurement["allocated_workspace_peak_bytes"]
                    ),
                    "reserved_workspace_peak_bytes": int(
                        measurement["reserved_workspace_peak_bytes"]
                    ),
                }
            )
    workloads = []
    for name, observations in sorted(by_workload.items()):
        allocated = [
            int(item["allocated_workspace_peak_bytes"]) for item in observations
        ]
        workloads.append(
            {
                "name": name,
                "observation_count": len(observations),
                "allocated_workspace_min_bytes": min(allocated),
                "allocated_workspace_max_bytes": max(allocated),
                "allocated_workspace_range_bytes": max(allocated) - min(allocated),
                "portable_across_observed_stacks": len(set(allocated)) == 1,
                "observations": observations,
            }
        )
    return {
        "schema_version": "fakegpu.workspace_capture_summary.v1",
        "status": "success",
        "report_count": len(reports),
        "workloads": workloads,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for workload in summary["workloads"]:
        values = ", ".join(
            f"{item['profile_id']}={item['allocated_workspace_peak_bytes']:,}"
            for item in workload["observations"]
        )
        rows.append(
            f"| `{workload['name']}` | {values} | "
            f"{workload['allocated_workspace_range_bytes']:,} | "
            f"{'yes' if workload['portable_across_observed_stacks'] else 'no'} |"
        )
    return "\n".join(
        [
            "# Backend Workspace Profile Validation",
            "",
            f"- Reports: {summary['report_count']}",
            "",
            "| Workload | Allocated workspace bytes | Range | Exact across stacks |",
            "|---|---|---:|---|",
            *rows,
            "",
        ]
    )


def _measure_workload(torch: Any, name: str, *, warmups: int) -> dict[str, Any]:
    function, inputs, operator = _workload(torch, name)
    for _ in range(max(1, warmups)):
        warmup_output = function(*inputs)
        torch.cuda.synchronize()
        del warmup_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = int(torch.cuda.memory_allocated())
    before_reserved = int(torch.cuda.memory_reserved())
    output = function(*inputs)
    torch.cuda.synchronize()
    after_allocated = int(torch.cuda.memory_allocated())
    after_reserved = int(torch.cuda.memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    output_bytes = _unique_storage_bytes(output)
    measurement = {
        "name": name,
        "operator": operator,
        "input_shapes": [_shape(tensor) for tensor in inputs],
        "input_dtypes": [str(tensor.dtype) for tensor in inputs],
        "input_bytes": sum(_unique_storage_bytes(tensor) for tensor in inputs),
        "output_shapes": [_shape(tensor) for tensor in _tensor_leaves(output)],
        "output_dtypes": [str(tensor.dtype) for tensor in _tensor_leaves(output)],
        "output_bytes": output_bytes,
        "before_allocated_bytes": before_allocated,
        "after_allocated_bytes": after_allocated,
        "peak_allocated_bytes": peak_allocated,
        "before_reserved_bytes": before_reserved,
        "after_reserved_bytes": after_reserved,
        "peak_reserved_bytes": peak_reserved,
        "allocated_workspace_peak_bytes": max(
            0,
            peak_allocated - max(before_allocated, after_allocated),
        ),
        "reserved_workspace_peak_bytes": max(
            0,
            peak_reserved - max(before_reserved, after_reserved),
        ),
    }
    del output
    del inputs
    gc.collect()
    return measurement


def _workload(torch: Any, name: str):
    device = torch.device("cuda")
    if name == "mm-fp32-1024":
        inputs = (
            torch.randn((1024, 1024), device=device, dtype=torch.float32),
            torch.randn((1024, 1024), device=device, dtype=torch.float32),
        )
        return torch.mm, inputs, MM_OPERATOR
    if name == "mm-bf16-1024":
        inputs = (
            torch.randn((1024, 1024), device=device, dtype=torch.bfloat16),
            torch.randn((1024, 1024), device=device, dtype=torch.bfloat16),
        )
        return torch.mm, inputs, MM_OPERATOR
    if name in {
        "conv-fp32-n2-c64-h128-k64",
        "conv-bf16-n2-c64-h128-k64",
    }:
        dtype = torch.bfloat16 if "bf16" in name else torch.float32
        inputs = (
            torch.randn((2, 64, 128, 128), device=device, dtype=dtype),
            torch.randn((64, 64, 3, 3), device=device, dtype=dtype),
        )
        return (
            lambda value, weight: torch.nn.functional.conv2d(
                value,
                weight,
                padding=1,
            ),
            inputs,
            CONV_OPERATOR,
        )
    if name == "conv-fp32-n4-c128-h64-k128":
        inputs = (
            torch.randn((4, 128, 64, 64), device=device, dtype=torch.float32),
            torch.randn((128, 128, 3, 3), device=device, dtype=torch.float32),
        )
        return (
            lambda value, weight: torch.nn.functional.conv2d(
                value,
                weight,
                padding=1,
            ),
            inputs,
            CONV_OPERATOR,
        )
    raise ValueError(f"unknown workspace workload {name!r}")


def _catalog_profile_id(
    report: dict[str, Any],
    measurement: dict[str, Any],
) -> str:
    torch_version = str(report["torch_version"]).split("+", 1)[0]
    cuda_version = str(report["torch_cuda_version"] or "none")
    raw = (
        f"{measurement['name']}.{report['profile_id']}."
        f"torch-{torch_version}.cuda-{cuda_version}"
    )
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).lower()


def _tensor_leaves(value: Any) -> list[Any]:
    try:
        import torch
    except Exception:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensor_leaves(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_tensor_leaves(item))
        return result
    return []


def _unique_storage_bytes(value: Any) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in _tensor_leaves(value):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return total


def _shape(tensor: Any) -> list[int]:
    return [int(value) for value in tuple(tensor.shape)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--profile-id", required=True)
    capture_parser.add_argument("--warmups", type=int, default=3)
    capture_parser.add_argument("--output", type=Path, required=True)

    build_parser = subparsers.add_parser("build-catalog")
    build_parser.add_argument("reports", nargs="+", type=Path)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--summary", type=Path)
    build_parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.action == "capture":
        report = capture(profile_id=args.profile_id, warmups=args.warmups)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    catalog = build_catalog(reports)
    summary = summarize_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = args.summary or args.output.with_name(
        f"{args.output.stem}-summary.json"
    )
    markdown_path = args.markdown or summary_path.with_suffix(".md")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
