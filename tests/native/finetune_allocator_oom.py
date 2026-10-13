"""An isolated native-run allocator-budget failure, never a physical-card OOM test."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _budget(calibrations, expect):
    """Choose both controls before observing either OOM-control outcome."""
    quantum = 256 * 1024**2
    if expect == "oom":
        return (
            math.floor(
                0.75
                * min(run["peak_allocated_bytes"] for run in calibrations)
                / quantum
            )
            * quantum,
            "floor(0.75 * minimum_calibration_peak_allocated / 256MiB) * 256MiB",
        )
    if expect == "success":
        return (
            math.ceil(
                1.25 * max(run["peak_reserved_bytes"] for run in calibrations) / quantum
            )
            * quantum,
            "ceil(1.25 * maximum_calibration_peak_reserved / 256MiB) * 256MiB",
        )
    raise ValueError("expected outcome must be oom or success")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework", choices=["huggingface", "llamafactory", "axolotl"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--cuda-device",
        help="Optional CUDA_VISIBLE_DEVICES selection; exactly one device must be visible.",
    )
    parser.add_argument("--calibration-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-regime",
        choices=["empty_kernel_cache", "populated_kernel_cache"],
        required=True,
    )
    parser.add_argument("--expect", choices=["oom", "success"], required=True)
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if args.output.exists():
        raise ValueError("Preserve all evidence: probe output must be new")
    from fakegpu._finetune_model import native_config_signature
    from fakegpu.finetune_run_estimator import validate_native_run

    if len(args.calibration_run) < 2:
        raise ValueError(
            "Budget selection requires at least two native calibration runs"
        )
    raw = json.loads(args.config.read_text())
    calibrations = []
    for path in args.calibration_run:
        content = path.read_bytes()
        report = json.loads(content)
        validate_native_run(report, require_fixed_shape=False)
        if report["framework"] != args.framework or native_config_signature(
            report["native_config"]
        ) != native_config_signature(raw):
            raise ValueError(
                "OOM control must retain the native calibration execution configuration"
            )
        calibrations.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "run_identity": report["run_identity"],
                "peak_allocated_bytes": report["observation"]["peak_allocated_bytes"],
                "peak_reserved_bytes": report["observation"]["peak_reserved_bytes"],
            }
        )
    if len({run["run_identity"]["uuid"] for run in calibrations}) != len(calibrations):
        raise ValueError("OOM budget calibration UUIDs must be independent")
    budget, formula = _budget(calibrations, args.expect)
    import torch
    from fakegpu.finetune_runner import run_finetuning

    if torch.cuda.device_count() != 1:
        raise ValueError("Probe requires one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    total = properties.total_memory
    if not 0 < budget < total:
        raise ValueError(
            "Allocator budget must be positive and smaller than physical VRAM"
        )
    fraction = budget / total
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    evidence = {
        "schema_version": "fakegpu.native_allocator_oom_probe.v1",
        "identity": {
            "uuid": str(uuid4()),
            "pid": os.getpid(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "framework": args.framework,
        "config": str(args.config.resolve()),
        "hardware": {
            "name": properties.name,
            "total_memory_bytes": total,
            "physical_gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "allocator": {
            "requested_budget_bytes": budget,
            "requested_fraction": fraction,
            "reported_fraction": torch.cuda.get_per_process_memory_fraction(0),
        },
        "budget_selection": {"formula": formula, "calibration_runs": calibrations},
        "torch_version": torch.__version__,
        "physical_card_oom_test": False,
        "scope": "PyTorch per-process allocator-fraction budget only",
        "expected_outcome": args.expect,
    }
    peaks = {"allocated_bytes": 0, "reserved_bytes": 0}
    native_reset = torch.cuda.reset_peak_memory_stats

    def record_peaks():
        peaks["allocated_bytes"] = max(
            peaks["allocated_bytes"], torch.cuda.max_memory_allocated()
        )
        peaks["reserved_bytes"] = max(
            peaks["reserved_bytes"], torch.cuda.max_memory_reserved()
        )

    def preserve_peak_reset(*reset_args, **reset_kwargs):
        record_peaks()
        return native_reset(*reset_args, **reset_kwargs)

    torch.cuda.reset_peak_memory_stats = preserve_peak_reset
    started = time.monotonic()
    code = 0
    try:
        report = run_finetuning(
            args.config,
            framework=args.framework,
            profile_id=args.profile,
            warmup_steps=4,
            cache_regime=args.cache_regime,
        )
    except torch.cuda.OutOfMemoryError as exc:
        evidence.update(
            outcome="observed_allocator_budget_oom",
            exception_class=f"{type(exc).__module__}.{type(exc).__name__}",
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            traceback_frames=[
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(exc.__traceback__)
            ],
        )
        code = 86 if args.expect == "oom" else 1
    except Exception as exc:
        evidence.update(
            outcome="unexpected_failure",
            exception_class=f"{type(exc).__module__}.{type(exc).__name__}",
            exception_message=str(exc),
            traceback=traceback.format_exc(),
        )
        code = 1
    else:
        report_path = args.output.with_name(args.output.stem + "-native-report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        evidence.update(
            outcome="observed_native_success", native_report=str(report_path)
        )
        code = 0 if args.expect == "success" else 2
    finally:
        record_peaks()
        torch.cuda.reset_peak_memory_stats = native_reset
    evidence.update(
        elapsed_wall_seconds=time.monotonic() - started,
        final_allocated_bytes=torch.cuda.memory_allocated(),
        final_reserved_bytes=torch.cuda.memory_reserved(),
        allocator_peak_allocated_bytes=peaks["allocated_bytes"],
        allocator_peak_reserved_bytes=peaks["reserved_bytes"],
        peak_capture="maximum_across_intercepted_native_peak_resets",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": evidence["outcome"], "output": str(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
