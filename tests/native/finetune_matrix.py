r"""Collect independent CUDA fine-tuning observations and compare their predictions.

Requires one real CUDA GPU selected with CUDA_VISIBLE_DEVICES, a local Qwen3.5
checkpoint/tokenizer, and a Python environment with compatible torch,
Transformers, Accelerate and PEFT. --qlora additionally requires bitsandbytes.
This is an explicit GPU integration script, separate from CPU smoke tests.

Example:
    CUDA_VISIBLE_DEVICES=1 python tests/native/finetune_matrix.py \
        --model /models/Qwen3.5-0.8B --profile "$TARGET_GPU_PROFILE" \
        --report-dir /tmp/finetune-matrix --qlora

Each method runs calibration and holdouts in separate fresh processes. Defaults
are one holdout and four measured updates per process, after two warmup updates.
Use --holdout-runs and --holdout-steps for more independent, longer holdouts.
Results report errors without a universal accuracy threshold across stacks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


WARMUP_STEPS = 2
MEASURE_STEPS = 4
ACCUMULATION = 2
SEQUENCE_LENGTH = 128


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _configuration(model: Path, method: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "framework": "huggingface",
        "model_name_or_path": str(model),
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "sequence_length": SEQUENCE_LENGTH,
        "num_samples": 1024,
        "training_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": ACCUMULATION,
            "num_train_epochs": 3,
            "gradient_checkpointing": False,
            "bf16": True,
            "tf32": False,
            "optim": "adamw_torch",
            "lr_scheduler_type": "constant",
            "learning_rate": 2e-5,
        },
    }
    if method in {"lora", "qlora"}:
        config["peft"] = {
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
            "bias": "none", "target_modules": ["q_proj", "v_proj"],
        }
    if method == "qlora":
        config["quantization_config"] = {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
        }
    return config


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return float(value)


def _observation(path: Path, *, measure_steps: int = MEASURE_STEPS) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "fakegpu.finetune_observation.v1":
        raise ValueError(f"{path}: unexpected observation schema")
    if payload.get("source") != "fakegpu.huggingface_trainer_cuda_probe":
        raise ValueError(f"{path}: observation is not from the real Trainer probe")
    _positive(payload.get("training_loss"), "training_loss")
    _positive(payload.get("peak_allocated_bytes"), "peak_allocated_bytes")
    _positive(payload.get("peak_reserved_bytes"), "peak_reserved_bytes")
    if payload["peak_reserved_bytes"] < payload["peak_allocated_bytes"]:
        raise ValueError(f"{path}: reserved peak must cover allocated peak")
    updates = payload.get("optimizer_update_seconds")
    microsteps = payload.get("seconds_per_microstep")
    if not isinstance(updates, list) or len(updates) != measure_steps:
        raise ValueError(f"{path}: expected exactly {measure_steps} measured updates")
    if not isinstance(microsteps, list) or len(microsteps) != measure_steps:
        raise ValueError(f"{path}: missing per-microstep timing samples")
    for update, microstep in zip(updates, microsteps):
        _positive(update, "optimizer_update_seconds")
        _positive(microstep, "seconds_per_microstep")
        if not math.isclose(update, microstep * ACCUMULATION, rel_tol=1e-9):
            raise ValueError(f"{path}: accumulation/timing contract mismatch")
    workload = payload.get("workload", {})
    expected = {
        "sequence_length": SEQUENCE_LENGTH,
        "samples": (WARMUP_STEPS + measure_steps) * ACCUMULATION,
        "warmup_optimizer_steps": WARMUP_STEPS,
        "measured_optimizer_steps": measure_steps,
    }
    if any(workload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path}: measured workload differs from the requested matrix")
    hardware = payload.get("hardware", {})
    if not hardware.get("name") or not hardware.get("cuda_version"):
        raise ValueError(f"{path}: missing real CUDA hardware metadata")
    _positive(hardware.get("total_memory_bytes"), "total_memory_bytes")
    if not payload.get("software_versions") or not payload.get("signature"):
        raise ValueError(f"{path}: missing software identity or workload signature")
    return payload


def _comparison(predicted: float, observed: float) -> dict[str, float]:
    _positive(predicted, "predicted")
    _positive(observed, "observed")
    return {
        "predicted": predicted,
        "observed": observed,
        "absolute_percentage_error": abs(predicted - observed) / observed * 100,
        "underprediction_percent": max(observed - predicted, 0) / observed * 100,
    }


def _compare_holdout(
    calibration: dict[str, Any], holdout: dict[str, Any], analytical: dict[str, Any],
) -> dict[str, dict[str, float]]:
    return {
        "optimizer_update_seconds": _comparison(
            statistics.mean(calibration["optimizer_update_seconds"]),
            statistics.mean(holdout["optimizer_update_seconds"]),
        ),
        "allocated_peak_bytes": _comparison(
            calibration["peak_allocated_bytes"], holdout["peak_allocated_bytes"],
        ),
        "reserved_peak_bytes": _comparison(
            calibration["peak_reserved_bytes"], holdout["peak_reserved_bytes"],
        ),
        "analytical_allocated_peak_bytes": _comparison(
            analytical["memory"]["peak_bytes"]["expected"], holdout["peak_allocated_bytes"],
        ),
    }


def _run(
    arguments: list[str], *, log: Path, root: Path,
    environment: dict[str, str], timeout: float,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "fakegpu", *arguments]
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.run(
            command, cwd=root, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"{arguments[0]} exited {process.returncode}; see {log}")
    return {"argv": command, "log": str(log), "duration_seconds": time.monotonic() - started}


def _method(
    method: str, *, model: Path, profile: str, directory: Path,
    root: Path, environment: dict[str, str], timeout: float,
    holdout_runs: int = 1, holdout_steps: int = MEASURE_STEPS,
) -> dict[str, Any]:
    directory.mkdir()
    config_path = directory / "config.json"
    _write_json(config_path, _configuration(model, method))
    common = ["--config", str(config_path), "--profile", profile, "--text-only"]
    commands = []
    observations = {}
    runs = [("calibration", 42, MEASURE_STEPS)] + [
        ("holdout" if index == 0 else f"holdout-{index + 1}", 43 + index, holdout_steps)
        for index in range(holdout_runs)
    ]
    for role, seed, steps in runs:
        print(f"{method}: collecting {role} in a fresh CUDA process", flush=True)
        path = directory / f"{role}.json"
        commands.append(_run([
            "probe-finetune", *common, "--warmup-steps", str(WARMUP_STEPS),
            "--measure-steps", str(steps), "--seed", str(seed), "--output", str(path),
        ], log=directory / f"{role}.log", root=root, environment=environment, timeout=timeout))
        observations[role] = _observation(path, measure_steps=steps)
    calibration = observations["calibration"]
    for role, observation in observations.items():
        if calibration["signature"] != observation["signature"]:
            raise ValueError(f"{method}: calibration and {role} execution signatures differ")
    predictions = {}
    for role in ("analytical", "calibrated"):
        path = directory / f"{role}-estimate.json"
        options = ["--observation", str(directory / "calibration.json")] if role == "calibrated" else []
        commands.append(_run(
            ["estimate-finetune", *common, *options, "--json", str(path)],
            log=directory / f"{role}-estimate.log", root=root, environment=environment, timeout=timeout,
        ))
        predictions[role] = json.loads(path.read_text())
    analytical, calibrated = predictions["analytical"], predictions["calibrated"]
    if calibrated["timing"]["evidence"] != "matched_real_measurement_extrapolation":
        raise ValueError(f"{method}: estimator did not verify the real-GPU observation metadata")
    comparisons_by_run = {
        role: _compare_holdout(calibration, observation, analytical)
        for role, observation in observations.items() if role != "calibration"
    }
    worst_errors = {
        metric: {
            "maximum_absolute_percentage_error": max(
                comparisons[metric]["absolute_percentage_error"]
                for comparisons in comparisons_by_run.values()
            ),
            "maximum_underprediction_percent": max(
                comparisons[metric]["underprediction_percent"]
                for comparisons in comparisons_by_run.values()
            ),
        }
        for metric in comparisons_by_run["holdout"]
    }
    return {
        "status": "completed",
        "config": str(config_path),
        "observations": observations,
        "commands": commands,
        "predictions": {
            "analytical": str(directory / "analytical-estimate.json"),
            "calibrated": str(directory / "calibrated-estimate.json"),
        },
        "holdout_comparisons": comparisons_by_run["holdout"],
        "holdout_comparisons_by_run": comparisons_by_run,
        "worst_holdout_errors": worst_errors,
        "target_workload": calibrated["workload"],
        "extrapolated_training_loop_seconds": calibrated["timing"]["training_loop_seconds"],
        "calibrated_memory_fit": calibrated["memory"]["fit_status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local Qwen3.5 checkpoint and tokenizer directory.")
    parser.add_argument("--profile", required=True, help="Profile matching the selected physical CUDA GPU.")
    parser.add_argument("--report-dir", required=True, help="New or empty directory for raw logs, observations and summary.")
    parser.add_argument("--qlora", action="store_true", help="Also run NF4 QLoRA through bitsandbytes and PEFT.")
    parser.add_argument("--holdout-runs", type=int, default=1,
                        help="Independent holdout processes per method (>=1, default: 1).")
    parser.add_argument("--holdout-steps", type=int, default=MEASURE_STEPS,
                        help="Measured optimizer updates per holdout (>=2, default: 4); calibration always measures 4.")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="Timeout per child process (default: 1800).")
    args = parser.parse_args(argv)
    model = Path(args.model).expanduser().resolve()
    directory = Path(args.report_dir).expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error("--model must contain config.json and local pretrained weights/tokenizer")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    if args.holdout_runs < 1:
        parser.error("--holdout-runs must be >= 1")
    if args.holdout_steps < 2:
        parser.error("--holdout-steps must be >= 2")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or visible == "-1":
        parser.error("select one physical GPU first, for example CUDA_VISIBLE_DEVICES=1")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("--report-dir must be new or empty; existing evidence will not be overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), environment.get("PYTHONPATH"))))
    summary: dict[str, Any] = {
        "schema_version": "fakegpu.finetune_matrix.v1",
        "status": "running", "model": str(model), "profile_id": args.profile,
        "python": sys.executable, "cuda_visible_devices": visible,
        "methodology": (
            f"Each method uses {1 + args.holdout_runs} independent CUDA processes: "
            f"one calibration with {WARMUP_STEPS} warmup + {MEASURE_STEPS} measured updates, "
            f"and {args.holdout_runs} holdouts with {WARMUP_STEPS} warmup + "
            f"{args.holdout_steps} measured updates each; every process uses a different seed."
        ),
        "sampling_plan": {
            "calibration_runs": 1, "holdout_runs": args.holdout_runs,
            "warmup_optimizer_steps": WARMUP_STEPS,
            "calibration_measured_optimizer_steps": MEASURE_STEPS,
            "holdout_measured_optimizer_steps": args.holdout_steps,
        },
        "accuracy_gate": "report_only_no_universal_threshold",
        "methods": {},
        "limitations": [
            "Synthetic fixed-length text and single-GPU Hugging Face Trainer only.",
            f"Each method has {1 + args.holdout_runs} independent process runs; measured updates within a process are not independent process observations.",
            "The holdout validates step-time and peak-memory extrapolation, not a complete multi-epoch target run.",
            "Total training-loop estimates exclude model initialization, tokenization, evaluation and checkpoint I/O.",
            "Allocator peaks exclude driver context and other non-PyTorch GPU allocations.",
            "Successful collection is not an accuracy certification or model-quality assessment.",
        ],
    }
    summary_path = directory / "summary.json"
    _write_json(summary_path, summary)
    try:
        for method in (["full", "lora", "qlora"] if args.qlora else ["full", "lora"]):
            summary["methods"][method] = _method(
                method, model=model, profile=args.profile, directory=directory / method,
                root=root, environment=environment, timeout=args.timeout_seconds,
                holdout_runs=args.holdout_runs, holdout_steps=args.holdout_steps,
            )
            _write_json(summary_path, summary)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as exc:
        summary.update(status="failed", error=str(exc))
        _write_json(summary_path, summary)
        print(f"Matrix failed: {exc}; partial evidence: {summary_path}", file=sys.stderr)
        return 1
    summary["status"] = "completed"
    _write_json(summary_path, summary)
    print(f"Observations and comparisons written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
