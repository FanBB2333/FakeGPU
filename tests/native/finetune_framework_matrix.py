r"""Run native framework calibration and holdouts in separate CUDA processes.

Example:
    python tests/native/finetune_framework_matrix.py \
        --framework axolotl --config /tmp/axolotl.yaml \
        --python /tmp/axolotl/venv/bin/python --output-dir /tmp/axolotl-matrix

Requires an installed native framework, local model and dataset, and physical GPU.
Calibration defaults to six optimizer updates; holdouts default to 32 each.
The first two updates are warmup by default, included in native wall time.
Sequence length, batch size, accumulation and seed come from the source config.
No training starts when this module is imported.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any


CALIBRATION_STEPS = 6
WARMUP_STEPS = 2


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def _comparison(predicted: float, actual: float) -> dict[str, float]:
    _number(predicted, "prediction", positive=True)
    _number(actual, "actual", positive=True)
    return {
        "predicted": predicted,
        "actual": actual,
        "absolute_percentage_error": abs(predicted - actual) / actual * 100,
        "underprediction_percent": max(actual - predicted, 0) / actual * 100,
    }


def _validate_workload(report: dict[str, Any], steps: int) -> dict[str, Any]:
    from fakegpu.finetune_run_estimator import estimate_finetuning_run

    estimate = estimate_finetuning_run(report, max_steps=steps)
    for measured, projected in (
        ("optimizer_steps", "optimizer_steps"),
        ("microsteps", "microsteps"),
        ("samples", "processed_samples"),
    ):
        if report["workload"][measured] != estimate["workload"][projected]:
            raise ValueError(
                f"native {measured} differs from the configured training workload"
            )
    return estimate


def _read_report(
    path: Path, steps: int, warmup_steps: int = WARMUP_STEPS
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema_version") != "fakegpu.finetune_run.v1":
        raise ValueError(f"{path}: unexpected native run schema")
    observation = report["observation"]
    updates = observation["optimizer_update_seconds"]
    if len(updates) != steps - warmup_steps:
        raise ValueError(f"{path}: incorrect number of measured optimizer updates")
    if report["workload"]["warmup_optimizer_steps"] != warmup_steps:
        raise ValueError(f"{path}: incorrect number of warmup optimizer updates")
    for value in updates:
        _number(value, "optimizer_update_seconds", positive=True)
    _number(report["timing"]["total_wall_seconds"], "total_wall_seconds", positive=True)
    for key in (
        "setup_seconds",
        "trainer_wall_seconds",
        "training_loop_seconds",
        "evaluation_seconds",
        "checkpoint_seconds",
        "finalization_seconds",
    ):
        _number(report["timing"][key], key)
    _validate_workload(report, steps)
    return report


def _relocate_config(
    raw: dict[str, Any], source: Path, framework: str
) -> dict[str, Any]:
    """Keep local paths relative to their original config, preserving Hub IDs."""
    config = copy.deepcopy(raw)
    base = source.parent

    def local_path(value: Any, *, always_local: bool = False) -> Any:
        if not isinstance(value, str) or not value or "://" in value:
            return value
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else base / path
        if always_local or value.startswith((".", "~", "/")) or candidate.exists():
            return str(candidate.resolve())
        return value

    for field in (
        "model_name_or_path",
        "base_model",
        "base_model_config",
        "tokenizer_config",
    ):
        if field in config:
            config[field] = local_path(config[field])
    for field in (
        "cache_dir",
        "tokenized_path",
        "dataset_prepared_path",
        "logging_dir",
    ):
        if field in config:
            config[field] = local_path(config[field], always_local=True)
    if framework == "llamafactory":
        config["dataset_dir"] = local_path(
            config.get("dataset_dir", "data"), always_local=True
        )
        if "media_dir" in config:
            config["media_dir"] = local_path(config["media_dir"], always_local=True)
    for field in ("datasets", "test_datasets", "pretraining_dataset"):
        entries = config.get(field, [])
        if not isinstance(entries, list):
            continue
        for dataset in entries:
            if not isinstance(dataset, dict):
                continue
            original_path = dataset.get("path")
            resolved_path = local_path(original_path)
            if "path" in dataset:
                dataset["path"] = resolved_path
            # A Hub dataset's data_files are repository paths, not local paths.
            local_files = isinstance(original_path, str) and original_path in {
                "json",
                "csv",
                "text",
                "parquet",
                "arrow",
            }
            local_files = local_files or resolved_path != original_path
            if isinstance(resolved_path, str):
                local_files = local_files or Path(resolved_path).is_absolute()
            if local_files and "data_files" in dataset:
                files = dataset["data_files"]
                dataset["data_files"] = (
                    [local_path(item, always_local=True) for item in files]
                    if isinstance(files, list)
                    else local_path(files, always_local=True)
                )
            if "chat_template_jinja" in dataset:
                template = dataset["chat_template_jinja"]
                if isinstance(template, str) and not any(
                    marker in template for marker in ("{{", "{%", "{#", "\n")
                ):
                    dataset["chat_template_jinja"] = local_path(template)
    return config


def _run(
    command: list[str],
    *,
    log: Path,
    root: Path,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f"native training exited {returncode}; see {log}")
    return {
        "argv": command,
        "log": str(log),
        "pid": process.pid,
        "process_wall_seconds": time.perf_counter() - started,
    }


def _summarize(
    calibration: dict[str, Any], holdout: dict[str, Any], holdout_steps: int
) -> dict[str, Any]:
    from fakegpu.finetune_run_estimator import estimate_finetuning_run

    if calibration["observation"]["signature"] != holdout["observation"]["signature"]:
        raise ValueError("calibration and holdout workload signatures differ")
    cal_timing, held_timing = calibration["timing"], holdout["timing"]
    mean_update = statistics.mean(
        calibration["observation"]["optimizer_update_seconds"]
    )
    held_mean = statistics.mean(holdout["observation"]["optimizer_update_seconds"])
    estimate = estimate_finetuning_run(calibration, max_steps=holdout_steps)
    for measured, projected in (
        ("microsteps", "microsteps"),
        ("samples", "processed_samples"),
    ):
        if holdout["workload"][measured] != estimate["workload"][projected]:
            raise ValueError(f"holdout {measured} differs from the projected workload")
    pipeline = estimate["timing"]["native_pipeline"]
    if not pipeline["available"]:
        raise ValueError(
            "native pipeline projection unavailable: " + "; ".join(pipeline["reasons"])
        )
    fixed = pipeline["fixed_cost_seconds"]
    predicted = pipeline["seconds"]["expected"]
    return {
        "optimizer_update_seconds": _comparison(mean_update, held_mean),
        "seconds_per_microstep": _comparison(
            estimate["timing"]["seconds_per_microstep"]["expected"],
            statistics.mean(holdout["observation"]["seconds_per_microstep"]),
        ),
        "steady_training_seconds": _comparison(
            estimate["timing"]["seconds_per_microstep"]["expected"]
            * holdout["workload"]["measured_microsteps"],
            sum(holdout["observation"]["optimizer_update_seconds"]),
        ),
        "native_pipeline_seconds": _comparison(
            predicted, held_timing["total_wall_seconds"]
        ),
        "calibration_fixed_overhead_seconds": fixed,
        "calibration_setup_seconds": cal_timing["setup_seconds"],
        "calibration_checkpoint_seconds": cal_timing["checkpoint_seconds"],
        "calibration_finalization_seconds": cal_timing["finalization_seconds"],
        "holdout_timing": held_timing,
        "holdout_workload": {
            key: holdout["workload"][key]
            for key in (
                "optimizer_steps",
                "microsteps",
                "samples",
                "measured_microsteps",
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--framework", required=True, choices=("llamafactory", "axolotl")
    )
    parser.add_argument(
        "--config", required=True, help="Native YAML or JSON framework config."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python with the selected framework installed.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="New or empty evidence directory."
    )
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell")
    parser.add_argument("--holdout-runs", type=int, default=3)
    parser.add_argument("--calibration-steps", type=int, default=CALIBRATION_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument(
        "--holdout-steps",
        type=int,
        default=32,
        help="Total optimizer updates per holdout, including warmup updates.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=1,
        help="Physical GPU index exposed to each child process.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.holdout_runs < 1 or args.warmup_steps < 1:
        parser.error("require --holdout-runs >= 1 and --warmup-steps >= 1")
    if min(args.calibration_steps, args.holdout_steps) < args.warmup_steps + 2:
        parser.error(
            "calibration and holdout each need at least two measured updates after warmup"
        )
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fakegpu.structured_io import load_mapping

    source = Path(args.config).expanduser().resolve()
    directory = Path(args.output_dir).expanduser().resolve()
    try:
        raw_config = load_mapping(source)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    shape_fields = (
        ("sequence_len", "micro_batch_size")
        if args.framework == "axolotl"
        else ("cutoff_len", "per_device_train_batch_size")
    )
    for key in (*shape_fields, "gradient_accumulation_steps"):
        value = raw_config.get(key)
        minimum = 2 if key == shape_fields[0] else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            parser.error(f"matrix requires integer {key} >= {minimum}; got {value!r}")
    if raw_config.get("resume_from_checkpoint"):
        parser.error("matrix requires fresh training without resume_from_checkpoint")
    relocated_config = _relocate_config(raw_config, source, args.framework)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("--output-dir must be new or empty; prior evidence is preserved")
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_index),
        PYTHONNOUSERSITE="1",
        AXOLOTL_DO_NOT_TRACK="1",
        TOKENIZERS_PARALLELISM="false",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root), environment.get("PYTHONPATH")))
    )
    summary: dict[str, Any] = {
        "schema_version": "fakegpu.finetune_framework_matrix.v1",
        "status": "running",
        "framework": args.framework,
        "source_config": str(source),
        "profile_id": args.profile,
        "python": args.python,
        "cuda_visible_devices": str(args.gpu_index),
        "sampling_plan": {
            "calibration_optimizer_steps": args.calibration_steps,
            "warmup_optimizer_steps_per_run": args.warmup_steps,
            "holdout_runs": args.holdout_runs,
            "holdout_optimizer_steps_per_run": args.holdout_steps,
        },
        "prediction_formula": "estimate_finetuning_run: calibration fixed-cost residual + target actual microsteps * calibration steady mean seconds per microstep",
        "accuracy_gate": "report_only_no_universal_threshold",
        "runs": {},
        "holdout_comparisons": {},
        "limitations": [
            "Each run uses an independent process; filesystem and compilation caches may persist between runs.",
            "Fixed overhead is inferred from one calibration, includes warmup and I/O, and is not a confidence interval.",
            "Framework-specific tokenization and optional kernels can differ; matching sequence lengths do not establish identical workloads across frameworks.",
            "Native wall time includes setup, training and final saving; child startup and report serialization are recorded separately as process wall time.",
            "These runs measure throughput on the supplied fixture, not training convergence or model quality.",
            "Optimizer updates can contain partial accumulation windows; per-microstep comparisons and production workload counts retain those windows.",
        ],
    }
    summary_path = directory / "summary.json"
    _write_json(directory / "source-config.json", raw_config)
    _write_json(summary_path, summary)
    calibration = None
    plan = [("calibration", args.calibration_steps)] + [
        (f"holdout-{index + 1}", args.holdout_steps)
        for index in range(args.holdout_runs)
    ]
    try:
        for role, steps in plan:
            run_dir = directory / role
            run_dir.mkdir()
            config = copy.deepcopy(relocated_config)
            config.update(
                output_dir=str(run_dir / "model"),
                max_steps=steps,
                save_strategy="no",
                eval_strategy="no",
            )
            config_path, report_path = run_dir / "config.json", run_dir / "report.json"
            _write_json(config_path, config)
            command = [
                args.python,
                "-m",
                "fakegpu",
                "run-finetune",
                "--config",
                str(config_path),
                "--framework",
                args.framework,
                "--profile",
                args.profile,
                "--warmup-steps",
                str(args.warmup_steps),
                "--output",
                str(report_path),
            ]
            summary["runs"][role] = {
                "status": "running",
                "config": str(config_path),
                "report": str(report_path),
            }
            _write_json(summary_path, summary)
            print(
                f"{args.framework}: {role}, {steps} updates, GPU {args.gpu_index}",
                flush=True,
            )
            execution = _run(
                command,
                log=run_dir / "run.log",
                root=root,
                environment=environment,
                timeout=args.timeout_seconds,
            )
            report = _read_report(report_path, steps, args.warmup_steps)
            summary["runs"][role].update(status="completed", **execution)
            if calibration is None:
                calibration = report
            else:
                summary["holdout_comparisons"][role] = _summarize(
                    calibration, report, steps
                )
            _write_json(summary_path, summary)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as exc:
        summary.update(status="failed", error=str(exc))
        _write_json(summary_path, summary)
        print(
            f"Matrix failed: {exc}; partial evidence: {summary_path}", file=sys.stderr
        )
        return 1
    summary["worst_holdout_errors"] = {
        metric: max(
            value[metric]["absolute_percentage_error"]
            for value in summary["holdout_comparisons"].values()
        )
        for metric in (
            "optimizer_update_seconds",
            "seconds_per_microstep",
            "steady_training_seconds",
            "native_pipeline_seconds",
        )
    }
    summary["status"] = "completed"
    _write_json(summary_path, summary)
    print(f"Native framework evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
