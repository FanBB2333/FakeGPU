r"""Validate complete dynamic accumulation windows in native CUDA trainers.

The source config supplies the real model, dataset, sampler, seed and collator.
Only epochs/output paths change. Two independent calibration processes precede
longer holdouts, whose extra epochs exercise sampler orders absent from calibration.
A separate optional primer populates shared caches and is retained as evidence.

Example:
    python tests/native/finetune_window_matrix.py \
        --framework llamafactory --config /tmp/native.json \
        --python /tmp/factory/venv/bin/python --output-dir /tmp/window-matrix \
        --calibration-epochs 24 --holdout-epochs 40 --prime-cache

Every reported comparison is for whole optimizer-update windows. Neither batch
shape timing nor per-shape VRAM is inferred. No training starts on import.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

if __package__:
    from .finetune_framework_matrix import (
        _comparison,
        _relocate_config,
        _run,
        _write_json,
    )
else:
    from finetune_framework_matrix import (
        _comparison,
        _relocate_config,
        _run,
        _write_json,
    )


def _window_key(update: dict[str, Any]) -> str:
    return json.dumps(
        [
            {
                key: value
                for key, value in batch.items()
                if key not in {"epoch_index", "epoch_microstep"}
            }
            for batch in update["batches"]
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _retained_updates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Independent reference: discard warmup and each window's first occurrence."""
    seen = set()
    retained = []
    for update in report["observation"]["updates"]:
        key = _window_key(update)
        first = key not in seen
        seen.add(key)
        if not update["warmup"] and not first:
            retained.append(update)
    return retained


def _coverage(report: dict[str, Any]) -> list[dict[str, Any]]:
    counts = Counter(_window_key(update) for update in _retained_updates(report))
    all_keys = {_window_key(update) for update in report["observation"]["updates"]}
    return [
        {"batches": json.loads(key), "retained_updates": counts[key]}
        for key in sorted(all_keys)
    ]


def _read_report(path: Path, epochs: int, warmup_steps: int) -> dict[str, Any]:
    from fakegpu._finetune_batches import validate_batch_profile

    report = json.loads(path.read_text())
    if report.get("schema_version") != "fakegpu.finetune_run.v1":
        raise ValueError(f"{path}: unexpected native run schema")
    observation = report["observation"]
    validate_batch_profile(observation)
    config = report["training_config"]
    rows = config["num_samples"]
    batch = config["micro_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    microsteps_per_epoch = math.ceil(rows / batch)
    updates_per_epoch = math.ceil(microsteps_per_epoch / accumulation)
    expected = {
        "optimizer_steps": epochs * updates_per_epoch,
        "microsteps": epochs * microsteps_per_epoch,
        "samples": epochs * rows,
        "warmup_optimizer_steps": warmup_steps,
    }
    for field, value in expected.items():
        if report["workload"][field] != value:
            raise ValueError(
                f"{path}: actual {field} differs from native epoch schedule"
            )
    updates = observation["updates"]
    if updates[-1]["batches"][-1]["epoch_index"] != epochs:
        raise ValueError(f"{path}: missing final epoch metadata")
    if not any(
        record["shape"][0] < batch for update in updates for record in update["batches"]
    ):
        raise ValueError(f"{path}: fixture did not exercise partial microbatches")
    if not any(update["microsteps"] < accumulation for update in updates):
        raise ValueError(
            f"{path}: fixture did not exercise partial accumulation windows"
        )
    if report["workload"]["fixed_shape"]:
        raise ValueError(f"{path}: fixture must exercise dynamic shapes")
    return report


def _reference_comparison(
    calibrations: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    after_epoch: int,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Compute a heldout suffix independently from per-run window means."""
    per_run = []
    for report in calibrations:
        groups = defaultdict(list)
        for update in _retained_updates(report):
            groups[_window_key(update)].append(update["seconds"])
        per_run.append(groups)
    target_updates = [
        update
        for update in _retained_updates(target)
        if update["batches"][0]["epoch_index"] > after_epoch
    ]
    if not target_updates:
        raise ValueError("holdout has no updates beyond the calibration epochs")
    counts = Counter(_window_key(update) for update in target_updates)
    means = {}
    for key in counts:
        if any(len(groups.get(key, [])) < 2 for groups in per_run):
            raise ValueError(
                "holdout suffix window lacks repeated independent calibration"
            )
        means[key] = statistics.mean(statistics.mean(groups[key]) for groups in per_run)
    predicted = sum(means[key] * count for key, count in counts.items())
    actual = sum(update["seconds"] for update in target_updates)
    projected_updates = [
        update
        for update in projection["target_updates"]
        if update["eligible"] and update["epoch_index"] > after_epoch
    ]
    if [update["optimizer_step"] for update in projected_updates] != [
        update["optimizer_step"] for update in target_updates
    ]:
        raise ValueError(
            "production projection selects different holdout suffix updates"
        )
    projected_seconds = sum(
        update["predicted_seconds"]["expected"] for update in projected_updates
    )
    projected_actual = sum(update["observed_seconds"] for update in projected_updates)
    if not math.isclose(predicted, projected_seconds, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "production suffix prediction differs from independent window means"
        )
    if not math.isclose(actual, projected_actual, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "production suffix timing differs from original target updates"
        )
    result = {
        "epoch_start": after_epoch + 1,
        "epoch_end": max(
            update["batches"][0]["epoch_index"] for update in target_updates
        ),
        "optimizer_steps": len(target_updates),
        "microsteps": sum(update["microsteps"] for update in target_updates),
        "samples": sum(update["samples"] for update in target_updates),
        "training_update_seconds": _comparison(predicted, actual),
        "window_count": len(counts),
    }
    calibration_stream = [
        _window_key(update) for update in calibrations[0]["observation"]["updates"]
    ]
    suffix_stream = [_window_key(update) for update in target_updates]
    result["descriptor_stream_differs_from_calibration_prefix"] = (
        suffix_stream != calibration_stream[: len(suffix_stream)]
    )
    return result


def _check_process_identity(
    report: dict[str, Any],
    execution: dict[str, Any],
    *,
    started_at: float,
    finished_at: float,
    seen: set[str],
) -> dict[str, Any]:
    identity = report.get("run_identity")
    if not isinstance(identity, dict) or identity.get("pid") != execution["pid"]:
        raise ValueError(
            "report run_identity.pid differs from the actual child process"
        )
    timestamp = identity.get("started_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("report run_identity requires its actual UTC start timestamp")
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("report run_identity start timestamp must use UTC")
    if not started_at <= parsed.timestamp() <= finished_at:
        raise ValueError(
            "report run_identity timestamp falls outside child process lifetime"
        )
    run_uuid = identity.get("uuid")
    if not isinstance(run_uuid, str) or not run_uuid or run_uuid in seen:
        raise ValueError(
            "report run_identity UUID is missing or reused by another child"
        )
    seen.add(run_uuid)
    return copy.deepcopy(identity)


def _source_hashes(root: Path) -> dict[str, str]:
    paths = set()
    for pattern in ("fakegpu/finetune*.py", "fakegpu/_finetune*.py"):
        paths.update(root.glob(pattern))
    paths.update(
        root / relative
        for relative in (
            "fakegpu/_local_checkpoint.py",
            "fakegpu/_calibration_cuda.py",
            "tests/native/finetune_window_matrix.py",
            "tests/native/finetune_framework_matrix.py",
        )
    )
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", required=True, choices=("llamafactory", "axolotl")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell-72gb")
    parser.add_argument("--calibration-runs", type=int, default=2)
    parser.add_argument("--holdout-runs", type=int, default=2)
    parser.add_argument("--calibration-epochs", type=int, default=24)
    parser.add_argument("--holdout-epochs", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--prime-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.calibration_runs < 2 or args.holdout_runs < 1:
        parser.error("require at least two calibration runs and one holdout")
    if not 1 <= args.calibration_epochs < args.holdout_epochs:
        parser.error("require 1 <= calibration epochs < holdout epochs")
    if args.warmup_steps < 1 or args.gpu_index < 0:
        parser.error("warmup steps must be positive and GPU index nonnegative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be finite and positive")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fakegpu.structured_io import load_mapping

    source = Path(args.config).expanduser().resolve()
    directory = Path(args.output_dir).expanduser().resolve()
    try:
        raw = load_mapping(source)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if raw.get("resume_from_checkpoint") or raw.get("dataloader_drop_last"):
        parser.error("window matrix requires fresh epochs retaining tail batches")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    config_template = _relocate_config(raw, source, args.framework)
    epoch_key = "num_epochs" if args.framework == "axolotl" else "num_train_epochs"
    plan = (
        ([("primer", args.calibration_epochs)] if args.prime_cache else [])
        + [
            (f"calibration-{index + 1}", args.calibration_epochs)
            for index in range(args.calibration_runs)
        ]
        + [
            (f"holdout-{index + 1}", args.holdout_epochs)
            for index in range(args.holdout_runs)
        ]
    )
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
        "schema_version": "fakegpu.finetune_window_matrix.v1",
        "status": "prepared" if args.prepare_only else "running",
        "framework": args.framework,
        "source_config": str(source),
        "source_file_sha256": _source_hashes(root),
        "sampling_plan": vars(args),
        "runs": {},
        "holdout_comparisons": {},
        "limitations": [
            "Same seed, native sampler and dataset across processes; only epochs/output paths differ.",
            "Holdout suffix uses extra real sampler epochs, not a fabricated batch schedule.",
            "Shared compilation/filesystem caches persist; an optional primer is archived separately, never used for calibration.",
            "Each window's first occurrence and global warmup updates are excluded from calibration and holdout comparisons.",
            "Predictions cover measured optimizer-update work only, excluding setup, final saving, observation overhead and CUDA allocations outside PyTorch.",
            "Short synthetic fixtures validate accounting and timing, not convergence or model quality.",
        ],
    }
    summary_path = directory / "summary.json"
    calibrations = []
    identities: set[str] = set()
    _write_json(directory / "source-config.json", raw)
    _write_json(summary_path, summary)
    try:
        for role, epochs in plan:
            run_dir = directory / role
            run_dir.mkdir()
            config = copy.deepcopy(config_template)
            config.update(output_dir=str(run_dir / "model"), max_steps=-1)
            config[epoch_key] = epochs
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
            entry: dict[str, Any] = {
                "status": "prepared" if args.prepare_only else "running",
                "config": str(config_path),
                "report": str(report_path),
                "argv": command,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
            summary["runs"][role] = entry
            _write_json(summary_path, summary)
            if args.prepare_only:
                continue
            if _source_hashes(root) != summary["source_file_sha256"]:
                raise ValueError("execution source changed during the matrix")
            print(
                f"{args.framework}: {role}, {epochs} epochs, GPU {args.gpu_index}",
                flush=True,
            )
            started_at = time.time()
            execution = _run(
                command,
                log=run_dir / "run.log",
                root=root,
                environment=environment,
                timeout=args.timeout_seconds,
            )
            finished_at = time.time()
            entry.update(
                started_at_unix=started_at, finished_at_unix=finished_at, **execution
            )
            report = _read_report(report_path, epochs, args.warmup_steps)
            identity = _check_process_identity(
                report,
                execution,
                started_at=started_at,
                finished_at=finished_at,
                seen=identities,
            )
            entry.update(
                status="completed",
                run_identity=identity,
                coverage=_coverage(report),
                report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            if _source_hashes(root) != summary["source_file_sha256"]:
                raise ValueError("execution source changed during the matrix")
            if role.startswith("calibration-"):
                calibrations.append(report)
            elif role.startswith("holdout-"):
                from fakegpu.finetune_window_estimator import (
                    estimate_finetuning_windows,
                )

                projection = estimate_finetuning_windows(calibrations, report)
                suffix = _reference_comparison(
                    calibrations,
                    report,
                    after_epoch=args.calibration_epochs,
                    projection=projection,
                )
                if not suffix["descriptor_stream_differs_from_calibration_prefix"]:
                    raise ValueError(
                        "holdout suffix merely replays the calibration descriptor prefix"
                    )
                summary["holdout_comparisons"][role] = {
                    "estimate": projection,
                    "unseen_epoch_suffix": suffix,
                }
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
        print(f"Window matrix failed: {exc}; evidence: {summary_path}", file=sys.stderr)
        return 1
    if not args.prepare_only:
        summary["status"] = "completed"
    _write_json(summary_path, summary)
    print(f"Native window evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
