r"""Compare native periodic evaluation/checkpoint events in independent runs.

The source configuration owns the dataset and evaluation/checkpoint schedules.
The matrix changes only max_steps and fresh output paths. Its independent event
reference describes the installed Transformers 5.6/5.14 DefaultFlowCallback and
the checked Factory/Axolotl finalization paths; other versions must establish
their own reference before reusing this acceptance fixture.

Example:
    python tests/native/finetune_event_matrix.py \
        --framework llamafactory --config /tmp/events.json \
        --python /tmp/factory/venv/bin/python --output-dir /tmp/event-matrix

No training starts on import. Event comparisons do not predict total wall time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

if __package__:
    from .finetune_framework_matrix import _relocate_config, _run, _write_json
    from .finetune_window_matrix import _check_process_identity, _source_hashes
else:
    from finetune_framework_matrix import _relocate_config, _run, _write_json
    from finetune_window_matrix import _check_process_identity, _source_hashes


def _expected_calls(
    framework: str,
    steps: int,
    *,
    strategy: str,
    updates_per_epoch: int,
    eval_steps: int = 4,
    save_steps: int = 6,
) -> list[dict[str, Any]]:
    """Reference includes terminal-step and post-train calls, without deduping."""
    if framework not in {"llamafactory", "axolotl"}:
        raise ValueError("unsupported native framework")
    if strategy not in {"steps", "epoch"}:
        raise ValueError("event matrix requires step or epoch schedules")
    if min(steps, updates_per_epoch, eval_steps, save_steps) < 1:
        raise ValueError("event schedule counts must be positive")
    calls = []

    def add(operation: str, step: int, inside: bool = True):
        calls.append(
            {"operation": operation, "optimizer_step": step, "inside_train": inside}
        )

    # Axolotl's native argument builder enables eval_on_start for both strategies.
    if framework == "axolotl":
        add("evaluate", 0)
    for step in range(1, steps + 1):
        terminal = step == steps
        if strategy == "epoch":
            evaluate = save = terminal or step % updates_per_epoch == 0
        else:
            evaluate = terminal or step % eval_steps == 0
            save = terminal or step % save_steps == 0
        if evaluate:
            add("evaluate", step)
        if save:
            add("_save_checkpoint", step)
    if framework == "llamafactory":
        add("save_model", steps, False)
        add("evaluate", steps, False)
    else:
        add("save_pretrained", steps, False)
    return calls


def _check_calls(report: dict[str, Any], expected: list[dict[str, Any]]) -> None:
    roots = [
        event
        for event in report["observation"]["phase_events"]
        if event["parent_event_id"] is None
    ]
    actual = [
        {
            field: event[field]
            for field in ("operation", "optimizer_step", "inside_train")
        }
        for event in roots
    ]
    if actual != expected:
        raise ValueError(
            f"native event calls differ from independent reference: {actual}"
        )
    if any(event["status"] != "succeeded" for event in roots):
        raise ValueError("native event ledger contains an unsuccessful root event")


def _checkpoint_inventory(
    output: Path, expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Inspect every actual saved checkpoint, including optimizer/RNG state."""
    steps = [
        row["optimizer_step"]
        for row in expected
        if row["operation"] == "_save_checkpoint"
    ]
    directories = sorted(output.glob("checkpoint-*"))
    if {path.name for path in directories} != {f"checkpoint-{step}" for step in steps}:
        raise ValueError(
            "saved checkpoint directories differ from calls; possible rotation"
        )
    result = []
    for directory in directories:
        required = {
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        }
        files = {
            str(path.relative_to(directory)): path
            for path in directory.rglob("*")
            if path.is_file()
        }
        if not required <= files.keys():
            raise ValueError(f"{directory}: missing resumable checkpoint files")
        state = json.loads(files["trainer_state.json"].read_text())
        if state["global_step"] != int(directory.name.removeprefix("checkpoint-")):
            raise ValueError("saved Trainer state disagrees with checkpoint directory")
        result.append(
            {
                "directory": str(directory),
                "global_step": state["global_step"],
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files.values()),
                "files": {
                    name: {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in sorted(files.items())
                },
            }
        )
    if not (output / "adapter_model.safetensors").is_file():
        raise ValueError("native final adapter was not saved")
    return result


def _reference_groups(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Independent event grouping; deliberately does not import estimator helpers."""
    groups = defaultdict(list)
    seen = set()
    events = report["observation"]["phase_events"]
    children = defaultdict(list)
    for event in events:
        children[event["parent_event_id"]].append(event)

    def subtree(parent):
        result = []
        for child in children[parent]:
            result.append(child)
            result.extend(subtree(child["event_id"]))
        return sorted(result, key=lambda event: event["event_id"])

    for event in children[None]:
        work = event["workload"]
        if event["kind"] == "evaluation":
            descriptor = {
                "dataset": {
                    field: work["dataset"][field]
                    for field in ("content_sha256", "num_samples")
                },
                "batches": [
                    {
                        key: value
                        for key, value in batch.items()
                        if key not in {"epoch_index", "epoch_microstep"}
                    }
                    for batch in work["batches"]
                ],
                **{
                    field: work[field]
                    for field in (
                        "consumed_content_sha256",
                        "prediction_loss_only",
                        "predict_with_generate",
                        "compute_metrics",
                        "eval_batch_size",
                        "dataloader_drop_last",
                    )
                },
            }
        else:
            descriptor = {
                "save_only_model": work["save_only_model"],
                "artifact_kind": work["artifact_kind"],
                "file_layout": sorted(
                    re.sub(
                        r"(^|/)checkpoint-\d+(/|$)", r"\1checkpoint-N\2", row["path"]
                    )
                    for row in work["files"]
                ),
            }
        nested = subtree(event["event_id"])
        operation_indices = {event["event_id"]: 0}
        operation_indices.update(
            (child["event_id"], index) for index, child in enumerate(nested, 1)
        )
        key_data = {
            "kind": event["kind"],
            "operation": event["operation"],
            "inside_train": event["inside_train"],
            "workload": descriptor,
            "nested_operations": [
                {
                    "kind": child["kind"],
                    "operation": child["operation"],
                    "parent_operation_index": operation_indices[
                        child["parent_event_id"]
                    ],
                }
                for child in nested
            ],
        }
        base = json.dumps(
            key_data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if event["optimizer_step"] == 0:
            phase = "initial"
        elif not event["inside_train"]:
            phase = "final"
        else:
            phase = "repeated" if base in seen else "first"
            seen.add(base)
        key_data["phase"] = phase
        key = json.dumps(
            key_data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        groups[key].append(event)
    return groups


def _reference_comparison(calibrations, target, projection):
    calibration_groups = [_reference_groups(report) for report in calibrations]
    groups = _reference_groups(target)
    predicted, actual = 0.0, 0.0
    expected_ids = []
    projected_groups = {
        group["event_group_id"]: group for group in projection["event_groups"]
    }
    for key, events in groups.items():
        group_id = hashlib.sha256(key.encode()).hexdigest()
        expected_ids.append(group_id)
        phase = json.loads(key)["phase"]
        minimum = 2 if phase == "repeated" else 1
        if any(len(sample.get(key, [])) < minimum for sample in calibration_groups):
            raise ValueError("independent event reference lacks calibration coverage")
        per_run = [
            statistics.mean(event["seconds"] for event in sample[key])
            for sample in calibration_groups
        ]
        group_predicted = statistics.mean(per_run) * len(events)
        group_actual = sum(event["seconds"] for event in events)
        projected = projected_groups.get(group_id)
        if projected is None or projected["target_event_ids"] != [
            event["event_id"] for event in events
        ]:
            raise ValueError(
                "production event grouping differs from independent reference"
            )
        for expected_value, projected_value in (
            (group_predicted, projected["predicted_seconds"]["expected"]),
            (group_actual, projected["comparison"]["actual"]),
        ):
            if not math.isclose(
                expected_value, projected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "production event timing differs from independent reference"
                )
        predicted += group_predicted
        actual += group_actual
    if set(expected_ids) != set(projected_groups):
        raise ValueError("production event grouping contains unexpected groups")
    if not math.isclose(
        predicted,
        projection["timing"]["root_event_seconds"]["expected"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("production event total differs from independent reference")
    return {
        "root_events": sum(map(len, groups.values())),
        "event_groups": len(groups),
        "predicted_seconds": predicted,
        "observed_seconds": actual,
        "absolute_percentage_error": abs(predicted - actual) / actual * 100,
    }


def _validate_template(raw: dict[str, Any], framework: str) -> str:
    strategy = raw.get("eval_strategy")
    if strategy not in {"steps", "epoch"} or raw.get("save_strategy") != strategy:
        raise ValueError(
            "matrix requires matching explicit step/epoch eval/save strategies"
        )
    if strategy == "steps" and any(
        type(raw.get(field)) is not int or raw[field] < 1
        for field in ("eval_steps", "save_steps")
    ):
        raise ValueError("step matrix requires positive integer eval/save intervals")
    if strategy == "epoch" and (raw.get("eval_steps") or raw.get("save_steps")):
        raise ValueError("epoch matrix must omit step intervals")
    if raw.get("save_total_limit") != 100 or raw.get("save_only_model") is not False:
        raise ValueError(
            "matrix requires save_total_limit=100 and complete checkpoint state"
        )
    if raw.get("resume_from_checkpoint") or raw.get("dataloader_drop_last"):
        raise ValueError("matrix requires fresh training retaining tail batches")
    if framework == "llamafactory" and (
        raw.get("do_eval") is not True or raw.get("eval_on_start", False)
    ):
        raise ValueError(
            "Factory matrix requires final evaluation and no initial evaluation"
        )
    return strategy


def _read_report(
    path: Path, framework: str, steps: int, strategy: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from fakegpu._finetune_batches import validate_batch_profile

    report = json.loads(path.read_text())
    if report.get("schema_version") != "fakegpu.finetune_run.v1":
        raise ValueError("unexpected native run schema")
    validate_batch_profile(report["observation"])
    if report["workload"]["optimizer_steps"] != steps:
        raise ValueError("native optimizer steps differ from requested run length")
    cfg = report["training_config"]
    schedule = report["schedule"]
    if any(schedule[field] != strategy for field in ("eval_strategy", "save_strategy")):
        raise ValueError("native event strategies differ from the source configuration")
    intervals = {}
    if strategy == "steps":
        for field in ("eval_steps", "save_steps"):
            configured = report["native_config"][field]
            requested, resolved = (
                schedule[f"requested_{field}"],
                schedule[f"resolved_{field}"],
            )
            if any(
                type(value) is not int or value < 1
                for value in (configured, requested, resolved)
            ):
                raise ValueError("native event intervals must remain positive integers")
            if configured != requested or requested != resolved:
                raise ValueError(
                    "native requested/resolved event intervals differ from configuration"
                )
            intervals[field] = resolved
    updates_per_epoch = math.ceil(
        math.ceil(cfg["num_samples"] / cfg["micro_batch_size"])
        / cfg["gradient_accumulation_steps"]
    )
    expected = _expected_calls(
        framework,
        steps,
        strategy=strategy,
        updates_per_epoch=updates_per_epoch,
        **intervals,
    )
    _check_calls(report, expected)
    return report, expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", required=True, choices=("llamafactory", "axolotl")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell-72gb")
    parser.add_argument("--calibration-steps", type=int, default=20)
    parser.add_argument("--holdout-steps", type=int, default=29)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="One accounting-only run; no event estimate.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.calibration_steps, args.holdout_steps) < args.warmup_steps + 2
        or args.warmup_steps < 1
    ):
        parser.error("runs need at least two optimizer updates after positive warmup")
    if (
        args.gpu_index < 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("GPU index must be nonnegative and timeout finite and positive")
    if not args.smoke_only and args.holdout_steps <= args.calibration_steps:
        parser.error("holdout must extend beyond calibration")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fakegpu.structured_io import load_mapping

    source, directory = (
        Path(args.config).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
    try:
        raw = load_mapping(source)
        strategy = _validate_template(raw, args.framework)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error("output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    template = _relocate_config(raw, source, args.framework)
    plan = (
        [("smoke", args.calibration_steps)]
        if args.smoke_only
        else [
            (f"{role}-{index}", steps)
            for role, steps in (
                ("calibration", args.calibration_steps),
                ("holdout", args.holdout_steps),
            )
            for index in (1, 2)
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

    def hashes():
        result = _source_hashes(root)
        result[str(Path(__file__).relative_to(root))] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return result

    summary = {
        "schema_version": "fakegpu.finetune_event_matrix.v1",
        "framework": args.framework,
        "status": "prepared" if args.prepare_only else "running",
        "sampling_plan": vars(args),
        "source_file_sha256": hashes(),
        "runs": {},
        "holdout_comparisons": {},
        "limitations": [
            "Comparison covers separately observed events, not overall training wall time or model quality.",
            "Shared compilation and filesystem caches persist; no controlled cold-cache claim.",
            "Full periodic checkpoints retain optimizer/scheduler/RNG state; final model saves have different contents.",
        ],
    }
    summary_path = directory / "summary.json"
    _write_json(directory / "source-config.json", raw)
    _write_json(summary_path, summary)
    calibrations, identities = [], set()
    try:
        for role, steps in plan:
            run_dir = directory / role
            run_dir.mkdir()
            config = copy.deepcopy(template)
            config.update(output_dir=str(run_dir / "model"), max_steps=steps)
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
            entry = {
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
            if hashes() != summary["source_file_sha256"]:
                raise ValueError("execution source changed during matrix")
            print(
                f"{args.framework}: {role}, {steps} updates, GPU {args.gpu_index}",
                flush=True,
            )
            started = time.time()
            execution = _run(
                command,
                log=run_dir / "run.log",
                root=root,
                environment=environment,
                timeout=args.timeout_seconds,
            )
            finished = time.time()
            report, expected = _read_report(
                report_path, args.framework, steps, strategy
            )
            identity = _check_process_identity(
                report,
                execution,
                started_at=started,
                finished_at=finished,
                seen=identities,
            )
            entry.update(
                status="completed",
                run_identity=identity,
                started_at_unix=started,
                finished_at_unix=finished,
                expected_calls=expected,
                checkpoint_inventory=_checkpoint_inventory(run_dir / "model", expected),
                root_event_counts=dict(Counter(row["operation"] for row in expected)),
                report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                **execution,
            )
            if hashes() != summary["source_file_sha256"]:
                raise ValueError("execution source changed during matrix")
            if role.startswith("calibration-"):
                calibrations.append(report)
            elif role.startswith("holdout-"):
                from fakegpu.finetune_event_estimator import estimate_finetuning_events

                projection = estimate_finetuning_events(calibrations, report)
                summary["holdout_comparisons"][role] = {
                    "estimate": projection,
                    "independent_reference": _reference_comparison(
                        calibrations, report, projection
                    ),
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
        print(f"Event matrix failed: {exc}; evidence: {summary_path}", file=sys.stderr)
        return 1
    if not args.prepare_only:
        summary["status"] = "completed"
    _write_json(summary_path, summary)
    print(f"Native event evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
