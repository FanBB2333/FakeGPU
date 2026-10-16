"""Validate future native pipeline and memory budgets in independent processes.

Each framework runs two calibrations and three longer holdouts per available
kernel-cache regime. Predictions are sealed before holdout execution. The first
empty-cache calibration supplies the immutable populated-cache seed, if its
controlled directories actually contain kernel artifacts. No GPU runs on import.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

if __package__:
    from .finetune_event_matrix import (
        _check_calls,
        _checkpoint_inventory,
        _reference_groups,
    )
    from .finetune_event_window_matrix import _reference_windows
    from .finetune_framework_matrix import _relocate_config, _write_json
    from .finetune_window_matrix import _check_process_identity, _source_hashes
else:
    from finetune_event_matrix import (
        _check_calls,
        _checkpoint_inventory,
        _reference_groups,
    )
    from finetune_event_window_matrix import _reference_windows
    from finetune_framework_matrix import _relocate_config, _write_json
    from finetune_window_matrix import _check_process_identity, _source_hashes


_FIELDS = ("wall_seconds", "instrumentation_seconds", "seconds")
_WORK = {"optimizer_update", "evaluation", "checkpoint"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _native_command(args, config_path, report_path, mode):
    return [
        args.python,
        "-m",
        "fakegpu",
        "run-finetune",
        "--framework",
        args.framework,
        "--config",
        str(config_path),
        "--profile",
        args.profile,
        "--warmup-steps",
        str(args.warmup_steps),
        "--cache-regime",
        mode,
        "--output",
        str(report_path),
    ]


def _expected_calls(framework, steps):
    calls = []

    def add(operation, step, inside=True):
        calls.append(
            {"operation": operation, "optimizer_step": step, "inside_train": inside}
        )

    if framework == "axolotl":
        add("evaluate", 0)
    elif framework not in {"huggingface", "llamafactory"}:
        raise ValueError("unsupported framework")
    for step in range(1, steps + 1):
        if step % 2 == 0 or step == steps:
            add("evaluate", step)
        if step % 4 == 0 or step == steps:
            add("_save_checkpoint", step)
    add("save_pretrained" if framework == "axolotl" else "save_model", steps, False)
    if framework == "llamafactory":
        add("evaluate", steps, False)
    return calls


def _validate_template(raw, framework):
    sampler = "curriculum_sampling" if framework == "axolotl" else "disable_shuffling"
    if raw.get(sampler) is not True:
        raise ValueError(f"future matrix requires {sampler}=true")
    if any(
        raw.get(key)
        for key in (
            "packing",
            "sample_packing",
            "eval_sample_packing",
            "resume_from_checkpoint",
            "dataloader_drop_last",
            "load_best_model_at_end",
            "save_only_model",
        )
    ):
        raise ValueError("matrix requires unpacked fresh training and full checkpoints")
    for kind, steps in (("eval", 2), ("save", 4)):
        if (
            raw.get(f"{kind}_strategy") != "steps"
            or type(raw.get(f"{kind}_steps")) is not int
            or raw[f"{kind}_steps"] != steps
        ):
            raise ValueError("matrix requires integer eval_steps=2 and save_steps=4")
    if raw.get("save_total_limit") != 100:
        raise ValueError("matrix requires save_total_limit=100 to retain checkpoints")


def _cache_directories(directory, *, seed=None):
    from fakegpu._finetune_cache import CACHE_VARIABLES, capture_cache_regime

    directory.mkdir(parents=True)
    environment = {}
    for variable in CACHE_VARIABLES:
        destination = directory / variable
        if seed is None:
            destination.mkdir()
        else:
            shutil.copytree(seed / variable, destination, copy_function=shutil.copyfile)
            for path in [destination, *destination.rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o644)
        environment[variable] = str(destination)
    mode = "empty_kernel_cache" if seed is None else "populated_kernel_cache"
    manifest = capture_cache_regime(mode, environ=environment)
    _write_json(
        directory.parent / "cache-layout.json",
        {"environment": environment, "input_manifest": manifest},
    )
    return environment, manifest


def _seed_cache(source, destination):
    """Keep the first calibration's actual post-run artifacts; never fake a seed."""
    from fakegpu._finetune_cache import CACHE_VARIABLES, capture_cache_regime

    if not any(path.is_file() for path in source.rglob("*")):
        return None
    original = capture_cache_regime(
        "populated_kernel_cache",
        environ={name: str(source / name) for name in CACHE_VARIABLES},
    )
    destination.mkdir()
    for variable in CACHE_VARIABLES:
        shutil.copytree(source / variable, destination / variable)
    environment = {name: str(destination / name) for name in CACHE_VARIABLES}
    manifest = capture_cache_regime("populated_kernel_cache", environ=environment)
    if manifest != original:
        raise ValueError("immutable seed copy differs from original cache artifacts")
    for path in [*destination.rglob("*"), destination]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return manifest


def _execute(command, *, directory, root, environment, timeout, expected_code=0):
    """Persist PID and exit status even when a native process fails."""
    started = time.time()
    execution = {
        "argv": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_returncode": expected_code,
    }
    path = directory / "execution.json"
    _write_json(path, execution)
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        execution["pid"] = process.pid
        _write_json(path, execution)
        try:
            execution["returncode"] = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            execution.update(returncode=process.returncode, interrupted=True)
            raise
        finally:
            execution.update(
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                process_wall_seconds=time.time() - started,
            )
            _write_json(path, execution)
    if execution["returncode"] != expected_code:
        raise RuntimeError(
            f"native process exited {execution['returncode']}; preserved {directory / 'run.log'}"
        )
    return execution, started, time.time()


def _reference_components(run):
    """Partition original interval ledgers, independently of production helpers."""
    observation = run["observation"]
    timeline = observation["timeline"]
    events = {
        event["event_id"]: event
        for event in observation["phase_events"]
        if event["parent_event_id"] is None
    }
    event_descriptors = {
        event["event_id"]: json.loads(key)
        for key, members in _reference_groups(run).items()
        for event in members
    }
    windows = _reference_windows(run, epoch_context=True)
    work, seen = [], set()
    for update, window in zip(observation["updates"], windows):
        first = window["key"] not in seen
        seen.add(window["key"])
        descriptor = {
            "kind": "optimizer_update",
            "window": json.loads(window["key"]),
            "stage": {
                "warmup_index": update["optimizer_step"] if update["warmup"] else None,
                "first_descriptor_occurrence": first,
            },
        }
        work.append(
            (
                update["start_seconds"],
                update["end_seconds"],
                {
                    "kind": "optimizer_update",
                    "optimizer_step": update["optimizer_step"],
                },
                descriptor,
                update["batch_instrumentation_seconds"],
            )
        )
    for event in events.values():
        descriptor = {
            "kind": event["kind"],
            "event": event_descriptors[event["event_id"]],
        }
        work.append(
            (
                event["start_seconds"],
                event["end_seconds"],
                {"kind": event["kind"], "event_id": event["event_id"]},
                descriptor,
                event["instrumentation_seconds"],
            )
        )
    for name, moment in (
        ("trainer_start", timeline["trainer"]["start_seconds"]),
        ("trainer_end", timeline["trainer"]["end_seconds"]),
        ("native_end", timeline["span"]["end_seconds"]),
    ):
        work.append((moment, moment, {"kind": name}, {"kind": name}, 0.0))
    work.sort(key=lambda item: (item[0], item[2]["kind"] in _WORK))
    result, seen_gaps = [], set()
    previous, previous_ref, cursor, region = (
        {"kind": "native_start"},
        {"kind": "native_start"},
        0.0,
        "setup",
    )

    def append(kind, descriptor, reference, wall, instrumentation):
        result.append(
            {
                "component_id": len(result) + 1,
                "kind": kind,
                "group_id": _digest(descriptor),
                "descriptor": descriptor,
                "reference": reference,
                "observed": {
                    "wall_seconds": wall,
                    "instrumentation_seconds": instrumentation,
                    "seconds": wall - instrumentation,
                },
            }
        )

    for start, end, reference, descriptor, instrumentation in work:
        kind = reference["kind"]
        semantic = (
            {"kind": kind, "group_id": _digest(descriptor)}
            if kind in _WORK
            else descriptor
        )
        gap = {"kind": region, "left": previous, "right": semantic}
        key = _canonical(gap)
        gap["phase"] = "repeated" if key in seen_gaps else "first"
        seen_gaps.add(key)
        metadata = sum(
            interval["seconds"]
            for interval in observation["instrumentation_intervals"]
            if cursor <= interval["start_seconds"] and interval["end_seconds"] <= start
        )
        append(
            region,
            gap,
            {"left": previous_ref, "right": reference},
            start - cursor,
            metadata,
        )
        if kind in _WORK:
            append(kind, descriptor, reference, end - start, instrumentation)
        elif kind == "trainer_start":
            region = "trainer_other"
        elif kind == "trainer_end":
            region = "finalization"
        previous, previous_ref, cursor = semantic, reference, end
    for field in _FIELDS:
        if not math.isclose(
            sum(row["observed"][field] for row in result),
            timeline["totals"][field],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("independent partition does not cover the native timeline")
    return result


def _reference_comparison(calibrations, target, prediction):
    by_run = []
    for calibration in calibrations:
        groups = defaultdict(list)
        for row in _reference_components(calibration):
            groups[row["group_id"]].append(row)
        by_run.append(groups)
    target_rows = _reference_components(target)
    if [(row["component_id"], row["group_id"]) for row in target_rows] != [
        (row["component_id"], row["group_id"])
        for row in prediction["target_components"]
    ]:
        raise ValueError(
            "production component grouping differs from independent reference"
        )
    result = {"by_kind": {}, "intervals": {}, "by_kind_intervals": {}}
    for kind in [None, *sorted({row["kind"] for row in target_rows})]:
        selected = [row for row in target_rows if kind is None or row["kind"] == kind]
        comparisons, intervals = {}, {}
        production = (
            prediction["timing"]
            if kind is None
            else prediction["timing"]["by_kind"][kind]
        )
        for field in _FIELDS:
            per_component = [
                [
                    statistics.mean(
                        row["observed"][field]
                        for row in process[target_row["group_id"]]
                    )
                    for process in by_run
                ]
                for target_row in selected
            ]
            interval = {
                name: sum(function(values) for values in per_component)
                for name, function in [
                    ("min", min),
                    ("expected", statistics.mean),
                    ("max", max),
                ]
            }
            for name, value in interval.items():
                if not math.isclose(
                    value, production[field][name], rel_tol=1e-10, abs_tol=1e-10
                ):
                    raise ValueError(
                        "production pipeline means differ from independent raw intervals"
                    )
            predicted = interval["expected"]
            actual = sum(row["observed"][field] for row in selected)
            comparisons[field] = {
                "predicted": predicted,
                "actual": actual,
                "absolute_percentage_error": abs(predicted - actual) / actual * 100
                if actual
                else None,
                "underprediction_percent": max(actual - predicted, 0) / actual * 100
                if actual
                else None,
            }
            intervals[field] = interval
        if kind is None:
            result.update(comparisons)
            result["intervals"] = intervals
        else:
            result["by_kind"][kind] = comparisons
            result["by_kind_intervals"][kind] = intervals
    return result


def _semantic_plan(plan):
    """Align root IDs with a CPU plan while retaining all schedule semantics."""
    result = deepcopy(plan)
    result.pop("source")
    result.pop("projection", None)
    identifiers = {
        row["event_id"]: index for index, row in enumerate(result["root_events"], 1)
    }
    for root in result["root_events"]:
        root["event_id"] = identifiers[root["event_id"]]
    for update in result["updates"]:
        update["preceding_event_ids"] = [
            identifiers[value] for value in update["preceding_event_ids"]
        ]
    for operation in result["operations"]:
        if "event_id" in operation:
            operation["event_id"] = identifiers[operation["event_id"]]
    return result


def _check_seal(directory, record):
    for filename, field in (
        ("future-plan.json", "plan_sha256"),
        ("future-prediction.json", "prediction_sha256"),
    ):
        if (
            hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            != record[field]
        ):
            raise ValueError(
                "sealed future plan or prediction changed after calibration"
            )


def _verify_report(report, *, framework, steps, warmup, expected_cache):
    from fakegpu._finetune_pipeline_plan import pipeline_plan_from_run

    plan = pipeline_plan_from_run(report)
    if (
        report["framework"] != framework
        or report["workload"]["optimizer_steps"] != steps
        or report["workload"]["warmup_optimizer_steps"] != warmup
    ):
        raise ValueError(
            "native framework or update counts differ from requested workload"
        )
    if report["cache_regime"] != expected_cache:
        raise ValueError("native cache evidence differs from its sealed input manifest")
    config, schedule = report["training_config"], report["schedule"]
    if (
        config["num_samples"] != 13
        or config["micro_batch_size"] != 3
        or config["gradient_accumulation_steps"] != 2
    ):
        raise ValueError(
            "matrix fixture requires 13 rows, microbatch 3 and accumulation 2"
        )
    for kind, interval in (("eval", 2), ("save", 4)):
        if (
            schedule[f"{kind}_strategy"] != "steps"
            or schedule[f"requested_{kind}_steps"] != interval
            or schedule[f"resolved_{kind}_steps"] != interval
        ):
            raise ValueError(
                "native event schedule differs from the configured intervals"
            )
    calls = _expected_calls(framework, steps)
    _check_calls(report, calls)
    if report["workload"]["fixed_shape"] or not any(
        batch["shape"][0] == 1
        for update in report["observation"]["updates"]
        for batch in update["batches"]
    ):
        raise ValueError(
            "fixture must retain variable lengths and its partial tail batch"
        )
    if not any(
        update["microsteps"] == 1 for update in report["observation"]["updates"]
    ):
        raise ValueError("fixture must retain partial accumulation")
    for root in plan["root_events"]:
        if root["descriptor"]["kind"] == "evaluation":
            work = root["descriptor"]["workload"]
            if (
                work["dataset"]["num_samples"] != 8
                or work["prediction_loss_only"] != [True] * 4
                or work["compute_metrics"]
                or work["predict_with_generate"]
            ):
                raise ValueError(
                    "fixture requires one loss-only eight-row evaluation split"
                )
    return plan, calls


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", choices=["huggingface", "llamafactory", "axolotl"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell-72gb")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--calibration-steps", type=int, default=68)
    parser.add_argument("--holdout-steps", type=int, default=92)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--with-oom-controls", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fakegpu._finetune_cache import CACHE_VARIABLES, capture_cache_regime
    from fakegpu.finetune_memory_estimator import (
        compare_finetuning_memory,
        estimate_finetuning_memory,
    )
    from fakegpu.finetune_pipeline_estimator import (
        compare_finetuning_pipeline,
        estimate_finetuning_pipeline,
    )
    from fakegpu.finetune_pipeline_plan import plan_finetuning_pipeline
    from fakegpu.structured_io import load_mapping

    if (
        not 1 <= args.warmup_steps < args.calibration_steps - 1 < args.holdout_steps - 1
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or args.gpu_index < 0
    ):
        parser.error(
            "require positive warmup, longer holdouts and a finite positive timeout"
        )
    source, directory = args.config.resolve(), args.output_dir.resolve()
    raw = load_mapping(source)
    _validate_template(raw, args.framework)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error(
            "matrix output must be new or empty; existing evidence is preserved"
        )
    directory.mkdir(parents=True, exist_ok=True)
    template = _relocate_config(raw, source, args.framework)
    for key in ("train_dataset", "eval_dataset"):
        if args.framework == "huggingface" and key in template:
            path = Path(template[key]).expanduser()
            template[key] = str(
                (path if path.is_absolute() else source.parent / path).resolve()
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
        for path in (root / "fakegpu").rglob("*.py"):
            result[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        for name in (
            "finetune_event_matrix.py",
            "finetune_event_window_matrix.py",
            "finetune_pipeline_matrix.py",
            "finetune_allocator_oom.py",
        ):
            path = Path(__file__).with_name(name)
            result[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return result

    summary = {
        "schema_version": "fakegpu.finetune_pipeline_matrix.v1",
        "framework": args.framework,
        "status": "prepared" if args.prepare_only else "running",
        "calibration_steps": args.calibration_steps,
        "holdout_steps": args.holdout_steps,
        "calibration_runs_per_regime": 2,
        "holdout_runs_per_regime": 3,
        "source_file_sha256": hashes(),
        "regimes": {},
        "oom_controls": {},
        "limitations": [
            "Only the five named kernel-cache directories are controlled. Dataset caches, filesystem page caches, GPU clocks and system load are inherited.",
            "Complete native span excludes report serialization and process exit. Separate process wall time is recorded.",
            "Allocator-budget OOM controls do not demonstrate physical-card OOM or general fragmentation safety.",
        ],
    }
    summary_path = directory / "summary.json"
    _write_json(directory / "source-config.json", raw)
    _write_json(summary_path, summary)
    if args.prepare_only:
        return 0
    seen, seed_manifest, empty_calibrations = set(), None, []
    seed = directory / "populated-seed"

    def check_sources():
        if hashes() != summary["source_file_sha256"]:
            raise RuntimeError(
                "collection or acceptance source changed during the matrix"
            )
        if seed_manifest is not None:
            current = capture_cache_regime(
                "populated_kernel_cache",
                environ={name: str(seed / name) for name in CACHE_VARIABLES},
            )
            if current != seed_manifest:
                raise RuntimeError("immutable populated kernel-cache seed changed")

    try:
        for mode in ("empty_kernel_cache", "populated_kernel_cache"):
            if mode == "populated_kernel_cache" and seed_manifest is None:
                summary["regimes"][mode] = {
                    "status": "unavailable",
                    "reason": "The first empty-cache calibration produced no files in any controlled kernel-cache directory; no dummy seed was created.",
                }
                _write_json(summary_path, summary)
                continue
            cohort = directory / mode
            cohort.mkdir()
            cohort_result = {"status": "running", "runs": {}, "holdout_comparisons": {}}
            summary["regimes"][mode] = cohort_result
            calibrations = []
            frozen_plan = frozen_prediction = None
            roles = [
                (f"calibration-{index}", args.calibration_steps) for index in (1, 2)
            ] + [(f"holdout-{index}", args.holdout_steps) for index in (1, 2, 3)]
            for role, steps in roles:
                check_sources()
                if frozen_prediction is not None:
                    _check_seal(cohort, cohort_result["sealed_prediction"])
                run_dir = cohort / role
                run_dir.mkdir()
                config = deepcopy(template)
                config.update(max_steps=steps, output_dir=str(run_dir / "model"))
                config_path, report_path = (
                    run_dir / "config.json",
                    run_dir / "report.json",
                )
                _write_json(config_path, config)
                cache_env, cache_input = _cache_directories(
                    run_dir / "caches",
                    seed=seed if mode == "populated_kernel_cache" else None,
                )
                if mode == "populated_kernel_cache" and cache_input != seed_manifest:
                    raise RuntimeError("copied cache differs from its immutable seed")
                print(f"START {args.framework} {mode} {role} steps={steps}", flush=True)
                command = _native_command(args, config_path, report_path, mode)
                execution, started, finished = _execute(
                    command,
                    directory=run_dir,
                    root=root,
                    environment=environment | cache_env,
                    timeout=args.timeout_seconds,
                )
                check_sources()
                if frozen_prediction is not None:
                    _check_seal(cohort, cohort_result["sealed_prediction"])
                report = json.loads(report_path.read_text())
                actual_plan, calls = _verify_report(
                    report,
                    framework=args.framework,
                    steps=steps,
                    warmup=args.warmup_steps,
                    expected_cache=cache_input,
                )
                identity = _check_process_identity(
                    report,
                    execution,
                    started_at=started,
                    finished_at=finished,
                    seen=seen,
                )
                inventory = _checkpoint_inventory(run_dir / "model", calls)
                cohort_result["runs"][role] = {
                    "report": str(report_path),
                    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    "execution": execution,
                    "identity": identity,
                    "checkpoints": inventory,
                    "event_counts": report["timing"]["eventcounts"],
                    "peaks": {
                        key: report["observation"][key]
                        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
                    },
                }
                if role.startswith("calibration"):
                    calibrations.append(report)
                    if mode == "empty_kernel_cache":
                        empty_calibrations.append(report_path)
                    if mode == "empty_kernel_cache" and role == "calibration-1":
                        seed_manifest = _seed_cache(run_dir / "caches", seed)
                        summary["populated_seed_manifest"] = seed_manifest
                    if len(calibrations) == 2:
                        frozen_plan = plan_finetuning_pipeline(
                            calibrations[0], max_steps=args.holdout_steps
                        )
                        frozen_prediction = {
                            "pipeline": estimate_finetuning_pipeline(
                                calibrations, frozen_plan
                            ),
                            "memory": estimate_finetuning_memory(
                                calibrations, frozen_plan
                            ),
                        }
                        _write_json(cohort / "future-plan.json", frozen_plan)
                        _write_json(
                            cohort / "future-prediction.json", frozen_prediction
                        )
                        cohort_result["sealed_prediction"] = {
                            "plan_sha256": hashlib.sha256(
                                (cohort / "future-plan.json").read_bytes()
                            ).hexdigest(),
                            "prediction_sha256": hashlib.sha256(
                                (cohort / "future-prediction.json").read_bytes()
                            ).hexdigest(),
                            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                else:
                    if _semantic_plan(actual_plan) != _semantic_plan(frozen_plan):
                        raise ValueError(
                            "actual holdout workload differs from the sealed future plan"
                        )
                    comparison = {
                        "pipeline": compare_finetuning_pipeline(calibrations, report),
                        "memory": compare_finetuning_memory(calibrations, report),
                    }
                    for field in _FIELDS:
                        if (
                            comparison["pipeline"]["timing"][field]
                            != frozen_prediction["pipeline"]["timing"][field]
                        ):
                            raise ValueError(
                                "replay estimate differs from the sealed prior prediction"
                            )
                    if (
                        comparison["memory"]["estimate"]["memory"]
                        != frozen_prediction["memory"]["memory"]
                    ):
                        raise ValueError(
                            "memory replay differs from the sealed prior envelope"
                        )
                    comparison["independent_reference"] = _reference_comparison(
                        calibrations, report, frozen_prediction["pipeline"]
                    )
                    _write_json(run_dir / "comparison.json", comparison)
                    cohort_result["holdout_comparisons"][role] = {
                        "path": str(run_dir / "comparison.json"),
                        "pipeline": comparison["pipeline"]["timing"]["comparison"],
                        "memory": comparison["memory"]["comparison"],
                        "independent_reference": comparison["independent_reference"],
                    }
                _write_json(summary_path, summary)
                print(
                    f"COMPLETE {args.framework} {mode} {role} seconds={report['timing']['total_wall_seconds']:.3f}",
                    flush=True,
                )
            cohort_result["status"] = "complete"
            _write_json(summary_path, summary)
        if args.with_oom_controls:
            for expected in ("oom", "success"):
                check_sources()
                run_dir = directory / f"allocator-control-{expected}"
                run_dir.mkdir()
                config = deepcopy(template)
                config.update(max_steps=14, output_dir=str(run_dir / "model"))
                _write_json(run_dir / "config.json", config)
                cache_env, _ = _cache_directories(run_dir / "caches")
                command = [
                    args.python,
                    str(Path(__file__).with_name("finetune_allocator_oom.py")),
                    "--framework",
                    args.framework,
                    "--config",
                    str(run_dir / "config.json"),
                    "--profile",
                    args.profile,
                    "--cuda-device",
                    str(args.gpu_index),
                    "--output",
                    str(run_dir / "result.json"),
                    "--expect",
                    expected,
                    "--cache-regime",
                    "empty_kernel_cache",
                ]
                for path in empty_calibrations:
                    command.extend(["--calibration-run", str(path)])
                print(
                    f"START {args.framework} allocator-control-{expected}", flush=True
                )
                execution, _, _ = _execute(
                    command,
                    directory=run_dir,
                    root=root,
                    environment=environment | cache_env,
                    timeout=args.timeout_seconds,
                    expected_code=86 if expected == "oom" else 0,
                )
                check_sources()
                evidence = json.loads((run_dir / "result.json").read_text())
                if evidence["identity"]["pid"] != execution["pid"]:
                    raise ValueError(
                        "allocator control PID differs from its subprocess"
                    )
                if (
                    expected == "oom"
                    and evidence["outcome"] != "observed_allocator_budget_oom"
                ):
                    raise ValueError(
                        "low-budget control did not produce an actual CUDA OOM"
                    )
                if expected == "success":
                    native = json.loads(Path(evidence["native_report"]).read_text())
                    _check_calls(native, _expected_calls(args.framework, 14))
                    _checkpoint_inventory(
                        run_dir / "model", _expected_calls(args.framework, 14)
                    )
                summary["oom_controls"][expected] = {
                    "path": str(run_dir / "result.json"),
                    "execution": execution,
                    "outcome": evidence["outcome"],
                    "allocator": evidence["allocator"],
                }
                _write_json(summary_path, summary)
                print(
                    f"COMPLETE {args.framework} allocator-control-{expected}",
                    flush=True,
                )
        check_sources()
        summary["status"] = "complete"
    except BaseException:
        summary.update(status="failed", failure=traceback.format_exc())
        _write_json(summary_path, summary)
        raise
    _write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
