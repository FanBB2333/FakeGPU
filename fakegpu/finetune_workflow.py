"""Reproducible native calibration workspaces with explicit resource limits.

Training runs only in fresh child processes. A workspace retains every attempt;
CPU budget reuse never starts a process or requires the training environment.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time
from uuid import UUID, uuid4

from ._finetune_cache import CACHE_VARIABLES, capture_cache_regime
from ._finetune_transfer_plan import transfer_plan_from_run
from .structured_io import load_mapping
from ._finetune_workflow_inputs import _prepare_inputs, _runtime_identity
from ._finetune_workflow_store import (
    _digest,
    _now,
    _write,
    _file,
    _tree,
    _source_identity,
    _locked,
    _live,
    _execute,
)

SCHEMA_VERSION = "fakegpu.finetune_workflow.v1"
_FRAMEWORKS = {"huggingface", "llamafactory", "axolotl"}
_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
)


def _positive(value, field, *, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    if integer and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _duration(epochs, max_steps):
    if (epochs is None) == (max_steps is None):
        raise ValueError("provide exactly one of epochs or max_steps")
    return {
        "epochs": _positive(epochs, "epochs") if epochs is not None else None,
        "max_steps": _positive(max_steps, "max_steps", integer=True)
        if max_steps is not None
        else None,
    }


def _memory_limits(memory_budget_bytes, reserve_bytes):
    for value, field in (
        (memory_budget_bytes, "memory_budget_bytes"),
        (reserve_bytes, "reserve_bytes"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{field} must be a nonnegative integer")
    if memory_budget_bytes == 0:
        raise ValueError("memory_budget_bytes must be positive")
    if reserve_bytes is None:
        raise ValueError("reserve_bytes must be a nonnegative integer")


def _native_command(request, directory):
    command = [
        request["python_executable"],
        "-m",
        "fakegpu",
        "run-finetune",
        "--framework",
        request["framework"],
        "--config",
        str(directory / "config.json"),
        "--profile",
        request["profile_id"],
        "--warmup-steps",
        str(request["warmup_steps"]),
        "--cache-regime",
        request["cache_regime"],
        "--output",
        str(directory / "run.json"),
    ]
    duration = request["planning_duration"]
    if request.get("length_grouping_seed") is not None:
        command.extend(["--length-grouping-seed", str(request["length_grouping_seed"])])
    if request.get("packing_seed") is not None:
        command.extend(["--packing-seed", str(request["packing_seed"])])
    flag, value = (
        ("--planning-epochs", duration["epochs"])
        if duration["epochs"] is not None
        else ("--planning-max-steps", duration["max_steps"])
    )
    return [*command, flag, str(value)]


def _validate_run(report, execution, request, directory):
    transfer_plan_from_run(report)
    if report.get("length_grouping_seed") != request.get("length_grouping_seed"):
        raise ValueError("native length grouping seed differs from the request")
    if report.get("packing_seed") != request.get("packing_seed"):
        raise ValueError("native packing seed differs from the request")
    identity = report["run_identity"]
    if (
        str(UUID(identity["uuid"])) != identity["uuid"]
        or identity["pid"] != execution["pid"]
    ):
        raise ValueError("native report identity differs from the launched process")
    timestamp = datetime.fromisoformat(identity["started_at_utc"])
    if timestamp.utcoffset() != timezone.utc.utcoffset(
        timestamp
    ) or not datetime.fromisoformat(
        execution["started_at_utc"]
    ) <= timestamp <= datetime.fromisoformat(execution["finished_at_utc"]):
        raise ValueError("native report start falls outside the child process lifetime")
    if (
        report.get("native_config") != load_mapping(directory / "config.json")
        or Path(report.get("config_path", "")).resolve()
        != (directory / "config.json").resolve()
    ):
        raise ValueError("native report configuration differs from the launched input")
    if (
        report["workload"]["optimizer_steps"] != request["calibration_steps"]
        or report["workload"]["warmup_optimizer_steps"] != request["warmup_steps"]
    ):
        raise ValueError(
            "native calibration update/warmup counts differ from the request"
        )
    versions = request["runtime"]["versions"]
    if any(
        versions.get(name, value) != value
        for name, value in report["training_config"]["software_versions"].items()
        if name in versions
    ):
        raise ValueError("native software versions changed after workflow preflight")
    if (
        report["framework"] != request["framework"]
        or report["profile_id"] != request["profile_id"]
    ):
        raise ValueError("native report framework/profile differs from the request")
    if report["schedule"]["max_steps"] != request["calibration_steps"]:
        raise ValueError(
            "native calibration did not execute the requested update count"
        )
    if report.get("cache_regime", {}).get("mode") != request["cache_regime"]:
        raise ValueError("native report cache regime differs from the request")


def _attempt_artifacts(directory):
    output = _tree(directory / "model")
    if not output:
        raise ValueError("successful calibration requires saved model artifacts")
    return {
        **{
            name: _file(directory / name)
            for name in (
                "config.json",
                "execution.json",
                "run.log",
                "run.json",
                "cache-layout.json",
            )
        },
        **{f"model/{name}": value for name, value in output.items()},
    }


def _reports(root, manifest):
    reports = []
    for slot in (1, 2):
        successes = [
            attempt
            for attempt in manifest["attempts"]
            if attempt["slot"] == slot and attempt["status"] == "succeeded"
        ]
        if len(successes) != 1:
            raise ValueError(
                "workspace requires one completed calibration in each slot"
            )
        attempt = successes[0]
        directory = root / attempt["directory"]
        if _attempt_artifacts(directory) != attempt["artifacts"]:
            raise ValueError(
                "completed calibration artifact SHA256 mismatch; refusing reuse"
            )
        execution = load_mapping(directory / "execution.json")
        if execution.get("status") != "succeeded" or execution.get("returncode") != 0:
            raise ValueError("calibration execution did not complete successfully")
        report = load_mapping(directory / "run.json")
        _validate_run(report, execution, manifest["request"], directory)
        reports.append(report)
    if len({report["run_identity"]["uuid"] for report in reports}) != 2:
        raise ValueError("calibrations must have distinct native run identities")
    return reports


def _read_manifest(root):
    manifest = load_mapping(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "request_sha256"
    ) != _digest(manifest.get("request")):
        raise ValueError("invalid calibration workspace manifest")
    return manifest


def _coverage_and_budget(
    reports, duration, memory_budget_bytes, reserve_bytes, extra_reports
):
    from .finetune_transfer_estimator import (
        estimate_finetuning_transfer,
        estimate_finetuning_transfer_memory,
        inspect_finetuning_transfer_coverage,
    )
    from .finetune_transfer_plan import plan_finetuning_transfer

    plan = plan_finetuning_transfer(reports[0], **duration)
    work_pool = [*reports, *extra_reports]
    coverage = inspect_finetuning_transfer_coverage(
        work_pool, plan, setup_calibration_runs=reports
    )
    if not coverage["complete"]:
        return plan, coverage, None
    budget = {
        "schema_version": "fakegpu.finetune_transfer_budget.v1",
        "pipeline": estimate_finetuning_transfer(
            work_pool, plan, setup_calibration_runs=reports
        ),
        "memory": estimate_finetuning_transfer_memory(
            work_pool,
            plan,
            setup_calibration_runs=reports,
            memory_budget_bytes=memory_budget_bytes,
            reserve_bytes=reserve_bytes,
        ),
    }
    return plan, coverage, budget


def _import_reports(root, manifest, paths):
    if paths is not None:
        if isinstance(paths, (str, Path)):
            raise ValueError(
                "reuse_calibration_runs must be a sequence of report paths"
            )
        selection = []
        for value in paths:
            source = Path(value).expanduser().resolve()
            original = _file(source)
            report = load_mapping(source)
            transfer_plan_from_run(report)
            if _file(source) != original:
                raise ValueError("imported calibration changed while reading")
            path = root / "imported-calibrations" / f"{_digest(report)}.json"
            if path.exists():
                if load_mapping(path) != report:
                    raise ValueError("imported calibration snapshot SHA256 mismatch")
            else:
                path.parent.mkdir(exist_ok=True)
                _write(path, report)
            selection.append(
                {
                    "source_path": str(source),
                    "source_identity": original,
                    "directory": path.relative_to(root).as_posix(),
                    "artifact": _file(path),
                }
            )
        manifest["reuse_calibration_runs"] = selection
        _write(root / "manifest.json", manifest)
    reports = []
    for entry in manifest.get("reuse_calibration_runs", []):
        path = root / entry["directory"]
        if _file(path) != entry["artifact"]:
            raise ValueError("imported calibration snapshot SHA256 mismatch")
        reports.append(load_mapping(path))
    return reports


def _budget(
    root,
    manifest,
    *,
    epochs,
    max_steps,
    memory_budget_bytes,
    reserve_bytes,
    reuse_calibration_runs,
):
    duration = _duration(epochs, max_steps)
    _memory_limits(memory_budget_bytes, reserve_bytes)
    reports = _reports(root, manifest)
    extra_reports = _import_reports(root, manifest, reuse_calibration_runs)
    request = {
        **duration,
        "memory_budget_bytes": memory_budget_bytes,
        "reserve_bytes": reserve_bytes,
        "calibration_report_sha256": [_digest(report) for report in reports],
        "reused_report_sha256": [_digest(report) for report in extra_reports],
        "estimator_source": _source_identity(),
    }
    request_hash = _digest(request)
    directory = root / "predictions" / request_hash
    record = next(
        (
            entry
            for entry in manifest["predictions"]
            if entry["directory"] == directory.relative_to(root).as_posix()
        ),
        None,
    )
    if record is not None and not directory.exists():
        staging = root / record["staging_directory"]
        if _file(staging / "seal.json") != record["seal"]:
            raise ValueError(
                "pending prediction seal differs from the workspace manifest"
            )
        seal = load_mapping(staging / "seal.json")
        if any(
            _file(staging / name) != identity
            for name, identity in seal["artifacts"].items()
        ):
            raise ValueError("pending prediction artifact SHA256 mismatch")
        staging.rename(directory)
    if directory.exists():
        if record is None or _file(directory / "seal.json") != record["seal"]:
            raise ValueError("prediction seal differs from the workspace manifest")
        seal = load_mapping(directory / "seal.json")
        if seal["request_sha256"] != request_hash or any(
            _file(directory / name) != identity
            for name, identity in seal["artifacts"].items()
        ):
            raise ValueError("sealed prediction artifact SHA256 mismatch")
        result = load_mapping(directory / "result.json")
        manifest.update(status=result["status"], updated_at_utc=_now())
        _write(root / "manifest.json", manifest)
        return result
    plan, coverage, budget = _coverage_and_budget(
        reports, duration, memory_budget_bytes, reserve_bytes, extra_reports
    )
    staging = directory.with_name(f".{request_hash}.{uuid4().hex}.pending")
    staging.mkdir(parents=True)
    result = {
        "schema_version": "fakegpu.finetune_workflow_result.v1",
        "status": "complete" if budget is not None else "coverage_incomplete",
        "workspace": str(root),
        "prediction_directory": str(directory),
        "request_sha256": request_hash,
        "coverage": coverage,
        "budget": budget,
    }
    documents = {
        "request.json": request,
        "plan.json": plan,
        "coverage.json": coverage,
        "result.json": result,
    }
    if budget is not None:
        documents["budget.json"] = budget
    for name, value in documents.items():
        _write(staging / name, value)
    _write(
        staging / "seal.json",
        {
            "schema_version": "fakegpu.finetune_prediction_seal.v1",
            "sealed_at_utc": _now(),
            "request_sha256": request_hash,
            "artifacts": {name: _file(staging / name) for name in documents},
        },
    )
    manifest.update(status=result["status"], updated_at_utc=_now())
    manifest["predictions"].append(
        {
            "directory": directory.relative_to(root).as_posix(),
            "status": result["status"],
            "seal": _file(staging / "seal.json"),
            "staging_directory": staging.relative_to(root).as_posix(),
        }
    )
    _write(root / "manifest.json", manifest)
    staging.rename(directory)
    return result


def budget_finetuning_workspace(
    workspace,
    *,
    epochs=None,
    max_steps=None,
    memory_budget_bytes=None,
    reserve_bytes=0,
    reuse_calibration_runs=None,
):
    """Recompute a sealed target budget from successful reports using only CPU.

    The recorded sampling horizon limits the available future dataset schedule.
    No training process, framework import, or device inspection occurs here.
    """
    root = Path(workspace).expanduser().resolve()
    with _locked(root):
        return _budget(
            root,
            _read_manifest(root),
            epochs=epochs,
            max_steps=max_steps,
            memory_budget_bytes=memory_budget_bytes,
            reserve_bytes=reserve_bytes,
            reuse_calibration_runs=reuse_calibration_runs,
        )


def calibrate_finetuning(
    config_path,
    *,
    framework,
    profile_id,
    workspace,
    cuda_visible_devices,
    calibration_steps,
    epochs=None,
    max_steps=None,
    python_executable=None,
    warmup_steps=2,
    cache_regime="empty_kernel_cache",
    timeout_seconds=None,
    calibration_budget_seconds=None,
    resume=False,
    memory_budget_bytes=None,
    reserve_bytes=0,
    reuse_calibration_runs=None,
    length_grouping_seed=None,
    packing_seed=None,
):
    """Run two predeclared independent calibrations, retaining failures and seals.

    ``calibration_budget_seconds`` limits child-process wall time newly spent in
    this invocation. It never changes the configured update count or admits an
    incomplete run. Resuming requires unchanged local inputs and execution code.
    """
    if framework not in _FRAMEWORKS:
        raise ValueError("unsupported native calibration framework")
    from ._finetune_sampler import validate_length_grouping_seed

    validate_length_grouping_seed(length_grouping_seed)
    if packing_seed is not None and (
        isinstance(packing_seed, bool)
        or not isinstance(packing_seed, int)
        or not 0 <= packing_seed < 2**63
        or length_grouping_seed is not None
    ):
        raise ValueError(
            "packing_seed requires an integer in [0, 2**63) and no length_grouping_seed"
        )
    if cache_regime not in {"uncontrolled", "empty_kernel_cache"}:
        raise ValueError(
            "workflow cache regime must be uncontrolled or empty_kernel_cache"
        )
    if (
        not isinstance(cuda_visible_devices, str)
        or not cuda_visible_devices
        or "," in cuda_visible_devices
    ):
        raise ValueError("select exactly one CUDA device using an index or UUID")
    _positive(warmup_steps, "warmup_steps", integer=True)
    _positive(calibration_steps, "calibration_steps", integer=True)
    if calibration_steps < warmup_steps + 2:
        raise ValueError("calibration requires at least two updates after warmup")
    for value, field in (
        (timeout_seconds, "timeout_seconds"),
        (calibration_budget_seconds, "calibration_budget_seconds"),
    ):
        if value is not None:
            _positive(value, field)
    duration = _duration(epochs, max_steps)
    _memory_limits(memory_budget_bytes, reserve_bytes)
    root = Path(workspace).expanduser().resolve()
    config, inputs = _prepare_inputs(config_path, framework, workspace=root)
    if any(root == Path(path) or root in Path(path).parents for path in inputs):
        raise ValueError("calibration inputs must be outside the generated workspace")
    executable = str(Path(python_executable or sys.executable).expanduser().absolute())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_visible_devices)
    runtime = _runtime_identity(executable, framework, environment)
    request = {
        "framework": framework,
        "profile_id": profile_id,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "resolved_config": config,
        "python_executable": executable,
        "executable_identity": _file(executable),
        "runtime": runtime,
        "environment": {name: environment.get(name) for name in _ENVIRONMENT},
        "calibration_steps": calibration_steps,
        "warmup_steps": warmup_steps,
        "planning_duration": duration,
        "cache_regime": cache_regime,
        "inputs": inputs,
        "source": _source_identity(),
    }
    if length_grouping_seed is not None:
        request["length_grouping_seed"] = length_grouping_seed
    if packing_seed is not None:
        request["packing_seed"] = packing_seed
    with _locked(root, create=True):
        if (root / "manifest.json").exists():
            if not resume:
                raise ValueError(
                    "workspace already exists; use resume or a new workspace"
                )
            manifest = _read_manifest(root)
            if request != manifest["request"]:
                raise ValueError(
                    "calibration inputs or execution identity changed; use a new workspace"
                )
        else:
            if any(path.name != ".workflow.lock" for path in root.iterdir()):
                raise ValueError("new calibration workspace must be empty")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "request": request,
                "request_sha256": _digest(request),
                "created_at_utc": _now(),
                "status": "prepared",
                "attempts": [],
                "invocations": [],
                "predictions": [],
                "limitations": [
                    "Only the five named kernel-cache directories are controlled in empty_kernel_cache mode; OS pages, dataset caches, clocks and system load are not controlled.",
                    "Calibration uses two fixed independent runs. Missing coverage does not trigger automatic extra training.",
                ],
            }
        _import_reports(root, manifest, reuse_calibration_runs)
        invocation = {
            "started_at_utc": _now(),
            "timeout_seconds": timeout_seconds,
            "calibration_budget_seconds": calibration_budget_seconds,
            "child_process_seconds": 0.0,
        }
        manifest["invocations"].append(invocation)
        _write(root / "manifest.json", manifest)
        try:
            for slot in (1, 2):
                previous = [
                    attempt
                    for attempt in manifest["attempts"]
                    if attempt["slot"] == slot
                ]
                for attempt in previous:
                    execution_path = root / attempt["directory"] / "execution.json"
                    if execution_path.exists() and _live(load_mapping(execution_path)):
                        raise RuntimeError(
                            "a previous native calibration process is still running"
                        )
                successes = [
                    attempt for attempt in previous if attempt["status"] == "succeeded"
                ]
                if successes:
                    if (
                        len(successes) != 1
                        or _attempt_artifacts(root / successes[0]["directory"])
                        != successes[0]["artifacts"]
                    ):
                        raise ValueError(
                            "completed calibration artifact SHA256 mismatch; refusing reuse"
                        )
                    continue
                remaining = (
                    None
                    if calibration_budget_seconds is None
                    else calibration_budget_seconds
                    - invocation["child_process_seconds"]
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "calibration wall-clock budget exhausted before next process"
                    )
                timeout = (
                    min(
                        value
                        for value in (remaining, timeout_seconds)
                        if value is not None
                    )
                    if remaining is not None or timeout_seconds is not None
                    else None
                )
                directory = (
                    root
                    / "attempts"
                    / f"calibration-{slot}"
                    / f"attempt-{len(previous) + 1}"
                )
                directory.mkdir(parents=True)
                attempt = {
                    "slot": slot,
                    "number": len(previous) + 1,
                    "directory": directory.relative_to(root).as_posix(),
                    "status": "prepared",
                }
                manifest["attempts"].append(attempt)
                manifest["status"] = "calibrating"
                native = deepcopy(config)
                native.update(
                    max_steps=calibration_steps,
                    output_dir=str(directory / "model"),
                    logging_dir=str(directory / "logs"),
                )
                _write(directory / "config.json", native)
                child_environment = dict(environment)
                cache_environment = {}
                if cache_regime == "empty_kernel_cache":
                    for name in CACHE_VARIABLES:
                        cache = directory / "caches" / name
                        cache.mkdir(parents=True)
                        cache_environment[name] = str(cache)
                    child_environment.update(cache_environment)
                _write(
                    directory / "cache-layout.json",
                    {
                        "environment": cache_environment,
                        "input_manifest": capture_cache_regime(
                            cache_regime, environ=child_environment
                        ),
                    },
                )
                _write(root / "manifest.json", manifest)
                execution_started = time.monotonic()
                execution = None
                try:
                    execution = _execute(
                        _native_command(request, directory),
                        directory,
                        child_environment,
                        timeout,
                    )
                    report = load_mapping(directory / "run.json")
                    _validate_run(report, execution, request, directory)
                    if (
                        _prepare_inputs(config_path, framework, workspace=root)[1]
                        != inputs
                        or _source_identity() != request["source"]
                    ):
                        raise ValueError(
                            "calibration input or source changed during execution"
                        )
                    attempt.update(
                        status="succeeded",
                        run_uuid=report["run_identity"]["uuid"],
                        artifacts=_attempt_artifacts(directory),
                    )
                except BaseException as exc:
                    attempt.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
                    raise
                finally:
                    record = execution or load_mapping(directory / "execution.json")
                    invocation["child_process_seconds"] += record.get(
                        "process_wall_seconds", time.monotonic() - execution_started
                    )
                    _write(root / "manifest.json", manifest)
            manifest["status"] = "calibration_complete"
            _write(root / "manifest.json", manifest)
            return _budget(
                root,
                manifest,
                epochs=epochs,
                max_steps=max_steps,
                memory_budget_bytes=memory_budget_bytes,
                reserve_bytes=reserve_bytes,
                reuse_calibration_runs=None,
            )
        except BaseException as exc:
            manifest.update(
                status="calibration_failed", last_error=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            invocation["finished_at_utc"] = _now()
            manifest["updated_at_utc"] = _now()
            _write(root / "manifest.json", manifest)
