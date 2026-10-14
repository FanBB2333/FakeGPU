"""Sequential parameter sweeps with immutable requests and preserved attempts.

Only declared inputs, executable, FakeGPU source and selected environment are
attested. Authors must pin external packages and include their native runtime
identity in result artifacts. Commands must run in the foreground.
"""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import re
import shutil
from string import Formatter
import sys

from ._finetune_workflow_store import (
    _canonical,
    _digest,
    _execute,
    _file,
    _live,
    _locked,
    _now,
    _source_identity,
    _tree,
    _write,
)
from .structured_io import load_mapping


SCHEMA = "fakegpu.experiment_matrix.v1"
_ENVIRONMENT = (
    "PATH",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError(f"invalid parameter/input name: {value!r}")
    return value


def _relative(value):
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or any(p in (".", "..") for p in value.split("/"))
    ):
        raise ValueError(
            f"artifact must be a relative path without traversal: {value!r}"
        )
    return value


def _render(value, context):
    for _, field, spec, conversion in Formatter().parse(value):
        if field is not None and (field not in context or spec or conversion):
            raise ValueError(f"unknown or decorated command field: {field!r}")
    return value.format_map(context)


def _input(path):
    if path.is_file():
        return {"kind": "file", **_file(path)}
    if path.is_dir():
        return {"kind": "directory", "files": _tree(path)}
    raise ValueError(f"declared input does not exist: {path}")


def _verify_inputs(request):
    for identity in request["inputs"].values():
        if _input(Path(identity["path"])) != {
            k: v for k, v in identity.items() if k != "path"
        }:
            raise ValueError("declared experiment input changed during sweep")


def _prepare(config, workspace):
    config = Path(config).expanduser().resolve()
    spec = load_mapping(config)
    allowed = {
        "schema_version",
        "command",
        "parameters",
        "repetitions",
        "max_runs",
        "inputs",
        "artifacts",
        "metrics",
        "timeout_seconds",
        "environment",
        "working_directory",
    }
    if set(spec) - allowed:
        raise ValueError(f"unknown experiment fields: {sorted(set(spec) - allowed)}")
    if spec.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    command = spec.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(v, str) or not v for v in command)
    ):
        raise ValueError("command must be a nonempty argv array of nonempty strings")
    if "{" in command[0] and command[0] != "{python}":
        raise ValueError("executable must be literal or {python}")
    parameters = spec.get("parameters", {})
    inputs = spec.get("inputs", {})
    if not isinstance(parameters, dict) or not isinstance(inputs, dict):
        raise ValueError("parameters and inputs must be objects")
    reserved = {"python", "run_dir", "repetition"}
    for name in [*parameters, *inputs]:
        _name(name)
    if set(parameters) & set(inputs) or (set(parameters) | set(inputs)) & reserved:
        raise ValueError("parameter/input names must be distinct and not reserved")
    count = _positive(spec.get("repetitions", 1), "repetitions")
    for name, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter {name} needs a nonempty list")
        for value in values:
            if type(value) not in (str, int, float, bool) or (
                type(value) is float and not math.isfinite(value)
            ):
                raise ValueError("parameter values must be finite JSON scalars")
        if len({_canonical(v) for v in values}) != len(values):
            raise ValueError(f"duplicate values for parameter {name}")
        count *= len(values)
    if count > _positive(spec.get("max_runs", 100), "max_runs"):
        raise ValueError(f"sweep has {count} runs; exceeds max_runs")
    timeout = spec.get("timeout_seconds")
    if timeout is not None and (
        type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    artifacts = spec.get("artifacts", [])
    if not isinstance(artifacts, list) or len(set(artifacts)) != len(artifacts):
        raise ValueError("artifacts must be a list without duplicates")
    for value in artifacts:
        _relative(value)
    metrics = spec.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    for name, metric in metrics.items():
        _name(name)
        if (
            not isinstance(metric, dict)
            or set(metric) - {"artifact", "path", "unit"}
            or metric.get("artifact") not in artifacts
        ):
            raise ValueError(f"metric {name} must reference a declared JSON artifact")
        path = metric.get("path")
        if (
            not isinstance(path, list)
            or not path
            or any(
                type(p) not in (str, int) or (type(p) is int and p < 0) for p in path
            )
        ):
            raise ValueError(f"metric {name} needs a nonempty JSON key/index path")
        if not isinstance(metric.get("unit", ""), str):
            raise ValueError("metric unit must be a string")
    environment = spec.get("environment", {})
    if not isinstance(environment, dict) or any(
        not isinstance(k, str)
        or not isinstance(v, str)
        or not k
        or "=" in k
        or "\0" in k + v
        for k, v in environment.items()
    ):
        raise ValueError("environment must map valid names to strings")
    resource_env = {k: os.environ.get(k) for k in _ENVIRONMENT}
    resource_env.update(environment)
    executable = (
        sys.executable
        if command[0] == "{python}"
        else shutil.which(
            command[0], path=environment.get("PATH", os.environ.get("PATH"))
        )
    )
    if not executable:
        raise ValueError(f"executable not found: {command[0]}")
    executable = str(Path(executable).resolve())
    cwd = (config.parent / spec.get("working_directory", ".")).resolve()
    if not cwd.is_dir():
        raise ValueError(f"working_directory does not exist: {cwd}")
    resolved_inputs = {}
    for name, value in inputs.items():
        if not isinstance(value, str):
            raise ValueError("input paths must be strings")
        path = (config.parent / value).resolve()
        if path.is_dir() and workspace.is_relative_to(path):
            raise ValueError("workspace must be outside declared input directories")
        resolved_inputs[name] = {"path": str(path), **_input(path)}
    request = {
        "schema_version": SCHEMA,
        "spec": spec,
        "config": str(config),
        "working_directory": str(cwd),
        "inputs": resolved_inputs,
        "executable": {"path": executable, **_file(executable)},
        "environment": resource_env,
        "implementation": _source_identity(),
    }
    request_id = _digest(request)
    jobs = []
    keys = sorted(parameters)
    for values in itertools.product(*(parameters[k] for k in keys)):
        params = dict(zip(keys, values))
        for repetition in range(1, spec.get("repetitions", 1) + 1):
            job_id = _digest(
                {
                    "request_id": request_id,
                    "parameters": params,
                    "repetition": repetition,
                }
            )[:24]
            context = {
                **params,
                **{k: v["path"] for k, v in resolved_inputs.items()},
                "python": sys.executable,
                "repetition": repetition,
                "run_dir": str(workspace / "runs" / job_id / "attempt-0001"),
            }
            argv = [_render(value, context) for value in command]
            argv[0] = executable
            jobs.append(
                {
                    "run_id": job_id,
                    "parameters": params,
                    "repetition": repetition,
                    "first_attempt_argv": argv,
                }
            )
    return {
        "schema_version": SCHEMA,
        "request_id": request_id,
        "request": request,
        "jobs": jobs,
    }


def plan_experiments(config, workspace):
    """Read and hash inputs, expand the finite grid, and show argv without execution."""
    return _prepare(config, Path(workspace).expanduser().resolve())


def _artifact(attempt, relative):
    path = attempt / relative
    if not path.resolve().is_relative_to(attempt.resolve()) or any(
        p.is_symlink() for p in [path, *path.parents] if p != attempt.parent
    ):
        raise ValueError(
            f"artifact must stay inside its attempt without symlinks: {relative}"
        )
    if not path.is_file():
        raise ValueError(f"artifact is missing or not an ordinary file: {relative}")
    return _file(path)


def _results(attempt, spec):
    identities = {name: _artifact(attempt, name) for name in spec.get("artifacts", [])}
    metrics = {}
    documents = {}
    for name, definition in spec.get("metrics", {}).items():
        artifact = definition["artifact"]
        if artifact not in documents:
            documents[artifact] = load_mapping(attempt / artifact)
        value = documents[artifact]
        try:
            for key in definition["path"]:
                value = value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"metric {name} path is missing") from exc
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"metric {name} must be a finite number")
        metrics[name] = value
    # Detect changes while parsing, rather than attesting different bytes.
    if any(
        _artifact(attempt, name) != identity for name, identity in identities.items()
    ):
        raise ValueError("artifact changed while extracting metrics")
    return identities, metrics


def _check_manifest(value):
    if (
        value.get("schema_version") != SCHEMA
        or _digest(value["request"]) != value["request_id"]
    ):
        raise ValueError("experiment request identity is invalid")
    expected = []
    spec = value["request"]["spec"]
    keys = sorted(spec.get("parameters", {}))
    for values in itertools.product(*(spec["parameters"][k] for k in keys)):
        params = dict(zip(keys, values))
        for repetition in range(1, spec.get("repetitions", 1) + 1):
            expected.append(
                (
                    _digest(
                        {
                            "request_id": value["request_id"],
                            "parameters": params,
                            "repetition": repetition,
                        }
                    )[:24],
                    params,
                    repetition,
                )
            )
    if [
        (j["run_id"], j["parameters"], j["repetition"]) for j in value["jobs"]
    ] != expected:
        raise ValueError("experiment jobs do not match the request")


def _verify_history(root, state):
    for name, records in state.get("history", {}).items():
        for relative, identity in records.items():
            if _artifact(root / name, relative) != identity:
                raise ValueError(
                    f"experiment attempt record changed: {root / name / relative}"
                )


def _verify_attempt(root, state, spec):
    attempt = root / state["attempt"]
    identities, metrics = _results(attempt, spec)
    if identities != state["artifacts"] or metrics != state["metrics"]:
        raise ValueError(f"completed experiment artifacts changed: {attempt}")


def run_experiments(config, workspace, *, resume=False, retry_failed=False):
    """Execute sequentially; resume skips successes, retries require an explicit flag."""
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    root = Path(workspace).expanduser().resolve()
    plan = _prepare(config, root)
    with _locked(root, create=True):
        manifest_path = root / "experiments.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError(
                    "experiment workspace exists; use resume or a new directory"
                )
            manifest = load_mapping(manifest_path)
            _check_manifest(manifest)
            if manifest != plan:
                raise ValueError(
                    "experiment inputs, executable, source, environment or request changed"
                )
        else:
            if resume:
                raise ValueError("cannot resume a workspace without experiments.json")
            if any(p.name != ".workflow.lock" for p in root.iterdir()):
                raise ValueError("new experiment workspace must be empty")
            _write(manifest_path, plan)
        spec = plan["request"]["spec"]
        for job in plan["jobs"]:
            directory = root / "runs" / job["run_id"]
            directory.mkdir(parents=True, exist_ok=True)
            state_path = directory / "state.json"
            state = (
                load_mapping(state_path)
                if state_path.exists()
                else {"status": "pending", "attempts": []}
            )
            _verify_history(directory, state)
            if state["status"] == "succeeded":
                _verify_attempt(directory, state, spec)
                continue
            if state.get("attempt"):
                execution_path = directory / state["attempt"] / "execution.json"
                if execution_path.exists() and _live(load_mapping(execution_path)):
                    raise RuntimeError(
                        f"experiment process is still alive: {directory}"
                    )
                if state["status"] == "running":
                    state["status"] = "interrupted"
                    _write(state_path, state)
                if not retry_failed:
                    continue
            # Inputs are re-attested between runs; never silently continue after mutation.
            _verify_inputs(plan["request"])
            name = f"attempt-{len(state['attempts']) + 1:04d}"
            attempt = directory / name
            attempt.mkdir()
            state.update(
                status="running",
                attempt=name,
                attempts=[*state["attempts"], name],
                started_at_utc=_now(),
            )
            state.pop("error", None)
            state.pop("artifacts", None)
            state.pop("metrics", None)
            _write(state_path, state)
            context = {
                **job["parameters"],
                **{k: v["path"] for k, v in plan["request"]["inputs"].items()},
                "python": sys.executable,
                "repetition": job["repetition"],
                "run_dir": str(attempt),
            }
            argv = [_render(value, context) for value in spec["command"]]
            argv[0] = plan["request"]["executable"]["path"]
            environment = dict(os.environ)
            environment.update(spec.get("environment", {}))
            try:
                _execute(
                    argv,
                    attempt,
                    environment,
                    spec.get("timeout_seconds"),
                    cwd=plan["request"]["working_directory"],
                    purpose="experiment",
                )
                identities, metrics = _results(attempt, spec)
                _verify_inputs(plan["request"])
                state.update(status="succeeded", artifacts=identities, metrics=metrics)
            except Exception as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            except BaseException:
                state.update(status="interrupted", finished_at_utc=_now())
                _write(state_path, state)
                raise
            state["finished_at_utc"] = _now()
            _write(
                attempt / "outcome.json",
                {k: v for k, v in state.items() if k not in {"attempts", "history"}},
            )
            state.setdefault("history", {})[name] = {
                p.name: _file(p)
                for p in (
                    attempt / "execution.json",
                    attempt / "run.log",
                    attempt / "outcome.json",
                )
            }
            _write(state_path, state)
    return compare_experiments([root])


def compare_experiments(workspaces):
    """Verify stored artifacts and aggregate repetitions separately per workspace."""
    roots = [Path(p).expanduser().resolve() for p in workspaces]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("provide distinct experiment workspaces")
    reports = []
    for root in roots:
        manifest = load_mapping(root / "experiments.json")
        _check_manifest(manifest)
        spec = manifest["request"]["spec"]
        runs, grouped = [], {}
        for job in manifest["jobs"]:
            directory = root / "runs" / job["run_id"]
            path = directory / "state.json"
            state = load_mapping(path) if path.exists() else {"status": "pending"}
            _verify_history(directory, state)
            if state["status"] == "succeeded":
                _verify_attempt(directory, state, spec)
                key = _canonical(job["parameters"])
                group = grouped.setdefault(
                    key, {"parameters": job["parameters"], "metrics": {}}
                )
                for name, value in state["metrics"].items():
                    group["metrics"].setdefault(name, []).append(
                        {
                            "run_id": job["run_id"],
                            "repetition": job["repetition"],
                            "value": value,
                        }
                    )
            runs.append({**job, **state, "directory": str(directory)})
        for group in grouped.values():
            for name, samples in list(group["metrics"].items()):
                values = [s["value"] for s in samples]
                group["metrics"][name] = {
                    "unit": spec["metrics"][name].get("unit", ""),
                    "count": len(values),
                    "mean": math.fsum(v / len(values) for v in values),
                    "min": min(values),
                    "max": max(values),
                    "samples": samples,
                }
        counts = {
            s: sum(r["status"] == s for r in runs)
            for s in ("pending", "running", "succeeded", "failed", "interrupted")
        }
        reports.append(
            {
                "workspace": str(root),
                "request_id": manifest["request_id"],
                "request": manifest["request"],
                "status_counts": counts,
                "complete": counts["succeeded"] == len(runs),
                "runs": runs,
                "groups": list(grouped.values()),
            }
        )
    return {
        "schema_version": "fakegpu.experiment_comparison.v1",
        "complete": all(r["complete"] for r in reports),
        "workspaces": reports,
        "scope": "Declared inputs/executable/FakeGPU source and selected environment are hashed; external packages must be pinned by the author. Ranges describe observed repetitions, not confidence intervals. Failed attempts are preserved and never retried implicitly.",
    }


def plot_experiments(comparison, output, *, x, metric):
    """Write a standalone Matplotlib figure; optional dependency is imported here."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    series = {}
    units = set()
    for workspace in comparison["workspaces"]:
        for group in workspace["groups"]:
            value = group["parameters"].get(x)
            if type(value) not in (int, float) or metric not in group["metrics"]:
                raise ValueError(
                    "plot needs a numeric x parameter and a recorded metric in every group"
                )
            other = {k: v for k, v in group["parameters"].items() if k != x}
            label = f"{Path(workspace['workspace']).name} / {workspace['request_id'][:8]} / {_canonical(other)}"
            data = group["metrics"][metric]
            units.add(data["unit"])
            series.setdefault(label, []).append((value, data))
    if not series or len(units) != 1:
        raise ValueError("plot requires observations with a common metric unit")
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
    try:
        for label, points in series.items():
            points.sort(key=lambda p: p[0])
            means = [p[1]["mean"] for p in points]
            axes.errorbar(
                [p[0] for p in points],
                means,
                yerr=[
                    [v - p[1]["min"] for p, v in zip(points, means)],
                    [p[1]["max"] - v for p, v in zip(points, means)],
                ],
                marker="o",
                capsize=3,
                label=label,
            )
        axes.set(
            xlabel=x,
            ylabel=f"{metric} ({next(iter(units))})",
            title="Observed repetition means and ranges",
        )
        axes.legend(fontsize="small")
        target = Path(output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target)
        return target
    finally:
        plt.close(figure)


__all__ = [
    "plan_experiments",
    "run_experiments",
    "compare_experiments",
    "plot_experiments",
]
