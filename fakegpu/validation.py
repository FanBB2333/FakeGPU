from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "fakegpu.validation_manifest.v1"
REPORT_SCHEMA_VERSION = "fakegpu.validation_report.v1"
CHECK_OPERATORS = {"eq", "ne", "lt", "le", "gt", "ge", "contains", "approx"}
CASE_STATUSES = {"passed", "failed", "skipped", "dry_run"}


class ValidationManifestError(ValueError):
    pass


def load_validation_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValidationManifestError(f"manifest not found: {resolved}")
    payload = _read_structured_file(resolved)
    if not isinstance(payload, Mapping):
        raise ValidationManifestError("manifest root must be an object")
    manifest = dict(payload)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValidationManifestError(
            f"expected schema_version {MANIFEST_SCHEMA_VERSION!r}"
        )
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValidationManifestError("defaults must be an object")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValidationManifestError("cases must be a non-empty list")
    names: set[str] = set()
    normalized_cases = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            raise ValidationManifestError(f"cases[{index}] must be an object")
        case = _validate_case(dict(raw), index=index)
        if case["name"] in names:
            raise ValidationManifestError(f"duplicate case name {case['name']!r}")
        names.add(case["name"])
        normalized_cases.append(case)
    manifest["defaults"] = dict(defaults)
    manifest["cases"] = normalized_cases
    manifest["_path"] = str(resolved)
    return manifest


def expand_validation_cases(
    manifest: Mapping[str, Any],
    *,
    selected_cases: Sequence[str] = (),
    report_dir: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    selected = set(selected_cases)
    known = {str(case["name"]) for case in manifest["cases"]}
    unknown = sorted(selected - known)
    if unknown:
        raise ValidationManifestError(f"unknown selected cases: {', '.join(unknown)}")
    manifest_path = Path(str(manifest["_path"]))
    root_report_dir = Path(report_dir).expanduser().resolve()
    defaults = dict(manifest.get("defaults") or {})
    expanded: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for case in manifest["cases"]:
        if selected and case["name"] not in selected:
            continue
        axes = dict(case.get("matrix") or {})
        axis_names = list(axes)
        combinations = (
            itertools.product(*(axes[name] for name in axis_names))
            if axis_names
            else [()]
        )
        for values in combinations:
            matrix = dict(zip(axis_names, values))
            suffix = ",".join(
                f"{name}={_context_value(matrix[name])}" for name in axis_names
            )
            execution_id = f"{case['name']}[{suffix}]" if suffix else str(case["name"])
            slug = _slug(execution_id)
            if slug in used_ids:
                raise ValidationManifestError(
                    f"case expansion produces duplicate id {execution_id!r}"
                )
            used_ids.add(slug)
            execution_report_dir = root_report_dir / "cases" / slug
            context = {
                "case": str(case["name"]),
                "execution_id": execution_id,
                "manifest_dir": str(manifest_path.parent),
                "report_dir": str(execution_report_dir),
                "root_report_dir": str(root_report_dir),
                "python": sys.executable,
                **matrix,
            }
            expanded.append(
                _materialize_case(
                    case,
                    defaults=defaults,
                    context=context,
                    manifest_dir=manifest_path.parent,
                    execution_id=execution_id,
                    matrix=matrix,
                    report_dir=execution_report_dir,
                )
            )
    return expanded


def run_validation_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    report_dir: str | os.PathLike[str] = "validation-report",
    selected_cases: Sequence[str] = (),
    strict: bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> tuple[int, dict[str, Any]]:
    manifest = load_validation_manifest(manifest_path)
    output_dir = Path(report_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executions = expand_validation_cases(
        manifest,
        selected_cases=selected_cases,
        report_dir=output_dir,
    )
    started_ns = time.time_ns()
    results: list[dict[str, Any]] = []
    for execution in executions:
        result = _run_case(execution, dry_run=dry_run)
        results.append(result)
        if fail_fast and result["status"] == "failed":
            break

    counts = {status: 0 for status in sorted(CASE_STATUSES)}
    for result in results:
        counts[result["status"]] += 1
    success = counts["failed"] == 0 and (not strict or counts["skipped"] == 0)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed" if success else "failed",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "report_dir": str(output_dir),
        "strict": bool(strict),
        "dry_run": bool(dry_run),
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "duration_seconds": round((time.time_ns() - started_ns) / 1e9, 6),
        "host": {
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "platform_detail": platform.platform(),
            "python": sys.version.split()[0],
        },
        "git_commit": _git_commit(Path(manifest["_path"]).parent),
        "counts": counts,
        "executions": results,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_report.md").write_text(
        render_validation_markdown(report),
        encoding="utf-8",
    )
    return (0 if success else 1), report


def render_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for item in report.get("executions") or []:
        matrix = ", ".join(
            f"{key}={value}" for key, value in dict(item.get("matrix") or {}).items()
        )
        detail = "; ".join(
            [
                *[str(value) for value in item.get("skip_reasons") or []],
                *[str(value) for value in item.get("failures") or []],
            ]
        )
        rows.append(
            f"| `{item['id']}` | `{item['status']}` | "
            f"{matrix or '-'} | {item.get('duration_seconds', 0):.3f} | "
            f"{_markdown_cell(detail or '-')} |"
        )
    counts = dict(report.get("counts") or {})
    return "\n".join(
        [
            "# FakeGPU Validation Report",
            "",
            f"**Status:** `{report.get('status', 'unknown')}`",
            "",
            f"- Manifest: `{report.get('manifest', '')}`",
            f"- Git commit: `{report.get('git_commit') or 'unknown'}`",
            f"- Host: `{(report.get('host') or {}).get('hostname', 'unknown')}`",
            f"- Passed: {counts.get('passed', 0)}",
            f"- Failed: {counts.get('failed', 0)}",
            f"- Skipped: {counts.get('skipped', 0)}",
            f"- Dry-run: {counts.get('dry_run', 0)}",
            "",
            "| Execution | Status | Matrix | Seconds | Details |",
            "|---|---|---|---:|---|",
            *rows,
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu validate",
        description="Execute a declarative FakeGPU validation matrix.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-dir", default="validation-report")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named case; may be repeated.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat skipped prerequisites as a failed validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete report to stdout.",
    )
    args = parser.parse_args(argv)
    try:
        code, report = run_validation_manifest(
            args.manifest,
            report_dir=args.report_dir,
            selected_cases=args.case,
            strict=args.strict,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
    except ValidationManifestError as exc:
        parser.exit(2, f"error: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = report["counts"]
        print(
            "Validation {status}: {passed} passed, {failed} failed, "
            "{skipped} skipped, {dry_run} dry-run".format(
                status=report["status"],
                **counts,
            )
        )
        print(Path(report["report_dir"]) / "validation_report.json")
    return code


def _validate_case(case: dict[str, Any], *, index: int) -> dict[str, Any]:
    prefix = f"cases[{index}]"
    name = case.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValidationManifestError(f"{prefix}.name must be a non-empty string")
    case["name"] = name.strip()
    command = case.get("command")
    if not isinstance(command, (str, list)) or (
        isinstance(command, list)
        and (
            not command
            or not all(isinstance(value, (str, int, float)) for value in command)
        )
    ):
        raise ValidationManifestError(
            f"{prefix}.command must be a string or non-empty scalar list"
        )
    matrix = case.get("matrix", {})
    if not isinstance(matrix, Mapping):
        raise ValidationManifestError(f"{prefix}.matrix must be an object")
    for axis, values in matrix.items():
        if not isinstance(axis, str) or not axis:
            raise ValidationManifestError(
                f"{prefix}.matrix keys must be non-empty strings"
            )
        if not isinstance(values, list) or not values:
            raise ValidationManifestError(
                f"{prefix}.matrix.{axis} must be a non-empty list"
            )
        if not all(isinstance(value, (str, int, float, bool)) for value in values):
            raise ValidationManifestError(
                f"{prefix}.matrix.{axis} values must be scalar"
            )
    for key in ("env", "requires", "expect"):
        value = case.get(key, {})
        if not isinstance(value, Mapping):
            raise ValidationManifestError(f"{prefix}.{key} must be an object")
    timeout = case.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValidationManifestError(f"{prefix}.timeout_seconds must be positive")
    return case


def _materialize_case(
    case: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    context: Mapping[str, Any],
    manifest_dir: Path,
    execution_id: str,
    matrix: Mapping[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    default_cwd = defaults.get("cwd", ".")
    cwd_text = _format_value(case.get("cwd", default_cwd), context)
    cwd = Path(str(cwd_text))
    if not cwd.is_absolute():
        cwd = (manifest_dir / cwd).resolve()

    command_raw = case["command"]
    if isinstance(command_raw, str):
        command = shlex.split(str(_format_value(command_raw, context)))
    else:
        command = [str(_format_value(value, context)) for value in command_raw]

    env = {
        **{
            str(key): str(_format_value(value, context))
            for key, value in dict(defaults.get("env") or {}).items()
        },
        **{
            str(key): str(_format_value(value, context))
            for key, value in dict(case.get("env") or {}).items()
        },
    }
    requires = _format_value(
        {
            **dict(defaults.get("requires") or {}),
            **dict(case.get("requires") or {}),
        },
        context,
    )
    expect = _format_value(
        {
            **dict(defaults.get("expect") or {}),
            **dict(case.get("expect") or {}),
        },
        context,
    )
    timeout = float(
        case.get(
            "timeout_seconds",
            defaults.get("timeout_seconds", 300),
        )
    )
    return {
        "id": execution_id,
        "case": str(case["name"]),
        "matrix": dict(matrix),
        "command": command,
        "cwd": str(cwd),
        "env": env,
        "requires": requires,
        "expect": expect,
        "timeout_seconds": timeout,
        "report_dir": str(report_dir),
    }


def _run_case(execution: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    case_dir = Path(str(execution["report_dir"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "id": execution["id"],
        "case": execution["case"],
        "matrix": dict(execution["matrix"]),
        "command": list(execution["command"]),
        "cwd": execution["cwd"],
        "env": dict(execution["env"]),
        "timeout_seconds": execution["timeout_seconds"],
        "stdout_log": str(case_dir / "stdout.log"),
        "stderr_log": str(case_dir / "stderr.log"),
        "skip_reasons": [],
        "failures": [],
    }
    skip_reasons = _prerequisite_failures(execution)
    if skip_reasons:
        return {
            **base,
            "status": "skipped",
            "exit_code": None,
            "duration_seconds": 0.0,
            "skip_reasons": skip_reasons,
        }
    if dry_run:
        return {
            **base,
            "status": "dry_run",
            "exit_code": None,
            "duration_seconds": 0.0,
        }

    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in execution["env"].items()})
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            list(execution["command"]),
            cwd=str(execution["cwd"]),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(execution["timeout_seconds"]),
            check=False,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
    duration = time.monotonic() - started
    Path(base["stdout_log"]).write_text(stdout, encoding="utf-8")
    Path(base["stderr_log"]).write_text(stderr, encoding="utf-8")

    failures = []
    if timed_out:
        failures.append(f"command timed out after {execution['timeout_seconds']:.3f}s")
    else:
        failures.extend(
            _expectation_failures(
                execution,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
            )
        )
    return {
        **base,
        "status": "failed" if failures else "passed",
        "exit_code": exit_code,
        "duration_seconds": round(duration, 6),
        "failures": failures,
    }


def _prerequisite_failures(execution: Mapping[str, Any]) -> list[str]:
    requires = dict(execution.get("requires") or {})
    cwd = Path(str(execution["cwd"]))
    reasons = []
    platforms = _string_list(requires.get("platforms"))
    if platforms and sys.platform not in platforms:
        reasons.append(f"platform {sys.platform!r} is not in {', '.join(platforms)}")
    for command in _string_list(requires.get("commands")):
        if shutil.which(command) is None:
            reasons.append(f"required command not found: {command}")
    for module in _string_list(requires.get("python_modules")):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            reasons.append(f"required Python module not found: {module}")
    for name in _string_list(requires.get("env")):
        if not os.environ.get(name):
            reasons.append(f"required environment variable is unset: {name}")
    for path_text in _string_list(requires.get("files")):
        path = Path(path_text)
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            reasons.append(f"required file not found: {path}")
    return reasons


def _expectation_failures(
    execution: Mapping[str, Any],
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    duration: float,
) -> list[str]:
    expect = dict(execution.get("expect") or {})
    failures = []
    expected_exit = int(expect.get("exit_code", 0))
    if exit_code != expected_exit:
        failures.append(f"exit code {exit_code}, expected {expected_exit}")
    for value in _string_list(expect.get("stdout_contains")):
        if value not in stdout:
            failures.append(f"stdout does not contain {value!r}")
    for value in _string_list(expect.get("stderr_contains")):
        if value not in stderr:
            failures.append(f"stderr does not contain {value!r}")
    for value in _string_list(expect.get("stdout_not_contains")):
        if value in stdout:
            failures.append(f"stdout unexpectedly contains {value!r}")
    for value in _string_list(expect.get("stderr_not_contains")):
        if value in stderr:
            failures.append(f"stderr unexpectedly contains {value!r}")
    duration_max = expect.get("duration_seconds_max")
    if duration_max is not None and duration > float(duration_max):
        failures.append(f"duration {duration:.6f}s exceeds {float(duration_max):.6f}s")
    cwd = Path(str(execution["cwd"]))
    for path_text in _string_list(expect.get("files_exist")):
        path = Path(path_text)
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            failures.append(f"expected file does not exist: {path}")
    checks = expect.get("json_checks", [])
    if not isinstance(checks, list):
        failures.append("expect.json_checks must be a list")
    else:
        for index, check in enumerate(checks):
            try:
                _evaluate_json_check(check, cwd=cwd)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                failures.append(f"json_checks[{index}]: {exc}")
    return failures


def _evaluate_json_check(check: Any, *, cwd: Path) -> None:
    if not isinstance(check, Mapping):
        raise TypeError("check must be an object")
    path = Path(str(check.get("path", "")))
    if not path.is_absolute():
        path = cwd / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    pointer = str(check.get("pointer", ""))
    observed = _json_pointer(payload, pointer)
    operator = str(check.get("op", "eq"))
    if operator not in CHECK_OPERATORS:
        raise ValueError(f"unsupported operator {operator!r}")
    expected = check.get("value")
    tolerance = float(check.get("tolerance", 0))
    passed = _compare_value(
        observed,
        expected,
        operator=operator,
        tolerance=tolerance,
    )
    if not passed:
        raise ValueError(
            f"{path}{pointer}: observed {observed!r} {operator} "
            f"expected {expected!r} failed"
        )


def _json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = payload
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, Mapping):
            current = current[token]
        else:
            raise KeyError(f"cannot descend through {type(current).__name__}")
    return current


def _compare_value(
    observed: Any,
    expected: Any,
    *,
    operator: str,
    tolerance: float,
) -> bool:
    if operator == "eq":
        return observed == expected
    if operator == "ne":
        return observed != expected
    if operator == "lt":
        return observed < expected
    if operator == "le":
        return observed <= expected
    if operator == "gt":
        return observed > expected
    if operator == "ge":
        return observed >= expected
    if operator == "contains":
        return expected in observed
    if operator == "approx":
        return abs(float(observed) - float(expected)) <= tolerance
    return False


def _read_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationManifestError(f"invalid JSON: {exc}") from exc
    if suffix == ".toml":
        try:
            import tomllib
        except ImportError as exc:  # pragma: no cover - Python >=3.11 locally
            raise ValidationManifestError(
                "TOML manifests require Python 3.11 or tomli"
            ) from exc
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ValidationManifestError(f"invalid TOML: {exc}") from exc
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ValidationManifestError(
                "YAML manifests require PyYAML; install "
                "'fakegpu[validation]' or use JSON/TOML"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValidationManifestError(f"invalid YAML: {exc}") from exc
    raise ValidationManifestError("manifest must use .json, .toml, .yaml, or .yml")


def _format_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(_StrictFormatMap(context))
        except KeyError as exc:
            raise ValidationManifestError(
                f"unknown manifest placeholder {exc.args[0]!r}"
            ) from exc
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _format_value(item, context) for key, item in value.items()}
    return value


class _StrictFormatMap(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def _context_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _slug(value: str) -> str:
    result = []
    for character in value:
        if character.isalnum() or character in {"-", "_", "."}:
            result.append(character)
        else:
            result.append("-")
    return "".join(result).strip("-") or "case"


def _git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
