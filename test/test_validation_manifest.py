from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from fakegpu.validation import (
    MANIFEST_SCHEMA_VERSION,
    ValidationManifestError,
    load_validation_manifest,
    main,
    run_validation_manifest,
)


def _write_manifest(
    path: Path, cases: list[dict], defaults: dict | None = None
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "defaults": defaults or {},
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_validation_manifest_expands_matrix_and_checks_json(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "name": "matrix",
                "matrix": {"value": [1, 2]},
                "command": [
                    "{python}",
                    "-c",
                    (
                        "import json,os,pathlib;"
                        "pathlib.Path(os.environ['OUT']).write_text("
                        "json.dumps(dict(value=int(os.environ['VALUE']))));"
                        "print('value=' + os.environ['VALUE'])"
                    ),
                ],
                "env": {
                    "OUT": "{report_dir}/result.json",
                    "VALUE": "{value}",
                },
                "expect": {
                    "stdout_contains": "value={value}",
                    "files_exist": "{report_dir}/result.json",
                    "json_checks": [
                        {
                            "path": "{report_dir}/result.json",
                            "pointer": "/value",
                            "op": "eq",
                            "value": "{value}",
                        }
                    ],
                },
            }
        ],
    )
    # JSON placeholders are strings. Compare against strings written by the
    # probe so the manifest remains format-agnostic.
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0]["command"][2] = (
        "import json,os,pathlib;"
        "pathlib.Path(os.environ['OUT']).write_text("
        "json.dumps(dict(value=os.environ['VALUE'])));"
        "print('value=' + os.environ['VALUE'])"
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    code, report = run_validation_manifest(
        manifest,
        report_dir=tmp_path / "report",
        strict=True,
    )
    assert code == 0
    assert report["status"] == "passed"
    assert report["counts"]["passed"] == 2
    assert [item["matrix"]["value"] for item in report["executions"]] == [1, 2]
    assert (tmp_path / "report" / "validation_report.json").is_file()
    assert (tmp_path / "report" / "validation_report.md").is_file()


def test_validation_manifest_skip_is_optional_or_strict(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "name": "missing",
                "command": [sys.executable, "-c", "raise SystemExit(99)"],
                "requires": {"commands": ["fakegpu-command-that-does-not-exist"]},
            }
        ],
    )
    code, report = run_validation_manifest(
        manifest,
        report_dir=tmp_path / "normal",
    )
    assert code == 0
    assert report["counts"]["skipped"] == 1

    code, report = run_validation_manifest(
        manifest,
        report_dir=tmp_path / "strict",
        strict=True,
    )
    assert code == 1
    assert report["status"] == "failed"


def test_validation_manifest_records_expectation_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "name": "failure",
                "command": [sys.executable, "-c", "print('actual')"],
                "expect": {"stdout_contains": "expected"},
            }
        ],
    )
    code, report = run_validation_manifest(
        manifest,
        report_dir=tmp_path / "report",
    )
    assert code == 1
    result = report["executions"][0]
    assert result["status"] == "failed"
    assert "stdout does not contain" in result["failures"][0]
    assert Path(result["stdout_log"]).read_text(encoding="utf-8") == "actual\n"


def test_validation_manifest_dry_run_and_case_selection(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {"name": "one", "command": [sys.executable, "-c", "pass"]},
            {"name": "two", "command": [sys.executable, "-c", "pass"]},
        ],
    )
    code, report = run_validation_manifest(
        manifest,
        report_dir=tmp_path / "report",
        selected_cases=["two"],
        dry_run=True,
    )
    assert code == 0
    assert report["counts"]["dry_run"] == 1
    assert report["executions"][0]["case"] == "two"


def test_validation_manifest_rejects_unknown_placeholder(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "name": "bad-placeholder",
                "command": [sys.executable, "-c", "{missing}"],
            }
        ],
    )
    with pytest.raises(ValidationManifestError, match="unknown manifest placeholder"):
        run_validation_manifest(manifest, report_dir=tmp_path / "report")


def test_validation_cli_json_output(tmp_path: Path, capsys) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"name": "ok", "command": [sys.executable, "-c", "pass"]}],
    )
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--report-dir",
                str(tmp_path / "report"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"


def test_checked_in_yaml_manifest_loads() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    manifest = load_validation_manifest(
        root / "verification" / "data" / "validation_smoke.yaml"
    )
    assert {case["name"] for case in manifest["cases"]} == {
        "allocator-api",
        "profile-doctor",
        "workspace-catalog",
    }


@pytest.mark.skipif(
    shutil.which("python3") is None,
    reason="python3 command unavailable",
)
def test_validation_manifest_schema_accepts_example() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "validation_manifest.schema.json").read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(
        (root / "verification" / "data" / "validation_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(manifest, schema)
