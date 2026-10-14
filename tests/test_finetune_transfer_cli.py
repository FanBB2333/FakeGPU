import json

import pytest

from fakegpu import estimate_finetuning_transfer, plan_finetuning_transfer
from fakegpu.finetune_transfer_cli import estimate_main, plan_main, validate_main
from test_finetune_run_estimator import native_run as native_run
from test_finetune_transfer_plan import transfer_run


def test_transfer_cli_plans_budgets_and_checks_an_independent_run(native_run, tmp_path):
    runs = [
        transfer_run(native_run, n, factor=n, epochs=32 if n == 3 else 24)
        for n in (1, 2, 3)
    ]
    paths = []
    for n, run in enumerate(runs):
        path = tmp_path / f"run-{n}.json"
        path.write_text(json.dumps(run))
        paths.append(str(path))
    plan_path = tmp_path / "plan.json"
    assert (
        plan_main(
            ["--run-report", paths[0], "--epochs", "32", "--json", str(plan_path)]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text())
    assert plan == plan_finetuning_transfer(runs[0], epochs=32)
    arguments = [
        value
        for path in paths[:2]
        for value in ("--calibration-run", path, "--setup-calibration-run", path)
    ]
    budget_path = tmp_path / "budget.json"
    assert (
        estimate_main(
            [*arguments, "--plan", str(plan_path), "--json", str(budget_path)]
        )
        == 0
    )
    budget = json.loads(budget_path.read_text())
    assert budget["coverage"]["complete"]
    assert budget["pipeline"] == estimate_finetuning_transfer(
        runs[:2], plan, setup_calibration_runs=runs[:2]
    )
    comparison_path = tmp_path / "comparison.json"
    assert (
        validate_main(
            [*arguments, "--holdout-run", paths[2], "--json", str(comparison_path)]
        )
        == 0
    )
    comparison = json.loads(comparison_path.read_text())
    for field in ("seconds", "wall_seconds", "instrumentation_seconds"):
        assert (
            comparison["pipeline"]["timing"][field]
            == budget["pipeline"]["timing"][field]
        )
    assert set(comparison["memory"]["comparison"]) == {
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }


def test_transfer_cli_incomplete_evidence_has_no_total_budget(native_run, tmp_path):
    run = transfer_run(native_run, 1)
    source, plan_path, budget = [
        tmp_path / name for name in ("run.json", "plan.json", "budget.json")
    ]
    source.write_text(json.dumps(run))
    plan_path.write_text(json.dumps(plan_finetuning_transfer(run, epochs=32)))
    assert (
        estimate_main(
            [
                "--calibration-run",
                str(source),
                "--setup-calibration-run",
                str(source),
                "--plan",
                str(plan_path),
                "--json",
                str(budget),
            ]
        )
        == 1
    )
    result = json.loads(budget.read_text())
    assert not result["coverage"]["complete"]
    assert result["coverage"]["missing_groups"]
    assert "pipeline" not in result and "memory" not in result


def test_transfer_cli_rejects_missing_saved_sampling_horizon(native_run, tmp_path):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(transfer_run(native_run, 1)))
    with pytest.raises(SystemExit) as error:
        plan_main(["--run-report", str(source), "--max-steps", "97"])
    assert error.value.code == 2
