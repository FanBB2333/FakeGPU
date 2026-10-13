import json

import pytest

from fakegpu import (
    compare_finetuning_memory,
    compare_finetuning_pipeline,
    estimate_finetuning_memory,
    estimate_finetuning_pipeline,
    plan_finetuning_pipeline,
)
from fakegpu.finetune_pipeline_cli import estimate_main, plan_main, validate_main
from test_finetune_pipeline_plan import sequential_run
from test_finetune_run_estimator import native_run as native_run


def test_public_api_and_cli_plan_budget_and_independent_validation(
    native_run, tmp_path
):
    runs = [
        sequential_run(native_run, i, factor=factor, epochs=epochs)
        for i, factor, epochs in ((1, 1, 24), (2, 3, 24), (3, 2.2, 32))
    ]
    paths = []
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    plan_path = tmp_path / "plan.json"
    assert (
        plan_main(
            ["--run-report", str(paths[0]), "--epochs", "32", "--json", str(plan_path)]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text())
    assert plan == plan_finetuning_pipeline(runs[0], epochs=32)
    calibration_args = [
        "--calibration-run",
        str(paths[0]),
        "--calibration-run",
        str(paths[1]),
    ]
    budget_path = tmp_path / "budget.json"
    assert (
        estimate_main(
            [
                *calibration_args,
                "--plan",
                str(plan_path),
                "--reserve-gib",
                "0.5",
                "--memory-budget-gib",
                "1",
                "--json",
                str(budget_path),
            ]
        )
        == 0
    )
    budget = json.loads(budget_path.read_text())
    assert budget["pipeline"] == estimate_finetuning_pipeline(runs[:2], plan)
    assert budget["memory"] == estimate_finetuning_memory(
        runs[:2], plan, memory_budget_bytes=1024**3, reserve_bytes=512 * 1024**2
    )
    validation_path = tmp_path / "validation.json"
    assert (
        validate_main(
            [
                *calibration_args,
                "--holdout-run",
                str(paths[2]),
                "--json",
                str(validation_path),
            ]
        )
        == 0
    )
    validation = json.loads(validation_path.read_text())
    assert validation["pipeline"] == compare_finetuning_pipeline(runs[:2], runs[2])
    assert validation["memory"] == compare_finetuning_memory(runs[:2], runs[2])


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_cli_rejects_invalid_memory_margin(native_run, tmp_path, value):
    runs = [sequential_run(native_run, i, factor=i) for i in (1, 2)]
    files = []
    for i, run in enumerate(runs):
        path = tmp_path / f"run-{i}.json"
        path.write_text(json.dumps(run))
        files.append(path)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_finetuning_pipeline(runs[0], epochs=32)))
    with pytest.raises(SystemExit) as error:
        estimate_main(
            [
                "--calibration-run",
                str(files[0]),
                "--calibration-run",
                str(files[1]),
                "--plan",
                str(plan),
                f"--reserve-gib={value}",
            ]
        )
    assert error.value.code == 2
