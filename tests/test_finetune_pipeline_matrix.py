from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fakegpu._finetune_cache import CACHE_VARIABLES, capture_cache_regime
from fakegpu._finetune_pipeline_plan import pipeline_plan_from_run
from fakegpu.finetune_pipeline_estimator import compare_finetuning_pipeline
from native import finetune_pipeline_matrix as matrix
from native.finetune_allocator_oom import _budget
from test_finetune_pipeline_estimator import _pipeline_run
from test_finetune_run_estimator import native_run as native_run


def test_native_command_passes_actual_cli_parser_without_starting_cuda(
    tmp_path, monkeypatch
):
    from fakegpu import finetune_cli, finetune_runner

    captured = []
    monkeypatch.setattr(
        finetune_runner,
        "run_finetuning",
        lambda *args, **kwargs: (
            captured.append((args, kwargs)) or {"native": "fixture"}
        ),
    )
    args = SimpleNamespace(
        python=sys.executable, framework="huggingface", profile="a100", warmup_steps=4
    )
    output = tmp_path / "report.json"
    command = matrix._native_command(
        args, tmp_path / "config.json", output, "empty_kernel_cache"
    )
    assert finetune_cli.run_main(command[4:]) == 0
    assert captured[0][1]["cache_regime"] == "empty_kernel_cache"
    assert json.loads(output.read_text()) == {"native": "fixture"}


def _template(framework):
    return {
        (
            "curriculum_sampling" if framework == "axolotl" else "disable_shuffling"
        ): True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": 2,
        "save_steps": 4,
        "save_total_limit": 100,
        "model_name_or_path": "/local/model",
    }


@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
def test_matrix_prepare_only_records_sources_and_never_launches_training(
    tmp_path, monkeypatch, framework
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_template(framework)))

    def fail(*args, **kwargs):
        pytest.fail("prepare-only launched a process")

    monkeypatch.setattr(matrix, "_execute", fail)
    assert (
        matrix.main(
            [
                "--framework",
                framework,
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "matrix"),
                "--prepare-only",
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "matrix/summary.json").read_text())
    assert result["status"] == "prepared"
    assert result["calibration_runs_per_regime"] == 2
    assert result["holdout_runs_per_regime"] == 3
    assert result["calibration_steps"] == 68
    assert result["holdout_steps"] == 92
    assert "tests/native/finetune_allocator_oom.py" in result["source_file_sha256"]
    assert "fakegpu/_finetune_allocator.py" in result["source_file_sha256"]
    assert not result["regimes"]


def test_existing_evidence_is_never_reused_or_deleted(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_template("llamafactory")))
    output = tmp_path / "matrix"
    output.mkdir()
    evidence = output / "old.log"
    evidence.write_text("keep me")
    with pytest.raises(SystemExit):
        matrix.main(
            [
                "--framework",
                "llamafactory",
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--prepare-only",
            ]
        )
    assert evidence.read_text() == "keep me"


@pytest.mark.parametrize(
    "changes",
    [
        {"disable_shuffling": False},
        {"save_only_model": True},
        {"save_total_limit": 1},
        {"eval_steps": 2.0},
        {"eval_steps": True},
        {"save_steps": 6},
        {"resume_from_checkpoint": "/old"},
        {"sample_packing": True},
        {"dataloader_drop_last": True},
    ],
)
def test_matrix_rejects_uncontrolled_sampler_rotation_and_interval(changes):
    with pytest.raises(ValueError):
        matrix._validate_template(_template("llamafactory") | changes, "llamafactory")


def test_empty_seed_is_unavailable_without_a_dummy_file(tmp_path):
    environment, before = matrix._cache_directories(tmp_path / "run/caches")
    assert before["files"] == []
    assert len(environment) == 5
    assert matrix._seed_cache(tmp_path / "run/caches", tmp_path / "seed") is None
    assert not (tmp_path / "seed").exists()


def test_real_cache_seed_is_identical_for_every_fresh_process_and_unchanged(tmp_path):
    environment, _ = matrix._cache_directories(tmp_path / "first/caches")
    original = Path(environment["TRITON_CACHE_DIR"]) / "kernel.cubin"
    original.write_bytes(b"actual fixture kernel artifact")
    seed = tmp_path / "seed"
    manifest = matrix._seed_cache(tmp_path / "first/caches", seed)
    assert manifest["files"][0]["path"] == "kernel.cubin"
    one, one_manifest = matrix._cache_directories(tmp_path / "one/caches", seed=seed)
    two, two_manifest = matrix._cache_directories(tmp_path / "two/caches", seed=seed)
    assert one_manifest == two_manifest == manifest
    (Path(one["TRITON_CACHE_DIR"]) / "kernel.cubin").write_bytes(b"run-local mutation")
    assert (
        Path(two["TRITON_CACHE_DIR"]) / "kernel.cubin"
    ).read_bytes() == original.read_bytes()
    assert (
        capture_cache_regime(
            "populated_kernel_cache",
            environ={name: str(seed / name) for name in CACHE_VARIABLES},
        )
        == manifest
    )


def test_symlink_cache_artifacts_are_not_silently_copied_as_regular_seed(tmp_path):
    environment, _ = matrix._cache_directories(tmp_path / "run/caches")
    external = tmp_path / "external"
    external.write_bytes(b"not a managed cache artifact")
    (Path(environment["TRITON_CACHE_DIR"]) / "link").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        matrix._seed_cache(tmp_path / "run/caches", tmp_path / "seed")


@pytest.mark.parametrize(
    "framework,first,final",
    [
        ("huggingface", ("evaluate", 2), ["save_model"]),
        ("llamafactory", ("evaluate", 2), ["save_model", "evaluate"]),
        ("axolotl", ("evaluate", 0), ["save_pretrained"]),
    ],
)
def test_independent_calls_keep_initial_forced_terminal_and_extra_final_calls(
    framework, first, final
):
    calls = matrix._expected_calls(framework, 5)
    assert (calls[0]["operation"], calls[0]["optimizer_step"]) == first
    assert [row["operation"] for row in calls if not row["inside_train"]] == final
    assert [
        row["operation"]
        for row in calls
        if row["optimizer_step"] == 5 and row["inside_train"]
    ] == ["evaluate", "_save_checkpoint"]


def test_complete_reference_recomputes_groups_and_equal_process_means(native_run):
    runs = [
        _pipeline_run(native_run, number, epochs=epochs, factor=factor)
        for number, epochs, factor in [(1, 24, 1), (2, 28, 3), (3, 32, 2.2)]
    ]
    prediction = compare_finetuning_pipeline(runs[:2], runs[2])
    before = deepcopy(runs)
    result = matrix._reference_comparison(runs[:2], runs[2], prediction)
    for field in matrix._FIELDS:
        assert result[field]["predicted"] == pytest.approx(
            prediction["timing"][field]["expected"]
        )
        assert result[field]["actual"] == pytest.approx(
            runs[2]["observation"]["timeline"]["totals"][field]
        )
    assert runs == before
    prediction["timing"]["wall_seconds"]["expected"] += 1
    with pytest.raises(ValueError, match="means differ"):
        matrix._reference_comparison(runs[:2], runs[2], prediction)


def test_reference_rejects_a_mutated_component_group(native_run):
    runs = [_pipeline_run(native_run, number, factor=number) for number in (1, 2, 3)]
    prediction = compare_finetuning_pipeline(runs[:2], runs[2])
    prediction["target_components"][0]["group_id"] = "0" * 64
    with pytest.raises(ValueError, match="grouping differs"):
        matrix._reference_comparison(runs[:2], runs[2], prediction)


@pytest.mark.parametrize("kind", [None, "setup", "trainer_other", "checkpoint"])
def test_reference_checks_intervals_and_per_stage_means(native_run, kind):
    runs = [_pipeline_run(native_run, number, factor=number) for number in (1, 2, 3)]
    prediction = compare_finetuning_pipeline(runs[:2], runs[2])
    selected = (
        prediction["timing"] if kind is None else prediction["timing"]["by_kind"][kind]
    )
    selected["wall_seconds"]["max"] += 1
    with pytest.raises(ValueError, match="means differ"):
        matrix._reference_comparison(runs[:2], runs[2], prediction)


def test_plan_alignment_only_remaps_root_ids_and_source_metadata(native_run):
    plan = pipeline_plan_from_run(_pipeline_run(native_run, 1))
    before = deepcopy(plan)
    changed = deepcopy(plan)
    changed.update(source="cpu_native_schedule", projection={"max_steps": 72})
    for root in changed["root_events"]:
        root["event_id"] += 10000
    for update in changed["updates"]:
        update["preceding_event_ids"] = [
            value + 10000 for value in update["preceding_event_ids"]
        ]
    for operation in changed["operations"]:
        if "event_id" in operation:
            operation["event_id"] += 10000
    assert matrix._semantic_plan(plan) == matrix._semantic_plan(changed)
    assert plan == before
    changed["updates"][0]["batches"][0]["epoch_microstep"] += 1
    assert matrix._semantic_plan(plan) != matrix._semantic_plan(changed)


def test_sealed_predictions_detect_any_rewrite(tmp_path):
    record = {}
    for name, key in [
        ("future-plan.json", "plan_sha256"),
        ("future-prediction.json", "prediction_sha256"),
    ]:
        (tmp_path / name).write_text("{}\n")
        record[key] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    matrix._check_seal(tmp_path, record)
    (tmp_path / "future-prediction.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="sealed"):
        matrix._check_seal(tmp_path, record)


@pytest.mark.parametrize("code", [0, 86])
def test_subprocess_records_exact_pid_exit_and_all_stdout(tmp_path, code):
    execution, start, end = matrix._execute(
        [
            sys.executable,
            "-c",
            f"import os; print(os.getpid()); raise SystemExit({code})",
        ],
        directory=tmp_path,
        root=tmp_path,
        environment=dict(os.environ),
        timeout=10,
        expected_code=code,
    )
    assert (tmp_path / "run.log").read_text().strip() == str(execution["pid"])
    assert start <= end
    assert json.loads((tmp_path / "execution.json").read_text())["returncode"] == code


def test_failed_subprocess_preserves_exit_identity_and_log(tmp_path):
    with pytest.raises(RuntimeError, match="exited 3"):
        matrix._execute(
            [sys.executable, "-c", 'print("failure evidence"); raise SystemExit(3)'],
            directory=tmp_path,
            root=tmp_path,
            environment=dict(os.environ),
            timeout=10,
        )
    evidence = json.loads((tmp_path / "execution.json").read_text())
    assert evidence["returncode"] == 3 and evidence["pid"] > 0
    assert (tmp_path / "run.log").read_text() == "failure evidence\n"


def test_allocator_control_budgets_are_fixed_functions_of_real_calibration_peaks():
    runs = [
        {"peak_allocated_bytes": 3 * 1024**3, "peak_reserved_bytes": 4 * 1024**3},
        {"peak_allocated_bytes": 2 * 1024**3, "peak_reserved_bytes": 3 * 1024**3},
    ]
    low, formula_low = _budget(runs, "oom")
    high, formula_high = _budget(runs, "success")
    assert low == 1.5 * 1024**3
    assert high == 5 * 1024**3
    assert "minimum_calibration_peak_allocated" in formula_low
    assert "maximum_calibration_peak_reserved" in formula_high
    with pytest.raises(ValueError):
        _budget(runs, "unknown")
