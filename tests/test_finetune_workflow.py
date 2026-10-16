"""Workspace lifecycle tests use real small subprocesses and native-shaped reports."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from fakegpu import finetune_workflow as workflow
from fakegpu._process import process_marker
from fakegpu.finetune_workflow import _coverage_and_budget as calculate_budget
from test_finetune_transfer_plan import transfer_run
from test_finetune_run_estimator import native_run as native_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def workspace_case(tmp_path, native_run, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture weights")
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"text":"training"}\n')
    write_json(dataset / "dataset_info.json", {"train": {"file_name": "train.jsonl"}})
    template = transfer_run(native_run, 1)
    template["schedule"]["max_steps"] = 72
    config = deepcopy(template["native_config"])
    config.update(model_name_or_path="model", dataset_dir="data", dataset="train")
    source = write_json(tmp_path / "config.json", config)
    template_path = write_json(tmp_path / "report-template.json", template)
    worker = tmp_path / "worker.py"
    worker.write_text("""import json, os, pathlib, sys, uuid
from datetime import datetime, timezone
from fakegpu._finetune_model import native_config_signature, native_execution_signature
from fakegpu.finetune_estimator import finetune_workload_signature
folder = pathlib.Path(sys.argv[1])
report = json.loads(pathlib.Path(sys.argv[2]).read_text())
report["native_config"] = json.loads((folder / "config.json").read_text())
report["run_identity"] = {"uuid": str(uuid.uuid4()), "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()}
report["config_path"] = str(folder / "config.json")
report["cache_regime"] = json.loads((folder / "cache-layout.json").read_text())["input_manifest"]
config = report["training_config"]
config["native_execution_signature"] = native_execution_signature(native_config_signature(report["native_config"]), report["observation"]["observed_model"])
report["observation"]["signature"] = finetune_workload_signature(report["model_config"], config, profile_id=report["profile_id"])
(folder / "model").mkdir()
(folder / "model" / "adapter.safetensors").write_bytes(b"complete adapter fixture")
(folder / "run.json").write_text(json.dumps(report))
print("native fixture completed", flush=True)
""")
    launched = []

    def command(request, directory):
        launched.append(directory)
        return [sys.executable, str(worker), str(directory), str(template_path)]

    monkeypatch.setattr(workflow, "_native_command", command)
    monkeypatch.setattr(workflow, "_runtime_identity", lambda *args: {"versions": {}})

    # Test workspace orchestration independently from the transfer estimator's
    # own numerical tests, retaining real native report validation above.
    def budget(reports, duration, memory, reserve, extra_reports):
        assert len({report["run_identity"]["uuid"] for report in reports}) == 2
        return (
            {"duration": duration},
            {"complete": True},
            {"pipeline": {"seconds": 123}, "memory": {"reserve_bytes": reserve}},
        )

    monkeypatch.setattr(workflow, "_coverage_and_budget", budget)
    options = dict(
        framework="llamafactory",
        profile_id="a100",
        workspace=tmp_path / "workspace",
        cuda_visible_devices="1",
        calibration_steps=72,
        max_steps=96,
        warmup_steps=4,
    )
    return source, options, launched, command


def test_two_independent_processes_seal_and_cpu_only_reuse(workspace_case, monkeypatch):
    source, options, launched, _ = workspace_case
    result = workflow.calibrate_finetuning(source, **options)
    assert result["status"] == "complete"
    assert len(launched) == 2
    root = options["workspace"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(attempt["status"] == "succeeded" for attempt in manifest["attempts"])
    assert len({attempt["run_uuid"] for attempt in manifest["attempts"]}) == 2
    command = workflow._native_command
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("CPU reuse launched a subprocess"),
    )
    again = workflow.budget_finetuning_workspace(root, max_steps=96)
    assert again == result
    shorter = workflow.budget_finetuning_workspace(root, max_steps=80)
    assert shorter["prediction_directory"] != result["prediction_directory"]
    assert len(launched) == 2
    assert command is workflow._native_command


def test_resume_reuses_first_success_and_keeps_failed_attempt(
    workspace_case, monkeypatch
):
    source, options, launched, original = workspace_case

    def fail_second(request, directory):
        if len(launched) == 1:
            launched.append(directory)
            return [
                sys.executable,
                "-c",
                "print('intentional failure'); raise SystemExit(9)",
            ]
        return original(request, directory)

    monkeypatch.setattr(workflow, "_native_command", fail_second)
    with pytest.raises(RuntimeError, match="exited 9"):
        workflow.calibrate_finetuning(source, **options)
    failed = options["workspace"] / "attempts/calibration-2/attempt-1"
    before = workflow._tree(failed)
    monkeypatch.setattr(workflow, "_native_command", original)
    result = workflow.calibrate_finetuning(source, resume=True, **options)
    assert result["status"] == "complete"
    assert len(launched) == 3
    assert workflow._tree(failed) == before
    manifest = workflow._read_manifest(options["workspace"])
    assert [row["status"] for row in manifest["attempts"]] == [
        "succeeded",
        "failed",
        "succeeded",
    ]


@pytest.mark.parametrize(
    "artifact", ["run.json", "model/adapter.safetensors", "run.log", "execution.json"]
)
def test_completed_artifact_tampering_refuses_reuse_without_training(
    workspace_case, artifact
):
    source, options, launched, _ = workspace_case
    workflow.calibrate_finetuning(source, **options)
    path = launched[0] / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        workflow.budget_finetuning_workspace(options["workspace"], max_steps=96)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        workflow.calibrate_finetuning(source, resume=True, **options)
    assert len(launched) == 2


def test_prediction_seal_and_budget_tampering_rejected(workspace_case):
    source, options, _, _ = workspace_case
    result = workflow.calibrate_finetuning(source, **options)
    path = Path(result["prediction_directory"]) / "budget.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="sealed prediction"):
        workflow.budget_finetuning_workspace(options["workspace"], max_steps=96)


def test_changed_source_data_requires_new_workspace(workspace_case):
    source, options, launched, _ = workspace_case
    workflow.calibrate_finetuning(source, **options)
    (source.parent / "data/train.jsonl").write_text('{"text":"new"}\n')
    with pytest.raises(ValueError, match="inputs or execution identity changed"):
        workflow.calibrate_finetuning(source, resume=True, **options)
    assert len(launched) == 2
    # Historical CPU reports remain useful after the source dataset changes.
    assert (
        workflow.budget_finetuning_workspace(options["workspace"], max_steps=96)[
            "status"
        ]
        == "complete"
    )


def test_missing_coverage_preserves_two_runs_and_never_resamples(
    workspace_case, monkeypatch
):
    source, options, launched, _ = workspace_case
    monkeypatch.setattr(
        workflow,
        "_coverage_and_budget",
        lambda *args: (
            {},
            {"complete": False, "missing_groups": [{"kind": "evaluation"}]},
            None,
        ),
    )
    result = workflow.calibrate_finetuning(source, **options)
    assert result["status"] == "coverage_incomplete"
    assert result["budget"] is None
    assert len(launched) == 2
    assert not (Path(result["prediction_directory"]) / "budget.json").exists()
    assert workflow.calibrate_finetuning(source, resume=True, **options) == result
    assert len(launched) == 2


def test_timeout_terminates_process_and_preserves_failure(workspace_case, monkeypatch):
    source, options, _, _ = workspace_case
    monkeypatch.setattr(
        workflow,
        "_native_command",
        lambda request, directory: [
            sys.executable,
            "-c",
            "import time; print('waiting', flush=True); time.sleep(30)",
        ],
    )
    with pytest.raises(subprocess.TimeoutExpired):
        workflow.calibrate_finetuning(source, timeout_seconds=0.3, **options)
    root = options["workspace"]
    execution = json.loads(
        (root / "attempts/calibration-1/attempt-1/execution.json").read_text()
    )
    assert execution["status"] == "timed_out"
    assert execution["returncode"] != 0
    assert not workflow._live(execution)
    assert workflow._read_manifest(root)["status"] == "calibration_failed"


def test_total_budget_limits_child_lifetime(workspace_case, monkeypatch):
    source, options, _, _ = workspace_case
    monkeypatch.setattr(
        workflow,
        "_native_command",
        lambda request, directory: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        workflow.calibrate_finetuning(
            source, timeout_seconds=10, calibration_budget_seconds=0.1, **options
        )
    assert time.monotonic() - started < 5


def test_workspace_lock_excludes_concurrent_drivers(tmp_path):
    with workflow._locked(tmp_path):
        with pytest.raises(RuntimeError, match="another process"):
            with workflow._locked(tmp_path):
                pytest.fail("second writer obtained the lock")


def test_live_orphan_blocks_resume(workspace_case):
    source, options, launched, _ = workspace_case
    workflow.calibrate_finetuning(source, **options)
    manifest = workflow._read_manifest(options["workspace"])
    execution_path = launched[0] / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution.update(
        pid=os.getpid(),
        process_start_ticks=process_marker(os.getpid())["start_ticks"],
    )
    write_json(execution_path, execution)
    manifest["attempts"][0]["artifacts"] = workflow._attempt_artifacts(launched[0])
    write_json(options["workspace"] / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="still running"):
        workflow.calibrate_finetuning(source, resume=True, **options)
    assert len(launched) == 2


def test_relative_huggingface_and_axolotl_inputs_are_preserved(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"input_ids":[1,2]}\n')
    for framework, config in [
        (
            "huggingface",
            {
                "model_name_or_path": "model",
                "train_dataset": "rows.jsonl",
                "eval_dataset": "rows.jsonl",
            },
        ),
        (
            "axolotl",
            {
                "base_model": "model",
                "datasets": [{"path": "json", "data_files": {"train": ["rows.jsonl"]}}],
            },
        ),
    ]:
        source = write_json(tmp_path / f"{framework}.json", config)
        relocated, inputs = workflow._prepare_inputs(source, framework)
        assert str(rows) in inputs
        if framework == "huggingface":
            assert relocated["train_dataset"] == relocated["eval_dataset"] == str(rows)
        else:
            assert relocated["datasets"][0]["data_files"]["train"] == [str(rows)]


def test_remote_dataset_rejected_before_training(workspace_case):
    source, options, launched, _ = workspace_case
    write_json(
        source.parent / "data/dataset_info.json",
        {"train": {"hf_hub_url": "remote/data"}},
    )
    with pytest.raises(ValueError, match="local file_name"):
        workflow.calibrate_finetuning(source, **options)
    assert not launched


def test_native_command_keeps_training_steps_distinct_from_future_horizon(tmp_path):
    request = {
        "python_executable": sys.executable,
        "framework": "huggingface",
        "profile_id": "a100",
        "warmup_steps": 2,
        "cache_regime": "empty_kernel_cache",
        "planning_duration": {"epochs": 3.5, "max_steps": None},
    }
    command = workflow._native_command(request, tmp_path)
    assert command[-2:] == ["--planning-epochs", "3.5"]
    assert "--max-steps" not in command
    assert "--output" in command


def test_runtime_preflight_reads_metadata_without_importing_torch(monkeypatch):
    before = "torch" in sys.modules
    identity = workflow._runtime_identity(sys.executable, "huggingface", os.environ)
    assert identity["source"] == workflow._source_identity()
    assert "transformers" in identity["versions"]
    assert ("torch" in sys.modules) == before


def test_reused_work_pool_is_snapshotted_for_cpu_rebudget(
    workspace_case, native_run, monkeypatch
):
    from test_finetune_transfer_plan import transfer_run

    source, options, launched, _ = workspace_case
    imported = write_json(
        source.parent / "imported-transfer.json", transfer_run(native_run, 3, factor=3)
    )
    selected = []

    def budget(reports, duration, memory, reserve, extras):
        selected.append(deepcopy(extras))
        assert len(reports) == 2
        return {}, {"complete": True}, {"imported": len(extras)}

    monkeypatch.setattr(workflow, "_coverage_and_budget", budget)
    workflow.calibrate_finetuning(source, reuse_calibration_runs=[imported], **options)
    imported.unlink()
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("budget reuse started a subprocess"),
    )
    result = workflow.budget_finetuning_workspace(options["workspace"], max_steps=80)
    assert result["budget"]["imported"] == 1
    assert selected[0] == selected[1]
    assert len(launched) == 2
    manifest = workflow._read_manifest(options["workspace"])
    snapshot = options["workspace"] / manifest["reuse_calibration_runs"][0]["directory"]
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(ValueError, match="snapshot SHA256 mismatch"):
        workflow.budget_finetuning_workspace(options["workspace"], max_steps=80)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_budget_bytes": 0},
        {"reserve_bytes": -1},
        {"calibration_steps": 4},
        {"cuda_visible_devices": "0,1"},
    ],
)
def test_invalid_resource_limits_do_not_launch_training(workspace_case, kwargs):
    source, options, launched, _ = workspace_case
    with pytest.raises(ValueError):
        workflow.calibrate_finetuning(source, **{**options, **kwargs})
    assert launched == []


def test_workflow_cli_forwards_reused_reports_and_reports_incomplete(
    workspace_case, monkeypatch, capsys
):
    from fakegpu.finetune_workflow_cli import budget_main, calibrate_main

    source, options, _, _ = workspace_case
    calls = []

    def calibrate(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "coverage_incomplete", "coverage": {"complete": False}}

    monkeypatch.setattr(workflow, "calibrate_finetuning", calibrate)
    arguments = [
        "--workspace",
        str(options["workspace"]),
        "--max-steps",
        "96",
        "--reuse-calibration-run",
        "old.json",
    ]
    assert (
        calibrate_main(
            [
                *arguments,
                "--config",
                str(source),
                "--framework",
                "llamafactory",
                "--profile",
                "a100",
                "--cuda-device",
                "1",
                "--calibration-steps",
                "72",
            ]
        )
        == 1
    )
    assert calls[-1]["reuse_calibration_runs"] == ["old.json"]
    monkeypatch.setattr(workflow, "budget_finetuning_workspace", calibrate)
    assert budget_main(arguments) == 1
    assert calls[-1]["reuse_calibration_runs"] == ["old.json"]
    assert "coverage_incomplete" in capsys.readouterr().out


def test_real_transfer_planner_time_and_memory_interfaces(native_run):
    from test_finetune_transfer_plan import transfer_run

    reports = [transfer_run(native_run, 1), transfer_run(native_run, 2, factor=3)]
    plan, coverage, budget = workflow._coverage_and_budget(
        reports, {"epochs": None, "max_steps": 96}, None, 0, []
    )
    assert coverage["complete"]
    assert budget["schema_version"] == "fakegpu.finetune_transfer_budget.v1"
    assert budget["pipeline"]["timing"]["wall_seconds"]["expected"] > 0
    assert budget["memory"]
    assert plan["workload"]["optimizer_steps"] == 96
    with pytest.raises(ValueError, match="horizon"):
        workflow._coverage_and_budget(
            reports, {"epochs": None, "max_steps": 97}, None, 0, []
        )


def test_timeout_kills_descendants_even_if_leader_exits(tmp_path):
    from fakegpu._finetune_workflow_store import _execute

    child_pid = tmp_path / "child.pid"
    child_code = (
        "import signal,time,os,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path("
        + repr(str(child_pid))
        + ").write_text(str(os.getpid())); time.sleep(30)"
    )
    leader_code = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
        + repr(child_code)
        + "]); time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _execute([sys.executable, "-c", leader_code], tmp_path, os.environ, 0.3)
    pid = int(child_pid.read_text())
    for _ in range(50):
        marker = process_marker(pid)
        if marker is None or marker["state"] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("timed-out calibration left a live descendant")


def test_complete_workspace_uses_real_transfer_protocol(
    workspace_case, native_run, monkeypatch
):
    from test_finetune_transfer_plan import transfer_run

    # Keep the implementation reference independently of the lifecycle fixture's
    # stub; this test spans subprocess artifacts through plan/cost/memory APIs.
    source, options, launched, command = workspace_case
    monkeypatch.setattr(workflow, "_coverage_and_budget", calculate_budget)
    templates = []
    for index in (1, 2):
        report = transfer_run(native_run, index, factor=index)
        report["schedule"]["max_steps"] = 72
        templates.append(write_json(source.parent / f"transfer-{index}.json", report))

    def transfer_command(request, directory):
        argv = command(request, directory)
        argv[-1] = str(templates[len(launched) - 1])
        return argv

    monkeypatch.setattr(workflow, "_native_command", transfer_command)
    result = workflow.calibrate_finetuning(source, **options)
    assert result["status"] == "complete"
    assert result["budget"]["pipeline"]["empirical_approximation"]["enabled"]
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("reprojection started a subprocess"),
    )
    shortened = workflow.budget_finetuning_workspace(options["workspace"], max_steps=84)
    assert shortened["status"] == "complete"
    with pytest.raises(ValueError, match="horizon"):
        workflow.budget_finetuning_workspace(options["workspace"], max_steps=97)
    assert len(launched) == 2


def test_prediction_publication_can_resume_without_rerunning_training(
    workspace_case, monkeypatch
):
    source, options, launched, _ = workspace_case
    rename = Path.rename

    def interrupted(path, target):
        if path.name.endswith(".pending"):
            raise OSError("simulated coordinator interruption before publication")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="before publication"):
        workflow.calibrate_finetuning(source, **options)
    manifest = workflow._read_manifest(options["workspace"])
    assert len(manifest["predictions"]) == 1
    pending = options["workspace"] / manifest["predictions"][0]["staging_directory"]
    assert (pending / "seal.json").exists()
    monkeypatch.setattr(Path, "rename", rename)
    result = workflow.calibrate_finetuning(source, resume=True, **options)
    assert result["status"] == "complete"
    assert len(launched) == 2
    assert not pending.exists()
    assert workflow._read_manifest(options["workspace"])["status"] == "complete"


def test_invalid_sampler_attestation_fails_before_second_calibration(workspace_case):
    source, options, launched, _ = workspace_case
    path = source.parent / "report-template.json"
    template = json.loads(path.read_text())
    del template["observation"]["sampling_verification"]
    write_json(path, template)
    with pytest.raises(ValueError, match="verification|sampling|sampler"):
        workflow.calibrate_finetuning(source, **options)
    assert len(launched) == 1
    assert (launched[0] / "run.json").exists()
    assert (
        workflow._read_manifest(options["workspace"])["attempts"][0]["status"]
        == "failed"
    )
