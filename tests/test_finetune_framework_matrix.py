from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_estimator import finetune_workload_signature
from native import finetune_framework_matrix as matrix
from native import prepare_finetune_fixture as fixture
from test_finetune_run_estimator import native_run as native_run


def _report(native_run, *, windows=(2, 2, 2, 2), warmup=2, num_samples=12):
    report = deepcopy(native_run)
    config = report["training_config"]
    dataset = profile_tokenized_records(
        [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}] * num_samples,
        sequence_length=4,
        source_framework="llamafactory",
    )
    report["dataset"] = dataset
    digest = dataset["provenance"]["content_sha256"]
    config.update(
        num_samples=num_samples, data_signature=digest, max_steps=len(windows)
    )
    updates = [
        {
            "optimizer_step": index + 1,
            "seconds": count * (1.0 if index < warmup else 0.5),
            "microsteps": count,
            "samples": count,
            "losses": [0.9] * count,
            "warmup": index < warmup,
        }
        for index, count in enumerate(windows)
    ]
    report["workload"].update(
        optimizer_steps=len(windows),
        microsteps=sum(windows),
        samples=sum(windows),
        warmup_optimizer_steps=warmup,
        measured_optimizer_steps=len(windows) - warmup,
        warmup_microsteps=sum(windows[:warmup]),
        measured_microsteps=sum(windows[warmup:]),
        shape_counts=[{"shape": [1, 4], "microsteps": sum(windows)}],
        sha256=digest,
    )
    report["observation"].update(
        updates=updates,
        workload=deepcopy(report["workload"]),
        optimizer_update_seconds=[update["seconds"] for update in updates[warmup:]],
        seconds_per_microstep=[0.5] * (len(windows) - warmup),
        signature=finetune_workload_signature(
            report["model_config"], config, profile_id="a100"
        ),
    )
    seconds = sum(update["seconds"] for update in updates)
    report["timing"].update(
        total_wall_seconds=10.0 + seconds,
        setup_seconds=8.0,
        trainer_wall_seconds=seconds,
        training_loop_seconds=seconds,
        optimizer_update_sample_seconds=seconds,
        trainer_other_seconds=0.0,
        warmup_training_seconds=sum(update["seconds"] for update in updates[:warmup]),
        evaluation_seconds=0.0,
        checkpoint_seconds=1.0,
        finalization_seconds=1.0,
    )
    return report


def test_read_report_validates_partial_accumulation(native_run, tmp_path):
    report = _report(native_run, windows=(2, 1, 2, 1), num_samples=3)
    path = tmp_path / "report.json"
    matrix._write_json(path, report)
    assert matrix._read_report(path, 4)["workload"]["microsteps"] == 6


def test_read_report_rejects_self_consistent_but_wrong_training_count(
    native_run, tmp_path
):
    # All recorded counters agree, but six rows with accumulation=2 need eight
    # microsteps for four updates. A six-microstep report is not valid evidence.
    report = _report(native_run, windows=(2, 1, 2, 1), num_samples=6)
    path = tmp_path / "report.json"
    matrix._write_json(path, report)
    with pytest.raises(ValueError, match="microsteps differs"):
        matrix._read_report(path, 4)


def test_read_report_obeys_requested_warmup(native_run, tmp_path):
    report = _report(native_run, windows=(2, 2, 2, 2), warmup=1)
    path = tmp_path / "report.json"
    matrix._write_json(path, report)
    assert matrix._read_report(path, 4, 1) == report
    with pytest.raises(ValueError, match="measured optimizer"):
        matrix._read_report(path, 4, 2)


def test_summary_uses_production_microstep_projection(native_run):
    cal = _report(native_run, windows=(2, 1, 2, 1), num_samples=3)
    held = _report(native_run, windows=(2, 1, 2, 1, 2), num_samples=3)
    comparison = matrix._summarize(cal, held, 5)
    # Five optimizer updates only process eight microsteps, not 5 * 2.
    assert comparison["holdout_workload"]["microsteps"] == 8
    assert (
        comparison["native_pipeline_seconds"]["predicted"]
        == held["timing"]["total_wall_seconds"]
    )
    assert comparison["native_pipeline_seconds"]["absolute_percentage_error"] == 0
    assert comparison["steady_training_seconds"]["predicted"] == 2.5
    assert comparison["seconds_per_microstep"]["predicted"] == 0.5


def test_summary_rejects_different_workload_signature(native_run):
    cal = _report(native_run)
    held = deepcopy(cal)
    held["observation"]["signature"] = "different"
    with pytest.raises(ValueError, match="signatures differ"):
        matrix._summarize(cal, held, 4)


def test_relative_config_paths_survive_copy_without_rewriting_hub_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "local.jsonl"
    dataset.write_text("{}\n")
    raw = {
        "base_model": "model",
        "tokenizer_config": "org/tokenizer",
        "dataset_prepared_path": "cache/prepared",
        "seed": 97,
        "datasets": [
            {"path": "local.jsonl"},
            {"path": "org/dataset", "data_files": ["train/*.jsonl"]},
            {
                "path": "json",
                "data_files": ["./data/*.jsonl", "https://example.org/a.jsonl"],
            },
            {"path": "s3://bucket/dataset"},
        ],
    }
    copied = matrix._relocate_config(raw, tmp_path / "source.yaml", "axolotl")
    assert copied["base_model"] == str(model)
    assert copied["tokenizer_config"] == "org/tokenizer"
    assert copied["dataset_prepared_path"] == str(tmp_path / "cache/prepared")
    assert copied["datasets"][0]["path"] == str(dataset)
    assert copied["datasets"][1] == raw["datasets"][1]
    assert copied["datasets"][2]["data_files"] == [
        str(tmp_path / "data/*.jsonl"),
        "https://example.org/a.jsonl",
    ]
    assert copied["datasets"][3]["path"] == "s3://bucket/dataset"
    assert copied["seed"] == raw["seed"] == 97
    assert raw["base_model"] == "model"


def test_factory_default_dataset_directory_stays_relative_to_source(tmp_path):
    copied = matrix._relocate_config({}, tmp_path / "source.yaml", "llamafactory")
    assert copied["dataset_dir"] == str(tmp_path / "data")


def _arguments(tmp_path, config):
    source = tmp_path / "source.json"
    matrix._write_json(source, config)
    return [
        "--framework",
        "llamafactory",
        "--config",
        str(source),
        "--output-dir",
        str(tmp_path / "evidence"),
        "--holdout-runs",
        "1",
        "--calibration-steps",
        "4",
        "--holdout-steps",
        "5",
        "--warmup-steps",
        "1",
    ]


def test_matrix_preserves_seed_and_uses_requested_step_counts(
    native_run, tmp_path, monkeypatch
):
    config = {
        "cutoff_len": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "seed": 137,
    }
    calls = []

    def run(command, **kwargs):
        copied = json.loads(Path(command[command.index("--config") + 1]).read_text())
        steps = copied["max_steps"]
        assert copied["seed"] == 137
        assert Path(copied["output_dir"]).parent.is_dir()
        assert not Path(copied["output_dir"]).exists()
        assert command[command.index("--warmup-steps") + 1] == "1"
        matrix._write_json(
            Path(command[command.index("--output") + 1]),
            _report(native_run, windows=(2,) * steps, warmup=1),
        )
        calls.append(copied)
        return {"pid": len(calls), "process_wall_seconds": 1.0}

    monkeypatch.setattr(matrix, "_run", run)
    assert matrix.main(_arguments(tmp_path, config)) == 0
    assert [call["max_steps"] for call in calls] == [4, 5]
    summary = json.loads((tmp_path / "evidence/summary.json").read_text())
    assert summary["sampling_plan"]["warmup_optimizer_steps_per_run"] == 1
    assert summary["status"] == "completed"
    assert summary["worst_holdout_errors"]["native_pipeline_seconds"] == 0


@pytest.mark.parametrize("sequence,batch,accumulation", [(512, 2, 3), (1024, 1, 4)])
def test_matrix_accepts_requested_shapes(
    sequence, batch, accumulation, tmp_path, monkeypatch
):
    config = {
        "cutoff_len": sequence,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": accumulation,
    }
    observed = []

    def stopped_before_training(command, **kwargs):
        observed.append(
            json.loads(Path(command[command.index("--config") + 1]).read_text())
        )
        raise RuntimeError("test stops before training")

    monkeypatch.setattr(matrix, "_run", stopped_before_training)
    assert matrix.main(_arguments(tmp_path, config)) == 1
    assert observed[0]["cutoff_len"] == sequence
    assert observed[0]["per_device_train_batch_size"] == batch
    assert observed[0]["gradient_accumulation_steps"] == accumulation
    assert "seed" not in observed[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff_len", True),
        ("cutoff_len", 1),
        ("per_device_train_batch_size", 0),
        ("gradient_accumulation_steps", 1.5),
    ],
)
def test_matrix_rejects_invalid_shapes_before_output(field, value, tmp_path):
    config = {
        "cutoff_len": 128,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        field: value,
    }
    with pytest.raises(SystemExit):
        matrix.main(_arguments(tmp_path, config))
    assert not (tmp_path / "evidence").exists()


def test_matrix_preserves_prior_evidence(tmp_path):
    config = {
        "cutoff_len": 128,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
    }
    directory = tmp_path / "evidence"
    directory.mkdir()
    previous = directory / "summary.json"
    previous.write_text("previous")
    with pytest.raises(SystemExit):
        matrix.main(_arguments(tmp_path, config))
    assert previous.read_text() == "previous"


def test_timeout_terminates_process_group(tmp_path, monkeypatch):
    signals = []
    original = os.killpg

    def killpg(pid, sig):
        signals.append(sig)
        original(pid, sig)

    monkeypatch.setattr(matrix.os, "killpg", killpg)
    with pytest.raises(subprocess.TimeoutExpired):
        matrix._run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            log=tmp_path / "timeout.log",
            root=tmp_path,
            environment=dict(os.environ),
            timeout=0.1,
        )
    assert signals == [signal.SIGTERM]


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()


def test_default_fixture_retains_original_dataset_hash(tmp_path):
    manifest = fixture.prepare_training_dataset(_Tokenizer(), tmp_path)
    assert (
        manifest["dataset_sha256"]
        == "3689fcfe0194023646b6b5317aa8568ee95b848eef526489f8df9cddf9ec5c33"
    )
    assert manifest["dataset_samples"] == 64
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 64


@pytest.mark.parametrize("sequence", [512, 1024])
def test_long_fixture_is_reproducible_and_exceeds_cutoff(tmp_path, sequence):
    first = fixture.prepare_training_dataset(
        _Tokenizer(), tmp_path / "a", sequence_length=sequence, num_samples=7
    )
    second = fixture.prepare_training_dataset(
        _Tokenizer(), tmp_path / "b", sequence_length=sequence, num_samples=7
    )
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["dataset_samples"] == 7
    assert first["raw_concatenated_token_length_min"] > sequence
    assert (
        hashlib.sha256((tmp_path / "a/train.jsonl").read_bytes()).hexdigest()
        == first["dataset_sha256"]
    )


def test_fixture_refuses_to_overwrite_data(tmp_path):
    fixture.prepare_training_dataset(_Tokenizer(), tmp_path)
    previous = (tmp_path / "train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        fixture.prepare_training_dataset(_Tokenizer(), tmp_path, sequence_length=512)
    assert (tmp_path / "train.jsonl").read_bytes() == previous


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": True},
        {"sequence_length": 1},
        {"num_samples": 0},
        {"num_samples": 1.5},
    ],
)
def test_fixture_rejects_invalid_scenarios_before_writing(tmp_path, kwargs):
    directory = tmp_path / "data"
    with pytest.raises(ValueError, match="integer"):
        fixture.prepare_training_dataset(_Tokenizer(), directory, **kwargs)
    assert not directory.exists()
