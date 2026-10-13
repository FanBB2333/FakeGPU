from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fakegpu.finetune_cli import main
from fakegpu.finetune_estimator import OBSERVATION_SCHEMA_VERSION
from fakegpu._local_checkpoint import local_checkpoint_identity


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def training_case(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": True,
    }
    (model_dir / "config.json").write_text(json.dumps(model))
    (model_dir / "model.safetensors").write_bytes(b"weights version1")
    training = {
        "model_name_or_path": "./model",
        "torch_dtype": "bfloat16",
        "bf16": True,
        "attn_implementation": "sdpa",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "sequence_length": 32,
        "num_samples": 100,
        "optim": "adamw_torch",
        "peft": {"r": 4, "target_modules": ["q_proj", "v_proj"]},
        "software_versions": {"test_backend": "fixture-only"},
    }
    config = tmp_path / "training.json"
    config.write_text(json.dumps(training))
    return config, model_dir, training


def _report(config: Path, capsys, *options: str) -> dict:
    assert (
        main(["--config", str(config), "--profile", "a100", "--json", "-", *options])
        == 0
    )
    captured = capsys.readouterr()
    assert not captured.err
    return json.loads(captured.out)


@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
def test_registered_cli_estimates_each_framework(
    tmp_path, training_case, framework
) -> None:
    _, model_dir, _ = training_case
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fakegpu",
            "estimate-finetune",
            "--config",
            str(ROOT / "examples/finetuning" / f"{framework}-lora.yaml"),
            "--model",
            str(model_dir),
            "--profile",
            "a100",
            "--num-samples",
            "64",
            "--epochs",
            "2",
            "--sequence-length",
            "32",
            "--json",
            "-",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["source_framework"] == framework
    assert report["workload"]["optimizer_steps"] == 32
    assert report["workload"]["processed_samples"] == 128
    assert report["memory"]["peak_bytes"]["expected"] > 0
    assert report["timing"]["evidence"] == "analytical_scenario_range"


def test_cli_samples_epochs_scale_time_but_not_peak_vram(training_case, capsys) -> None:
    config, _, _ = training_case
    initial = _report(config, capsys)
    scaled = _report(config, capsys, "--num-samples", "200", "--epochs", "3")
    assert scaled["workload"]["microsteps"] == initial["workload"]["microsteps"] * 6
    assert scaled["timing"]["training_loop_seconds"]["expected"] == pytest.approx(
        initial["timing"]["training_loop_seconds"]["expected"] * 6
    )
    assert scaled["memory"] == initial["memory"]
    assert scaled["signature"] == initial["signature"]


def test_cli_max_steps_overrides_epochs(training_case, capsys) -> None:
    config, _, _ = training_case
    report = _report(config, capsys, "--max-steps", "3", "--epochs", "100")
    assert report["workload"]["stop_rule"] == "max_steps"
    assert report["workload"]["optimizer_steps"] == 3
    assert report["workload"]["microsteps"] == 6


def test_cli_json_artifact_is_parseable(training_case, tmp_path, capsys) -> None:
    config, _, _ = training_case
    output = tmp_path / "reports" / "estimate.json"
    assert (
        main(["--config", str(config), "--profile", "a100", "--json", str(output)]) == 0
    )
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text())["source_framework"] == "huggingface"


@pytest.mark.parametrize(
    "options, error",
    [
        (["--num-samples", "0"], "num_samples"),
        (["--epochs", "nan"], "epochs"),
        (["--epochs", "-1"], "epochs"),
        (["--sequence-length", "0"], "sequence_length"),
        (["--max-steps", "-1"], "max_steps"),
        (["--profile", "not-a-real-gpu"], "profile"),
        (["--model", "/does-not-exist/config.json"], "local model"),
    ],
)
def test_cli_bad_inputs_are_usage_errors(training_case, capsys, options, error) -> None:
    config, _, _ = training_case
    with pytest.raises(SystemExit) as caught:
        main(["--config", str(config), "--profile", "a100", *options])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert error in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_distributed_time_is_explicitly_unsupported(training_case, capsys) -> None:
    config, _, training = training_case
    training["world_size"] = 2
    config.write_text(json.dumps(training))
    with pytest.raises(SystemExit) as caught:
        main(["--config", str(config), "--profile", "a100"])
    assert caught.value.code == 2
    assert "training-plan" in capsys.readouterr().err


def test_cli_observation_extrapolates_only_matching_workload(
    training_case, capsys, monkeypatch, tmp_path
) -> None:
    config, model_dir, _ = training_case
    baseline = _report(config, capsys)
    baseline["signature"]["checkpoint_sha256"] = local_checkpoint_identity(model_dir)[
        "sha256"
    ]
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "signature": baseline["signature"],
        "seconds_per_microstep": [0.1, 0.2, 0.3],
        "peak_allocated_bytes": 1024**2,
        "peak_reserved_bytes": 2 * 1024**2,
    }
    path = tmp_path / "fixture-observation.json"
    path.write_text(json.dumps(observation))
    monkeypatch.setattr(
        "fakegpu.finetune_probe.installed_training_versions",
        lambda: {"test_backend": "fixture-only"},
    )
    measured = _report(config, capsys, "--observation", str(path), "--epochs", "2")
    assert measured["timing"]["evidence"] == "supplied_observation_extrapolation"
    assert measured["timing"]["training_loop_seconds"]["expected"] == pytest.approx(20)
    assert measured["memory"]["observed_peak_allocated_bytes"] == 1024**2
    assert (
        measured["timing"]["interval_kind"]
        == "observed_min_max_not_confidence_interval"
    )
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--config",
                str(config),
                "--profile",
                "a100",
                "--observation",
                str(path),
                "--sequence-length",
                "64",
            ]
        )
    assert caught.value.code == 2
    assert "signature mismatch" in capsys.readouterr().err
    # Replacing weights in the same path, even preserving size and mtime,
    # changes content identity and cannot reuse the previous observation.
    weights = model_dir / "model.safetensors"
    old_stat = weights.stat()
    weights.write_bytes(b"weights version2")
    os.utime(weights, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises(SystemExit) as caught:
        main(["--config", str(config), "--profile", "a100", "--observation", str(path)])
    assert caught.value.code == 2
    assert "signature mismatch" in capsys.readouterr().err


def test_estimation_does_not_import_training_frameworks(
    training_case, tmp_path
) -> None:
    config, _, _ = training_case
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    code = (
        "import sys; from fakegpu.__main__ import main; "
        "main(sys.argv[1:]); "
        "assert not {'torch', 'transformers', 'peft', 'bitsandbytes'} & set(sys.modules)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "estimate-finetune",
            "--config",
            str(config),
            "--profile",
            "a100",
            "--json",
            str(tmp_path / "no-imports.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
