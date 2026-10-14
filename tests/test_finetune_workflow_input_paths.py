"""Reject self-recording workspaces before interpreter probing or training."""

import json
from pathlib import Path

import pytest

from fakegpu import finetune_workflow as workflow
from fakegpu import _finetune_workflow_inputs as inputs


def _config(tmp_path, framework):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"local fixture weights")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"input_ids":[1,2]}\n')
    (data / "dataset_info.json").write_text(
        json.dumps({"train": {"file_name": "train.jsonl"}})
    )
    if framework == "huggingface":
        raw = {"model_name_or_path": "model", "train_dataset": "data/train.jsonl"}
    elif framework == "llamafactory":
        raw = {"model_name_or_path": "model", "dataset_dir": "data", "dataset": "train"}
    else:
        raw = {"base_model": "model", "datasets": [{"path": "data"}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path, raw


def _reject_before_runtime(monkeypatch, source, framework, workspace):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input workspace reached interpreter probing or training")

    monkeypatch.setattr(workflow, "_runtime_identity", forbidden)
    monkeypatch.setattr(workflow, "_native_command", forbidden)
    with pytest.raises(
        ValueError, match="workspace must be outside recorded input directories"
    ):
        workflow.calibrate_finetuning(
            source,
            framework=framework,
            profile_id="a100",
            workspace=workspace,
            cuda_visible_devices="1",
            calibration_steps=6,
            max_steps=20,
        )
    assert not (Path(workspace).resolve() / ".workflow.lock").exists()


@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
@pytest.mark.parametrize("placement", ["same", "child", "dotdot", "symlink"])
def test_model_workspace_is_rejected_before_probing(
    tmp_path, monkeypatch, framework, placement
):
    source, _ = _config(tmp_path, framework)
    model = tmp_path / "model"
    if placement == "same":
        workspace = model
    elif placement == "child":
        workspace = model / "experiments" / "calibration"
    elif placement == "dotdot":
        workspace = tmp_path / "uncreated" / ".." / "model" / "calibration"
    else:
        alias = tmp_path / "model-link"
        alias.symlink_to(model, target_is_directory=True)
        workspace = alias / "calibration"
    original_hash = inputs._file

    def no_model_hash(path):
        assert not Path(path).resolve().is_relative_to(model)
        return original_hash(path)

    monkeypatch.setattr(inputs, "_file", no_model_hash)
    _reject_before_runtime(monkeypatch, source, framework, workspace)


@pytest.mark.parametrize("field", ["datasets", "base_model_config", "tokenizer_config"])
def test_other_whole_directory_inputs_reject_descendant_workspace(
    tmp_path, monkeypatch, field
):
    source, raw = _config(tmp_path, "axolotl")
    selected = tmp_path / "data"
    if field != "datasets":
        selected = tmp_path / field
        selected.mkdir()
        (selected / "config.json").write_text("{}")
        raw[field] = str(selected)
        source.write_text(json.dumps(raw))
    _reject_before_runtime(monkeypatch, source, "axolotl", selected / "calibration")


def test_factory_selected_directory_is_recorded_as_a_whole(tmp_path, monkeypatch):
    source, _ = _config(tmp_path, "llamafactory")
    selected = tmp_path / "data" / "selected"
    selected.mkdir()
    (selected / "train.jsonl").write_text('{"text":"training"}\n')
    (tmp_path / "data/dataset_info.json").write_text(
        json.dumps({"train": {"file_name": "selected"}})
    )
    _reject_before_runtime(
        monkeypatch, source, "llamafactory", selected / "calibration"
    )


def test_factory_registry_container_allows_an_unrelated_workspace(tmp_path):
    source, _ = _config(tmp_path, "llamafactory")
    workspace = tmp_path / "data" / "calibration"
    config, identities = inputs._prepare_inputs(
        source, "llamafactory", workspace=workspace
    )
    assert config["dataset_dir"] == str(tmp_path / "data")
    assert str(tmp_path / "data/train.jsonl") in identities
    assert not any(Path(path).is_relative_to(workspace) for path in identities)
    assert not workspace.exists()


def test_separate_workspace_preserves_existing_input_contract(tmp_path):
    source, _ = _config(tmp_path, "huggingface")
    assert inputs._prepare_inputs(
        source, "huggingface", workspace=tmp_path / "calibration"
    ) == inputs._prepare_inputs(source, "huggingface")
