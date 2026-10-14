"""Workflow caches cannot silently replace the recorded raw dataset sources."""

import json
from pathlib import Path

import pytest

from fakegpu._finetune_workflow_inputs import _prepare_inputs


@pytest.mark.parametrize(
    "framework,field",
    [("axolotl", "dataset_prepared_path"), ("llamafactory", "tokenized_path")],
)
def test_workspaces_isolate_preprocessed_data_from_unverified_external_cache(
    tmp_path, framework, field
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text":"old"}\n')
    cache = tmp_path / "external-cache"
    cache.mkdir()
    sentinel = cache / "stale.arrow"
    sentinel.write_bytes(b"old dataset")
    if framework == "axolotl":
        raw = {
            "base_model": str(model),
            "datasets": [{"path": str(data / "train.jsonl"), "type": "alpaca"}],
        }
    else:
        (data / "dataset_info.json").write_text(
            json.dumps({"train": {"file_name": "train.jsonl"}})
        )
        raw = {
            "model_name_or_path": str(model),
            "dataset_dir": str(data),
            "dataset": "train",
        }
    raw[field] = str(cache)
    source = tmp_path / "config.json"
    source.write_text(json.dumps(raw))
    first, inputs = _prepare_inputs(source, framework, workspace=tmp_path / "first")
    assert Path(first[field]).is_relative_to(tmp_path / "first")
    (data / "train.jsonl").write_text('{"text":"new"}\n')
    second, changed = _prepare_inputs(source, framework, workspace=tmp_path / "second")
    assert first[field] != second[field]
    assert inputs[str(data / "train.jsonl")] != changed[str(data / "train.jsonl")]
    assert sentinel.read_bytes() == b"old dataset"
    assert json.loads(source.read_text()) == raw


def test_default_axolotl_preprocessing_cache_is_also_owned_by_workspace(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"base_model": str(model), "datasets": [{"path": str(data)}]})
    )
    resolved, _ = _prepare_inputs(config, "axolotl", workspace=tmp_path / "workspace")
    assert Path(resolved["dataset_prepared_path"]).is_relative_to(
        tmp_path / "workspace"
    )
