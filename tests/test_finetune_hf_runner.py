from dataclasses import make_dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from packaging.version import Version
import pytest

from fakegpu._finetune_hf_runner import (
    _argument_options,
    _load_model,
    _load_processed_dataset,
    validate_huggingface_config,
)
from fakegpu.finetune_config import normalize_finetune_config


def _config(tmp_path, **changes):
    row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    raw = {
        "model_name_or_path": str(tmp_path / "model"),
        "train_dataset": "train.jsonl",
        "output_dir": str(tmp_path / "output"),
        "sequence_length": 16,
        "attn_implementation": "eager",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "optim": "adamw_torch",
        "max_steps": 5,
    }
    raw.update(changes)
    return raw, normalize_finetune_config(
        raw, framework="huggingface", base_dir=tmp_path
    )


@pytest.fixture
def argument_type(monkeypatch):
    names = (
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "optim",
        "max_steps",
        "seed",
        "data_seed",
        "eval_strategy",
        "eval_steps",
        "do_eval",
        "do_train",
        "do_predict",
        "push_to_hub",
        "auto_find_batch_size",
        "dataloader_num_workers",
        "group_by_length",
        "gradient_checkpointing_kwargs",
    )
    cls = make_dataclass("TrainingArguments", [(name, object) for name in names])
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(TrainingArguments=cls)
    )
    return cls


def test_normalized_training_options_and_official_extras_are_preserved(
    tmp_path, argument_type
):
    raw, normalized = _config(tmp_path, data_seed=123, dataloader_num_workers=2)
    validate_huggingface_config(raw, normalized, tmp_path)
    options = _argument_options(raw, normalized, argument_type)
    assert options["data_seed"] == 123
    assert options["dataloader_num_workers"] == 2
    assert options["max_steps"] == 5
    assert options["per_device_train_batch_size"] == 2
    assert options["gradient_accumulation_steps"] == 2
    assert options["optim"] == "adamw_torch"
    assert options["report_to"] == []
    assert "train_dataset" not in options


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"training_args": {}}, "flat configuration"),
        ({"model_args": {}}, "flat configuration"),
        ({"data_args": {}}, "flat configuration"),
        ({"mystery_plugin": "x"}, "unsupported.*mystery_plugin"),
        ({"mixed_precision": "bf16"}, "unsupported.*mixed_precision"),
        ({"auto_find_batch_size": True}, "does not support auto_find_batch_size"),
        ({"push_to_hub": True}, "does not support push_to_hub"),
        ({"do_predict": True}, "does not support do_predict"),
        ({"do_train": False}, "requires training"),
        ({"disable_shuffling": "true"}, "must be boolean"),
        ({"pad_to_multiple_of": 0}, "positive integer"),
        ({"eval_strategy": "steps"}, "requires eval_dataset"),
        ({"do_eval": True}, "requires an eval_strategy"),
        ({"group_by_length": True, "disable_shuffling": True}, "conflicts"),
    ],
)
def test_unsupported_configuration_is_not_silently_ignored(
    tmp_path, argument_type, changes, match
):
    raw, normalized = _config(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        validate_huggingface_config(raw, normalized, tmp_path)


def test_unknown_peft_option_rejected_instead_of_using_a_different_adapter(
    tmp_path, argument_type
):
    raw, normalized = _config(
        tmp_path, peft={"target_modules": ["q_proj"], "init_lora_weights": "gaussian"}
    )
    with pytest.raises(ValueError, match="PEFT fields.*init_lora_weights"):
        validate_huggingface_config(raw, normalized, tmp_path)


def test_local_processed_rows_preserve_all_tokens_and_labels(tmp_path):
    raw, _ = _config(tmp_path)
    assert _load_processed_dataset(tmp_path / raw["train_dataset"], 16) == [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
    ]


@pytest.mark.parametrize(
    "row,match",
    [
        ({"text": "unprocessed"}, "requires exactly"),
        ({"input_ids": [1, 2], "labels": [1, 2]}, "requires exactly"),
        ({"input_ids": [1, 2], "attention_mask": [1, 2], "labels": [1, 2]}, "binary"),
        (
            {"input_ids": [1, 2], "attention_mask": [1], "labels": [1, 2]},
            "match input_ids",
        ),
        (
            {"input_ids": [1, True], "attention_mask": [1, 1], "labels": [1, 2]},
            "integer IDs",
        ),
        (
            {"input_ids": [1] * 17, "attention_mask": [1] * 17, "labels": [1] * 17},
            "does not truncate",
        ),
    ],
)
def test_processed_data_does_not_silently_retokenize_pack_or_truncate(
    tmp_path, row, match
):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=match):
        _load_processed_dataset(path, 16)


def test_quantized_loader_and_all_linear_peft_options_match_requested_execution(
    tmp_path, monkeypatch
):
    _, normalized = _config(
        tmp_path,
        method="qlora",
        lora_target_modules="all-linear",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bf16",
        bnb_4bit_quant_storage="uint8",
        gradient_checkpointing=True,
    )
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    loader = Mock(return_value=model)
    quantization = Mock(side_effect=lambda **kwargs: kwargs)
    lora = Mock(side_effect=lambda **kwargs: kwargs)
    prepare = Mock(side_effect=lambda value, **kwargs: value)
    wrap = Mock(side_effect=lambda value, adapter: value)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=lora,
            get_peft_model=wrap,
            prepare_model_for_kbit_training=prepare,
        ),
    )
    result = _load_model(
        normalized,
        SimpleNamespace(float32="fp32", float16="fp16", bfloat16="bf16", uint8="uint8"),
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            BitsAndBytesConfig=quantization,
        ),
    )
    assert result is model and model.config.use_cache is False
    options = loader.call_args.kwargs
    assert options["device_map"] == {"": 0}
    assert options["local_files_only"] is True and options["trust_remote_code"] is False
    assert options["quantization_config"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
        "bnb_4bit_quant_storage": "uint8",
    }
    assert lora.call_args.kwargs["target_modules"] == "all-linear"
    assert prepare.call_args.kwargs["use_gradient_checkpointing"] is True
    assert prepare.call_args.kwargs["gradient_checkpointing_kwargs"] == {
        "use_reentrant": False
    }


def test_huggingface_native_adapter_executes_real_cpu_trainer(tmp_path):
    for dependency in ("torch", "transformers", "accelerate", "peft", "tokenizers"):
        try:
            version = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            pytest.skip(f"optional CPU Trainer validation requires {dependency}")
        if dependency == "transformers" and Version(version) < Version("4.57"):
            pytest.skip("CPU Trainer validation requires Transformers >= 4.57")
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    env["PYTHONPATH"] = str(root)
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/support/finetune_native_hf_counts.py"),
            str(tmp_path),
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads((tmp_path / "native-hf-counts.json").read_text())
    assert set(reports) == {"full", "lora"}
    for report in reports.values():
        assert report["updates"] == 5
        assert report["microsteps"] == 8
        assert report["samples"] == 14
        assert report["ordered_lengths"] == [2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7]
        assert report["weight_change"] is True
        assert report["saved"] is True
