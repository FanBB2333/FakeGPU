from __future__ import annotations

import copy
import json

import pytest

from fakegpu.finetune_config import (
    FinetuneConfigError,
    load_finetune_config,
    normalize_finetune_config,
)
from fakegpu.training_plan import estimate_training_plan


def hf_config(**changes):
    return {"model_name_or_path": "Qwen/Qwen3.5-0.8B", **changes}


def test_huggingface_nested_trainer_peft_and_dataset() -> None:
    config = {
        "model_args": {
            "model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "torch_dtype": "float32",
        },
        "training_args": {
            "bf16": True,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 2.5,
            "max_steps": -1,
            "gradient_checkpointing": True,
        },
        "peft": {"peft_type": "LORA", "r": 16, "target_modules": ["q_proj", "v_proj"]},
        "dataset": {"num_samples": 512, "sequence_length": 256},
        "software_versions": {"transformers": "5.3.0"},
    }
    before = copy.deepcopy(config)
    result = normalize_finetune_config(config)
    assert result["source_framework"] == "huggingface"
    assert result["method"] == "lora"
    assert result["dtype"] == "bf16"
    assert result["parameter_dtype"] == "fp32"
    assert result["micro_batch_size"] == 2
    assert result["gradient_accumulation_steps"] == 4
    assert result["epochs"] == 2.5
    assert result["max_steps"] is None
    assert result["lora_rank"] == 16
    assert result["lora_target_modules"] == ["q_proj", "v_proj"]
    assert result["sequence_length"] == 256
    assert result["num_samples"] == 512
    assert result["software_versions"] == {"transformers": "5.3.0"}
    assert result["distributed_config"]["activation_checkpointing"] is True
    assert config == before


def test_huggingface_flat_defaults_do_not_invent_dataset_size() -> None:
    result = normalize_finetune_config(
        hf_config(bf16=False, fp16=False, max_train_samples=100)
    )
    assert result["dtype"] == "fp32"
    assert result["micro_batch_size"] == 8
    assert result["epochs"] == 3
    assert result["num_samples"] is None
    assert result["sequence_length"] is None
    assert result["max_samples"] == 100
    assert result["method"] == "full"
    assert result["lora_rank"] is None


def test_llamafactory_qlora_preserves_cap_and_defaults() -> None:
    result = normalize_finetune_config(
        hf_config(
            stage="sft",
            finetuning_type="lora",
            bf16=True,
            quantization_bit=4,
            lora_rank=8,
            lora_target="q_proj, v_proj",
            cutoff_len=1024,
            dataset="alpaca_en, identity",
            max_samples=1000,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )
    )
    assert result["source_framework"] == "llamafactory"
    assert result["method"] == "qlora"
    assert result["quantization_bits"] == 4
    assert result["lora_target_modules"] == ["q_proj", "v_proj"]
    assert result["dataset_sources"] == ["alpaca_en", "identity"]
    assert result["num_samples"] is None
    assert result["gradient_checkpointing"] is True
    assert result["packing"] is False


def test_llamafactory_pretraining_and_default_all_targets() -> None:
    result = normalize_finetune_config(hf_config(stage="pt", finetuning_type="lora"))
    assert result["packing"] is True
    assert result["sequence_length"] == 2048
    assert result["lora_target_modules"] == ["all-linear"]


def test_llamafactory_checkpoint_flag_matches_framework_behavior() -> None:
    config = hf_config(finetuning_type="full", gradient_checkpointing=False)
    with pytest.raises(
        FinetuneConfigError, match="disable_gradient_checkpointing=true"
    ):
        normalize_finetune_config(config)
    config["disable_gradient_checkpointing"] = True
    assert normalize_finetune_config(config)["gradient_checkpointing"] is False


def test_amp_precision_does_not_imply_low_precision_weight_storage() -> None:
    config = normalize_finetune_config(hf_config(bf16=True))
    assert config["dtype"] == "bf16"
    assert config["parameter_dtype"] == "fp32"


def test_optimizer_and_adapter_hyperparameters_preserve_source_defaults() -> None:
    hf = normalize_finetune_config(
        hf_config(peft={"r": 16, "target_modules": ["q_proj"]})
    )
    factory = normalize_finetune_config(hf_config(finetuning_type="lora", lora_rank=16))
    axolotl = normalize_finetune_config(
        {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "bf16": True,
            "adapter": "lora",
            "lora_r": 16,
            "lora_target_linear": True,
        }
    )
    assert hf["learning_rate"] == factory["learning_rate"] == 5e-5
    assert axolotl["learning_rate"] is None
    assert hf["lora_alpha"] == 8
    assert factory["lora_alpha"] == 32
    assert axolotl["lora_alpha"] is None
    assert all(config["lora_dropout"] == 0 for config in (hf, factory, axolotl))
    assert all(config["lora_bias"] == "none" for config in (hf, factory, axolotl))


def test_explicit_peft_hyperparameters_and_scientific_learning_rate() -> None:
    result = normalize_finetune_config(
        hf_config(
            learning_rate="2e-5",
            tf32=True,
            peft={
                "r": 16,
                "target_modules": ["q_proj"],
                "lora_alpha": 24,
                "lora_dropout": 0.15,
            },
        )
    )
    assert result["learning_rate"] == 2e-5
    assert result["lora_alpha"] == 24
    assert result["lora_dropout"] == 0.15
    assert result["tf32"] is True


def test_scheduler_defaults_and_explicit_optimizer_options() -> None:
    hf = normalize_finetune_config(hf_config())
    factory = normalize_finetune_config(hf_config(finetuning_type="full"))
    axolotl = normalize_finetune_config(
        {"base_model": "Qwen/Qwen3.5-0.8B", "bf16": True}
    )
    for config in (hf, factory):
        assert config["lr_scheduler_type"] == "linear"
        assert config["warmup_steps"] == 0
        assert config["warmup_ratio"] == 0
        assert config["weight_decay"] == 0
        assert config["max_grad_norm"] == 1
    assert axolotl["lr_scheduler_type"] == "cosine"
    assert axolotl["warmup_steps"] is None
    assert axolotl["warmup_ratio"] is None
    assert axolotl["max_grad_norm"] is None
    explicit = normalize_finetune_config(
        hf_config(
            lr_scheduler_type="cosine",
            warmup_steps=12,
            warmup_ratio=0.1,
            weight_decay=0.05,
            max_grad_norm=0,
        )
    )
    assert explicit["warmup_steps"] == 12
    assert explicit["warmup_ratio"] == 0.1
    assert explicit["weight_decay"] == 0.05
    assert explicit["max_grad_norm"] == 0
    assert any("takes precedence" in item for item in explicit["assumptions"])


def test_axolotl_warmup_mutual_exclusion_and_scheduler_alias() -> None:
    config = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "bf16": True,
        "lr_scheduler": "constant_with_warmup",
        "warmup_steps": 10,
    }
    result = normalize_finetune_config(config)
    assert result["lr_scheduler_type"] == "constant_with_warmup"
    assert result["warmup_steps"] == 10
    with pytest.raises(FinetuneConfigError, match="mutually exclusive"):
        normalize_finetune_config({**config, "warmup_ratio": 0.1})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"lr_scheduler_type": "custom"}, "lr_scheduler_type"),
        ({"lr_scheduler_type": "linear", "lr_scheduler": "cosine"}, "conflicting"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"warmup_steps": 1.5}, "warmup_steps"),
        ({"warmup_steps": True}, "warmup_steps"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.1}, "warmup_ratio"),
        ({"warmup_ratio": True}, "warmup_ratio"),
        ({"warmup_ratio": float("nan")}, "finite"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"max_grad_norm": -1}, "max_grad_norm"),
        ({"max_grad_norm": float("inf")}, "finite"),
        ({"lr_scheduler_kwargs": {"num_cycles": 3}}, "not supported"),
    ],
)
def test_invalid_scheduler_optimizer_settings_are_rejected(changes, message) -> None:
    with pytest.raises(FinetuneConfigError, match=message):
        normalize_finetune_config(hf_config(**changes))


@pytest.mark.parametrize(
    "options",
    [
        {"optim_args": "amsgrad=True"},
        {"optim_args": {"foreach": True}},
        {"optimizer_kwargs": {"momentum": 0.9}},
        {"optimizer_kwargs": {"amsgrad": False}},
    ],
)
@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
def test_custom_optimizer_options_are_not_silently_ignored(options, framework) -> None:
    config = (
        {"base_model": "Qwen/Qwen3.5-0.8B", "bf16": True}
        if framework == "axolotl"
        else hf_config()
    )
    if framework == "huggingface":
        config["training_args"] = options
    else:
        config.update(options)
    with pytest.raises(FinetuneConfigError, match="is not supported"):
        normalize_finetune_config(config, framework=framework)


@pytest.mark.parametrize("value", [None, {}, ""])
def test_empty_custom_optimizer_options_do_not_change_configuration(value) -> None:
    baseline = normalize_finetune_config(hf_config())
    result = normalize_finetune_config(
        hf_config(optim_args=value, optimizer_kwargs=value)
    )
    assert result == baseline


def test_huggingface_bitsandbytes_defaults_differ_from_framework_qlora_defaults() -> (
    None
):
    hf = normalize_finetune_config(
        hf_config(
            bf16=True,
            peft={"target_modules": ["q_proj"]},
            load_in_4bit=True,
        )
    )
    factory = normalize_finetune_config(
        hf_config(finetuning_type="lora", quantization_bit=4, bf16=True)
    )
    axolotl = normalize_finetune_config(
        {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "adapter": "qlora",
            "load_in_4bit": True,
            "lora_target_linear": True,
            "bf16": True,
        }
    )
    assert (
        hf["quantization_type"],
        hf["double_quantization"],
        hf["quantization_compute_dtype"],
        hf["quantization_storage"],
    ) == ("fp4", False, "fp32", "uint8")
    assert (
        factory["quantization_type"],
        factory["double_quantization"],
        factory["quantization_compute_dtype"],
        factory["quantization_storage"],
    ) == ("nf4", True, "bf16", "bf16")
    assert axolotl["quantization_storage"] == "bf16"


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_factory_quantized_storage_defaults_to_compute_dtype(dtype) -> None:
    options = {dtype: True} if dtype != "fp32" else {}
    result = normalize_finetune_config(
        hf_config(finetuning_type="lora", quantization_bit=4, **options)
    )
    assert result["quantization_compute_dtype"] == dtype
    assert result["quantization_storage"] == dtype


@pytest.mark.parametrize("method", [None, "bnb", "bitsandbytes", "hqq", {}, []])
def test_factory_quantization_method_matches_native_loader(method) -> None:
    config = hf_config(finetuning_type="lora", quantization_bit=4)
    if method is not None:
        config["quantization_method"] = method
    if method not in (None, "bnb"):
        with pytest.raises(
            FinetuneConfigError, match="requires quantization_method=bnb"
        ):
            normalize_finetune_config(config)
    else:
        assert normalize_finetune_config(config)["method"] == "qlora"


def test_explicit_quantization_settings_preserved_and_owned() -> None:
    quant = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "float16",
    }
    result = normalize_finetune_config(
        hf_config(peft={"target_modules": ["q_proj"]}, quantization_config=quant)
    )
    quant["bnb_4bit_quant_type"] = "fp4"
    assert result["quantization_type"] == "nf4"
    assert result["double_quantization"] is True
    assert result["quantization_compute_dtype"] == "bf16"
    assert result["quantization_storage"] == "fp16"


def test_quantization_and_lora_alias_conflicts_are_not_silently_overwritten() -> None:
    with pytest.raises(FinetuneConfigError, match="conflicts with quantization_bits"):
        normalize_finetune_config(
            hf_config(
                finetuning_type="lora",
                quantization_bit=4,
                load_in_4bit=False,
            )
        )
    with pytest.raises(FinetuneConfigError, match="conflicting values"):
        normalize_finetune_config(
            hf_config(
                lora_alpha=16,
                peft={"target_modules": ["q_proj"], "lora_alpha": 8},
            )
        )
    with pytest.raises(FinetuneConfigError, match="conflicting quantization_config"):
        normalize_finetune_config(
            hf_config(
                peft={"target_modules": ["q_proj"]},
                load_in_4bit=True,
                quantization_config={"bnb_4bit_quant_type": "nf4"},
                bnb_config_kwargs={"bnb_4bit_quant_type": "fp4"},
            )
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": "nan"}, "finite"),
        ({"learning_rate": True}, "learning_rate"),
        ({"lora_alpha": -1}, "lora_alpha"),
        ({"lora_dropout": 1.0}, "lora_dropout"),
        ({"lora_dropout": -0.1}, "lora_dropout"),
        ({"lora_dropout": True}, "lora_dropout"),
        ({"lora_bias": "all"}, "lora_bias"),
        ({"use_rslora": True}, "not supported"),
        ({"peft_use_dora": True}, "not supported"),
        ({"lora_modules_to_save": ["lm_head"]}, "not supported"),
        ({"quantization_config": {"bnb_4bit_quant_type": "int4"}}, "nf4 or fp4"),
        (
            {"quantization_config": {"bnb_4bit_use_double_quant": "false"}},
            "true or false",
        ),
        ({"quantization_config": {"bnb_4bit_quant_storage": "int8"}}, "dtype"),
        (
            {"quantization_config": {"llm_int8_enable_fp32_cpu_offload": True}},
            "not supported",
        ),
        (
            {"quantization_config": {"unknown_kernel_option": True}},
            "unsupported quantization settings",
        ),
        ({"tf32": "auto"}, "true or false"),
    ],
)
def test_unsupported_or_invalid_execution_hyperparameters_are_not_ignored(
    changes, message
) -> None:
    config = hf_config(
        peft={"target_modules": ["q_proj"]}, load_in_4bit=True, **changes
    )
    with pytest.raises(FinetuneConfigError, match=message):
        normalize_finetune_config(config)


def test_axolotl_config_and_packing_semantics(tmp_path) -> None:
    config = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "adapter": "qlora",
        "load_in_4bit": True,
        "bf16": True,
        "lora_r": 16,
        "lora_target_linear": True,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1.5,
        "sequence_len": 512,
        "sample_packing": True,
        "max_steps": 12,
        "datasets": [{"path": "./data/tiny.jsonl", "type": "alpaca"}],
        "flash_attention": True,
    }
    result = normalize_finetune_config(config, base_dir=tmp_path)
    assert result["source_framework"] == "axolotl"
    assert result["lora_target_modules"] == ["all-linear"]
    assert result["max_steps"] == 12
    assert result["attention_implementation"] == "flash_attention_2"
    assert result["dataset_sources"] == [str(tmp_path / "data/tiny.jsonl")]
    assert any("packed" in assumption for assumption in result["assumptions"])


def test_paths_and_deepspeed_auto_resolve_against_configuration(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    ds = {
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": "auto"},
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}},
    }
    (config_dir / "ds.json").write_text(json.dumps(ds))
    config = hf_config(
        model_name_or_path="../models/qwen",
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        world_size=2,
        deepspeed="ds.json",
        train_file="../data/train.jsonl",
    )
    (config_dir / "train.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path.parent)
    result = load_finetune_config(config_dir / "train.json")
    assert result["model_name_or_path"] == str(tmp_path / "models/qwen")
    assert result["dataset_sources"] == [str(tmp_path / "data/train.jsonl")]
    dist = result["distributed_config"]
    assert dist["framework"] == "deepspeed"
    assert dist["zero_stage"] == 2
    assert dist["world_size"] == 2
    assert dist["mixed_precision"] == "bf16"
    assert dist["optimizer_offload_device"] == "cpu"
    report = estimate_training_plan(dist, parameter_bytes=1024, activation_bytes=128)
    assert report["inputs"]["world_size"] == 2


def test_fsdp_strategy_and_axolotl_fsdp2() -> None:
    result = normalize_finetune_config(
        hf_config(
            fsdp="full_shard auto_wrap offload",
            fsdp_config={"use_orig_params": True},
            world_size=4,
        )
    )
    assert result["distributed_config"]["sharding_strategy"] == "full_shard"
    assert result["distributed_config"]["parameter_offload_device"] == "cpu"


def test_fsdp_activation_checkpointing_is_preserved() -> None:
    result = normalize_finetune_config(
        hf_config(
            fsdp="full_shard",
            fsdp_config={"activation_checkpointing": True},
        )
    )
    assert result["gradient_checkpointing"] is True
    assert result["distributed_config"]["activation_checkpointing"] is True


def test_published_example_configurations_agree_on_training_shape() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1] / "examples" / "finetuning"
    expected = None
    for name in ("huggingface-lora", "llamafactory-lora", "axolotl-lora"):
        result = load_finetune_config(root / f"{name}.yaml")
        shape = {
            key: result[key]
            for key in (
                "method",
                "dtype",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "sequence_length",
                "epochs",
                "lora_rank",
                "lora_target_modules",
            )
        }
        if expected is None:
            expected = shape
        assert shape == expected
    result = normalize_finetune_config(
        {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "bf16": True,
            "fsdp_config": {"reshard_after_forward": False, "offload_params": True},
        }
    )
    assert result["distributed_config"]["framework"] == "fsdp2"
    assert result["distributed_config"]["sharding_strategy"] == "shard_grad_op"
    assert result["distributed_config"]["parameter_offload_device"] == "cpu"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"bf16": True, "fp16": True}, "precision"),
        ({"bf16": "false"}, "true or false"),
        ({"num_train_epochs": float("nan")}, "finite"),
        ({"num_train_epochs": float("inf")}, "finite"),
        ({"num_train_epochs": -1}, "positive"),
        ({"per_device_train_batch_size": True}, "integer"),
        ({"gradient_accumulation_steps": 1.5}, "integer"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": -2}, "integer"),
        ({"num_samples": 0}, "integer"),
        ({"world_size": 0}, "integer"),
        ({"packing": 0}, "true or false"),
        ({"optim": "paged_adamw_8bit"}, "unsupported optimizer"),
        ({"stage": "dpo"}, "training stages"),
        ({"finetuning_type": "freeze"}, "fine-tuning method"),
        ({"load_in_4bit": True}, "full fine-tuning"),
        ({"load_in_4bit": True, "load_in_8bit": True}, "quantization"),
        ({"gradient_checkpointing": "offload"}, "true or false"),
        (
            {"gradient_checkpointing": True, "disable_gradient_checkpointing": True},
            "conflict",
        ),
        ({"sequence_length": 128, "max_seq_length": 256}, "conflicting"),
        (
            {"training_args": {"num_train_epochs": 2}, "num_train_epochs": 3},
            "conflicting",
        ),
        (
            {"deepspeed": {"zero_optimization": {"stage": 2}}, "fsdp": "full_shard"},
            "cannot both",
        ),
        ({"fsdp": "made_up"}, "unsupported FSDP"),
        ({"fsdp": "full_shard shard_grad_op"}, "multiple FSDP"),
        ({"fsdp_config": {"world_size": 4}}, "world_size conflicts"),
        (
            {"fsdp": "full_shard offload", "fsdp_config": {"offload_params": False}},
            "offload settings conflict",
        ),
        ({"fsdp_config": {"mixed_precision_policy": "bf16"}}, "precision conflicts"),
        ({"attention_implementation": {}}, "must be a string"),
        ({"deepspeed": {"zero_optimization": {"stage": 1.5}}}, "integer"),
        ({"deepspeed": {"zero_optimization": {"stage": 4}}}, "stage"),
        (
            {"deepspeed": {"zero_optimization": {"offload_param": {"device": "cuda"}}}},
            "offload_param",
        ),
        ({"deepspeed": {"train_batch_size": 500}}, "conflicts"),
        ({"deepspeed": {"bf16": {"enabled": True}}}, "precision"),
        ({"batch_size": 2}, "batch_size conflicts"),
        ({"use_dora": True}, "not supported"),
        ({"context_parallel_size": 2}, "not supported"),
    ],
)
def test_invalid_configuration_rejected(changes, message) -> None:
    with pytest.raises(FinetuneConfigError, match=message):
        normalize_finetune_config(hf_config(**changes))


@pytest.mark.parametrize(
    "peft, message",
    [
        ({"target_modules": "q_proj|v_proj"}, "regular expressions"),
        ({"target_modules": []}, "non-empty"),
        ({"r": 0, "target_modules": ["q_proj"]}, "integer"),
        ({"target_modules": ["q_proj", "q_proj"]}, "unique"),
        ({"peft_type": "IA3", "target_modules": ["q_proj"]}, "only LoRA"),
        ({"r": 8}, "model-dependent defaults"),
        (
            {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]},
            "not supported",
        ),
    ],
)
def test_peft_configuration_boundaries(peft, message) -> None:
    with pytest.raises(FinetuneConfigError, match=message):
        normalize_finetune_config(hf_config(peft=peft))


@pytest.mark.parametrize("key", ["peft", "peft_config"])
def test_empty_peft_configuration_does_not_silently_select_full_tuning(key) -> None:
    with pytest.raises(FinetuneConfigError, match="model-dependent defaults"):
        normalize_finetune_config(hf_config(**{key: {}}))
    with pytest.raises(FinetuneConfigError, match="full fine-tuning cannot use PEFT"):
        normalize_finetune_config(hf_config(method="full", **{key: {}}))


@pytest.mark.parametrize(
    "changes",
    [
        {"gradient_checkpointing_kwargs": {"use_reentrant": True}},
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        {"gradient_checkpointing_kwargs": {}},
        {"use_reentrant_gc": True},
        {"use_reentrant_gc": False},
    ],
)
def test_custom_checkpoint_implementations_are_explicitly_unsupported(changes) -> None:
    with pytest.raises(FinetuneConfigError, match="custom checkpoint implementations"):
        normalize_finetune_config(hf_config(gradient_checkpointing=True, **changes))


@pytest.mark.parametrize(
    "device_map", ["auto", {"": "cpu"}, {"model.layers.0": "disk"}, {"": 0}]
)
def test_model_device_placement_is_not_silently_ignored(device_map) -> None:
    with pytest.raises(FinetuneConfigError, match="device_map is not supported"):
        normalize_finetune_config(hf_config(device_map=device_map))


def test_explicit_checkpoint_identity_is_preserved_and_validated() -> None:
    digest = "ABCDEF0123456789" * 4
    config = normalize_finetune_config(
        hf_config(revision="v2", checkpoint_sha256=digest)
    )
    assert config["model_revision"] == "v2"
    assert config["checkpoint_sha256"] == digest.lower()
    with pytest.raises(FinetuneConfigError, match="conflicting values"):
        normalize_finetune_config(hf_config(model_revision="v1", revision="v2"))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model_revision": ""}, "model_revision"),
        ({"model_revision": 123}, "model_revision"),
        ({"checkpoint_sha256": "a" * 63}, "64 hexadecimal"),
        ({"checkpoint_sha256": "g" * 64}, "64 hexadecimal"),
        ({"checkpoint_sha256": 123}, "64 hexadecimal"),
    ],
)
def test_invalid_checkpoint_identity_is_rejected(changes, message) -> None:
    with pytest.raises(FinetuneConfigError, match=message):
        normalize_finetune_config(hf_config(**changes))


def test_framework_ambiguity_and_explicit_override() -> None:
    mixed = hf_config(base_model="Qwen/Qwen3.5-0.8B", finetuning_type="full", bf16=True)
    with pytest.raises(FinetuneConfigError, match="unambiguously"):
        normalize_finetune_config(mixed)
    assert (
        normalize_finetune_config(mixed, framework="axolotl")["source_framework"]
        == "axolotl"
    )
    with pytest.raises(FinetuneConfigError, match="framework argument conflicts"):
        normalize_finetune_config(hf_config(framework="hf"), framework="axolotl")


def test_axolotl_automatic_precision_requires_explicit_choice() -> None:
    for changes in ({}, {"bf16": "auto"}):
        with pytest.raises(FinetuneConfigError, match="[Aa]uto"):
            normalize_finetune_config({"base_model": "Qwen/Qwen3.5-0.8B", **changes})


def test_qlora_quantization_config_and_all_linear() -> None:
    result = normalize_finetune_config(
        hf_config(
            peft_config={"target_modules": "all-linear"},
            quantization_config={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
            torch_dtype="bfloat16",
        )
    )
    assert result["method"] == "qlora"
    assert result["quantization_bits"] == 4
    assert result["lora_target_modules"] == ["all-linear"]


def test_yaml_load_preserves_framework_fields(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "train.yaml"
    path.write_text(
        "model_name_or_path: Qwen/Qwen3.5-0.8B\nfinetuning_type: lora\nlora_target: all\ncutoff_len: 256\nbf16: true\n"
    )
    assert load_finetune_config(path)["source_framework"] == "llamafactory"
