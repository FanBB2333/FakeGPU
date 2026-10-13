from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fakegpu import finetune_cli, finetune_probe
from fakegpu._local_checkpoint import bind_local_checkpoint, local_checkpoint_identity


def _training() -> dict:
    return {
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "source_framework": "huggingface",
        "world_size": 1,
        "distributed_config": {"sharding_strategy": "replicated", "zero_stage": 0},
        "method": "full",
        "dtype": "bf16",
        "optimizer": "adamw_torch",
        "attention_implementation": "sdpa",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "sequence_length": 4,
        "gradient_checkpointing": False,
        "packing": False,
        "dataloader_drop_last": False,
        "learning_rate": 2e-5,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_bias": "none",
        "quantization_type": "nf4",
        "double_quantization": True,
        "quantization_compute_dtype": "bf16",
        "quantization_storage": "uint8",
        "tf32": False,
    }


def _model_config() -> dict:
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 32,
        "vocab_size": 64,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"source_framework": "llamafactory"}, "huggingface Trainer"),
        ({"source_framework": "axolotl"}, "huggingface Trainer"),
        ({"world_size": 2}, "single-GPU"),
        ({"distributed_config": {"sharding_strategy": "full_shard"}}, "single-GPU"),
        (
            {
                "distributed_config": {
                    "sharding_strategy": "replicated",
                    "zero_stage": 2,
                }
            },
            "single-GPU",
        ),
        ({"distributed_config": {"parameter_offload_device": "cpu"}}, "single-GPU"),
        ({"packing": True}, "packing=false"),
        ({"optimizer": "sgd"}, "adamw_torch"),
        ({"attention_implementation": "auto"}, "attn_implementation"),
        ({"sequence_length": 1}, "sequence_length"),
        ({"sequence_length": True}, "sequence_length"),
    ],
)
def test_probe_rejects_unmeasured_execution_contract_before_optional_imports(
    tmp_path: Path,
    monkeypatch,
    changes: dict,
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    training = _training() | changes
    with pytest.raises(ValueError, match=message):
        finetune_probe.measure_finetuning(
            _model_config(),
            training,
            model_dir=tmp_path,
            profile_id="a100",
        )


@pytest.mark.parametrize(
    "key, value",
    [
        ("warmup_steps", 0),
        ("warmup_steps", True),
        ("measure_steps", 1),
        ("measure_steps", 2.5),
    ],
)
def test_probe_requires_warmup_and_multiple_measurements(
    tmp_path: Path,
    key: str,
    value,
) -> None:
    with pytest.raises(ValueError, match=key):
        finetune_probe.measure_finetuning(
            _model_config(),
            _training(),
            model_dir=tmp_path,
            profile_id="a100",
            **{key: value},
        )


def test_probe_refuses_implicit_vision_exclusion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    multimodal = {"text_config": _model_config(), "vision_config": {"depth": 2}}
    with pytest.raises(ValueError, match="text-only"):
        finetune_probe.measure_finetuning(
            multimodal,
            _training(),
            model_dir=tmp_path,
            profile_id="a100",
        )


@pytest.fixture
def measured_runtime(monkeypatch, tmp_path):
    """Drive the observation protocol with a deterministic CPU Trainer double.

    This does not claim CUDA/model correctness. The double follows the actual
    Trainer order: prefetch an accumulation group, start callback, update, end.
    """
    import torch

    (tmp_path / "model.safetensors").write_bytes(b"fixture checkpoint bytes")

    for name in ("FAKEGPU_MODE", "FAKEGPU_RUNTIME", "LD_PRELOAD"):
        monkeypatch.delenv(name, raising=False)
    clock = SimpleNamespace(value=0.0)
    state = SimpleNamespace(
        trainer=None,
        reset_count=0,
        training_loss=0.5,
        property_name="Test Physical GPU",
        capability=(8, 0),
        count=1,
    )
    monkeypatch.setattr(finetune_probe.time, "perf_counter", lambda: clock.value)
    monkeypatch.setattr(
        finetune_probe,
        "installed_training_versions",
        lambda: {
            "torch": "test",
            "transformers": "test",
            "peft": "not-installed",
            "accelerate": "test",
            "bitsandbytes": "not-installed",
        },
    )
    monkeypatch.setattr(
        finetune_probe,
        "get_profile",
        lambda _: SimpleNamespace(
            name="Test Physical GPU",
            torch_name="Test Physical GPU",
            official_model="Test Physical GPU",
            compute_capability=(8, 0),
        ),
    )
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    for name in ("manual_seed_all", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: state.count)
    monkeypatch.setattr(torch.cuda, "_fakegpu_simulated", False, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name=state.property_name,
            total_memory=80 * 2**30,
            major=state.capability[0],
            minor=state.capability[1],
        ),
    )

    def reset_peak(*args, **kwargs):
        state.reset_count += 1

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 500)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 700)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.config = SimpleNamespace(use_cache=True)

        def to(self, *args, **kwargs):
            return self

    def load_model(*args, **kwargs):
        state.model_options = kwargs
        clock.value += 3.0
        return Model()

    def load_tokenizer(*args, **kwargs):
        clock.value += 2.0
        return SimpleNamespace(encode=lambda *a, **k: [1, 2, 3])

    class Trainer:
        def __init__(self, *, model, args, train_dataset, callbacks):
            self.model, self.args = model, args
            self.train_dataset, self.callbacks = train_dataset, callbacks
            self.optimizer = SimpleNamespace(
                state={
                    p: {"step": torch.tensor(3.0), "exp_avg": torch.ones_like(p)}
                    for p in model.parameters()
                }
            )
            state.trainer = self

        def get_batch_samples(self, *args, **kwargs):
            clock.value += 2.0
            return [1, 2], 8

        def train(self):
            train_state = SimpleNamespace(global_step=0)
            for _ in range(self.args.max_steps):
                self.get_batch_samples(
                    None, self.args.gradient_accumulation_steps, None
                )
                for callback in self.callbacks:
                    callback.on_step_begin(self.args, train_state, None)
                clock.value += 10.0
                with torch.no_grad():
                    self.model.weight.add_(0.1)
                train_state.global_step += 1
                for callback in self.callbacks:
                    callback.on_step_end(self.args, train_state, None)
            return SimpleNamespace(training_loss=state.training_loss)

    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(
        for_model=lambda kind, **kwargs: SimpleNamespace(model_type=kind, **kwargs)
    )
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.Trainer = Trainer
    transformers.TrainerCallback = object
    transformers.TrainingArguments = SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


@pytest.mark.parametrize(
    "targets,expected",
    [
        (["all-linear"], "all-linear"),
        (["q_proj", "v_proj"], ["q_proj", "v_proj"]),
    ],
)
def test_probe_restores_peft_all_linear_shorthand(
    tmp_path, monkeypatch, measured_runtime, targets, expected
):
    adapters = []
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    def apply_adapter(model, config):
        adapters.append(config)
        return model

    peft.get_peft_model = apply_adapter
    peft.prepare_model_for_kbit_training = lambda model, **kwargs: model
    monkeypatch.setitem(sys.modules, "peft", peft)
    training = _training() | {
        "method": "lora",
        "lora_rank": 8,
        "lora_target_modules": targets,
    }
    finetune_probe.measure_finetuning(
        _model_config(),
        training,
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    assert len(adapters) == 1
    assert adapters[0].target_modules == expected


@pytest.mark.parametrize(
    "explicit_kwargs", [None, {"use_reentrant": False}, {"use_reentrant": True}]
)
def test_probe_qlora_uses_the_same_checkpoint_mode_during_preparation_and_training(
    tmp_path, monkeypatch, measured_runtime, explicit_kwargs
):
    preparation = []
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    peft.get_peft_model = lambda model, config: model

    def prepare(model, **kwargs):
        preparation.append(kwargs)
        return model

    peft.prepare_model_for_kbit_training = prepare
    monkeypatch.setitem(sys.modules, "peft", peft)
    transformers = sys.modules["transformers"]
    transformers.BitsAndBytesConfig = SimpleNamespace
    load_model = transformers.AutoModelForCausalLM.from_pretrained

    def load_quantized_model(*args, **kwargs):
        model = load_model(*args, **kwargs)
        model.num_parameters = lambda: sum(p.numel() for p in model.parameters())
        return model

    transformers.AutoModelForCausalLM.from_pretrained = load_quantized_model
    versions = finetune_probe.installed_training_versions() | {
        "peft": "test",
        "bitsandbytes": "test",
    }
    monkeypatch.setattr(finetune_probe, "installed_training_versions", lambda: versions)
    training = _training() | {
        "method": "qlora",
        "lora_rank": 8,
        "lora_target_modules": ["q_proj"],
        "gradient_checkpointing": True,
    }
    if explicit_kwargs is not None:
        training["gradient_checkpointing_kwargs"] = explicit_kwargs
    finetune_probe.measure_finetuning(
        _model_config(),
        training,
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    expected = (
        explicit_kwargs if explicit_kwargs is not None else {"use_reentrant": False}
    )
    assert preparation == [
        {
            "use_gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": expected,
        }
    ]
    assert measured_runtime.trainer.args.gradient_checkpointing_kwargs == expected


def test_real_peft_non_reentrant_preparation_avoids_input_gradient_hook(tmp_path):
    for dependency in ("torch", "transformers", "peft"):
        try:
            metadata.version(dependency)
        except metadata.PackageNotFoundError:
            pytest.skip(f"optional CPU PEFT regression requires {dependency}")
    # Keep the actual optional stack separate from the deterministic Trainer
    # double. The marker exercises PEFT preparation only, not GPU quantization.
    code = """
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class Model(torch.nn.Module):
    is_loaded_in_4bit = True

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_kwargs = gradient_checkpointing_kwargs

for use_reentrant in (False, True):
    model = prepare_model_for_kbit_training(
        Model(),
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
    )
    assert model.checkpointing_kwargs == {"use_reentrant": use_reentrant}
    assert len(model.embedding._forward_hooks) == int(use_reentrant)
    assert model.embedding(torch.tensor([1])).requires_grad is use_reentrant

model = get_peft_model(
    torch.nn.Sequential(torch.nn.Linear(4, 4)),
    LoraConfig(r=2, target_modules="all-linear"),
)
assert any("lora_A" in name for name, _ in model.named_parameters())
try:
    get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4)),
        LoraConfig(r=2, target_modules=["all-linear"]),
    )
except ValueError as error:
    assert "not found" in str(error)
else:
    raise AssertionError("PEFT unexpectedly treats a list as the all-linear shorthand")
"""
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_probe_observation_includes_prefetch_and_accumulated_optimizer_work(
    tmp_path: Path,
    measured_runtime,
) -> None:
    training = _training()
    original = copy.deepcopy(training)
    report = finetune_probe.measure_finetuning(
        _model_config(),
        training,
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    # Each update has 2 seconds of prefetch plus 10 seconds of compute/update.
    assert report["optimizer_update_seconds"] == [12.0, 12.0]
    assert report["seconds_per_microstep"] == [6.0, 6.0]
    assert report["initialization_seconds"] == 5.0
    assert report["peak_allocated_bytes"] == 500
    assert report["peak_reserved_bytes"] == 700
    assert report["workload"]["samples"] == 12
    assert report["workload"]["measured_optimizer_steps"] == 2
    assert report["hardware"]["name"] == "Test Physical GPU"
    assert report["software_versions"]["transformers"] == "test"
    assert report["checkpoint_identity"]["algorithm"] == "sha256_content_manifest_v1"
    assert (
        report["signature"]["checkpoint_sha256"]
        == report["checkpoint_identity"]["sha256"]
    )
    assert report["checkpoint_hash_seconds"] == 0.0
    assert training == original
    trainer = measured_runtime.trainer
    assert trainer.args.optim == "adamw_torch"
    assert trainer.args.logging_nan_inf_filter is False
    assert len(trainer.train_dataset) == 12
    assert all(
        record["input_ids"] == record["labels"] == [1, 2, 3, 1]
        for record in trainer.train_dataset
    )
    assert measured_runtime.model_options["config"].model_type == "qwen3_5_text"
    assert measured_runtime.model_options["trust_remote_code"] is False


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_probe_does_not_emit_calibration_for_nonfinite_training(
    tmp_path: Path,
    measured_runtime,
    loss: float,
) -> None:
    measured_runtime.training_loss = loss
    with pytest.raises(RuntimeError, match="finite"):
        finetune_probe.measure_finetuning(
            _model_config(),
            _training(),
            model_dir=tmp_path,
            profile_id="a100",
            warmup_steps=1,
            measure_steps=2,
        )


@pytest.mark.parametrize(
    "name, value",
    [
        ("FAKEGPU_MODE", "simulate"),
        ("FAKEGPU_MODE", "hybrid"),
        ("FAKEGPU_RUNTIME", "fakecuda"),
    ],
)
def test_probe_rejects_simulated_environment_despite_cuda_availability(
    tmp_path: Path,
    measured_runtime,
    monkeypatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="require real CUDA"):
        finetune_probe.measure_finetuning(
            _model_config(),
            _training(),
            model_dir=tmp_path,
            profile_id="a100",
        )
    assert measured_runtime.trainer is None


def test_probe_keeps_startup_peak_when_steady_state_uses_less_memory(
    tmp_path: Path,
    measured_runtime,
    monkeypatch,
) -> None:
    import torch

    # The first query captures startup; later queries observe a lower peak.
    allocated = iter([900])
    reserved = iter([1200])
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda *args: next(allocated, 500)
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", lambda *args: next(reserved, 700)
    )
    report = finetune_probe.measure_finetuning(
        _model_config(),
        _training(),
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    assert report["peak_allocated_bytes"] == 900
    assert report["peak_reserved_bytes"] == 1200
    assert report["steady_peak_allocated_bytes"] == 500
    assert report["steady_peak_reserved_bytes"] == 700


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("count", 2, "exactly one"),
        ("property_name", "Different GPU", "does not match profile"),
        ("capability", (9, 0), "compute capability"),
    ],
)
def test_probe_rejects_different_hardware_before_loading_weights(
    tmp_path: Path,
    measured_runtime,
    field: str,
    value,
    message: str,
) -> None:
    setattr(measured_runtime, field, value)
    with pytest.raises(ValueError, match=message):
        finetune_probe.measure_finetuning(
            _model_config(),
            _training(),
            model_dir=tmp_path,
            profile_id="a100",
        )
    assert measured_runtime.trainer is None


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": _model_config(),
                "vision_config": {"depth": 2},
            }
        )
    )
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "framework": "huggingface",
                "model_name_or_path": str(model_dir),
                "per_device_train_batch_size": 1,
                "sequence_length": 8,
                "bf16": True,
            }
        )
    )
    return config, model_dir


def test_cli_preserves_multimodal_scope_until_text_only_is_explicit(
    tmp_path: Path,
) -> None:
    config, model_dir = _write_cli_inputs(tmp_path)
    args = argparse.Namespace(
        config=str(config),
        framework="auto",
        model=None,
        text_only=False,
        sequence_length=None,
    )
    model, training, root = finetune_cli._load_inputs(args)
    assert model["vision_config"] == {"depth": 2}
    assert root == model_dir
    assert training["sequence_length"] == 8
    args.text_only, args.sequence_length = True, 16
    text_model, text_training, text_root = finetune_cli._load_inputs(args)
    assert text_model == _model_config()
    assert text_training["sequence_length"] == 16
    assert text_root == model_dir
    assert "vision_config" in model  # Explicit selection did not mutate prior metadata.


def test_cli_model_override_updates_calibration_identity(tmp_path: Path) -> None:
    config, original_dir = _write_cli_inputs(tmp_path)
    replacement = tmp_path / "different-checkpoint"
    replacement.mkdir()
    (replacement / "config.json").write_text((original_dir / "config.json").read_text())
    args = argparse.Namespace(
        config=str(config),
        framework="auto",
        model=str(replacement),
        text_only=True,
        sequence_length=None,
    )
    _, training, root = finetune_cli._load_inputs(args)
    assert root == replacement
    assert training["model_name_or_path"] == str(replacement)


def test_cli_failed_probe_leaves_no_observation(tmp_path: Path, monkeypatch) -> None:
    config, _ = _write_cli_inputs(tmp_path)
    output = tmp_path / "failed.json"

    def fail(*args, **kwargs):
        raise RuntimeError("physical CUDA device unavailable")

    monkeypatch.setattr(finetune_probe, "measure_finetuning", fail)
    with pytest.raises(SystemExit) as exc:
        finetune_cli.probe_main(
            [
                "--config",
                str(config),
                "--profile",
                "a100",
                "--text-only",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert not output.exists()


def test_probe_binds_actual_model_dir_instead_of_declared_model_path(
    tmp_path, measured_runtime
) -> None:
    training = _training() | {"model_name_or_path": "/different/checkpoint"}
    observation = finetune_probe.measure_finetuning(
        _model_config(),
        training,
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    assert observation["signature"]["model_name_or_path"] == str(tmp_path.resolve())
    assert (
        observation["signature"]["checkpoint_sha256"]
        == local_checkpoint_identity(tmp_path)["sha256"]
    )
    assert training["model_name_or_path"] == "/different/checkpoint"


def test_probe_checks_expected_content_digest_before_loading(
    tmp_path, measured_runtime
) -> None:
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        finetune_probe.measure_finetuning(
            _model_config(),
            _training() | {"checkpoint_sha256": "0" * 64},
            model_dir=tmp_path,
            profile_id="a100",
        )
    assert measured_runtime.trainer is None


def test_checkpoint_identity_prefers_selected_safetensors_and_hashes_payload(
    tmp_path,
) -> None:
    payload = b"selected safetensors payload"
    (tmp_path / "model.safetensors").write_bytes(payload)
    (tmp_path / "pytorch_model.bin").write_bytes(b"unselected bin")
    (tmp_path / "unrelated.safetensors").write_bytes(b"unselected variant")
    identity = local_checkpoint_identity(tmp_path)
    assert identity["selection"] == "model.safetensors"
    assert identity["files"] == [
        {
            "name": "model.safetensors",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    (tmp_path / "pytorch_model.bin").write_bytes(b"unused changed")
    assert local_checkpoint_identity(tmp_path)["sha256"] == identity["sha256"]
    bound, observed = bind_local_checkpoint(
        {"checkpoint_sha256": identity["sha256"]}, tmp_path
    )
    assert bound["model_name_or_path"] == str(tmp_path.resolve())
    assert observed == identity


@pytest.mark.parametrize(
    "format, index_name, suffix",
    [
        ("safetensors", "model.safetensors.index.json", ".safetensors"),
        ("pytorch_bin", "pytorch_model.bin.index.json", ".bin"),
    ],
)
def test_checkpoint_identity_covers_index_and_all_referenced_shards(
    tmp_path, format, index_name, suffix
) -> None:
    first, second = "weights-1" + suffix, "weights-2" + suffix
    (tmp_path / first).write_bytes(b"first")
    (tmp_path / second).write_bytes(b"second")
    (tmp_path / index_name).write_text(
        json.dumps({"weight_map": {"a": second, "b": first, "c": first}})
    )
    identity = local_checkpoint_identity(tmp_path)
    assert identity["format"] == format
    assert [item["name"] for item in identity["files"]] == [index_name, first, second]
    (tmp_path / second).write_bytes(b"changed")
    assert local_checkpoint_identity(tmp_path)["sha256"] != identity["sha256"]


def test_checkpoint_identity_selects_single_file_before_sharded_index(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"single")
    (tmp_path / "model.safetensors.index.json").write_text("invalid unused index")
    assert local_checkpoint_identity(tmp_path)["selection"] == "model.safetensors"


@pytest.mark.parametrize(
    "shard", ["../weights.safetensors", "/tmp/weights.safetensors", "weights.bin"]
)
def test_checkpoint_identity_rejects_invalid_shard_paths(tmp_path, shard) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard}})
    )
    with pytest.raises(ValueError, match="shard path"):
        local_checkpoint_identity(tmp_path)


def test_checkpoint_identity_requires_weights_and_reads_bin_without_deserialization(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="no supported local checkpoint"):
        local_checkpoint_identity(tmp_path)
    # Deliberately not a pickle: fingerprinting must never execute/deserialise it.
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a pickle")
    assert local_checkpoint_identity(tmp_path)["format"] == "pytorch_bin"


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_probe_passes_explicit_checkpoint_mode(
    tmp_path, measured_runtime, use_reentrant
):
    training = _training() | {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": use_reentrant},
    }
    report = finetune_probe.measure_finetuning(
        _model_config(),
        training,
        model_dir=tmp_path,
        profile_id="a100",
        warmup_steps=1,
        measure_steps=2,
    )
    assert measured_runtime.trainer.args.gradient_checkpointing_kwargs == {
        "use_reentrant": use_reentrant
    }
    assert report["signature"]["gradient_checkpointing_kwargs"] == {
        "use_reentrant": use_reentrant
    }
