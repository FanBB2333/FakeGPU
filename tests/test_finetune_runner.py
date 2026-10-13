from collections import UserDict
from types import SimpleNamespace
import json
import math

import pytest

from fakegpu._finetune_observer import NativeTrainerObserver, capture_native_trainer
from fakegpu.finetune_runner import _execution_signature, _prepare_config


def _config(tmp_path, **changes):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    config = {
        "model_name_or_path": str(model),
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "cutoff_len": 128,
        "flash_attn": "sdpa",
        "bf16": True,
        "max_steps": 6,
        "output_dir": str(tmp_path / "output"),
        "disable_gradient_checkpointing": True,
        "gradient_checkpointing": False,
    }
    config.update(changes)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"packing": True}, "unpacked"),
        ({"max_steps": 3}, "two measured"),
        ({"trust_remote_code": True}, "trust_remote_code"),
        ({"flash_attn": "auto"}, "sdpa or eager"),
        ({"streaming": True}, "streaming"),
        ({"world_size": 2}, "one GPU"),
        ({"train_from_scratch": True}, "train_from_scratch"),
        ({"reinit_weights": True}, "reinit_weights"),
        ({"unfrozen_parameters": ["model.layers.0"]}, "unfrozen_parameters"),
        ({"pure_bf16": True}, "pure_bf16"),
        ({"export_quantization_bit": 4}, "export_quantization_bit"),
        (
            {"overrides_of_model_kwargs": {"use_safetensors": False}},
            "overrides_of_model_kwargs",
        ),
        ({"model_kwargs": {"use_safetensors": False}}, "model_kwargs"),
        (
            {"overrides_of_model_config": {"num_hidden_layers": 2}},
            "overrides_of_model_config",
        ),
        ({"model_config": {"num_hidden_layers": 2}}, "model_config"),
        ({"use_kernels": True}, "use_kernels"),
    ],
)
def test_unsupported_runs_rejected_before_framework_import(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        _prepare_config(_config(tmp_path, **changes), "llamafactory", 2)


def test_inert_native_checkpointing_option_and_relative_paths(tmp_path):
    path = _config(tmp_path, use_reentrant_gc=False, model_name_or_path="model")
    raw, config, _, model = _prepare_config(path, "llamafactory", 2)
    assert raw["use_reentrant_gc"] is False
    assert config["gradient_checkpointing"] is False
    assert model == tmp_path / "model"


def test_existing_output_cannot_silently_resume(tmp_path):
    path = _config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "checkpoint").write_text("existing")
    with pytest.raises(ValueError, match="fresh output"):
        _prepare_config(path, "llamafactory", 2)


def test_multimodal_checkpoint_requires_explicit_export(tmp_path):
    path = _config(tmp_path)
    (tmp_path / "model/config.json").write_text(
        json.dumps({"vision_config": {"hidden_size": 4}})
    )
    with pytest.raises(ValueError, match="pure text"):
        _prepare_config(path, "llamafactory", 2)


def test_native_four_bit_qlora_preflight_uses_unquantized_source_checkpoint(tmp_path):
    path = _config(tmp_path, quantization_bit=4, quantization_type="nf4")
    _, config, _, _ = _prepare_config(path, "llamafactory", 2)
    assert config["method"] == "qlora"
    assert config["quantization_bits"] == 4


def test_native_eight_bit_training_rejected_before_framework_import(tmp_path):
    with pytest.raises(ValueError, match="4-bit"):
        _prepare_config(_config(tmp_path, quantization_bit=8), "llamafactory", 2)


def test_prequantized_source_checkpoint_rejected_before_framework_import(tmp_path):
    path = _config(tmp_path, quantization_bit=4)
    (tmp_path / "model/config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {"quant_method": "bitsandbytes"},
            }
        )
    )
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        _prepare_config(path, "llamafactory", 2)


def test_native_identity_distinguishes_preprocessing_but_not_duration():
    base = {
        "template": "qwen3_nothink",
        "max_steps": 6,
        "num_train_epochs": 1,
        "output_dir": "/tmp/a",
        "use_v1_kernels": False,
    }
    duration = dict(base, max_steps=8, num_train_epochs=3, output_dir="/tmp/b")
    assert _execution_signature(base) == _execution_signature(duration)
    assert _execution_signature(base) != _execution_signature(
        dict(base, template="qwen3")
    )
    assert _execution_signature(base) != _execution_signature(
        dict(base, use_v1_kernels=True)
    )


class Clock:
    value = 0.0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Scalar:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def item(self):
        return self.value


class Callback:
    pass


def _instrumented_fixture(monkeypatch, *, fail=False, nonfinite=False, remainder=False):
    import fakegpu._finetune_observer as module

    clock = Clock()
    monkeypatch.setattr(module.time, "perf_counter", clock.now)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda: 100,
            max_memory_reserved=lambda: 200,
            reset_peak_memory_stats=lambda: None,
        )
    )

    class Model:
        weight = 0

        def save_pretrained(self, *_args):
            clock.advance(3)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.callbacks = []
            self.accelerator = SimpleNamespace(optimizer_step_was_skipped=False)
            self.state = SimpleNamespace(global_step=0)

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def remove_callback(self, callback):
            self.callbacks.remove(callback)

        def get_batch_samples(self, *_args):
            clock.advance(2)  # Deliberately outside on_step_begin.
            count = 1 if remainder and self.state.global_step == 3 else 2
            # BatchEncoding is a Mapping/UserDict rather than a plain dict.
            return [
                UserDict({"input_ids": SimpleNamespace(shape=(1, 8))})
                for _ in range(count)
            ]

        def training_step(self, _model, _inputs):
            clock.advance(3)
            return Scalar(math.nan if nonfinite else 1.0)

        def evaluate(self):
            clock.advance(5)

        def save_model(self):
            self.model.save_pretrained("output")

        def _save_checkpoint(self):
            self.save_model()  # Nested method wrappers must count once.

        def train(self):
            callback = self.callbacks[0]
            callback.on_train_begin(None, self.state, None)
            if fail:
                raise RuntimeError("deliberate training failure")
            for index in range(4):
                batches = self.get_batch_samples()
                callback.on_step_begin(None, self.state, None)
                for inputs in batches:
                    self.training_step(self.model, inputs)
                clock.advance(1)  # Optimizer work belongs in update timing.
                self.model.weight += 1
                self.state.global_step = index + 1
                callback.on_step_end(None, self.state, None)
                if index == 1:
                    self.evaluate()
                    self._save_checkpoint()
            callback.on_train_end(None, self.state, None)
            return SimpleNamespace(training_loss=1.0)

    trainer = Trainer()
    observer = NativeTrainerObserver(
        torch, warmup_steps=2, on_trainer=lambda _: None, on_train_begin=lambda _: None
    )
    observer._weight_samples = lambda: {"weight": trainer.model.weight}
    return Trainer, trainer, observer


def test_prefetch_timing_real_microsteps_and_nested_final_saves(monkeypatch):
    base, trainer, observer = _instrumented_fixture(monkeypatch, remainder=True)
    original = base.train
    with capture_native_trainer(base, Callback, observer):
        trainer.train()
        trainer.save_model()
    assert base.train is original
    assert not trainer.callbacks
    assert "training_step" not in trainer.__dict__
    assert "save_pretrained" not in trainer.model.__dict__
    assert [update["seconds"] for update in observer.updates] == [9, 9, 9, 6]
    assert [update["microsteps"] for update in observer.updates] == [2, 2, 2, 1]
    assert observer.microsteps == 7 and observer.samples == 7
    assert observer.phases == {"evaluation": 5, "checkpoint": 6}
    assert observer.events == {"evaluation": 1, "checkpoint": 2}
    assert observer.during_train_events == {"evaluation": 1, "checkpoint": 1}
    assert observer.parameter_changes["has_changed_trainable_parameters"]
    assert observer.peak_allocated == 100 and observer.peak_reserved == 200


@pytest.mark.parametrize(
    "options,match",
    [
        ({"fail": True}, "deliberate training failure"),
        ({"nonfinite": True}, "nonfinite microstep loss"),
    ],
)
def test_instrumentation_restored_on_training_failure(monkeypatch, options, match):
    base, trainer, observer = _instrumented_fixture(monkeypatch, **options)
    original = base.train
    with pytest.raises(RuntimeError, match=match):
        with capture_native_trainer(base, Callback, observer):
            trainer.train()
    assert base.train is original
    assert not trainer.callbacks
    assert "get_batch_samples" not in trainer.__dict__
    assert "save_pretrained" not in trainer.model.__dict__


def test_duplicate_native_train_call_cannot_mix_observations(monkeypatch):
    base, trainer, observer = _instrumented_fixture(monkeypatch)
    with capture_native_trainer(base, Callback, observer):
        trainer.train()
        with pytest.raises(RuntimeError, match="exactly one"):
            trainer.train()


def test_memory_tracker_reset_before_trainer_capture_preserves_loading_peak(
    monkeypatch,
):
    base, trainer, observer = _instrumented_fixture(monkeypatch)
    peak = {"allocated": 900, "reserved": 1200}
    cuda = observer.torch.cuda
    cuda.max_memory_allocated = lambda: peak["allocated"]
    cuda.max_memory_reserved = lambda: peak["reserved"]

    def reset():
        peak.update(allocated=100, reserved=200)

    cuda.reset_peak_memory_stats = reset
    with capture_native_trainer(base, Callback, observer):
        assert observer.trainer is None
        cuda.reset_peak_memory_stats()  # Model load completed; Trainer init resets.
        trainer.train()
    assert cuda.reset_peak_memory_stats is reset
    assert observer.peak_allocated == 900
    assert observer.peak_reserved == 1200


def test_peak_reset_hook_restored_when_native_initialization_fails(monkeypatch):
    base, _trainer, observer = _instrumented_fixture(monkeypatch)
    original = observer.torch.cuda.reset_peak_memory_stats
    with pytest.raises(ValueError, match="initialization failed"):
        with capture_native_trainer(base, Callback, observer):
            raise ValueError("initialization failed before Trainer exists")
    assert observer.torch.cuda.reset_peak_memory_stats is original
