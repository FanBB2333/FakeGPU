from types import SimpleNamespace

import pytest

from fakegpu._finetune_model import actual_training_config
from fakegpu._finetune_observer import capture_native_trainer
from fakegpu._finetune_sampler import SCHEMA_VERSION
from fakegpu.finetune_estimator import (
    FinetuneEstimateError,
    finetune_workload_signature,
)
from fakegpu.finetune_runner import run_finetuning
from test_finetune_model import _normalized, trainer as trainer
from test_finetune_runner import Callback, _instrumented_fixture


def test_input_matching_cost_is_counted_only_as_batch_instrumentation(monkeypatch):
    base, trainer, observer = _instrumented_fixture(monkeypatch)
    calls = []

    def observe(inputs, *, epoch_index, epoch_microstep, descriptor):
        calls.append((epoch_index, epoch_microstep))
        trainer.test_clock.advance(0.25)
        return {
            **descriptor,
            "input_sha256": "a" * 64,
            "row_indices": [epoch_microstep - 1],
        }

    observer.batch_matcher = SimpleNamespace(observe=observe)
    with capture_native_trainer(base, Callback, observer):
        trainer.train()
    assert calls == [(1, index) for index in range(1, 9)]
    assert [update["seconds"] for update in observer.updates] == [9.0] * 4
    assert [update["wall_seconds"] for update in observer.updates] == [9.5] * 4
    assert observer.batch_instrumentation_seconds == 2.0
    assert all(
        batch["input_sha256"] == "a" * 64
        for update in observer.updates
        for batch in update["batches"]
    )


def test_sampler_mismatch_aborts_before_training_and_restores_instrumentation(
    monkeypatch,
):
    base, trainer, observer = _instrumented_fixture(monkeypatch)
    original = base.train

    def reject(*_args, **_kwargs):
        raise FinetuneEstimateError("native batch tensor contents differ from CPU plan")

    observer.batch_matcher = SimpleNamespace(observe=reject)
    with pytest.raises(RuntimeError, match="tensor contents differ"):
        with capture_native_trainer(base, Callback, observer):
            trainer.train()
    assert observer.microsteps == 0
    assert not observer.updates
    assert trainer.model.weight == 0
    assert base.train is original
    assert not trainer.callbacks
    assert "training_step" not in trainer.__dict__
    assert "get_batch_samples" not in trainer.__dict__


def test_actual_model_reconciliation_preserves_sampler_protocol_in_signature(
    trainer,
):
    native_trainer, _, _ = trainer
    normalized = {**_normalized(), "sampler_observation_version": SCHEMA_VERSION}
    actual, _ = actual_training_config(native_trainer, normalized)
    assert actual["sampler_observation_version"] == SCHEMA_VERSION
    signature = finetune_workload_signature(
        {"model_type": "llama"}, actual, profile_id="a100"
    )
    assert signature["sampler_observation_version"] == SCHEMA_VERSION
    actual.pop("sampler_observation_version")
    legacy = finetune_workload_signature(
        {"model_type": "llama"}, actual, profile_id="a100"
    )
    assert "sampler_observation_version" not in legacy
    assert {
        key: value
        for key, value in signature.items()
        if key != "sampler_observation_version"
    } == legacy


@pytest.mark.parametrize(
    "options,match",
    [
        ({"planning_epochs": 1, "planning_max_steps": 2}, "one planning duration"),
        ({"planning_epochs": 0}, "finite and positive"),
        ({"planning_epochs": float("inf")}, "finite and positive"),
        ({"planning_epochs": True}, "finite and positive"),
        ({"planning_max_steps": 0}, "positive integer"),
        ({"planning_max_steps": 1.5}, "positive integer"),
        ({"planning_max_steps": True}, "positive integer"),
    ],
)
def test_invalid_planning_duration_rejected_before_config_or_gpu_access(options, match):
    with pytest.raises(ValueError, match=match):
        run_finetuning(
            "does-not-exist.json", framework="huggingface", profile_id="a100", **options
        )
