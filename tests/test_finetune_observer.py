"""Regression checks for invalid batches at the native training boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fakegpu._finetune_observer import NativeTrainerObserver, capture_native_trainer
from fakegpu._finetune_timeline import build_native_timeline
from test_finetune_runner import Callback, Scalar, _instrumented_fixture


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        [],
        {},
        {"input_ids": SimpleNamespace(shape=(2, 0))},
        {"input_ids": SimpleNamespace(shape=(2, 3, 4))},
    ],
)
def test_invalid_batch_fails_before_training_and_restores_instrumentation(
    monkeypatch, inputs
):
    base, trainer, observer = _instrumented_fixture(monkeypatch)
    original_train = base.train
    trainer.get_batch_samples = original_batches = Mock(return_value=[inputs])
    trainer.training_step = original_step = Mock()

    with pytest.raises(RuntimeError, match="invalid native text microbatch") as error:
        with capture_native_trainer(base, Callback, observer):
            trainer.train()

    assert isinstance(error.value.__cause__, ValueError)
    original_step.assert_not_called()
    assert observer.microsteps == observer.samples == 0
    assert observer.shapes == {}
    assert observer.updates == []
    assert base.train is original_train
    assert trainer.get_batch_samples is original_batches
    assert trainer.training_step is original_step
    assert not trainer.callbacks
    assert "save_pretrained" not in trainer.model.__dict__


def test_shared_clock_and_callback_metadata_partition_without_more_cuda_syncs(
    monkeypatch,
):
    base, trainer, original_observer = _instrumented_fixture(monkeypatch)
    clock = trainer.test_clock
    clock.advance(100)
    cuda = original_observer.torch.cuda
    cuda.synchronize = Mock()
    observer = NativeTrainerObserver(
        original_observer.torch,
        warmup_steps=2,
        on_trainer=lambda _: clock.advance(1),
        on_train_begin=lambda _: clock.advance(0.3),
        run_started=95,
    )

    def weights():
        clock.advance(0.2)
        return {"weight": trainer.model.weight}

    def loss_item(value):
        clock.advance(0.1)
        return value.value

    observer._weight_samples = weights
    monkeypatch.setattr(Scalar, "item", loss_item)
    with capture_native_trainer(base, Callback, observer):
        trainer.train()
    # Existing entry/exit, per-update/microstep and event syncs are unchanged.
    assert cuda.synchronize.call_count == 34
    assert observer.trainer_started == 101
    assert observer.updates[0]["start_seconds"] == pytest.approx(6.5)
    assert observer.updates[0]["end_seconds"] == pytest.approx(15.5)
    assert all(
        update["wall_seconds"] == pytest.approx(9) for update in observer.updates
    )
    assert [item["kind"] for item in observer.instrumentation_intervals] == [
        "trainer_inspection",
        "train_begin_inspection",
        *["loss_collection"] * 4,
        "train_end_parameter_check",
    ]
    assert sum(
        item["seconds"] for item in observer.instrumentation_intervals
    ) == pytest.approx(2.5)
    timeline = build_native_timeline(
        observer.updates,
        observer.phase_events,
        observer.instrumentation_intervals,
        trainer_start_seconds=observer.trainer_started - observer.run_started,
        trainer_end_seconds=observer.trainer_finished - observer.run_started,
        run_end_seconds=clock.now() - observer.run_started,
    )
    assert timeline["totals"]["wall_seconds"] == pytest.approx(51.5)
    assert timeline["totals"]["seconds"] == pytest.approx(49)
    assert timeline["totals"]["instrumentation_seconds"] == pytest.approx(
        observer.instrumentation_seconds
    )
    assert timeline["totals"]["by_kind"]["setup"]["wall_seconds"] == 6
    assert timeline["totals"]["by_kind"]["trainer_other"]["seconds"] == pytest.approx(0)
