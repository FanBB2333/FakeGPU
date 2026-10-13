"""Regression checks for invalid batches at the native training boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fakegpu._finetune_observer import capture_native_trainer
from test_finetune_runner import Callback, _instrumented_fixture


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
