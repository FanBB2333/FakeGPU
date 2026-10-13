"""Call-tree, metadata overhead, and artifact accounting independent of CUDA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakegpu._finetune_observer import NativeTrainerObserver


class Tensor:
    def __init__(self, values, clock=None):
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.clock = clock

    def detach(self):
        return self

    def cpu(self):
        if self.clock is not None:
            self.clock.advance(0.125)
        return self

    def tolist(self):
        return self.values


class Clock:
    now = 100.0

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def instrumented(monkeypatch, tmp_path):
    import fakegpu._finetune_event_observer as module

    clock = Clock()
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        )
    )
    dataset = [
        {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
        {"input_ids": [5, 6], "labels": [-100, 6]},
    ]
    inputs = {
        "input_ids": Tensor([[2, 3, 4], [5, 6, 0]], clock),
        "attention_mask": Tensor([[1, 1, 1], [1, 1, 0]], clock),
        "labels": Tensor([[-100, 3, 4], [-100, 6, -100]], clock),
    }

    class Model:
        def save_pretrained(self, save_directory):
            directory = Path(save_directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "adapter_model.safetensors").write_bytes(b"weights")
            clock.advance(3)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.args = SimpleNamespace(
                output_dir=str(tmp_path),
                save_only_model=False,
                per_device_eval_batch_size=2,
                dataloader_drop_last=False,
                predict_with_generate=False,
            )
            self.state = SimpleNamespace(global_step=4, epoch=1.5)
            self.eval_dataset = dataset
            self.compute_metrics = None
            self.fail = False

        def prediction_step(self, model, inputs, prediction_loss_only):
            clock.advance(2)
            if self.fail:
                raise RuntimeError("prediction failure")
            return (1.0, None, None)

        def evaluate(self, eval_dataset=None):
            if isinstance(eval_dataset, dict):
                return {
                    name: self.evaluate(rows) for name, rows in eval_dataset.items()
                }
            self.prediction_step(self.model, inputs, True)
            return {"eval_loss": 1.0}

        def predict(self, test_dataset):
            self.prediction_step(self.model, inputs, False)

        def save_model(self, output_dir=None):
            self.model.save_pretrained(output_dir or self.args.output_dir)

        def _save_checkpoint(self):
            directory = tmp_path / f"checkpoint-{self.state.global_step}"
            self.save_model(str(directory))
            (directory / "optimizer.pt").write_bytes(b"optimizer")
            (directory / "trainer_state.json").write_text("{}")
            clock.advance(1)

    trainer = Trainer()
    observer = NativeTrainerObserver(
        torch,
        warmup_steps=0,
        on_trainer=lambda _: None,
        on_train_begin=lambda _: None,
        run_started=90.0,
    )
    observer.trainer = trainer
    observer.event_observer.attach(trainer)
    yield trainer, observer, clock, inputs
    observer.restore()


def test_root_and_nested_saves_record_actual_artifacts_once(instrumented, tmp_path):
    trainer, observer, _, _ = instrumented
    observer._inside_train = True
    trainer._save_checkpoint()
    observer._inside_train = False
    trainer.save_model()
    ledger = observer.phase_events
    assert [event["operation"] for event in ledger] == [
        "_save_checkpoint",
        "save_model",
        "save_pretrained",
        "save_model",
        "save_pretrained",
    ]
    assert [event["parent_event_id"] for event in ledger] == [None, 1, 2, None, 4]
    assert [event["seconds"] for event in ledger] == [4, 3, 3, 3, 3]
    assert observer.phases == {"checkpoint": 7}
    assert observer.events == {"checkpoint": 2}
    assert observer.during_train_phases == {"checkpoint": 4}
    assert ledger[0]["optimizer_step"] == 4 and ledger[0]["epoch"] == 1.5
    assert ledger[0]["start_seconds"] == 10
    assert ledger[0]["workload"]["files"] == [
        {"path": "adapter_model.safetensors", "bytes": 7},
        {"path": "optimizer.pt", "bytes": 9},
        {"path": "trainer_state.json", "bytes": 2},
    ]
    assert ledger[0]["workload"]["total_file_bytes"] == 18
    assert ledger[0]["workload"]["artifact_kind"] == "adapter"
    assert ledger[3]["workload"]["files"] == [
        {"path": "adapter_model.safetensors", "bytes": 7}
    ]
    assert observer.event_observer.take_preceding_events() == [1]
    assert observer.event_observer.take_preceding_events() == []
    assert (tmp_path / "checkpoint-4/optimizer.pt").is_file()


def test_evaluation_consumed_batches_hashes_and_metadata_seconds(instrumented):
    trainer, observer, _, inputs = instrumented
    trainer.evaluate()
    event = observer.phase_events[0]
    workload = event["workload"]
    assert event["status"] == "succeeded" and event["seconds"] == 2
    assert event["wall_seconds"] == 2.625
    assert event["instrumentation_seconds"] == 0.625
    assert (
        observer.event_instrumentation_seconds
        == observer.instrumentation_seconds
        == 0.625
    )
    assert observer.phases == {"evaluation": 2.625}
    batch = workload["batches"][0]
    assert batch["shape"] == [2, 3]
    assert batch["nonpadding_tokens"] == 5
    assert batch["supervised_tokens"] == batch["causal_supervised_tokens"] == 3
    assert workload["prediction_loss_only"] == [True]
    assert workload["dataset"]["num_samples"] == 2

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"

    expected_dataset = hashlib.sha256(
        "".join(map(canonical, trainer.eval_dataset)).encode()
    ).hexdigest()
    expected_inputs = hashlib.sha256(
        canonical({key: value.values for key, value in inputs.items()}).encode()
    ).hexdigest()
    assert workload["dataset"]["content_sha256"] == expected_dataset
    assert workload["consumed_content_sha256"] == expected_inputs
    assert "input_ids" not in json.dumps(workload["batches"])


def test_nested_split_eval_does_not_double_count_metadata_or_root_time(instrumented):
    trainer, observer, _, _ = instrumented
    trainer.evaluate({"a": trainer.eval_dataset, "b": trainer.eval_dataset})
    parent, first, second = observer.phase_events
    assert parent["workload"]["dataset"]["status"] == "unknown"
    assert len(parent["workload"]["batches"]) == 2
    assert [first["parent_event_id"], second["parent_event_id"]] == [1, 1]
    assert parent["instrumentation_seconds"] == 1.25
    assert observer.event_instrumentation_seconds == 1.25
    assert parent["seconds"] == 4
    assert observer.phases == {"evaluation": 5.25}
    assert observer.events == {"evaluation": 1}


def test_same_step_calls_remain_separate_and_failed_prediction_not_consumed(
    instrumented,
):
    trainer, observer, _, _ = instrumented
    trainer.evaluate()
    trainer.evaluate()
    trainer.fail = True
    with pytest.raises(RuntimeError, match="prediction failure"):
        trainer.evaluate()
    assert [event["optimizer_step"] for event in observer.phase_events] == [4, 4, 4]
    assert observer.events["evaluation"] == 3
    failed = observer.phase_events[-1]
    assert failed["status"] == "failed"
    assert failed["workload"]["batches"] == []
    assert failed["workload"]["prediction_loss_only"] == []
    assert observer.event_observer.stack == []
    observer.restore()
    assert "prediction_step" not in trainer.__dict__
    assert "save_pretrained" not in trainer.model.__dict__


def test_prediction_and_unsupported_metadata_remain_explicit(instrumented):
    trainer, observer, _, inputs = instrumented
    inputs.pop("attention_mask")
    trainer.predict(trainer.eval_dataset)
    workload = observer.phase_events[0]["workload"]
    assert workload["prediction_loss_only"] == [False]
    assert workload["batches"][0]["nonpadding_tokens"] is None
    assert workload["batches"][0]["attention_mask_kind"] == "missing"


def test_dataset_fingerprinting_does_not_invoke_arbitrary_transforms(instrumented):
    trainer, observer, _, _ = instrumented

    class TransformedDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("instrumentation must not call random transform")

    trainer.eval_dataset = TransformedDataset()
    trainer.evaluate()
    assert observer.phase_events[0]["workload"]["dataset"] == {
        "status": "unknown",
        "reason": "unverified_map_style_access",
    }


def test_wait_for_gpu_work_is_not_subtracted_as_metadata(instrumented):
    trainer, observer, clock, _ = instrumented
    original = trainer.prediction_step
    pending = 3.0

    def synchronize():
        nonlocal pending
        clock.advance(pending)
        pending = 0.0

    observer.torch.cuda.synchronize = synchronize
    # Work queued before entering evaluation belongs to the preceding phase.
    trainer.evaluate()
    event = observer.phase_events[0]
    assert event["start_seconds"] == 13
    assert event["seconds"] == 2
    assert event["instrumentation_seconds"] == 0.625
    assert trainer.prediction_step is original


def test_missing_batch_shape_is_observed_without_claiming_known_workload(instrumented):
    trainer, observer, _, inputs = instrumented
    inputs.pop("input_ids")
    trainer.evaluate()
    workload = observer.phase_events[0]["workload"]
    assert workload["batches"] == []
    assert workload["unobserved_batches"] == 1
    assert workload["prediction_loss_only"] == [True]
    assert workload["consumed_content_status"] == "unknown"
    assert workload["consumed_content_sha256"] is None
