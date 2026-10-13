"""Audit a real CPU Trainer's evaluation/checkpoint event ledger.

Independent forward inputs, TrainerCallbacks and on-disk files establish counts;
CPU durations do not establish GPU evaluation or storage throughput accuracy.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

import transformers

from fakegpu._finetune_observer import NativeTrainerObserver, capture_native_trainer
from fakegpu._finetune_events import summarize_phase_events

import finetune_native_batch_counts as support


class Model(support.TinyModel):
    def __init__(self):
        super().__init__()
        self.eval_calls = []

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        if not self.training:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
            }
            self.eval_calls.append(
                {
                    key: value.tolist()
                    for key, value in inputs.items()
                    if value is not None
                }
            )
        return super().forward(input_ids, attention_mask, labels, position_ids)


class ReferenceCallback(transformers.TrainerCallback):
    def __init__(self):
        self.eval_boundaries = []
        self.checkpoints = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_boundaries.append(
            (state.global_step, len(kwargs["model"].eval_calls))
        )

    def on_save(self, args, state, control, **kwargs):
        self.checkpoints.append(state.global_step)


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    data = []
    for index, length in enumerate((2, 3, 5, 7, 11, 4, 8, 6, 10, 9, 12, 13, 14)):
        ids = [index + 1] * length
        data.append({"input_ids": ids, "labels": [-100] + ids[1:]})
    args = transformers.TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=7,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=2,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=3,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        data_seed=42,
    )
    model = Model()
    reference = ReferenceCallback()
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=data,
        eval_dataset=data[:5],
        data_collator=support.ReferenceCollator(),
        callbacks=[reference],
    )
    started = time.perf_counter()
    observer = NativeTrainerObserver(
        support.ObserverTorch(),
        warmup_steps=1,
        on_trainer=lambda _: None,
        on_train_begin=lambda _: None,
        run_started=started,
    )
    original_predict = trainer.prediction_step
    with capture_native_trainer(
        transformers.Trainer, transformers.TrainerCallback, observer
    ):
        trainer.train()
        trainer.evaluate()
        trainer.save_model(str(output / "final"))
    assert trainer.prediction_step == original_predict
    assert not observer._restores
    ledger = observer.phase_events
    assert all(event["status"] == "succeeded" for event in ledger)
    roots = [event for event in ledger if event["parent_event_id"] is None]
    evals = [event for event in roots if event["kind"] == "evaluation"]
    saves = [event for event in roots if event["kind"] == "checkpoint"]
    assert [event["optimizer_step"] for event in evals] == [
        step for step, _ in reference.eval_boundaries
    ]
    assert [event["optimizer_step"] for event in saves[:-1]] == reference.checkpoints
    previous = 0
    for event, (_, boundary) in zip(evals, reference.eval_boundaries):
        consumed = model.eval_calls[previous:boundary]
        encoded = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in consumed
        ).encode()
        workload = event["workload"]
        assert (
            workload["consumed_content_sha256"] == hashlib.sha256(encoded).hexdigest()
        )
        assert [batch["shape"] for batch in workload["batches"]] == [
            [2, 3],
            [2, 7],
            [1, 11],
        ]
        assert sum(batch["nonpadding_tokens"] for batch in workload["batches"]) == 28
        assert sum(batch["supervised_tokens"] for batch in workload["batches"]) == 23
        assert workload["prediction_loss_only"] == [True] * 3
        previous = boundary
    for event in saves:
        workload = event["workload"]
        root = Path(workload["output_directories"][0])
        expected = [
            {"path": path.name, "bytes": path.stat().st_size}
            for path in sorted(root.iterdir())
            if path.is_file()
        ]
        assert workload["files"] == expected
        assert workload["total_file_bytes"] == sum(item["bytes"] for item in expected)
        if event["operation"] == "_save_checkpoint":
            assert {
                "optimizer.pt",
                "scheduler.pt",
                "trainer_state.json",
                "rng_state.pth",
            } <= {item["path"] for item in expected}
    for update in observer.updates:
        expected = [
            event["event_id"]
            for event in roots
            if event["inside_train"]
            and event["optimizer_step"] == update["optimizer_step"] - 1
        ]
        assert update["preceding_event_ids"] == expected
    assert (
        abs(
            sum(event["instrumentation_seconds"] for event in roots)
            - observer.event_instrumentation_seconds
        )
        < 1e-6
    )
    summary = summarize_phase_events(ledger)
    report = {
        "transformers": transformers.__version__,
        "events": ledger,
        "profile": summary,
        "eval_steps": [e["optimizer_step"] for e in evals],
        "checkpoint_steps": reference.checkpoints,
        "updates": len(observer.updates),
        "passed": True,
    }
    (output / "native-event-counts.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("events", "profile")
            }
        )
    )


if __name__ == "__main__":
    main()
