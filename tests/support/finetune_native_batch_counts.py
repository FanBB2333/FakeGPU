"""Check observed batches against a real CPU Trainer in an isolated process.

The CUDA facade only adapts observer synchronization and allocator hooks. The
Trainer, autograd, optimizer, sampler, and collator execute normally on CPU;
these checks establish accounting correctness, not CUDA performance.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import transformers

from fakegpu._finetune_batches import summarize_batch_updates
from fakegpu._finetune_observer import NativeTrainerObserver, capture_native_trainer


class ObserverTorch:
    def __init__(self):
        self.cuda = SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
            reset_peak_memory_stats=lambda: None,
        )

    def __getattr__(self, name):
        return getattr(torch, name)


class ReferenceCollator:
    def __init__(self):
        self.calls = []

    def __call__(self, records):
        lengths = [len(record["input_ids"]) for record in records]
        width = max(lengths)
        ids, masks, labels, positions = [], [], [], []
        for record, length in zip(records, lengths):
            padding = width - length
            ids.append(record["input_ids"] + [0] * padding)
            masks.append([1] * length + [0] * padding)
            labels.append(record["labels"] + [-100] * padding)
            positions.append(list(range(length)) + [0] * padding)
        row_supervised = [
            sum(label != -100 for label in record["labels"]) for record in records
        ]
        row_causal = [
            sum(label != -100 for label in record["labels"][1:]) for record in records
        ]
        # Compute expectations from the original variable-length records, not
        # from the production inspector or its returned tensors.
        expected = {
            "shape": [len(records), width],
            "token_slots": len(records) * width,
            "attention_mask_kind": "binary_2d",
            "row_nonpadding_tokens": lengths,
            "nonpadding_tokens": sum(lengths),
            "padding_tokens": len(records) * width - sum(lengths),
            "labels_present": True,
            "labels_kind": "integer_2d",
            "row_supervised_tokens": row_supervised,
            "supervised_tokens": sum(row_supervised),
            "row_causal_supervised_tokens": row_causal,
            "causal_supervised_tokens": sum(row_causal),
            "position_ids_kind": "integer_2d",
            "position_reset_offsets": [[0] for _ in records],
        }
        self.calls.append({"input_ids": ids, "expected": expected})
        return {
            name: torch.tensor(value, dtype=torch.long)
            for name, value in (
                ("input_ids", ids),
                ("attention_mask", masks),
                ("labels", labels),
                ("position_ids", positions),
            )
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.output = torch.nn.Linear(4, 16)
        self.consumed = []

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        self.consumed.append(input_ids.tolist())
        logits = self.output(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1)
        )
        return {"loss": loss, "logits": logits}


def _run_case(output_dir, *, max_steps, expected_windows):
    dataset = []
    for index, length in enumerate((2, 3, 5, 7, 11, 4, 8, 6, 10, 9, 12, 13, 14)):
        ids = [index + 1] * length
        # Keep some first tokens supervised so causal and noncausal label
        # counts differ, and mask other prompts independently of padding.
        prompt = min(index % 3, length - 1)
        dataset.append({"input_ids": ids, "labels": [-100] * prompt + ids[prompt:]})
    model = TinyModel()
    collator = ReferenceCollator()
    observer_torch = ObserverTorch()
    trainer_calls = []
    observer = NativeTrainerObserver(
        observer_torch,
        warmup_steps=1,
        on_trainer=lambda trainer: trainer_calls.append("attach"),
        on_train_begin=lambda trainer: trainer_calls.append("begin"),
    )
    arguments = transformers.TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        max_steps=max_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        data_seed=42,
    )
    trainer = transformers.Trainer(
        model=model, args=arguments, train_dataset=dataset, data_collator=collator
    )
    original_train = transformers.Trainer.train
    original_step = trainer.training_step
    original_batches = trainer.get_batch_samples
    original_reset = observer_torch.cuda.reset_peak_memory_stats
    real_cuda_reset = torch.cuda.reset_peak_memory_stats
    original_callbacks = list(trainer.callback_handler.callbacks)
    with capture_native_trainer(
        transformers.Trainer, transformers.TrainerCallback, observer
    ):
        result = trainer.train()
    assert transformers.Trainer.train is original_train
    assert trainer.training_step == original_step
    assert trainer.get_batch_samples == original_batches
    assert observer_torch.cuda.reset_peak_memory_stats is original_reset
    assert torch.cuda.reset_peak_memory_stats is real_cuda_reset
    assert trainer.callback_handler.callbacks == original_callbacks
    assert observer._restores == []
    assert trainer_calls == ["attach", "begin"]
    assert next(model.parameters()).device.type == "cpu"
    assert math.isfinite(result.training_loss)
    assert observer.parameter_changes["has_changed_trainable_parameters"]

    batches = [batch for update in observer.updates for batch in update["batches"]]
    assert [update["microsteps"] for update in observer.updates] == expected_windows
    assert observer.microsteps == len(model.consumed) == len(batches)
    assert trainer.state.global_step == len(observer.updates) == len(expected_windows)
    assert observer.samples == sum(len(ids) for ids in model.consumed)
    assert all(update["seconds"] > 0 for update in observer.updates)
    assert all(
        len(update["losses"]) == len(update["batches"]) for update in observer.updates
    )
    assert [batch["shape"][0] for batch in batches] == [3, 3, 3, 3, 1, 3, 3, 3, 3, 1][
        : len(batches)
    ]
    assert [batch["epoch_index"] for batch in batches] == [
        index // 5 + 1 for index in range(len(batches))
    ]
    assert [batch["epoch_microstep"] for batch in batches] == [
        index % 5 + 1 for index in range(len(batches))
    ]

    # Accelerate may collate the next batch ahead of consumption. Match each
    # actual model call in order and never count an unconsumed collator call.
    reference_index = 0
    expected = []
    for ids, batch in zip(model.consumed, batches):
        while collator.calls[reference_index]["input_ids"] != ids:
            reference_index += 1
        reference = collator.calls[reference_index]["expected"]
        reference_index += 1
        for field, value in reference.items():
            assert batch[field] == value, (field, batch[field], value)
        expected.append(reference)
    profile = summarize_batch_updates(observer.updates)
    assert profile["schema_version"] == "fakegpu.finetune_batches.v1"
    for field in (
        "token_slots",
        "nonpadding_tokens",
        "padding_tokens",
        "supervised_tokens",
        "causal_supervised_tokens",
    ):
        assert profile["totals"][field] == sum(batch[field] for batch in expected)
    assert profile["totals"]["samples"] == sum(batch["shape"][0] for batch in expected)
    assert profile["totals"]["microsteps"] == len(expected)
    assert all(count == len(expected) for count in profile["coverage"].values())
    shapes = Counter(tuple(batch["shape"]) for batch in expected)
    assert profile["shape_counts"] == [
        {"shape": list(shape), "microsteps": count}
        for shape, count in sorted(shapes.items())
    ]
    return {
        "max_steps": max_steps,
        "optimizer_steps": trainer.state.global_step,
        "microsteps": observer.microsteps,
        "samples": observer.samples,
        "collated_batches": len(collator.calls),
        "expected_nonpadding_tokens": sum(
            batch["nonpadding_tokens"] for batch in expected
        ),
        "expected_supervised_tokens": sum(
            batch["supervised_tokens"] for batch in expected
        ),
        "expected_causal_supervised_tokens": sum(
            batch["causal_supervised_tokens"] for batch in expected
        ),
        "profile": profile,
    }


def main(output_dir: Path):
    torch.set_num_threads(1)
    results = [
        _run_case(
            output_dir / "epochs", max_steps=-1, expected_windows=[2, 2, 1, 2, 2, 1]
        ),
        _run_case(output_dir / "max-steps", max_steps=4, expected_windows=[2, 2, 1, 2]),
    ]
    (output_dir / "native-batch-counts.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cases": results,
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
