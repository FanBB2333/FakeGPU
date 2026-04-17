"""P6-1: HuggingFace Trainer end-to-end compatibility tests.

Validates that the ``transformers.Trainer`` pipeline works under FakeGPU:
  - Training loop
  - Evaluation with custom metrics
  - Checkpoint save / load
  - Training + evaluation combined

Requires: ``transformers``, ``accelerate``
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")

import unittest

import fakegpu

fakegpu.patch_torch()

import numpy as np
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def _make_tiny_config() -> DistilBertConfig:
    return DistilBertConfig(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
        num_labels=2,
    )


class _TinyDataset(torch.utils.data.Dataset):
    """Synthetic dataset with random token ids."""

    def __init__(self, size: int = 32, seq_len: int = 16, vocab_size: int = 1000):
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len))
        self.attention_mask = torch.ones(size, seq_len, dtype=torch.long)
        self.labels = torch.randint(0, 2, (size,))

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }

    def __len__(self):
        return len(self.labels)


def _base_training_args(**overrides) -> TrainingArguments:
    defaults = dict(
        output_dir=tempfile.mkdtemp(prefix="fakegpu_trainer_"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        logging_steps=99,  # suppress log noise
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        dataloader_pin_memory=False,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


class TestHFTrainerBasic(unittest.TestCase):
    """Core Trainer workflows under FakeGPU."""

    def setUp(self):
        self.model = DistilBertForSequenceClassification(_make_tiny_config())
        self.train_ds = _TinyDataset(32)
        self.eval_ds = _TinyDataset(16)

    def test_training_loop(self):
        """Basic training completes without error."""
        trainer = Trainer(
            model=self.model,
            args=_base_training_args(),
            train_dataset=self.train_ds,
        )
        result = trainer.train()
        self.assertIsInstance(result.training_loss, float)
        self.assertGreater(result.training_loss, 0)

    def test_evaluation(self):
        """Evaluation with compute_metrics returns accuracy."""

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float((preds == labels).mean())}

        trainer = Trainer(
            model=self.model,
            args=_base_training_args(
                eval_strategy="epoch",
                per_device_eval_batch_size=8,
            ),
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertIn("eval_loss", metrics)
        self.assertIn("eval_accuracy", metrics)

    def test_checkpoint_save_load(self):
        """Model can be saved and reloaded from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=self.model,
                args=_base_training_args(output_dir=tmpdir),
                train_dataset=self.train_ds,
            )
            trainer.train()
            trainer.save_model(tmpdir)

            loaded = DistilBertForSequenceClassification.from_pretrained(tmpdir)
            orig_params = sum(p.numel() for p in self.model.parameters())
            loaded_params = sum(p.numel() for p in loaded.parameters())
            self.assertEqual(orig_params, loaded_params)

    def test_predict(self):
        """Trainer.predict() returns logits with correct shape."""
        trainer = Trainer(
            model=self.model,
            args=_base_training_args(),
        )
        pred = trainer.predict(self.eval_ds)
        self.assertEqual(pred.predictions.shape, (len(self.eval_ds), 2))


if __name__ == "__main__":
    unittest.main()
