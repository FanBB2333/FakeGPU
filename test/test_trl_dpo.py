from __future__ import annotations

import math
import tempfile

import pytest

from hf_test_utils import (
    make_dpo_dataset,
    make_tiny_causal_lm,
    make_tiny_tokenizer,
    patch_fakegpu,
)


def test_trl_dpo_tiny_flow() -> None:
    trl = pytest.importorskip("trl")

    patch_fakegpu(profile="a100", device_count=1)

    tokenizer = make_tiny_tokenizer()
    model = make_tiny_causal_lm(vocab_size=len(tokenizer))
    ref_model = make_tiny_causal_lm(vocab_size=len(tokenizer))

    with tempfile.TemporaryDirectory(prefix="fakegpu_dpo_") as tmpdir:
        trainer = trl.DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=trl.DPOConfig(
                output_dir=tmpdir,
                max_steps=1,
                per_device_train_batch_size=1,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=False,
                tf32=True,
                dataloader_pin_memory=False,
                gradient_checkpointing=False,
                max_length=32,
            ),
            train_dataset=make_dpo_dataset(),
            processing_class=tokenizer,
        )
        result = trainer.train()
        assert isinstance(result.training_loss, float)
        assert math.isfinite(result.training_loss)

        trainer.model.save_pretrained(tmpdir)
