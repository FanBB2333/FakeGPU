from __future__ import annotations

import tempfile

import pytest

from hf_test_utils import (
    make_sft_dataset,
    make_tiny_causal_lm,
    make_tiny_tokenizer,
    patch_fakegpu,
)


def test_trl_sft_tiny_flow() -> None:
    trl = pytest.importorskip("trl")
    from transformers import AutoModelForCausalLM

    patch_fakegpu(profile="a100", device_count=1)

    tokenizer = make_tiny_tokenizer()
    model = make_tiny_causal_lm(vocab_size=len(tokenizer))

    with tempfile.TemporaryDirectory(prefix="fakegpu_sft_") as tmpdir:
        trainer = trl.SFTTrainer(
            model=model,
            args=trl.SFTConfig(
                output_dir=tmpdir,
                max_steps=1,
                per_device_train_batch_size=2,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=False,
                tf32=True,
                dataloader_pin_memory=False,
                gradient_checkpointing=False,
                max_length=32,
                dataset_text_field="text",
            ),
            train_dataset=make_sft_dataset(),
            processing_class=tokenizer,
        )
        result = trainer.train()
        assert isinstance(result.training_loss, float)
        assert result.training_loss > 0

        trainer.model.save_pretrained(tmpdir)
        reloaded = AutoModelForCausalLM.from_pretrained(tmpdir).cuda()
        batch = tokenizer("User: hello world", return_tensors="pt")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        outputs = reloaded(**batch)
        assert outputs.logits.device.type == "cuda"
