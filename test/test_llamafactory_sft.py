"""LLaMA-Factory LoRA SFT / DPO compatibility — tiny model, fake CUDA."""
from __future__ import annotations

import tempfile

import pytest

from hf_test_utils import (
    make_dpo_dataset,
    make_sft_dataset,
    make_tiny_causal_lm,
    make_tiny_tokenizer,
    patch_fakegpu,
)


def test_llamafactory_style_lora_sft() -> None:
    """Simulate the LLaMA-Factory LoRA SFT workflow with underlying libraries."""
    peft = pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    patch_fakegpu(profile="a100", device_count=1)

    tokenizer = make_tiny_tokenizer()
    model = make_tiny_causal_lm(vocab_size=len(tokenizer))

    # LLaMA-Factory applies LoRA then uses SFTTrainer
    peft_model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="fakegpu_llama_factory_") as tmpdir:
        trainer = trl.SFTTrainer(
            model=peft_model,
            args=trl.SFTConfig(
                output_dir=tmpdir,
                max_steps=2,
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

        # Save adapter (LLaMA-Factory pattern)
        trainer.model.save_pretrained(tmpdir)

        # Reload and merge (LLaMA-Factory export pattern)
        base = make_tiny_causal_lm(vocab_size=len(tokenizer))
        reloaded = peft.PeftModel.from_pretrained(base, tmpdir, torch_device="cpu")
        merged = reloaded.merge_and_unload().cuda()

        batch = tokenizer("User: hello world", return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = merged(**batch)
        assert outputs.logits.device.type == "cuda"


def test_llamafactory_style_lora_dpo() -> None:
    """Simulate the LLaMA-Factory LoRA DPO workflow."""
    peft = pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    patch_fakegpu(profile="a100", device_count=1)

    tokenizer = make_tiny_tokenizer()
    model = make_tiny_causal_lm(vocab_size=len(tokenizer))
    ref_model = make_tiny_causal_lm(vocab_size=len(tokenizer))

    peft_model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="fakegpu_llama_dpo_") as tmpdir:
        trainer = trl.DPOTrainer(
            model=peft_model,
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
