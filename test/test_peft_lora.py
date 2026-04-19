from __future__ import annotations

import tempfile

import pytest

from hf_test_utils import (
    make_lm_dataset,
    make_tiny_causal_lm,
    make_tiny_tokenizer,
    make_training_args,
    patch_fakegpu,
)


def test_peft_lora_tiny_flow() -> None:
    peft = pytest.importorskip("peft")
    from transformers import Trainer

    patch_fakegpu(profile="a100", device_count=1)

    tokenizer = make_tiny_tokenizer()
    model = make_tiny_causal_lm(vocab_size=len(tokenizer))
    peft_model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="fakegpu_peft_") as tmpdir:
        trainer = Trainer(
            model=peft_model,
            args=make_training_args(output_dir=tmpdir),
            train_dataset=make_lm_dataset(vocab_size=len(tokenizer)),
        )
        result = trainer.train()
        assert isinstance(result.training_loss, float)
        assert result.training_loss > 0

        trainer.model.save_pretrained(tmpdir)

        reloaded = peft.PeftModel.from_pretrained(
            make_tiny_causal_lm(vocab_size=len(tokenizer)),
            tmpdir,
            torch_device="cpu",
        )
        merged = reloaded.merge_and_unload().cuda()
        batch = tokenizer("User: hello world", return_tensors="pt")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        outputs = merged(**batch)
        assert outputs.logits.device.type == "cuda"
