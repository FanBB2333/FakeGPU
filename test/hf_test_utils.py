from __future__ import annotations

import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def patch_fakegpu(*, profile: str = "a100", device_count: int = 1) -> None:
    os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TORCH_SDPA_KERNEL", "math")
    os.environ["FAKEGPU_STRICT_COMPAT"] = "1"
    os.environ["FAKEGPU_PROFILES"] = f"{profile}:{device_count}"
    os.environ["FAKEGPU_DEVICE_COUNT"] = str(device_count)

    import fakegpu

    fakegpu.patch_torch()


def make_tiny_qwen_config():
    from transformers import AutoConfig

    return AutoConfig.for_model(
        "qwen2",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=False,
    )


def make_tiny_causal_lm():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_config(make_tiny_qwen_config())


def make_lm_dataset(*, size: int = 8, seq_len: int = 16, vocab_size: int = 128):
    import torch

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return size

        def __getitem__(self, index: int):
            input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone(),
            }

    return _Dataset()


def make_training_args(**overrides):
    from transformers import TrainingArguments

    defaults = dict(
        output_dir=tempfile.mkdtemp(prefix="fakegpu_hf_cuda_"),
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        no_cuda=False,
        use_cpu=False,
        tf32=True,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)
