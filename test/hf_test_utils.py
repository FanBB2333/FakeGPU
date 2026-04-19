from __future__ import annotations

import os
import inspect
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
    import torch

    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
        if hasattr(torch.backends.mps, "is_built"):
            torch.backends.mps.is_built = lambda: False
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        torch.mps.is_available = lambda: False


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


def make_tiny_causal_lm(*, vocab_size: int = 128):
    from transformers import AutoModelForCausalLM

    config = make_tiny_qwen_config()
    config.vocab_size = vocab_size
    return AutoModelForCausalLM.from_config(config)


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


def make_tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "User": 4,
        "Assistant": 5,
        ":": 6,
        "hello": 7,
        "world": 8,
        "what": 9,
        "is": 10,
        "1": 11,
        "+": 12,
        "2": 13,
        "?": 14,
        "The": 15,
        "answer": 16,
        "blue": 17,
        "green": 18,
        "sky": 19,
        "hi": 20,
        "there": 21,
        ".": 22,
    }

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


def make_sft_dataset():
    from datasets import Dataset

    rows = [
        {
            "prompt": "User: what is 1 + 1 ?",
            "completion": "Assistant: The answer is 2 .",
            "text": "User: what is 1 + 1 ? Assistant: The answer is 2 .",
        },
        {
            "prompt": "User: hello world",
            "completion": "Assistant: hi there .",
            "text": "User: hello world Assistant: hi there .",
        },
        {
            "prompt": "User: what color is the sky ?",
            "completion": "Assistant: The sky is blue .",
            "text": "User: what color is the sky ? Assistant: The sky is blue .",
        },
        {
            "prompt": "User: what is 1 + 1 ?",
            "completion": "Assistant: 2 .",
            "text": "User: what is 1 + 1 ? Assistant: 2 .",
        },
    ]
    return Dataset.from_list(rows)


def make_dpo_dataset():
    from datasets import Dataset

    rows = [
        {
            "prompt": "User: what is 1 + 1 ?",
            "chosen": "Assistant: The answer is 2 .",
            "rejected": "Assistant: The answer is green .",
        },
        {
            "prompt": "User: what color is the sky ?",
            "chosen": "Assistant: The sky is blue .",
            "rejected": "Assistant: The sky is green .",
        },
    ]
    return Dataset.from_list(rows)


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
    supported = inspect.signature(TrainingArguments.__init__).parameters
    filtered = {key: value for key, value in defaults.items() if key in supported}
    return TrainingArguments(**filtered)
