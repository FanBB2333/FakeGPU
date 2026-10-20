"""Local Hugging Face Trainer execution on explicitly processed text rows.

The adapter owns model/data construction only. NativeTrainerObserver supplies
all timing, memory and event evidence through the shared runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import json
from pathlib import Path

from .finetune_data import _token_list


# These options are consumed by the shared normalizer or this adapter. Remaining
# keys must be constructor fields of the installed TrainingArguments dataclass.
_ADAPTER_OPTIONS = {
    "framework",
    "model_name_or_path",
    "base_model",
    "model_revision",
    "revision",
    "checkpoint_sha256",
    "parameter_dtype",
    "torch_dtype",
    "dtype",
    "attention_implementation",
    "attn_implementation",
    "method",
    "finetuning_type",
    "adapter",
    "peft",
    "peft_config",
    "lora_rank",
    "lora_r",
    "lora_target_modules",
    "lora_target",
    "lora_target_linear",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "quantization_bits",
    "quantization_bit",
    "quantization_method",
    "quantization_config",
    "bnb_config_kwargs",
    "load_in_4bit",
    "load_in_8bit",
    "quantization_type",
    "double_quantization",
    "quantization_compute_dtype",
    "quantization_storage",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_use_double_quant",
    "bnb_4bit_quant_storage",
    "sequence_length",
    "max_seq_length",
    "seq_length",
    "sequence_len",
    "cutoff_len",
    "block_size",
    "micro_batch_size",
    "batch_size",
    "epochs",
    "num_epochs",
    "optimizer",
    "lr_scheduler",
    "world_size",
    "num_processes",
    "disable_gradient_checkpointing",
    "packing",
    "sample_packing",
    "stage",
    "train_dataset",
    "eval_dataset",
    "disable_shuffling",
    "pad_to_multiple_of",
    "num_samples",
    "dataset_num_samples",
    "dataset_samples",
    "trust_remote_code",
}
_PEFT_OPTIONS = {
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "target_modules",
    "lora_dropout",
    "bias",
}
_UNSUPPORTED_ARGUMENTS = {
    "resume_from_checkpoint",
    "deepspeed",
    "fsdp",
    "fsdp_config",
    "accelerator_config",
    "auto_find_batch_size",
    "push_to_hub",
    "do_predict",
    "use_liger_kernel",
    "liger_kernel_config",
    "torch_empty_cache_steps",
}


def _dataset_path(value, base_dir, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hugging Face {field} must name a local processed JSONL file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    if path.suffix.lower() != ".jsonl" or not path.is_file():
        raise ValueError(
            f"Hugging Face {field} must be an existing processed .jsonl file"
        )
    return path


def _argument_options(raw, normalized, argument_type):
    supported = {field.name for field in fields(argument_type) if field.init}
    unknown = set(raw) - supported - _ADAPTER_OPTIONS
    if unknown:
        raise ValueError(
            "unsupported Hugging Face native configuration fields: "
            + ", ".join(sorted(unknown))
        )
    for key in sorted(_UNSUPPORTED_ARGUMENTS):
        if raw.get(key):
            raise ValueError(f"Hugging Face native runner does not support {key}")
    if raw.get("do_train") is False:
        raise ValueError(
            "Hugging Face native runner requires training; do_train=false is unsupported"
        )
    if raw.get("disable_shuffling") and (
        raw.get("group_by_length")
        or raw.get("train_sampling_strategy") == "group_by_length"
    ):
        raise ValueError("disable_shuffling conflicts with group_by_length")
    checkpointing = normalized.get("gradient_checkpointing_kwargs")
    if (
        checkpointing is None
        and normalized["gradient_checkpointing"]
        and normalized["method"] != "full"
    ):
        checkpointing = {"use_reentrant": False}
    options = {
        "report_to": [],
        "disable_tqdm": True,
        "logging_nan_inf_filter": False,
        **{key: value for key, value in raw.items() if key in supported},
        "per_device_train_batch_size": normalized["micro_batch_size"],
        "gradient_accumulation_steps": normalized["gradient_accumulation_steps"],
        "num_train_epochs": normalized["epochs"],
        "max_steps": normalized["max_steps"]
        if normalized["max_steps"] is not None
        else -1,
        "bf16": normalized["dtype"] == "bf16",
        "fp16": normalized["dtype"] == "fp16",
        "tf32": normalized["tf32"],
        "gradient_checkpointing": normalized["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": checkpointing,
        "optim": normalized["optimizer"],
        "learning_rate": normalized["learning_rate"],
        "lr_scheduler_type": normalized["lr_scheduler_type"],
        "warmup_steps": normalized["warmup_steps"],
        "warmup_ratio": normalized["warmup_ratio"],
        "weight_decay": normalized["weight_decay"],
        "max_grad_norm": normalized["max_grad_norm"],
        "dataloader_drop_last": normalized["dataloader_drop_last"],
    }
    return options


def validate_huggingface_config(raw, normalized, base_dir):
    """Validate the flat native contract before any model or dataset is loaded."""
    if any(key in raw for key in ("model_args", "training_args", "data_args")):
        raise ValueError(
            "Hugging Face native execution requires a flat configuration; nested model_args/training_args/data_args remain available for offline estimation only"
        )
    for key in ("disable_shuffling", "trust_remote_code"):
        if key in raw and not isinstance(raw[key], bool):
            raise ValueError(f"Hugging Face {key} must be boolean")
    if raw.get("trust_remote_code"):
        raise ValueError(
            "Hugging Face native runner does not support trust_remote_code"
        )
    multiple = raw.get("pad_to_multiple_of", 8)
    if multiple is not None and (
        isinstance(multiple, bool) or not isinstance(multiple, int) or multiple < 1
    ):
        raise ValueError(
            "Hugging Face pad_to_multiple_of must be a positive integer or null"
        )
    _dataset_path(raw.get("train_dataset"), base_dir, "train_dataset")
    if raw.get("eval_dataset") is not None:
        _dataset_path(raw["eval_dataset"], base_dir, "eval_dataset")
    strategy = raw.get("eval_strategy", "no")
    if strategy != "no" and raw.get("eval_dataset") is None:
        raise ValueError("Hugging Face periodic evaluation requires eval_dataset")
    if raw.get("do_eval") and strategy == "no":
        raise ValueError(
            "Hugging Face do_eval=true requires an eval_strategy; native finalization only saves the model"
        )
    for key in ("peft", "peft_config"):
        if key in raw:
            if not isinstance(raw[key], Mapping):
                raise ValueError(f"Hugging Face {key} must be a mapping")
            unknown = set(raw[key]) - _PEFT_OPTIONS
            if unknown:
                raise ValueError(
                    "unsupported Hugging Face PEFT fields: "
                    + ", ".join(sorted(unknown))
                )
    from transformers import TrainingArguments

    _argument_options(raw, normalized, TrainingArguments)


def _load_processed_dataset(path, sequence_length, *, packed=False):
    records = []
    with Path(path).expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = {
                "input_ids",
                "attention_mask",
                "labels",
            }
            if packed:
                expected.add("position_ids")
            if not isinstance(row, dict) or set(row) != expected:
                fields = (
                    "input_ids, attention_mask, labels and position_ids"
                    if packed
                    else "input_ids, attention_mask and labels"
                )
                raise ValueError(
                    f"processed dataset row {line_number} requires exactly {fields}"
                )
            for key in row:
                row[key] = _token_list(row[key], key, labels=key == "labels")
            length = len(row["input_ids"])
            if not 2 <= length <= sequence_length:
                raise ValueError(
                    f"processed dataset row {line_number} length must be between 2 and sequence_length; native execution does not truncate"
                )
            if any(len(row[key]) != length for key in ("attention_mask", "labels")):
                raise ValueError(
                    f"processed dataset row {line_number} mask and labels must match input_ids length"
                )
            if packed:
                from ._finetune_packed_batches import packed_source_membership

                packed_source_membership([row], framework="huggingface")
            elif any(value not in (0, 1) for value in row["attention_mask"]):
                raise ValueError(
                    f"processed dataset row {line_number} requires a binary attention_mask; packing is unsupported"
                )
            records.append(row)
    if not records:
        raise ValueError("Hugging Face processed dataset must contain at least one row")
    return records


def _load_model(normalized, torch, transformers):
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    options = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtypes[normalized["parameter_dtype"]],
        "attn_implementation": normalized["attention_implementation"],
    }
    if normalized["method"] == "qlora":
        options["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=normalized["quantization_type"],
            bnb_4bit_use_double_quant=normalized["double_quantization"],
            bnb_4bit_compute_dtype=dtypes[normalized["quantization_compute_dtype"]],
            bnb_4bit_quant_storage=(
                torch.uint8
                if normalized["quantization_storage"] == "uint8"
                else dtypes[normalized["quantization_storage"]]
            ),
        )
        options["device_map"] = {"": 0}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        normalized["model_name_or_path"], **options
    )
    model.config.use_cache = False
    if normalized["method"] in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if normalized["method"] == "qlora":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=normalized["gradient_checkpointing"],
                gradient_checkpointing_kwargs=normalized.get(
                    "gradient_checkpointing_kwargs", {"use_reentrant": False}
                ),
            )
        targets = normalized["lora_target_modules"]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=normalized["lora_rank"],
                lora_alpha=normalized["lora_alpha"],
                target_modules="all-linear" if targets == ["all-linear"] else targets,
                lora_dropout=normalized["lora_dropout"],
                bias=normalized["lora_bias"],
            ),
        )
    return model


def invoke_huggingface(raw, normalized):
    """Train and save through the ordinary Trainer lifecycle and shared hooks."""
    import torch
    import transformers
    from torch.utils.data import SequentialSampler

    options = _argument_options(raw, normalized, transformers.TrainingArguments)
    arguments = transformers.TrainingArguments(**options)
    transformers.set_seed(arguments.seed)
    train_dataset = _load_processed_dataset(
        raw["train_dataset"],
        normalized["sequence_length"],
        packed=normalized["packing"],
    )
    eval_dataset = (
        _load_processed_dataset(raw["eval_dataset"], normalized["sequence_length"])
        if raw.get("eval_dataset") is not None
        else None
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        normalized["model_name_or_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Hugging Face native dynamic padding requires a saved tokenizer pad token"
        )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=raw.get("pad_to_multiple_of", 8),
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    model = _load_model(normalized, torch, transformers)

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(
                self.train_dataset if train_dataset is None else train_dataset
            )

    trainer_type = (
        SequentialTrainer
        if raw.get("disable_shuffling", False)
        else transformers.Trainer
    )
    trainer = trainer_type(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
