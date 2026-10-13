"""Exercise the Hugging Face adapter with real CPU autograd in a fresh process."""

import json
from pathlib import Path
import sys

import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from fakegpu._finetune_hf_runner import invoke_huggingface, validate_huggingface_config
from fakegpu._finetune_observer import NativeTrainerObserver, capture_native_trainer
from fakegpu._finetune_timeline import build_native_timeline
from fakegpu.finetune_config import normalize_finetune_config

from finetune_native_batch_counts import ObserverTorch


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "model"
    torch.manual_seed(7)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).save_pretrained(model_dir)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    vocab.update({f"token{i}": i for i in range(3, 32)})
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    records = [
        {
            "input_ids": [index + 3] * length,
            "attention_mask": [1] * length,
            "labels": [-100] + [index + 3] * (length - 1),
        }
        for index, length in enumerate((2, 3, 5, 7, 9))
    ]
    data = root / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in records))
    reports = {}
    for method in ("full", "lora"):
        raw = {
            "model_name_or_path": str(model_dir),
            "train_dataset": str(data),
            "eval_dataset": str(data),
            "output_dir": str(root / method),
            "method": method,
            "sequence_length": 16,
            "parameter_dtype": "fp32",
            "attn_implementation": "eager",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_steps": 5,
            "optim": "adamw_torch",
            "learning_rate": 0.001,
            "disable_shuffling": True,
            "pad_to_multiple_of": 8,
            "eval_strategy": "steps",
            "eval_steps": 2,
            "save_strategy": "steps",
            "save_steps": 2,
            "logging_strategy": "no",
            "report_to": [],
            "use_cpu": True,
            "dataloader_pin_memory": False,
            "seed": 42,
        }
        if method == "lora":
            raw["peft"] = {"r": 2, "lora_alpha": 4, "target_modules": ["c_attn"]}
        normalized = normalize_finetune_config(
            raw, framework="huggingface", base_dir=root
        )
        validate_huggingface_config(raw, normalized, root)
        observer = NativeTrainerObserver(
            ObserverTorch(),
            warmup_steps=1,
            on_trainer=lambda _: None,
            on_train_begin=lambda _: None,
        )
        with capture_native_trainer(
            transformers.Trainer, transformers.TrainerCallback, observer
        ):
            invoke_huggingface(raw, normalized)
        assert observer.outcome is not None
        # The real Trainer, collator, SequentialSampler, optimizer and PEFT model
        # execute normally. Only CUDA timing/memory calls use the CPU facade.
        build_native_timeline(
            observer.updates,
            observer.phase_events,
            observer.instrumentation_intervals,
            trainer_start_seconds=observer.trainer_started - observer.run_started,
            trainer_end_seconds=observer.trainer_finished - observer.run_started,
            run_end_seconds=max(row["end_seconds"] for row in observer.phase_events),
        )
        roots = [row for row in observer.phase_events if row["parent_event_id"] is None]
        assert roots[-1]["operation"] == "save_model" and not roots[-1]["inside_train"]
        assert roots[-1]["workload"]["files_status"] == "known"
        assert not observer._restores
        model_file = (
            "adapter_model.safetensors" if method == "lora" else "model.safetensors"
        )
        reports[method] = {
            "updates": len(observer.updates),
            "microsteps": observer.microsteps,
            "samples": observer.samples,
            "ordered_lengths": [
                length
                for update in observer.updates
                for batch in update["batches"]
                for length in batch["row_nonpadding_tokens"]
            ],
            "weight_change": observer.parameter_changes[
                "has_changed_trainable_parameters"
            ],
            "saved": (root / method / model_file).is_file(),
        }
    (root / "native-hf-counts.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
