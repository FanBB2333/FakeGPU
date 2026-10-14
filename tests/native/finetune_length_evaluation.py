"""CPU native Trainer audit: epoch-seeded training and repeatable grouped eval."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("huggingface", "llamafactory", "axolotl"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", AXOLOTL_DO_NOT_TRACK="1")
    import torch
    import transformers
    from fakegpu._finetune_length_sampler import install_length_grouping
    from fakegpu._finetune_sampler import NativeBatchMatcher, plan_native_batches

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.dropout = torch.nn.Dropout(.25)
            self.output = torch.nn.Linear(8, 128)

        def forward(self, input_ids, attention_mask=None, labels=None):
            logits = self.output(self.dropout(self.embedding(input_ids)))
            return {"loss": torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1)), "logits": logits}

    class Epoch(transformers.TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.trainer.epoch_number += 1
            self.trainer.microstep = 0

    class Capture(transformers.Trainer):
        def training_step(self, model, inputs, *args, **kwargs):
            self.microstep += 1
            self.matcher.observe(inputs, epoch_index=self.epoch_number, epoch_microstep=self.microstep)
            return super().training_step(model, inputs, *args, **kwargs)

        def evaluate(self, *args, **kwargs):
            self.evaluations.append({"step": self.state.global_step, "row_ids": [], "batches": []})
            return super().evaluate(*args, **kwargs)

        def prediction_step(self, model, inputs, *args, **kwargs):
            event = self.evaluations[-1]
            event["row_ids"].extend(inputs["input_ids"][:, 0].tolist())
            content = {key: value.detach().cpu().tolist() for key, value in inputs.items()}
            event["batches"].append(hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            return super().prediction_step(model, inputs, *args, **kwargs)

    def order(lengths, width, seed):
        indices = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(seed)).tolist()
        width *= max(1, min(len(lengths) // (width * 4), 50))
        groups = [sorted(indices[start:start + width], key=lambda row: lengths[row], reverse=True)
                  for start in range(0, len(indices), width)]
        largest = max(range(len(groups)), key=lambda index: lengths[groups[index][0]])
        groups[0][0], groups[largest][0] = groups[largest][0], groups[0][0]
        return [row for group in groups for row in group]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.pad_token_id = 0
    if args.framework == "huggingface":
        collator = transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
        native_type = transformers.Trainer
    elif args.framework == "llamafactory":
        from llamafactory.data import SFTDataCollatorWith4DAttentionMask
        from llamafactory.data.template import TEMPLATES
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        collator = SFTDataCollatorWith4DAttentionMask(tokenizer, template=TEMPLATES["qwen3_nothink"],
                                                   pad_to_multiple_of=8, block_diag_attn=False, neat_packing=False)
        native_type = CustomSeq2SeqTrainer
    else:
        from axolotl.utils.collators import DataCollatorForSeq2Seq
        from axolotl.core.trainers.base import AxolotlTrainer
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        native_type = AxolotlTrainer

    report = {"framework": args.framework, "transformers": transformers.__version__,
              "torch": torch.__version__, "cases": [],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Real CPU Trainer, native framework sampler methods and collators. Not GPU cost accuracy."}
    for seed, persistent in ((0, False), (123, True)):
        training = [{"input_ids": [index + 1] * (3 + index * 7 % 53),
                     "attention_mask": [1] * (3 + index * 7 % 53),
                     "labels": [index + 1] * (3 + index * 7 % 53)} for index in range(17)]
        evaluation = deepcopy(training[2:13:2])
        arguments = transformers.TrainingArguments(
            output_dir=str(args.output.parent/f"model-{seed}"), use_cpu=True,
            train_sampling_strategy="group_by_length", per_device_train_batch_size=3,
            gradient_accumulation_steps=2, per_device_eval_batch_size=2,
            num_train_epochs=2.5, eval_strategy="steps", eval_steps=2,
            save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True,
            dataloader_num_workers=1 if persistent else 0,
            dataloader_persistent_workers=persistent, dataloader_prefetch_factor=32 if persistent else None,
            dataloader_pin_memory=False, optim="adamw_torch", seed=42,
        )
        trainer = Capture(model=Model(), args=arguments, train_dataset=training,
                          eval_dataset=evaluation, data_collator=collator)
        trainer.epoch_number, trainer.microstep, trainer.evaluations = 0, 0, []
        trainer.add_callback(Epoch(trainer))
        if native_type is not transformers.Trainer:
            adapter = native_type.__new__(native_type)
            adapter.args, adapter.train_dataset, adapter.processing_class = arguments, training, None
            if args.framework == "llamafactory":
                adapter.finetuning_args = SimpleNamespace(disable_shuffling=False)
            else:
                arguments.curriculum_sampling = arguments.sample_packing = arguments.pretraining = False
            trainer._get_train_sampler = adapter._get_train_sampler
            trainer._get_eval_sampler = adapter._get_eval_sampler
        policy = install_length_grouping(trainer, seed=seed, replace=setattr)
        steps_per_epoch = math.ceil(math.ceil(len(training)/3)/2)
        steps = math.ceil(2.5 * steps_per_epoch)
        plan = plan_native_batches(trainer, max_steps=steps+steps_per_epoch, framework=args.framework, sequence_length=64)
        for epoch in range(math.ceil((steps+steps_per_epoch)/steps_per_epoch)):
            expected = order([len(row["input_ids"]) for row in training], 6, seed+epoch)
            actual = [index for update in plan["updates"] for batch in update["batches"]
                      if batch["epoch_index"] == epoch+1 for index in batch["row_indices"]]
            assert actual == expected[:len(actual)]
        trainer.matcher = NativeBatchMatcher(plan)
        trainer.train()
        trainer.evaluate()
        torch.rand(357)
        trainer.evaluate()
        expected_eval = [evaluation[index]["input_ids"][0]
                         for index in order([len(row["input_ids"]) for row in evaluation], 2, seed)]
        assert all(event["row_ids"] == expected_eval for event in trainer.evaluations)
        assert all(event["batches"] == trainer.evaluations[0]["batches"] for event in trainer.evaluations)
        verification = trainer.matcher.finish(expected_updates=steps)
        assert verification["matched"]
        report["cases"].append({"seed":seed, "persistent_workers":persistent, "policy":policy,
                                "evaluations":trainer.evaluations, "training_verification":verification,
                                "cpu_plan":plan, "passed":True})
        # Cached worker processes should not outlive this audit case.
        del trainer
        gc.collect()
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed":True, "framework":args.framework, "cases":len(report["cases"])}))


if __name__ == "__main__":
    main()
