"""Run real CPU Trainer accounting checks in a fresh, CUDA-hidden process.

The pytest caller checks optional dependency versions before launching this
script. Import failures here are failures, not reasons to skip the regression.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import transformers

from fakegpu.finetune_config import normalize_finetune_config
from fakegpu.finetune_estimator import estimate_finetuning


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)
        self.batch_sizes = []

    def forward(self, input_ids, labels=None):
        self.batch_sizes.append(input_ids.shape[0])
        logits = self.output(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 8), labels.reshape(-1)
        )
        return {"loss": loss, "logits": logits}


def main(output_dir: Path) -> None:
    torch.set_num_threads(1)
    cases = [
        # (samples, batch size, accumulation, epochs, max_steps, drop_last)
        (19, 2, 4, 2.0, None, False),
        (19, 2, 4, 2.0, None, True),
        (19, 2, 4, 0.5, None, False),
        (19, 2, 4, 1.5, None, False),
        (19, 2, 4, 100.0, 4, False),
        (19, 2, 4, 0.1, 7, True),
        (3, 2, 8, 2.0, None, False),
        (3, 2, 8, 2.0, None, True),
        (1, 8, 4, 1.0, None, False),
        (16, 2, 4, 1.0, None, False),
        (13, 3, 1, 0.2, None, False),
        (13, 3, 3, 2.1, None, True),
    ]
    model_config = {
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
    }
    results = []
    for index, (
        samples,
        batch,
        accumulation,
        epochs,
        max_steps,
        drop_last,
    ) in enumerate(cases):
        model = TinyModel()
        dataset = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]} for _ in range(samples)
        ]
        arguments = transformers.TrainingArguments(
            output_dir=str(output_dir / str(index)),
            use_cpu=True,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accumulation,
            num_train_epochs=epochs,
            max_steps=max_steps or -1,
            dataloader_drop_last=drop_last,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
        )
        trainer = transformers.Trainer(
            model=model, args=arguments, train_dataset=dataset
        )
        trainer.train()
        config = normalize_finetune_config(
            {
                "framework": "huggingface",
                "model_name_or_path": "fixture/counting-model",
                "sequence_length": 3,
                "num_samples": samples,
                "per_device_train_batch_size": batch,
                "gradient_accumulation_steps": accumulation,
                "num_train_epochs": epochs,
                "max_steps": max_steps,
                "dataloader_drop_last": drop_last,
            }
        )
        work = estimate_finetuning(model_config, config, profile_id="a100")["workload"]
        observed = {
            "optimizer_steps": trainer.state.global_step,
            "microsteps": len(model.batch_sizes),
            "processed_samples": sum(model.batch_sizes),
        }
        for key, actual in observed.items():
            assert work[key] == actual, (cases[index], key, work[key], actual)
        assert next(model.parameters()).device.type == "cpu"
        results.append({"input": cases[index], "observed": observed})
    (output_dir / "trainer-counts.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cases": results,
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
