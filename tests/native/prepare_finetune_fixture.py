"""Export a local Qwen3.5 text checkpoint and synthetic framework SFT data.

Run with a Transformers version supporting Qwen3_5ForCausalLM, for example::

    python tests/native/prepare_finetune_fixture.py \
        --model /models/Qwen3.5-0.8B \
        --output-dir /tmp/fakegpu-phase2/qwen35-text \
        --dataset-dir /tmp/fakegpu-phase2/data

The source checkpoint is read only. Export and verification always run on CPU;
no training or accuracy claim is made by this fixture preparation script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verify_loading(info: dict, stage: str) -> dict[str, list]:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    problems = {key: info[key] for key in keys if info.get(key)}
    if problems:
        raise RuntimeError(f"{stage} checkpoint loading was not exact: {problems}")
    # Transformers releases use either sets or lists for empty loading issues.
    return {key: [] for key in keys}


def prepare(model_dir: Path, output_dir: Path, dataset_dir: Path) -> dict:
    # Set this before importing Torch so this script cannot initialize a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("output_dir must be outside the original checkpoint")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be absent or empty")
    if dataset_dir == model_dir or model_dir in dataset_dir.parents:
        raise ValueError("dataset_dir must be outside the original checkpoint")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (dataset_dir / name).exists() for name in ("train.jsonl", "dataset_info.json")
    ):
        raise ValueError("dataset files already exist; choose a new dataset_dir")
    torch.set_num_threads(4)
    original = json.loads((model_dir / "config.json").read_text())
    if original.get("model_type") != "qwen3_5" or not original.get("text_config"):
        raise ValueError(
            "this fixture exporter requires a local Qwen3.5 multimodal checkpoint"
        )
    text_config = dict(original["text_config"])
    architecture = AutoConfig.for_model(text_config.pop("model_type"), **text_config)
    architecture.use_cache = False
    model, source_loading = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=architecture,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    source_loading = _verify_loading(source_loading, "source text")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("fixture export must use CPU parameters")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    tokens = tokenizer.encode(
        "Question: What is 2 + 3? Answer: 5.", add_special_tokens=False
    )
    input_ids = torch.tensor([tokens[:8]], dtype=torch.long)
    if input_ids.shape != (1, 8):
        raise RuntimeError("expected at least eight verification tokens")
    with torch.inference_mode():
        original_logits = model(input_ids=input_ids, use_cache=False).logits.float()
    if not torch.isfinite(original_logits).all():
        raise RuntimeError("source text model produced nonfinite logits")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model
    gc.collect()

    reloaded, exported_loading = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    exported_loading = _verify_loading(exported_loading, "exported text")
    if reloaded.config.model_type != "qwen3_5_text":
        raise RuntimeError("export must resolve to the pure text model")
    if getattr(reloaded.config, "vision_config", None) is not None:
        raise RuntimeError("export unexpectedly includes a vision configuration")
    if sum(parameter.numel() for parameter in reloaded.parameters()) != parameter_count:
        raise RuntimeError("parameter count changed after export")
    reloaded.eval()
    with torch.inference_mode():
        exported_logits = reloaded(input_ids=input_ids, use_cache=False).logits.float()
    torch.testing.assert_close(exported_logits, original_logits, rtol=0, atol=0)
    if list(exported_logits.shape) != [1, 8, reloaded.config.vocab_size]:
        raise RuntimeError("export produced an unexpected logits shape")
    if torch.cuda.is_initialized():
        raise RuntimeError("fixture preparation unexpectedly initialized CUDA")

    records = []
    for index in range(64):
        left, right = index % 13, index % 7
        records.append(
            {
                "instruction": f"What is {left} + {right}? Explain the arithmetic.",
                "input": "Use whole numbers.",
                "output": (
                    f"Adding {left} and {right} gives {left + right}. " * 48
                ).strip(),
            }
        )
    lengths = [
        len(tokenizer.encode("\n".join(record.values()), add_special_tokens=False))
        for record in records
    ]
    if min(lengths) <= 128:
        raise RuntimeError("all synthetic samples must exceed the training cutoff")
    serialized = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    (dataset_dir / "train.jsonl").write_text(serialized)
    _write_json(
        dataset_dir / "dataset_info.json",
        {
            "synthetic_arithmetic": {
                "file_name": "train.jsonl",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        },
    )
    manifest = {
        "source_model_dir": str(model_dir),
        "exported_model_dir": str(output_dir),
        "model_class": type(reloaded).__name__,
        "model_type": reloaded.config.model_type,
        "parameter_count": parameter_count,
        "parameter_dtype": str(next(reloaded.parameters()).dtype),
        "source_loading_info": source_loading,
        "exported_loading_info": exported_loading,
        "verification_input_ids": input_ids.tolist(),
        "verification_logits_shape": list(exported_logits.shape),
        "verification_logits_exact_match": True,
        "cuda_initialized": torch.cuda.is_initialized(),
        "dataset_file": str(dataset_dir / "train.jsonl"),
        "dataset_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "dataset_samples": len(records),
        "raw_concatenated_token_length_min": min(lengths),
        "raw_concatenated_token_length_max": max(lengths),
        "software_versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "safetensors", "tokenizers")
        },
    }
    _write_json(output_dir / "fixture_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.output_dir, args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
