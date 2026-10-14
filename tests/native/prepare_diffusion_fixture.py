"""Prepare small reproducible cached-feature fixtures without loading a GPU.

Latents, noise and text features are random synthetic tensors. They verify
resource measurement and integration; they are not image/caption training data
and do not establish convergence. The original model directory is read only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--method", choices=("full", "lora"), default="lora")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "num_samples",
        "height",
        "width",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
    ):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.height % 64 or args.width % 64:
        parser.error("height/width must be multiples of 64")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0,2**63)")
    model = args.model_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if model == out or model in out.parents:
        parser.error("output must be outside the model directory")
    if out.exists() and any(out.iterdir()):
        parser.error("choose a new or empty output directory")
    config = json.loads((model / "unet/config.json").read_text())
    scheduler = json.loads((model / "scheduler/scheduler_config.json").read_text())
    if config.get("_class_name") != "UNet2DConditionModel" or config.get(
        "addition_embed_type"
    ) not in (None, "text_time"):
        parser.error(
            "fixture requires a standard UNet with optional text_time conditioning"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from safetensors.torch import save_file
    from fakegpu.diffusion_training import plan_diffusion_training
    from fakegpu.structured_io import write_json

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    tensors = {
        "latents": torch.randn(args.num_samples, 4, args.height // 8, args.width // 8),
        "noise": torch.randn(args.num_samples, 4, args.height // 8, args.width // 8),
        "encoder_hidden_states": torch.randn(
            args.num_samples, 77, config["cross_attention_dim"]
        ),
        "timesteps": torch.randint(
            0, scheduler["num_train_timesteps"], (args.num_samples,), dtype=torch.int64
        ),
    }
    if config.get("addition_embed_type") == "text_time":
        width = (
            config["projection_class_embeddings_input_dim"]
            - 6 * config["addition_time_embed_dim"]
        )
        tensors.update(
            pooled_text_embeds=torch.randn(args.num_samples, width),
            time_ids=torch.tensor(
                [[args.height, args.width, 0, 0, args.height, args.width]]
                * args.num_samples,
                dtype=torch.float32,
            ),
        )
    out.mkdir(parents=True, exist_ok=True)
    dataset = out / "cached.safetensors"
    save_file(tensors, str(dataset))
    plan = plan_diffusion_training(
        model,
        dataset,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        method=args.method,
        lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        ema_decay=args.ema_decay,
    )
    write_json(out / "plan.json", plan)
    assert not torch.cuda.is_initialized()
    write_json(
        out / "fixture.json",
        {
            "scope": __doc__,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "height": args.height,
            "width": args.width,
            "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
            "plan_sha256": plan["plan_sha256"],
            "cuda_initialized": False,
        },
    )
    print(
        json.dumps(
            {
                "plan": str(out / "plan.json"),
                "dataset": str(dataset),
                "cuda_initialized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
