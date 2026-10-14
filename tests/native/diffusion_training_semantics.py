"""Check native UNet accumulation, checkpointing and weight reload on CPU.

Requires the optional Diffusers/PEFT environment. These tiny random models test
training semantics, not pretrained quality, GPU timing or allocator accuracy.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from peft import LoraConfig
    from fakegpu.diffusion_training import TARGETS
    from fakegpu.diffusion_training_runner import (
        _backward_window,
        _update_ema,
        _save_ema,
    )

    if args.work_dir.exists():
        raise ValueError("choose a new semantic-check work directory")
    args.work_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    records = []
    for text_time in (False, True):
        for method in ("full", "lora"):
            for checkpointing in (False, True):
                torch.manual_seed(42)
                options = (
                    {
                        "addition_embed_type": "text_time",
                        "addition_time_embed_dim": 8,
                        "projection_class_embeddings_input_dim": 64,
                    }
                    if text_time
                    else {}
                )
                base = UNet2DConditionModel(
                    sample_size=8,
                    in_channels=4,
                    out_channels=4,
                    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                    block_out_channels=(32, 64),
                    layers_per_block=1,
                    cross_attention_dim=16,
                    attention_head_dim=8,
                    norm_num_groups=8,
                    **options,
                )
                base_state = deepcopy(base.state_dict())
                model = deepcopy(base)
                if method == "lora":
                    model.requires_grad_(False)
                    model.add_adapter(
                        LoraConfig(
                            r=2,
                            lora_alpha=2,
                            target_modules=list(TARGETS),
                            bias="none",
                            lora_dropout=0.0,
                        )
                    )
                reference = deepcopy(model)
                if checkpointing:
                    model.enable_gradient_checkpointing()
                model.train()
                reference.train()
                scheduler = DDPMScheduler(num_train_timesteps=1000)
                batch = {
                    "latents": torch.randn(3, 4, 8, 8),
                    "noise": torch.randn(3, 4, 8, 8),
                    "timesteps": torch.tensor([1, 51, 999]),
                    "encoder_hidden_states": torch.randn(3, 5, 16),
                }
                if text_time:
                    batch.update(
                        pooled_text_embeds=torch.randn(3, 16),
                        time_ids=torch.tensor(
                            [[64, 64, 0, 0, 64, 64]] * 3, dtype=torch.float32
                        ),
                    )
                slices = [
                    {key: value[start:end] for key, value in batch.items()}
                    for start, end in ((0, 2), (2, 3))
                ]
                observed = _backward_window(
                    model,
                    scheduler,
                    slices,
                    torch=torch,
                    device=torch.device("cpu"),
                    autocast=False,
                )
                added = (
                    {
                        "text_embeds": batch["pooled_text_embeds"],
                        "time_ids": batch["time_ids"],
                    }
                    if text_time
                    else None
                )
                noisy = scheduler.add_noise(
                    batch["latents"], batch["noise"], batch["timesteps"]
                )
                output = reference(
                    noisy,
                    batch["timesteps"],
                    encoder_hidden_states=batch["encoder_hidden_states"],
                    added_cond_kwargs=added,
                ).sample
                expected = torch.nn.functional.mse_loss(output, batch["noise"])
                expected.backward()
                assert abs(observed - float(expected.detach())) < 1e-6
                maximum = 0.0
                for (name, left), (other, right) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    assert name == other and (left.grad is None) == (right.grad is None)
                    if left.grad is not None:
                        torch.testing.assert_close(
                            left.grad, right.grad, atol=3e-6, rtol=3e-4
                        )
                        maximum = max(
                            maximum, float((left.grad - right.grad).abs().max())
                        )
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=2e-5,
                    weight_decay=0.0,
                    foreach=False,
                )
                named = [
                    (name, p) for name, p in model.named_parameters() if p.requires_grad
                ]
                shadows = [p.detach().clone() for _, p in named]
                previous = [value.clone() for value in shadows]
                optimizer.step()
                _update_ema(shadows, [p for _, p in named], 0.9, torch=torch)
                for old, shadow, (_, parameter) in zip(previous, shadows, named):
                    torch.testing.assert_close(
                        shadow, old * 0.9 + parameter * 0.1, atol=1e-7, rtol=1e-6
                    )
                assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
                directory = args.work_dir / f"{text_time}-{method}-{checkpointing}"
                if method == "full":
                    model.save_pretrained(directory, safe_serialization=True)
                    reloaded = UNet2DConditionModel.from_pretrained(
                        directory, local_files_only=True
                    )
                else:
                    model.save_lora_adapter(str(directory), safe_serialization=True)
                    reloaded = UNet2DConditionModel.from_config(base.config)
                    reloaded.load_state_dict(base_state, strict=True)
                    reloaded.load_lora_adapter(
                        str(directory),
                        prefix=None,
                        weight_name="pytorch_lora_weights.safetensors",
                        use_safetensors=True,
                        local_files_only=True,
                    )
                trained = [p.detach().clone() for _, p in named]
                ema_directory = directory / "ema"
                _save_ema(
                    model,
                    [name for name, _ in named],
                    shadows,
                    ema_directory,
                    directory,
                    method=method,
                    decay=0.9,
                    updates=1,
                )
                assert all(
                    torch.equal(before, p) for before, (_, p) in zip(trained, named)
                )
                ema_expected = deepcopy(model)
                with torch.no_grad():
                    for p, shadow in zip(
                        [p for p in ema_expected.parameters() if p.requires_grad],
                        shadows,
                    ):
                        p.copy_(shadow)
                if method == "full":
                    ema_reloaded = UNet2DConditionModel.from_pretrained(
                        ema_directory, local_files_only=True
                    )
                else:
                    ema_reloaded = UNet2DConditionModel.from_config(base.config)
                    ema_reloaded.load_state_dict(base_state, strict=True)
                    ema_reloaded.load_lora_adapter(
                        str(ema_directory),
                        prefix=None,
                        weight_name="pytorch_lora_weights.safetensors",
                        use_safetensors=True,
                        local_files_only=True,
                    )
                model.eval()
                reloaded.eval()
                ema_expected.eval()
                ema_reloaded.eval()
                with torch.inference_mode():
                    left = model(
                        noisy,
                        batch["timesteps"],
                        encoder_hidden_states=batch["encoder_hidden_states"],
                        added_cond_kwargs=added,
                    ).sample
                    right = reloaded(
                        noisy,
                        batch["timesteps"],
                        encoder_hidden_states=batch["encoder_hidden_states"],
                        added_cond_kwargs=added,
                    ).sample
                    ema_left = ema_expected(
                        noisy,
                        batch["timesteps"],
                        encoder_hidden_states=batch["encoder_hidden_states"],
                        added_cond_kwargs=added,
                    ).sample
                    ema_right = ema_reloaded(
                        noisy,
                        batch["timesteps"],
                        encoder_hidden_states=batch["encoder_hidden_states"],
                        added_cond_kwargs=added,
                    ).sample
                torch.testing.assert_close(left, right, atol=0, rtol=0)
                torch.testing.assert_close(ema_left, ema_right, atol=0, rtol=0)
                records.append(
                    {
                        "method": method,
                        "text_time": text_time,
                        "checkpointing": checkpointing,
                        "loss_absolute_error": abs(observed - float(expected.detach())),
                        "maximum_gradient_absolute_error": maximum,
                        "reload_maximum_error": float((left - right).abs().max()),
                        "ema_reload_maximum_error": float(
                            (ema_left - ema_right).abs().max()
                        ),
                    }
                )
    assert not torch.cuda.is_initialized()
    args.output.write_text(
        json.dumps(
            {"scope": __doc__, "cases": records, "cuda_initialized": False}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(records),
                "max_gradient_error": max(
                    r["maximum_gradient_absolute_error"] for r in records
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
