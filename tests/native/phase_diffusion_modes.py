"""Check public sampler configuration against native Diffusers on CPU/meta.

Run with the optional Diffusers environment and a local SD or SDXL model-dir.
No checkpoint payload or CUDA device is used by this check.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import diffusers
import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from fakegpu.phase_memory_runner import _configure_diffusion


def pipeline(root):
    index = json.loads((root / "model_index.json").read_text())
    classes = {
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    }
    cls = classes[index["_class_name"]]
    with torch.device("meta"):
        vae = AutoencoderKL.from_config(
            json.loads((root / "vae/config.json").read_text())
        ).to(dtype=torch.float16)
        unet = UNet2DConditionModel.from_config(
            json.loads((root / "unet/config.json").read_text())
        ).to(dtype=torch.float16)
    scheduler = getattr(diffusers, index["scheduler"][1]).from_config(
        json.loads((root / "scheduler/scheduler_config.json").read_text())
    )
    components = dict(
        vae=vae, unet=unet, scheduler=scheduler, text_encoder=None, tokenizer=None
    )
    if cls is StableDiffusionXLPipeline:
        components.update(text_encoder_2=None, tokenizer_2=None, add_watermarker=False)
    else:
        components.update(
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    return cls(**components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    rows = []
    for backend, sliced, vae_sliced, tiled in itertools.product(
        ("sdpa", "eager"), (False, True), (False, True), (False, True)
    ):
        native = pipeline(args.model_dir)
        workload = dict(
            attention_backend=backend,
            attention_slicing=sliced,
            vae_slicing=vae_sliced,
            vae_tiling=tiled,
            vae_tile_size=256,
        )
        runtime = _configure_diffusion(native, workload)
        processor = "AttnProcessor2_0" if backend == "sdpa" else "AttnProcessor"
        assert runtime["vae_processors"] == [processor]
        assert runtime["unet_processors"] == [
            "SlicedAttnProcessor" if sliced else processor
        ]
        assert native.vae.use_slicing is vae_sliced
        assert native.vae.use_tiling is tiled
        if tiled:
            assert native.vae.tile_sample_min_size == 256
            assert native.vae.tile_latent_min_size == 256 // native.vae_scale_factor
        if isinstance(native, StableDiffusionXLPipeline):
            native.upcast_vae()
            half = sum(
                p.numel() for p in native.vae.parameters() if p.dtype == torch.float16
            )
            expected = (
                sum(
                    p.numel()
                    for name, p in native.vae.named_parameters()
                    if name.startswith(
                        ("post_quant_conv.", "decoder.conv_in.", "decoder.mid_block.")
                    )
                )
                if backend == "sdpa"
                else 0
            )
            assert half == expected
        rows.append({"workload": workload, "runtime": runtime})

    # Exercise the actual optimized VAE paths with allocated CPU tensors too.
    numerical = []
    for backend, sliced, tiled in itertools.product(
        ("sdpa", "eager"), (False, True), (False, True)
    ):
        torch.manual_seed(42)
        vae = AutoencoderKL(
            block_out_channels=(8, 16),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            norm_num_groups=4,
            sample_size=32,
        ).eval()
        # DiffusionPipeline's public VAE methods are inherited by both SD families.
        native = pipeline(args.model_dir)
        native.vae = vae
        native.vae_scale_factor = 2
        runtime = _configure_diffusion(
            native,
            dict(
                attention_backend=backend,
                attention_slicing=False,
                vae_slicing=sliced,
                vae_tiling=tiled,
                vae_tile_size=16,
            ),
        )
        with torch.inference_mode():
            result = vae.decode(torch.randn(3, 4, 12, 8)).sample
        assert result.shape == (3, 3, 24, 16)
        assert bool(torch.isfinite(result).all())
        numerical.append(dict(runtime=runtime, shape=list(result.shape), finite=True))
    assert not torch.cuda.is_initialized()
    report = dict(
        torch_version=torch.__version__,
        diffusers_version=diffusers.__version__,
        model_dir=str(args.model_dir),
        meta_cases=rows,
        numerical_vae_cases=numerical,
        cuda_initialized=False,
        passed=True,
    )
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "meta_cases": len(rows),
                "numerical_vae_cases": len(numerical),
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
