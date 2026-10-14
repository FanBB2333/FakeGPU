"""Observe native SD/SDXL phase peaks and verify primary outputs are finite.

Run in a real CUDA Diffusers environment. Per-forward checks and synchronizations
are instrumentation; recorded wall time is not an uninstrumented latency claim.
No model files are changed, and safety checking is disabled for these synthetic
benchmark prompts. Original cohort reports remain separate from later checks.
"""

import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _primary_tensor(output, torch):
    if isinstance(output, torch.Tensor):
        return output
    values = output.to_tuple() if hasattr(output, "to_tuple") else output
    if (
        not isinstance(values, tuple)
        or not values
        or not isinstance(values[0], torch.Tensor)
    ):
        raise ValueError(
            "native phase must return a Tensor, tuple or Transformers/Diffusers ModelOutput"
        )
    return values[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    a = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    import diffusers
    from fakegpu._calibration_cuda import (
        _reject_simulated_cuda_environment,
        _real_cuda_environment,
    )

    _reject_simulated_cuda_environment(torch)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    started = time.perf_counter()
    pipeline = diffusers.DiffusionPipeline.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    peaks = {}
    calls = {}
    restores = []
    finite_checks = {}

    def install(owner, name, phase):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            result = original(*args, **kwargs)
            tensor = _primary_tensor(result, torch)
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{phase} nonfinite primary output")
            finite_checks[phase] = finite_checks.get(phase, 0) + 1
            torch.cuda.synchronize()
            duration = time.perf_counter() - t
            observed = {
                k: int(getattr(torch.cuda, "max_memory_" + k)())
                for k in ("allocated", "reserved")
            }
            for k, v in observed.items():
                peaks.setdefault(phase, {})[k] = max(peaks.get(phase, {}).get(k, 0), v)
            calls.setdefault(phase, []).append(
                {"seconds_with_finite_check": duration, **observed}
            )
            return result

        setattr(owner, name, measured)
        restores.append((owner, name, original))

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        component = getattr(pipeline, name, None)
        if component is not None:
            install(component, "forward", "text_encode")
    install(pipeline.unet, "forward", "denoise")
    install(pipeline.vae, "decode", "vae_decode")
    setup = time.perf_counter() - started
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        with torch.inference_mode():
            out = pipeline(
                prompt="A red cube on a wooden table, studio lighting.",
                height=a.height,
                width=a.width,
                num_inference_steps=8,
                guidance_scale=7.5,
                output_type="np",
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        images = out.images
        assert images.shape == (1, a.height, a.width, 3) and np.isfinite(images).all()
    finally:
        for owner, name, original in restores:
            setattr(owner, name, original)
    report = {
        "schema_version": "fakegpu.diffusion_generation_observation.v1",
        "scope": "Native Diffusers generation; phase peaks include resident weights and per-forward finite checks; safety checker disabled; no offload/slicing/tiling",
        "model": str(Path(a.model).resolve()),
        "pipeline_class": type(pipeline).__name__,
        "index": a.index,
        "height": a.height,
        "width": a.width,
        "batch_size": 1,
        "dtype": "bfloat16",
        "guidance_scale": 7.5,
        "requested_steps": 8,
        "attention_backend": "sdpa",
        "environment": _real_cuda_environment(
            torch,
            framework="diffusers",
            framework_version=diffusers.__version__,
            software={"transformers": transformers.__version__},
        ),
        "setup_seconds": setup,
        "pipeline_seconds_with_phase_instrumentation": seconds,
        "peaks": peaks,
        "calls": calls,
        "image_shape": list(images.shape),
        "image_float32_sha256": hashlib.sha256(
            images.astype("<f4").tobytes()
        ).hexdigest(),
        "all_image_values_finite": True,
        "primary_output_finite_checks": finite_checks,
        "component_dtypes": {
            name: str(next(getattr(pipeline, name).parameters()).dtype)
            for name in ("text_encoder", "unet", "vae")
        },
        "vae_force_upcast": pipeline.vae.config.force_upcast,
    }
    assert finite_checks == {phase: len(values) for phase, values in calls.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "index": a.index,
                "model": a.model,
                "peaks": peaks,
                "calls": {k: len(v) for k, v in calls.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
