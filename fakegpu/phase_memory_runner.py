"""One fresh CUDA process measures both metrics of each supported native phase."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from uuid import uuid4

from ._calibration_cuda import (
    _next_transformers_token,
    _real_cuda_environment,
    _reject_simulated_cuda_environment,
    _synthetic_token_id,
    _transformers_serving_workload,
)
from ._diffusion_pipeline import (
    _select_component_checkpoint,
    inspect_diffusion_pipeline,
)
from ._finetune_allocator import capture_memory_allocator
from ._finetune_workflow_store import (
    _file,
    _now,
    _source_identity,
)
from ._process import process_marker
from .phase_memory import SAMPLE_SCHEMA, _sample
from .structured_io import canonical_sha256 as _digest


def _model_inputs(plan):
    schema = plan.get("schema_version")
    if schema == "fakegpu.llm_serving_plan.v1":
        workload = _transformers_serving_workload(plan, model_dir=None)
        root = workload["model_dir"]
        index = root / "model.safetensors.index.json"
        if index.is_file():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        elif (root / "model.safetensors").is_file():
            names = ["model.safetensors"]
        else:
            raise ValueError(
                "phase serving runner requires the native model.safetensors checkpoint family"
            )
        weights = [root / name for name in names]
        kind = "transformers"
    elif schema == "fakegpu.diffusion_generation_estimate.v1":
        previous = plan.get("pipeline_inspection")
        if not previous or previous.get("weight_variant") not in {None, "auto"}:
            raise ValueError(
                "phase diffusion runner requires an unvarianted local pipeline"
            )
        root = Path(previous["model_dir"])
        inspected = inspect_diffusion_pipeline(root)
        if {k: v for k, v in inspected.items() if k != "profile"} != previous:
            raise ValueError(
                "diffusion pipeline config or headers no longer match the prediction"
            )
        if inspected["pipeline_class"] not in {
            "StableDiffusionPipeline",
            "StableDiffusionXLPipeline",
        }:
            raise ValueError("phase diffusion runner supports SD and SDXL only")
        inputs = plan["inputs"]
        if "model_dir" in inputs:
            raise ValueError(
                "phase diffusion inputs cannot override the inspected model_dir"
            )
        if inputs["offload"] != "none":
            raise ValueError("phase diffusion runner does not support offload")
        if inputs["attention_backend"] not in {"sdpa", "eager"}:
            raise ValueError("phase diffusion runner requires SDPA or eager attention")
        for name in ("attention_slicing", "vae_slicing", "vae_tiling"):
            if type(inputs[name]) is not bool:
                raise ValueError(f"phase diffusion {name} must be a boolean")
        if inputs["vae_tiling"]:
            tile = inputs["vae_tile_size"]
            scale = inspected["profile"]["latent"]["scale_factor"]
            if type(tile) is not int or tile <= 0 or tile % (4 * scale):
                raise ValueError(
                    "phase diffusion vae_tile_size must be a positive multiple of 4 * latent scale"
                )
        if inputs["text_tokens"] != 77:
            raise ValueError("phase diffusion runner uses native 77-token conditioning")
        weights = []
        for name in inspected["components"]:
            names, _, variant = _select_component_checkpoint(
                root / name, weight_variant=None
            )
            if variant is not None:
                raise ValueError(
                    "phase diffusion runner requires default safetensors weights"
                )
            weights.extend(root / name / value for value in names)
        workload = {**inputs, "model_dir": root}
        kind = "diffusers"
    else:
        raise ValueError(
            "phase sampling requires a homogeneous serving or local SD/SDXL prediction"
        )
    files = set(weights)
    files.update(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".txt", ".model"}
    )
    for path in files:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(
                "checkpoint inputs must stay inside the declared model directory"
            )
    return kind, workload, root, sorted(files)


def _marker(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _Peaks:
    def __init__(self, torch):
        self.torch = torch
        self.phases = {}

    def start(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def finish(self, phase):
        self.torch.cuda.synchronize()
        value = {
            name: int(getattr(self.torch.cuda, "max_memory_" + name)())
            for name in ("allocated", "reserved")
        }
        self.phases[phase] = {
            name: max(value[name], self.phases.get(phase, {}).get(name, 0))
            for name in value
        }


def _serving(torch, workload, peaks):
    import transformers

    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(workload["model_dir"]),
            dtype=getattr(torch, workload["dtype"]),
            attn_implementation=workload["attention_implementation"],
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    token = _synthetic_token_id(model.config)
    input_ids = torch.full(
        (workload["active_sequences"], workload["prompt_tokens"]),
        token,
        dtype=torch.long,
        device="cuda",
    )
    with torch.inference_mode():
        peaks.start()
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        peaks.finish("prefill")
        if not bool(torch.isfinite(output.logits[:, -1]).all()):
            raise RuntimeError("nonfinite final prefill logits")
        cache, next_token = _next_transformers_token(output, phase="prefill")
        del output, input_ids
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        peaks.start()
        for _ in range(workload["generated_tokens"] - 1):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, next_token = _next_transformers_token(output, phase="decode")
            if not bool(torch.isfinite(output.logits[:, -1]).all()):
                raise RuntimeError("nonfinite decode logits")
            del output
        peaks.finish("decode")
    return {
        "synthetic_token_id": token,
        "generated_tokens": workload["generated_tokens"],
        "last_token_ids": next_token.cpu().tolist(),
        "finite_check": "Final prefill logits and each decode logits; prefill peak is captured before its finite check.",
    }


def _configure_diffusion(pipeline, workload):
    """Apply the planned native modes and describe their effective settings."""
    from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

    processor = (
        AttnProcessor2_0 if workload["attention_backend"] == "sdpa" else AttnProcessor
    )
    pipeline.unet.set_attn_processor(processor())
    pipeline.vae.set_attn_processor(processor())
    if workload["attention_slicing"]:
        pipeline.enable_attention_slicing("auto")
    if workload["vae_slicing"]:
        pipeline.enable_vae_slicing()
    if workload["vae_tiling"]:
        pipeline.enable_vae_tiling()
        pipeline.vae.tile_sample_min_size = workload["vae_tile_size"]
        pipeline.vae.tile_latent_min_size = (
            workload["vae_tile_size"] // pipeline.vae_scale_factor
        )
    return {
        "unet_processors": sorted(
            {type(p).__name__ for p in pipeline.unet.attn_processors.values()}
        ),
        "vae_processors": sorted(
            {type(p).__name__ for p in pipeline.vae.attn_processors.values()}
        ),
        "vae_slicing": pipeline.vae.use_slicing,
        "vae_tiling": pipeline.vae.use_tiling,
        "vae_tile_sample_min_size": pipeline.vae.tile_sample_min_size,
        "vae_tile_latent_min_size": pipeline.vae.tile_latent_min_size,
    }


def _diffusion(torch, workload, peaks):
    import diffusers
    import numpy as np

    pipeline = diffusers.DiffusionPipeline.from_pretrained(
        str(workload["model_dir"]),
        torch_dtype=getattr(torch, workload["dtype"]),
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    runtime = _configure_diffusion(pipeline, workload)
    calls = {}
    originals = []
    vae_dtypes = {}

    def install(owner, method, phase):
        original = getattr(owner, method)

        def measured(*args, **kwargs):
            if phase == "vae_decode":
                vae_dtypes.update(
                    {name: str(p.dtype) for name, p in owner.named_parameters()}
                )
            peaks.start()
            result = original(*args, **kwargs)
            primary = (
                result
                if isinstance(result, torch.Tensor)
                else (result.to_tuple() if hasattr(result, "to_tuple") else result)[0]
            )
            if not isinstance(primary, torch.Tensor) or not bool(
                torch.isfinite(primary).all()
            ):
                raise RuntimeError(f"{phase} nonfinite or unsupported primary output")
            peaks.finish(phase)
            calls[phase] = calls.get(phase, 0) + 1
            return result

        setattr(owner, method, measured)
        originals.append((owner, method, original))

    for name in ("text_encoder", "text_encoder_2"):
        owner = getattr(pipeline, name, None)
        if owner is not None:
            install(owner, "forward", "text_encode")
    install(pipeline.unet, "forward", "denoise")
    install(pipeline.vae, "decode", "vae_decode")
    try:
        with torch.inference_mode():
            result = pipeline(
                prompt=["A red cube on a wooden table, studio lighting."]
                * workload["batch_size"],
                height=workload["height"],
                width=workload["width"],
                num_inference_steps=workload["steps"],
                guidance_scale=workload["guidance_scale"],
                output_type="np",
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
        if not np.isfinite(result.images).all():
            raise RuntimeError("nonfinite final images")
        if result.images.shape != (
            workload["batch_size"],
            workload["height"],
            workload["width"],
            3,
        ):
            raise RuntimeError("final image shape does not match the prediction")
        import hashlib

        return {
            "calls": calls,
            "runtime": runtime,
            "vae_parameter_dtypes_at_decode": vae_dtypes,
            "image_shape": list(result.images.shape),
            "all_image_values_finite": True,
            "image_sha256": hashlib.sha256(
                result.images.astype("<f4").tobytes()
            ).hexdigest(),
            "finite_check": "Each primary output inside its phase; final NumPy images checked separately.",
        }
    finally:
        for owner, method, original in originals:
            setattr(owner, method, original)


def sample_phase_memory(prediction):
    """Measure one sample; invoke in a fresh process with one visible CUDA GPU."""
    started = _now()
    import torch
    import transformers

    _reject_simulated_cuda_environment(torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "phase memory sampling requires exactly one visible real CUDA GPU"
        )
    if torch.cuda.is_initialized():
        raise ValueError("phase memory sampling requires a fresh CUDA process")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    kind, workload, root, paths = _model_inputs(prediction)
    model_files = {str(path.relative_to(root)): _file(path) for path in paths}
    markers = {path: _marker(path) for path in paths}
    software = {"transformers": transformers.__version__}
    version = transformers.__version__
    if kind == "diffusers":
        import diffusers

        version = diffusers.__version__
    identity = {
        "environment": _real_cuda_environment(
            torch, framework=kind, framework_version=version, software=software
        ),
        "allocator": capture_memory_allocator(torch),
        "model_files": model_files,
        "implementation": _source_identity(),
        "execution": {
            "kind": kind,
            "protocol": "fresh_cuda_one_sample.v1",
            "seed": 42,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_version": torch.backends.cudnn.version(),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cache_regime": "uncontrolled",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    torch.manual_seed(42)
    peaks = _Peaks(torch)
    execution = (
        _serving(torch, workload, peaks)
        if kind == "transformers"
        else _diffusion(torch, workload, peaks)
    )
    if kind == "diffusers":
        identity["execution"]["diffusion"] = execution["runtime"]
    if any(_marker(path) != marker for path, marker in markers.items()):
        raise ValueError("model input changed during native sampling")
    if _source_identity() != identity["implementation"]:
        raise ValueError("implementation changed during native sampling")
    marker = process_marker(os.getpid())
    report = {
        "schema_version": SAMPLE_SCHEMA,
        "run_id": str(uuid4()),
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "process": {"pid": os.getpid(), "start_ticks": marker["start_ticks"]},
        "prediction": deepcopy(prediction),
        "prediction_sha256": _digest(prediction),
        "identity": identity,
        "identity_sha256": _digest(identity),
        "phases": peaks.phases,
        "execution": execution,
    }
    return _sample(report)


__all__ = ["sample_phase_memory"]
