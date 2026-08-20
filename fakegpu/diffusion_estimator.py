from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    usage_error,
)
from .llm_estimator import _DTYPE_BYTES as _SAFETENSORS_DTYPE_BYTES
from .llm_estimator import inspect_safetensors_checkpoint
from .profile_catalog import get_profile
from .structured_io import emit_json


SCHEMA_VERSION = "fakegpu.diffusion_generation_estimate.v1"
INSPECTION_SCHEMA_VERSION = "fakegpu.diffusion_pipeline_inspection.v1"
PROFILE_SCHEMA_VERSION = "fakegpu.diffusion_profiles.v1"
SUPPORTED_DTYPES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}
SUPPORTED_ATTENTION_BACKENDS = {"eager", "sdpa"}
SUPPORTED_OFFLOAD_MODES = {"none", "model"}
SUPPORTED_DENOISER_ARCHITECTURES = {"unet", "transformer"}

_CHECKPOINT_VARIANTS = {"bf16", "fp16", "fp32"}
_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$")
_FLOAT_SAFETENSORS_DTYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "F16",
    "BF16",
    "F32",
    "F64",
}
_TRANSFORMER_JOINT_CLASSES = {
    "FluxTransformer2DModel",
    "Flux2Transformer2DModel",
    "SD3Transformer2DModel",
}
_PIPELINE_DEFAULTS = {
    "StableDiffusionPipeline": {
        "steps": 50,
        "guidance_scale": 7.5,
        "text_tokens": 77,
    },
    "StableDiffusionXLPipeline": {
        "steps": 50,
        "guidance_scale": 5.0,
        "text_tokens": 77,
    },
    "PixArtAlphaPipeline": {
        "steps": 20,
        "guidance_scale": 4.5,
        "text_tokens": 120,
    },
    "PixArtSigmaPipeline": {
        "steps": 20,
        "guidance_scale": 4.5,
        "text_tokens": 300,
    },
    "StableDiffusion3Pipeline": {
        "steps": 28,
        "guidance_scale": 7.0,
        "text_tokens": 256,
    },
    "FluxPipeline": {
        "steps": 28,
        "guidance_scale": 3.5,
        "text_tokens": 512,
    },
}


class DiffusionEstimateError(ValueError):
    pass


def load_diffusion_profiles(
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and validate checked-in diffusion reference profiles."""

    profile_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else (
            Path(__file__).resolve().parent
            / "data"
            / "diffusion_profiles.json"
        )
    )
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiffusionEstimateError(
            f"cannot read diffusion profiles: {profile_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DiffusionEstimateError(
            f"invalid diffusion profile JSON: {profile_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DiffusionEstimateError(
            "diffusion profile root must be an object"
        )
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise DiffusionEstimateError(
            "unsupported diffusion profile schema"
        )
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise DiffusionEstimateError(
            "diffusion profiles must be a non-empty object"
        )

    validated: dict[str, dict[str, Any]] = {}
    for profile_id, raw_profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise DiffusionEstimateError(
                "diffusion profile IDs must be non-empty strings"
            )
        if not isinstance(raw_profile, Mapping):
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r} must be an object"
            )
        profile = copy.deepcopy(dict(raw_profile))
        _validate_profile(profile_id, profile)
        profile["id"] = profile_id
        validated[profile_id] = profile
    return validated


def inspect_diffusion_pipeline(
    model_dir: str | Path,
    *,
    weight_variant: str | None = None,
) -> dict[str, Any]:
    """Inspect a local Diffusers pipeline without importing model code.

    Pipeline and architecture metadata come from JSON configuration files.
    Parameter and checkpoint byte counts come from safetensors headers; tensor
    payloads are never materialized.
    """

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise DiffusionEstimateError(
            f"diffusion model directory does not exist: {root}"
        )
    if weight_variant is not None:
        weight_variant = str(weight_variant).lower()
        if weight_variant not in _CHECKPOINT_VARIANTS:
            raise DiffusionEstimateError(
                "weight_variant must be bf16, fp16, or fp32"
            )

    model_index_path = root / "model_index.json"
    model_index = _read_json_object(model_index_path)
    pipeline_class = str(model_index.get("_class_name") or "")
    if not pipeline_class:
        raise DiffusionEstimateError(
            f"missing _class_name in {model_index_path}"
        )

    declared: dict[str, dict[str, Any]] = {}
    for component_name, declaration in model_index.items():
        if component_name.startswith("_") or declaration is None:
            continue
        if (
            not isinstance(declaration, list)
            or len(declaration) < 2
            or not isinstance(declaration[1], str)
        ):
            continue
        component_dir = root / component_name
        config_path = component_dir / "config.json"
        if not config_path.is_file():
            continue
        declared[component_name] = {
            "name": component_name,
            "class_name": declaration[1],
            "config": _read_json_object(config_path),
            "config_path": config_path,
            "directory": component_dir,
        }

    denoiser_names = [
        name
        for name, component in declared.items()
        if name in {"unet", "transformer"}
        or "UNet" in str(component["class_name"])
        or "Transformer" in str(component["class_name"])
    ]
    if len(denoiser_names) != 1:
        raise DiffusionEstimateError(
            "expected exactly one UNet or 2D transformer component in "
            f"{model_index_path}; found {denoiser_names}"
        )
    denoiser_name = denoiser_names[0]

    vae_names = [
        name
        for name, component in declared.items()
        if name == "vae"
        or "Autoencoder" in str(component["class_name"])
        or "VQModel" in str(component["class_name"])
    ]
    if len(vae_names) != 1:
        raise DiffusionEstimateError(
            "expected exactly one VAE component in "
            f"{model_index_path}; found {vae_names}"
        )
    vae_name = vae_names[0]

    conditioning_names = sorted(
        name
        for name, component in declared.items()
        if name.startswith("text_encoder")
        or "TextModel" in str(component["class_name"])
        or "T5Encoder" in str(component["class_name"])
    )
    role_names = {
        "conditioning": conditioning_names,
        "denoiser": [denoiser_name],
        "vae": [vae_name],
    }

    components: dict[str, dict[str, Any]] = {}
    for role, component_names in role_names.items():
        for component_name in component_names:
            component = declared[component_name]
            selected_files, selected_family, selected_variant = (
                _select_component_checkpoint(
                    component["directory"],
                    weight_variant=weight_variant,
                )
            )
            try:
                checkpoint = inspect_safetensors_checkpoint(
                    component["directory"],
                    files=selected_files,
                )
            except (OSError, ValueError) as exc:
                raise DiffusionEstimateError(
                    f"cannot inspect {component_name!r}: {exc}"
                ) from exc
            checkpoint["selected_family"] = selected_family
            checkpoint["selected_variant"] = selected_variant
            components[component_name] = {
                "role": role,
                "class_name": component["class_name"],
                "config_path": str(
                    component["config_path"].relative_to(root)
                ),
                "checkpoint": checkpoint,
            }

    profile = _local_profile(
        root=root,
        pipeline_class=pipeline_class,
        declared=declared,
        components=components,
        role_names=role_names,
    )
    return {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "model_dir": str(root),
        "pipeline_class": pipeline_class,
        "weight_variant": weight_variant or "auto",
        "roles": copy.deepcopy(role_names),
        "components": components,
        "architecture": {
            "family": profile["denoiser"]["architecture"],
            "model_class": profile["denoiser"]["model_class"],
            "attention_pattern": profile["denoiser"].get(
                "attention_pattern"
            ),
            "guidance_mode": profile["denoiser"]["guidance_mode"],
        },
        "profile": profile,
        "evidence": {
            "pipeline": "model_index.json",
            "architecture": "component config.json files",
            "weights": "selected safetensors headers",
            "tensor_payloads_loaded": False,
        },
    }


def estimate_diffusion_generation(
    model_profile: str | None = None,
    *,
    model_dir: str | Path | None = None,
    weight_variant: str | None = None,
    height: int | None = None,
    width: int | None = None,
    batch_size: int = 1,
    steps: int | None = None,
    guidance_scale: float | None = None,
    text_tokens: int | None = None,
    dtype: str = "float16",
    attention_backend: str = "sdpa",
    attention_slicing: bool = False,
    vae_slicing: bool = False,
    vae_tiling: bool = False,
    vae_tile_size: int = 512,
    offload: str = "none",
    runtime_overhead_bytes: int = 0,
    target_profile: str | None = None,
    profiles_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate phase-aware memory for a reference diffusion pipeline.

    Component parameter counts come from fixed safetensors headers. Activation
    and workspace values are analytical shape models rather than GPU
    observations.
    """

    if (model_profile is None) == (model_dir is None):
        raise DiffusionEstimateError(
            "provide exactly one of model_profile or model_dir"
        )
    inspection = None
    if model_dir is not None:
        inspection = inspect_diffusion_pipeline(
            model_dir,
            weight_variant=weight_variant,
        )
        profile = copy.deepcopy(_mapping(inspection["profile"]))
        selected_model_id = Path(model_dir).expanduser().resolve().name
        model_source_kind = "local_diffusers_pipeline"
    else:
        profiles = load_diffusion_profiles(profiles_path)
        try:
            profile = profiles[str(model_profile)]
        except KeyError as exc:
            choices = ", ".join(sorted(profiles))
            raise DiffusionEstimateError(
                f"unknown diffusion profile {model_profile!r}; "
                f"available profiles: {choices}"
            ) from exc
        selected_model_id = str(model_profile)
        model_source_kind = "reference_profile"

    defaults = _mapping(profile["default_generation"])
    image_height = _generation_dimension(
        height,
        defaults.get("height"),
        "height",
    )
    image_width = _generation_dimension(
        width,
        defaults.get("width"),
        "width",
    )
    image_batch = _positive_integer(batch_size, "batch_size")
    denoising_steps = (
        _positive_integer(steps, "steps")
        if steps is not None
        else int(defaults["steps"])
    )
    selected_guidance = (
        _nonnegative_number(guidance_scale, "guidance_scale")
        if guidance_scale is not None
        else float(defaults["guidance_scale"])
    )
    selected_dtype = str(dtype).lower()
    if selected_dtype not in SUPPORTED_DTYPES:
        raise DiffusionEstimateError(
            "dtype must be float16, bfloat16, or float32"
        )
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise DiffusionEstimateError(
            "attention_backend must be eager or sdpa"
        )
    if offload not in SUPPORTED_OFFLOAD_MODES:
        raise DiffusionEstimateError(
            "offload must be none or model"
        )
    if not isinstance(attention_slicing, bool):
        raise DiffusionEstimateError(
            "attention_slicing must be a boolean"
        )
    if not isinstance(vae_slicing, bool):
        raise DiffusionEstimateError(
            "vae_slicing must be a boolean"
        )
    if not isinstance(vae_tiling, bool):
        raise DiffusionEstimateError(
            "vae_tiling must be a boolean"
        )
    tile_size = _positive_integer(vae_tile_size, "vae_tile_size")
    overhead_bytes = _nonnegative_integer(
        runtime_overhead_bytes,
        "runtime_overhead_bytes",
    )

    latent = _mapping(profile["latent"])
    latent_scale = int(latent["scale_factor"])
    latent_channels = int(latent["channels"])
    if (
        image_height % latent_scale != 0
        or image_width % latent_scale != 0
    ):
        raise DiffusionEstimateError(
            "height and width must be divisible by the latent scale "
            f"factor ({latent_scale})"
        )
    latent_height = image_height // latent_scale
    latent_width = image_width // latent_scale

    conditioning = _mapping(profile["conditioning"])
    conditioning_enabled = bool(conditioning.get("enabled", True))
    maximum_tokens = int(conditioning["max_tokens"])
    if conditioning_enabled:
        selected_tokens = (
            _positive_integer(text_tokens, "text_tokens")
            if text_tokens is not None
            else int(
                conditioning.get("default_tokens", maximum_tokens)
            )
        )
        if selected_tokens > maximum_tokens:
            raise DiffusionEstimateError(
                f"text_tokens must not exceed {maximum_tokens} "
                f"for {selected_model_id}"
            )
    else:
        if text_tokens is not None:
            raise DiffusionEstimateError(
                "text_tokens is not valid for a pipeline without a "
                "text-conditioning component"
            )
        selected_tokens = 0

    element_bytes = SUPPORTED_DTYPES[selected_dtype]
    denoiser_profile = _mapping(profile["denoiser"])
    guidance_mode = str(
        denoiser_profile.get("guidance_mode")
        or "classifier_free"
    )
    classifier_free_guidance = (
        guidance_mode == "classifier_free"
        and selected_guidance > 1.0
    )
    denoiser_batch = image_batch * (
        2 if classifier_free_guidance else 1
    )
    component_parameter_counts = {
        "conditioning": int(conditioning["parameter_count"]),
        "denoiser": int(denoiser_profile["parameter_count"]),
        "vae": int(_mapping(profile["vae"])["parameter_count"]),
    }
    component_checkpoint_bytes = {
        name: int(
            _mapping(profile[profile_key]).get(
                "checkpoint_bytes",
                component_parameter_counts[name] * element_bytes,
            )
        )
        for name, profile_key in (
            ("conditioning", "conditioning"),
            ("denoiser", "denoiser"),
            ("vae", "vae"),
        )
    }
    component_checkpoint_dtype_bytes = {
        name: copy.deepcopy(
            dict(
                _mapping(profile[profile_key]).get(
                    "checkpoint_dtype_bytes",
                    {},
                )
            )
        )
        for name, profile_key in (
            ("conditioning", "conditioning"),
            ("denoiser", "denoiser"),
            ("vae", "vae"),
        )
    }
    component_weight_bytes = {
        name: _runtime_checkpoint_bytes(
            parameter_count=component_parameter_counts[name],
            dtype_bytes=component_checkpoint_dtype_bytes[name],
            runtime_float_bytes=element_bytes,
        )
        for name in component_parameter_counts
    }
    total_weight_bytes = sum(component_weight_bytes.values())

    latent_bytes = (
        image_batch
        * latent_channels
        * latent_height
        * latent_width
        * element_bytes
    )
    denoiser_latent_bytes = latent_bytes * (
        2 if classifier_free_guidance else 1
    )
    conditioning_output_bytes = (
        denoiser_batch
        * selected_tokens
        * int(conditioning["width"])
        * element_bytes
    )
    text_activation = _text_encoder_activation(
        batch_size=denoiser_batch,
        tokens=selected_tokens,
        width=int(
            conditioning.get("encoder_width", conditioning["width"])
        ),
        element_bytes=element_bytes,
    )
    denoiser_activation = _denoiser_activation(
        profile=denoiser_profile,
        latent_height=latent_height,
        latent_width=latent_width,
        batch_size=denoiser_batch,
        text_tokens=selected_tokens,
        element_bytes=element_bytes,
        attention_backend=attention_backend,
        attention_slicing=attention_slicing,
    )
    vae_activation = _vae_decode_activation(
        profile=_mapping(profile["vae"]),
        height=image_height,
        width=image_width,
        batch_size=image_batch,
        latent_scale=latent_scale,
        element_bytes=element_bytes,
        slicing=vae_slicing,
        tiling=vae_tiling,
        tile_size=tile_size,
    )
    output_image_bytes = (
        image_batch
        * image_height
        * image_width
        * 3
        * element_bytes
    )
    scheduler_bytes = 3 * denoiser_latent_bytes

    resident_weights = {
        phase: (
            total_weight_bytes
            if offload == "none"
            else component_weight_bytes[component]
        )
        for phase, component in (
            ("text_encode", "conditioning"),
            ("denoise", "denoiser"),
            ("vae_decode", "vae"),
        )
    }
    phases = [
        _phase(
            "text_encode",
            {
                "resident_weights": resident_weights["text_encode"],
                "text_encoder_activations": text_activation[
                    "peak_bytes"
                ],
                "conditioning_output": conditioning_output_bytes,
                "runtime_overhead": overhead_bytes,
            },
        ),
        _phase(
            "denoise",
            {
                "resident_weights": resident_weights["denoise"],
                "latents": denoiser_latent_bytes,
                "conditioning": conditioning_output_bytes,
                "retained_residuals": denoiser_activation[
                    "retained_residual_bytes"
                ],
                "working_tensors": denoiser_activation[
                    "working_tensor_bytes"
                ],
                "attention_workspace": denoiser_activation[
                    "attention_workspace_bytes"
                ],
                "scheduler_buffers": scheduler_bytes,
                "runtime_overhead": overhead_bytes,
            },
        ),
        _phase(
            "vae_decode",
            {
                "resident_weights": resident_weights["vae_decode"],
                "latents": latent_bytes,
                "decoded_image": output_image_bytes,
                "retained_features": vae_activation[
                    "retained_feature_bytes"
                ],
                "working_tensors": vae_activation[
                    "working_tensor_bytes"
                ],
                "runtime_overhead": overhead_bytes,
            },
        ),
    ]
    peak_phase = max(phases, key=lambda item: item["peak_bytes"])
    peak_bytes = int(peak_phase["peak_bytes"])

    fit = None
    if target_profile is not None:
        gpu = get_profile(target_profile)
        fit = {
            "target_profile": gpu.id,
            "target_name": gpu.name,
            "profile_status": gpu.profile_status,
            "capacity_bytes": gpu.memory_bytes,
            "estimated_peak_bytes": peak_bytes,
            "fits": peak_bytes <= gpu.memory_bytes,
            "headroom_bytes": max(0, gpu.memory_bytes - peak_bytes),
            "overflow_bytes": max(0, peak_bytes - gpu.memory_bytes),
            "utilization_percent": (
                peak_bytes / gpu.memory_bytes * 100
            ),
        }

    latent_pixel_steps = (
        denoiser_batch
        * latent_height
        * latent_width
        * denoising_steps
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "safetensors_headers_plus_architecture_specific_"
            "phase_shape_memory_model"
        ),
        "model_profile": {
            "id": selected_model_id,
            "display_name": profile["display_name"],
            "pipeline_class": profile["pipeline_class"],
            "status": profile["status"],
            "source_kind": model_source_kind,
            "source": copy.deepcopy(profile["source"]),
        },
        "architecture": {
            "denoiser_family": denoiser_profile["architecture"],
            "model_class": denoiser_profile.get("model_class"),
            "attention_pattern": denoiser_profile.get(
                "attention_pattern"
            ),
            "guidance_mode": guidance_mode,
            "configuration_source": (
                "local_component_config"
                if inspection is not None
                else "fixed_revision_profile"
            ),
        },
        "inputs": {
            "height": image_height,
            "width": image_width,
            "batch_size": image_batch,
            "steps": denoising_steps,
            "guidance_scale": selected_guidance,
            "guidance_mode": guidance_mode,
            "classifier_free_guidance": classifier_free_guidance,
            "denoiser_batch_size": denoiser_batch,
            "text_tokens": selected_tokens,
            "text_conditioning_enabled": conditioning_enabled,
            "dtype": selected_dtype,
            "element_bytes": element_bytes,
            "attention_backend": attention_backend,
            "attention_slicing": attention_slicing,
            "vae_slicing": vae_slicing,
            "vae_tiling": vae_tiling,
            "vae_tile_size": tile_size if vae_tiling else None,
            "offload": offload,
            "runtime_overhead_bytes": overhead_bytes,
            "target_profile": target_profile,
        },
        "latent": {
            "scale_factor": latent_scale,
            "channels": latent_channels,
            "height": latent_height,
            "width": latent_width,
            "image_batch_bytes": latent_bytes,
            "denoiser_batch_bytes": denoiser_latent_bytes,
        },
        "weights": {
            "parameter_counts": component_parameter_counts,
            "checkpoint_bytes": component_checkpoint_bytes,
            "checkpoint_dtype_bytes": (
                component_checkpoint_dtype_bytes
            ),
            "total_checkpoint_bytes": sum(
                component_checkpoint_bytes.values()
            ),
            "runtime_bytes": component_weight_bytes,
            "total_parameter_count": sum(
                component_parameter_counts.values()
            ),
            "total_runtime_bytes": total_weight_bytes,
            "runtime_dtype": selected_dtype,
            "source": (
                "local_safetensors_headers"
                if inspection is not None
                else "fixed_revision_safetensors_headers"
            ),
        },
        "pipeline_inspection": (
            {
                key: copy.deepcopy(value)
                for key, value in inspection.items()
                if key != "profile"
            }
            if inspection is not None
            else None
        ),
        "activation_models": {
            "text_encoder": text_activation,
            "denoiser": denoiser_activation,
            "vae_decode": vae_activation,
        },
        "memory_timeline": {
            "unit": "bytes",
            "phases": phases,
            "peak_phase": peak_phase["phase"],
            "peak_bytes": peak_bytes,
        },
        "workload_scale": {
            "denoiser_evaluations": denoising_steps,
            "latent_pixel_steps": latent_pixel_steps,
            "description": (
                "Shape-based work indicator; not a FLOP or latency claim."
            ),
        },
        "offload": {
            "mode": offload,
            "resident_weight_bytes_by_phase": resident_weights,
            "maximum_resident_weight_bytes": max(
                resident_weights.values()
            ),
            "maximum_cpu_weight_bytes": (
                0
                if offload == "none"
                else max(
                    total_weight_bytes - value
                    for value in resident_weights.values()
                )
            ),
        },
        "fit": fit,
        "tracking_confidence": (
            "L3_exact_headers_and_local_architecture_shape_model"
            if inspection is not None
            else "L2_reference_component_and_architecture_shape_model"
        ),
        "validation_status": "Modeled",
        "accuracy": {
            "status": "uncalibrated",
            "point_estimate_bytes": peak_bytes,
            "prediction_interval_bytes": None,
            "observed_peak_bytes": None,
            "absolute_percentage_error_percent": None,
            "reason": (
                "No matching real-GPU observation was supplied for this "
                "model revision, shape, dtype, backend, and GPU."
            ),
        },
        "unmodeled_components": [
            "cuda_context_and_loaded_modules",
            "allocator_fragmentation",
            "backend_convolution_workspaces",
            "backend_attention_kernel_workspaces",
            "scheduler_implementation_metadata",
            "offload_transfer_overlap_and_staging",
            "safety_checker_and_watermarker",
        ],
        "notes": [
            "Component parameter and checkpoint byte counts come from selected safetensors headers.",
            "Runtime weight bytes reflect the requested runtime dtype and are separate from checkpoint storage bytes.",
            "Classifier-free guidance duplicates denoiser inputs only for architectures that use a two-branch CFG batch.",
            "UNet and transformer denoisers use separate residual, token, and attention workspace formulas.",
            "VAE slicing processes one image at a time; VAE tiling limits the modeled decode tile.",
            "Model offload keeps only the active text encoder, denoiser, or VAE weights on the GPU.",
            "The point estimate has no accuracy percentage until a matching real-GPU observation is compared.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Estimate architecture-aware, phase-specific memory for "
            "Diffusers generation pipelines."
        ),
    )
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model-profile")
    model_source.add_argument(
        "--model-dir",
        help=(
            "Inspect a local Diffusers pipeline through model_index.json, "
            "component configs, and safetensors headers."
        ),
    )
    parser.add_argument(
        "--weight-variant",
        choices=sorted(_CHECKPOINT_VARIANTS),
        help=(
            "Select bf16, fp16, or fp32 component checkpoint files when "
            "a local directory contains multiple variants."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available diffusion model profiles and exit.",
    )
    parser.add_argument("--height", type=_positive_argument)
    parser.add_argument("--width", type=_positive_argument)
    parser.add_argument(
        "--batch-size",
        type=_positive_argument,
        default=1,
    )
    parser.add_argument("--steps", type=_positive_argument)
    parser.add_argument(
        "--guidance-scale",
        type=_nonnegative_float_argument,
    )
    parser.add_argument("--text-tokens", type=_positive_argument)
    parser.add_argument(
        "--dtype",
        choices=sorted(SUPPORTED_DTYPES),
        default="float16",
    )
    parser.add_argument(
        "--attention-backend",
        choices=sorted(SUPPORTED_ATTENTION_BACKENDS),
        default="sdpa",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
    )
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument(
        "--vae-tile-size",
        type=_positive_argument,
        default=512,
    )
    parser.add_argument(
        "--offload",
        choices=sorted(SUPPORTED_OFFLOAD_MODES),
        default="none",
    )
    parser.add_argument(
        "--runtime-overhead-bytes",
        type=_nonnegative_integer_argument,
        default=0,
    )
    parser.add_argument("--target-profile")
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        profiles = load_diffusion_profiles()
        if args.list_profiles:
            for profile_id, profile in sorted(profiles.items()):
                defaults = _mapping(profile["default_generation"])
                print(
                    f"{profile_id}\t{profile['display_name']}\t"
                    f"{defaults['height']}x{defaults['width']}"
                )
            return 0
        if not args.model_profile and not args.model_dir:
            parser.error(
                "--model-profile or --model-dir is required unless "
                "--list-profiles is used"
            )
        report = estimate_diffusion_generation(
            args.model_profile,
            model_dir=args.model_dir,
            weight_variant=args.weight_variant,
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            text_tokens=args.text_tokens,
            dtype=args.dtype,
            attention_backend=args.attention_backend,
            attention_slicing=args.attention_slicing,
            vae_slicing=args.vae_slicing,
            vae_tiling=args.vae_tiling,
            vae_tile_size=args.vae_tile_size,
            offload=args.offload,
            runtime_overhead_bytes=args.runtime_overhead_bytes,
            target_profile=args.target_profile,
        )
        if args.json_path:
            output = emit_json(args.json_path, report)
            if output is not None:
                print(f"Diffusion estimate: {output}")
        else:
            _print_report(report)
        return 0
    except (OSError, ValueError) as exc:
        usage_error(parser, exc)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiffusionEstimateError(f"cannot read {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiffusionEstimateError(
            f"invalid JSON in {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DiffusionEstimateError(f"expected a JSON object in {path}")
    return payload


def _select_component_checkpoint(
    component_dir: Path,
    *,
    weight_variant: str | None,
) -> tuple[list[str], str, str | None]:
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise DiffusionEstimateError(
            f"no safetensors checkpoint found under {component_dir}"
        )

    families: dict[str, dict[str, Any]] = {}
    for path in files:
        stem = path.name[: -len(".safetensors")]
        family = _SHARD_SUFFIX.sub("", stem)
        parts = family.rsplit(".", 1)
        variant = (
            parts[1]
            if len(parts) == 2 and parts[1] in _CHECKPOINT_VARIANTS
            else None
        )
        families.setdefault(
            family,
            {"variant": variant, "files": []},
        )["files"].append(path.name)

    if weight_variant is not None:
        candidates = [
            family
            for family, metadata in families.items()
            if metadata["variant"] == weight_variant
        ]
    else:
        candidates = [
            family
            for family, metadata in families.items()
            if metadata["variant"] is None
        ]
        if not candidates:
            candidates = list(families)

    if len(candidates) > 1:
        preferred = [
            family
            for family in candidates
            if family in {"diffusion_pytorch_model", "model"}
        ]
        if len(preferred) == 1:
            candidates = preferred
    if len(candidates) != 1:
        choices = ", ".join(sorted(candidates or families))
        requested = weight_variant or "auto"
        raise DiffusionEstimateError(
            f"ambiguous safetensors families under {component_dir} "
            f"for variant {requested!r}: {choices}"
        )

    selected_family = candidates[0]
    metadata = families[selected_family]
    return (
        sorted(metadata["files"]),
        selected_family,
        metadata["variant"],
    )


def _local_profile(
    *,
    root: Path,
    pipeline_class: str,
    declared: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    role_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    denoiser_name = role_names["denoiser"][0]
    vae_name = role_names["vae"][0]
    denoiser_component = declared[denoiser_name]
    vae_component = declared[vae_name]
    denoiser_config = _mapping(denoiser_component["config"])
    vae_config = _mapping(vae_component["config"])

    vae_channels = _config_positive_list(
        vae_config.get("block_out_channels"),
        "vae.block_out_channels",
    )
    latent_channels = _first_positive_config(
        vae_config,
        ("latent_channels", "z_channels"),
        "VAE latent channels",
    )
    latent_scale = 2 ** (len(vae_channels) - 1)
    vae_stats = _component_role_stats(
        components,
        role_names["vae"],
    )
    denoiser_stats = _component_role_stats(
        components,
        role_names["denoiser"],
    )
    conditioning_stats = _component_role_stats(
        components,
        role_names["conditioning"],
    )

    text_widths: list[int] = []
    text_layers: list[int] = []
    text_capacities: list[int] = []
    for component_name in role_names["conditioning"]:
        config = _mapping(declared[component_name]["config"])
        width = _optional_positive_config(
            config,
            ("hidden_size", "d_model", "projection_dim"),
        )
        layers = _optional_positive_config(
            config,
            (
                "num_hidden_layers",
                "num_layers",
                "num_decoder_layers",
            ),
        )
        capacity = _optional_positive_config(
            config,
            (
                "max_position_embeddings",
                "n_positions",
                "model_max_length",
            ),
        )
        if width is not None:
            text_widths.append(width)
        if layers is not None:
            text_layers.append(layers)
        if capacity is not None and capacity <= 1_000_000:
            text_capacities.append(capacity)

    pipeline_defaults = _pipeline_defaults(pipeline_class)
    default_tokens = int(pipeline_defaults["text_tokens"])
    maximum_tokens = max([default_tokens, *text_capacities])
    conditioning_width = _optional_positive_config(
        denoiser_config,
        (
            "joint_attention_dim",
            "cross_attention_dim",
            "caption_projection_dim",
            "encoder_hid_dim",
        ),
    )
    if conditioning_width is None:
        if role_names["conditioning"] and not text_widths:
            raise DiffusionEstimateError(
                "could not determine the conditioning width: none of the "
                "pipeline's text encoder config(s) declare hidden_size, "
                "d_model, or projection_dim, and the denoiser config does "
                "not declare joint_attention_dim/cross_attention_dim/"
                "caption_projection_dim/encoder_hid_dim"
            )
        conditioning_width = max(text_widths, default=1)
    encoder_width = max(text_widths, default=conditioning_width)
    encoder_layers = max(text_layers, default=1)

    configured_class = str(
        denoiser_config.get("_class_name")
        or denoiser_component["class_name"]
    )
    if "UNet" in configured_class:
        denoiser = _local_unet_profile(
            config=denoiser_config,
            model_class=configured_class,
            stats=denoiser_stats,
            conditioning_width=conditioning_width,
        )
        packing_factor = 1
    elif "Transformer" in configured_class:
        denoiser = _local_transformer_profile(
            config=denoiser_config,
            model_class=configured_class,
            stats=denoiser_stats,
            latent_channels=latent_channels,
            has_conditioning=bool(role_names["conditioning"]),
        )
        packing_factor = int(denoiser["packing_factor"])
    else:
        raise DiffusionEstimateError(
            f"unsupported denoiser class {configured_class!r}"
        )

    sample_height, sample_width = _sample_dimensions(
        denoiser_config.get("sample_size")
    )
    default_height = (
        sample_height * latent_scale * packing_factor
        if sample_height is not None
        else None
    )
    default_width = (
        sample_width * latent_scale * packing_factor
        if sample_width is not None
        else None
    )

    return {
        "display_name": root.name,
        "pipeline_class": pipeline_class,
        "status": "local-inspected",
        "default_generation": {
            "height": default_height,
            "width": default_width,
            "steps": int(pipeline_defaults["steps"]),
            "guidance_scale": float(
                pipeline_defaults["guidance_scale"]
            ),
        },
        "latent": {
            "scale_factor": latent_scale,
            "channels": latent_channels,
        },
        "conditioning": {
            "enabled": bool(role_names["conditioning"]),
            "parameter_count": conditioning_stats[
                "parameter_count"
            ],
            "checkpoint_bytes": conditioning_stats[
                "checkpoint_bytes"
            ],
            "checkpoint_dtype_bytes": conditioning_stats[
                "dtype_bytes"
            ],
            "max_tokens": maximum_tokens,
            "default_tokens": default_tokens,
            "width": conditioning_width,
            "encoder_width": encoder_width,
            "layers": encoder_layers,
            "components": list(role_names["conditioning"]),
        },
        "denoiser": denoiser,
        "vae": {
            "parameter_count": vae_stats["parameter_count"],
            "checkpoint_bytes": vae_stats["checkpoint_bytes"],
            "checkpoint_dtype_bytes": vae_stats["dtype_bytes"],
            "block_out_channels": vae_channels,
            "layers_per_block": _layers_per_block(vae_config),
            "model_class": str(
                vae_config.get("_class_name")
                or vae_component["class_name"]
            ),
        },
        "source": {
            "repository": str(root),
            "revision": "local",
            "parameter_source": "selected local safetensors headers",
        },
    }


def _local_unet_profile(
    *,
    config: Mapping[str, Any],
    model_class: str,
    stats: Mapping[str, Any],
    conditioning_width: int,
) -> dict[str, Any]:
    channels = _config_positive_list(
        config.get("block_out_channels"),
        "denoiser.block_out_channels",
    )
    level_count = len(channels)
    down_block_types = list(config.get("down_block_types") or [])
    heads = _per_level_counts(
        config.get("num_attention_heads")
        or config.get("attention_head_dim")
        or 1,
        level_count=level_count,
        name="denoiser attention heads",
    )
    transformer_layers = _per_level_counts(
        config.get("transformer_layers_per_block") or 1,
        level_count=level_count,
        name="denoiser transformer layers",
    )
    if len(down_block_types) == level_count:
        for index, block_type in enumerate(down_block_types):
            if "Attn" not in str(block_type):
                heads[index] = 0
                transformer_layers[index] = 0
    layers_by_level = _per_level_counts(
        config.get("layers_per_block") or 1,
        level_count=level_count,
        name="denoiser layers per block",
    )
    return {
        "architecture": "unet",
        "model_class": model_class,
        "guidance_mode": "classifier_free",
        "parameter_count": int(stats["parameter_count"]),
        "checkpoint_bytes": int(stats["checkpoint_bytes"]),
        "checkpoint_dtype_bytes": copy.deepcopy(
            stats["dtype_bytes"]
        ),
        "block_out_channels": channels,
        "layers_per_block": max(layers_by_level),
        "layers_per_block_by_level": layers_by_level,
        "attention_heads": heads,
        "transformer_layers_per_block": transformer_layers,
        "cross_attention_dim": conditioning_width,
    }


def _local_transformer_profile(
    *,
    config: Mapping[str, Any],
    model_class: str,
    stats: Mapping[str, Any],
    latent_channels: int,
    has_conditioning: bool,
) -> dict[str, Any]:
    heads = _first_positive_config(
        config,
        ("num_attention_heads", "attention_heads", "num_heads"),
        "transformer attention heads",
    )
    head_dim = _optional_positive_config(
        config,
        ("attention_head_dim", "head_dim"),
    )
    if head_dim is None:
        inner_dim = _first_positive_config(
            config,
            ("inner_dim", "hidden_size"),
            "transformer hidden width",
        )
        if inner_dim % heads:
            raise DiffusionEstimateError(
                "transformer hidden width is not divisible by its heads"
            )
        head_dim = inner_dim // heads
    layers = _first_positive_config(
        config,
        ("num_layers", "num_hidden_layers", "depth"),
        "transformer layer count",
    )
    single_layers = _optional_positive_config(
        config,
        ("num_single_layers",),
    ) or 0
    patch_size = _first_positive_config(
        config,
        ("patch_size",),
        "transformer patch size",
    )
    in_channels = _first_positive_config(
        config,
        ("in_channels",),
        "transformer input channels",
    )
    packing_factor = 1
    if model_class.startswith("Flux") and in_channels > latent_channels:
        ratio = in_channels // latent_channels
        candidate = math.isqrt(ratio)
        if (
            in_channels % latent_channels == 0
            and candidate * candidate == ratio
        ):
            packing_factor = candidate

    if (
        model_class in _TRANSFORMER_JOINT_CLASSES
        or config.get("joint_attention_dim") is not None
    ):
        attention_pattern = "joint"
    elif has_conditioning:
        attention_pattern = "cross"
    else:
        attention_pattern = "self"
    guidance_mode = (
        "embedded"
        if model_class.startswith("Flux")
        else "classifier_free"
        if has_conditioning
        else "none"
    )
    return {
        "architecture": "transformer",
        "model_class": model_class,
        "attention_pattern": attention_pattern,
        "guidance_mode": guidance_mode,
        "parameter_count": int(stats["parameter_count"]),
        "checkpoint_bytes": int(stats["checkpoint_bytes"]),
        "checkpoint_dtype_bytes": copy.deepcopy(
            stats["dtype_bytes"]
        ),
        "num_layers": layers,
        "num_single_layers": single_layers,
        "attention_heads": heads,
        "attention_head_dim": head_dim,
        "patch_size": patch_size,
        "packing_factor": packing_factor,
        "mlp_ratio": float(config.get("mlp_ratio", 4.0)),
        "cross_attention_dim": _optional_positive_config(
            config,
            (
                "joint_attention_dim",
                "cross_attention_dim",
                "caption_projection_dim",
            ),
        )
        or heads * head_dim,
    }


def _component_role_stats(
    components: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, Any]:
    parameter_count = 0
    checkpoint_bytes = 0
    dtype_bytes: dict[str, int] = {}
    for name in names:
        checkpoint = _mapping(components[name]["checkpoint"])
        parameter_count += int(checkpoint["parameter_count"])
        checkpoint_bytes += int(checkpoint["checkpoint_bytes"])
        for dtype, byte_count in _mapping(
            checkpoint["dtype_bytes"]
        ).items():
            dtype_bytes[str(dtype)] = (
                dtype_bytes.get(str(dtype), 0) + int(byte_count)
            )
    return {
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_bytes,
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
    }


def _runtime_checkpoint_bytes(
    *,
    parameter_count: int,
    dtype_bytes: Mapping[str, Any],
    runtime_float_bytes: int,
) -> int:
    if not dtype_bytes:
        return parameter_count * runtime_float_bytes
    accounted_parameters = 0
    runtime_bytes = 0
    for dtype, raw_byte_count in dtype_bytes.items():
        byte_count = int(raw_byte_count)
        storage_width = _SAFETENSORS_DTYPE_BYTES.get(str(dtype))
        if storage_width is None or byte_count % storage_width:
            return parameter_count * runtime_float_bytes
        dtype_parameters = byte_count // storage_width
        accounted_parameters += dtype_parameters
        runtime_bytes += (
            dtype_parameters * runtime_float_bytes
            if str(dtype) in _FLOAT_SAFETENSORS_DTYPES
            else byte_count
        )
    if accounted_parameters > parameter_count:
        raise DiffusionEstimateError(
            "checkpoint dtype byte counts exceed parameter count"
        )
    return (
        runtime_bytes
        + (parameter_count - accounted_parameters)
        * runtime_float_bytes
    )


def _pipeline_defaults(pipeline_class: str) -> dict[str, Any]:
    if pipeline_class in _PIPELINE_DEFAULTS:
        return copy.deepcopy(_PIPELINE_DEFAULTS[pipeline_class])
    return {
        "steps": 50,
        "guidance_scale": 7.5,
        "text_tokens": 77,
    }


def _sample_dimensions(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value, value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item > 0
            for item in value
        )
    ):
        return int(value[0]), int(value[1])
    return None, None


def _layers_per_block(config: Mapping[str, Any]) -> int:
    value = config.get("layers_per_block", 1)
    if isinstance(value, list):
        values = [
            int(item)
            for item in value
            if isinstance(item, int)
            and not isinstance(item, bool)
            and item > 0
        ]
        if values:
            return max(values)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise DiffusionEstimateError(
        "layers_per_block must contain positive integers"
    )


def _per_level_counts(
    value: Any,
    *,
    level_count: int,
    name: str,
) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return [value] * level_count
    if not isinstance(value, list) or len(value) != level_count:
        raise DiffusionEstimateError(
            f"{name} must be an integer or an array of {level_count} items"
        )
    result: list[int] = []
    for item in value:
        if isinstance(item, list):
            if not item or any(
                not isinstance(entry, int)
                or isinstance(entry, bool)
                or entry < 0
                for entry in item
            ):
                raise DiffusionEstimateError(
                    f"{name} contains invalid nested values"
                )
            result.append(sum(item))
        elif (
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
        ):
            result.append(item)
        else:
            raise DiffusionEstimateError(
                f"{name} must contain non-negative integers"
            )
    return result


def _config_positive_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DiffusionEstimateError(
            f"{name} must be a non-empty integer array"
        )
    result: list[int] = []
    for item in value:
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
        ):
            raise DiffusionEstimateError(
                f"{name} must contain positive integers"
            )
        result.append(item)
    return result


def _optional_positive_config(
    config: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, list):
            values = [
                int(item)
                for item in value
                if isinstance(item, int)
                and not isinstance(item, bool)
                and item > 0
            ]
            if values:
                return max(values)
        elif (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            return value
    return None


def _first_positive_config(
    config: Mapping[str, Any],
    keys: Sequence[str],
    description: str,
) -> int:
    value = _optional_positive_config(config, keys)
    if value is None:
        joined = ", ".join(keys)
        raise DiffusionEstimateError(
            f"cannot derive {description}; expected one of {joined}"
        )
    return value


def _validate_profile(
    profile_id: str,
    profile: Mapping[str, Any],
) -> None:
    for key in (
        "display_name",
        "pipeline_class",
        "status",
        "default_generation",
        "latent",
        "conditioning",
        "denoiser",
        "vae",
        "source",
    ):
        if key not in profile:
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r} is missing {key!r}"
            )
    if profile["status"] not in {"reference", "synthetic"}:
        raise DiffusionEstimateError(
            f"diffusion profile {profile_id!r} has invalid status"
        )
    defaults = _mapping(profile["default_generation"])
    latent = _mapping(profile["latent"])
    conditioning = _mapping(profile["conditioning"])
    denoiser = _mapping(profile["denoiser"])
    vae = _mapping(profile["vae"])
    for name, value in (
        ("default_generation.height", defaults.get("height")),
        ("default_generation.width", defaults.get("width")),
        ("default_generation.steps", defaults.get("steps")),
        ("latent.scale_factor", latent.get("scale_factor")),
        ("latent.channels", latent.get("channels")),
        (
            "conditioning.parameter_count",
            conditioning.get("parameter_count"),
        ),
        ("conditioning.max_tokens", conditioning.get("max_tokens")),
        ("conditioning.width", conditioning.get("width")),
        ("conditioning.layers", conditioning.get("layers")),
        ("denoiser.parameter_count", denoiser.get("parameter_count")),
        ("vae.parameter_count", vae.get("parameter_count")),
        ("vae.layers_per_block", vae.get("layers_per_block")),
    ):
        try:
            _positive_integer(value, name)
        except DiffusionEstimateError as exc:
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r}: {exc}"
            ) from exc
    _nonnegative_number(
        defaults.get("guidance_scale"),
        "default_generation.guidance_scale",
    )
    default_tokens = int(
        conditioning.get("default_tokens", conditioning["max_tokens"])
    )
    if default_tokens <= 0 or default_tokens > int(
        conditioning["max_tokens"]
    ):
        raise DiffusionEstimateError(
            f"diffusion profile {profile_id!r}: "
            "conditioning.default_tokens must be in [1, max_tokens]"
        )
    architecture = str(denoiser.get("architecture") or "")
    if architecture not in SUPPORTED_DENOISER_ARCHITECTURES:
        raise DiffusionEstimateError(
            f"diffusion profile {profile_id!r}: unsupported denoiser "
            f"architecture {architecture!r}"
        )
    guidance_mode = str(
        denoiser.get("guidance_mode") or "classifier_free"
    )
    if guidance_mode not in {"classifier_free", "embedded", "none"}:
        raise DiffusionEstimateError(
            f"diffusion profile {profile_id!r}: invalid guidance_mode"
        )
    if architecture == "unet":
        _positive_integer(
            denoiser.get("layers_per_block"),
            "denoiser.layers_per_block",
        )
        block_channels = _positive_integer_list(
            denoiser.get("block_out_channels"),
            "denoiser.block_out_channels",
        )
        attention_heads = _nonnegative_integer_list(
            denoiser.get("attention_heads"),
            "denoiser.attention_heads",
        )
        transformer_layers = _nonnegative_integer_list(
            denoiser.get("transformer_layers_per_block"),
            "denoiser.transformer_layers_per_block",
        )
        if not (
            len(block_channels)
            == len(attention_heads)
            == len(transformer_layers)
        ):
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r}: denoiser block "
                "arrays must have equal lengths"
            )
        _positive_integer(
            denoiser.get("cross_attention_dim"),
            "denoiser.cross_attention_dim",
        )
    else:
        for name in (
            "num_layers",
            "attention_heads",
            "attention_head_dim",
            "patch_size",
            "packing_factor",
        ):
            _positive_integer(
                denoiser.get(name),
                f"denoiser.{name}",
            )
        _nonnegative_integer(
            denoiser.get("num_single_layers", 0),
            "denoiser.num_single_layers",
        )
        if denoiser.get("attention_pattern") not in {
            "self",
            "cross",
            "joint",
        }:
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r}: invalid transformer "
                "attention_pattern"
            )
    _positive_integer_list(
        vae.get("block_out_channels"),
        "vae.block_out_channels",
    )
    scale = int(latent["scale_factor"])
    if (
        int(defaults["height"]) % scale != 0
        or int(defaults["width"]) % scale != 0
    ):
        raise DiffusionEstimateError(
            f"diffusion profile {profile_id!r}: default dimensions "
            "must be divisible by latent.scale_factor"
        )
    source = _mapping(profile["source"])
    for key in ("repository", "revision", "parameter_source"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise DiffusionEstimateError(
                f"diffusion profile {profile_id!r}: "
                f"source.{key} must be a non-empty string"
            )


def _text_encoder_activation(
    *,
    batch_size: int,
    tokens: int,
    width: int,
    element_bytes: int,
) -> dict[str, int | str]:
    hidden_state_bytes = (
        batch_size * tokens * width * element_bytes
    )
    return {
        "hidden_state_bytes": hidden_state_bytes,
        "working_tensor_bytes": 4 * hidden_state_bytes,
        "peak_bytes": 5 * hidden_state_bytes,
        "formula": (
            "hidden_state + four hidden-state-sized attention/MLP "
            "working tensors"
        ),
    }


def _denoiser_activation(
    *,
    profile: Mapping[str, Any],
    latent_height: int,
    latent_width: int,
    batch_size: int,
    text_tokens: int,
    element_bytes: int,
    attention_backend: str,
    attention_slicing: bool,
) -> dict[str, Any]:
    architecture = str(profile.get("architecture") or "")
    kwargs = {
        "profile": profile,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "text_tokens": text_tokens,
        "element_bytes": element_bytes,
        "attention_backend": attention_backend,
        "attention_slicing": attention_slicing,
    }
    if architecture == "unet":
        return _unet_denoiser_activation(**kwargs)
    if architecture == "transformer":
        return _transformer_denoiser_activation(**kwargs)
    raise DiffusionEstimateError(
        f"unsupported denoiser architecture {architecture!r}"
    )


def _unet_denoiser_activation(
    *,
    profile: Mapping[str, Any],
    latent_height: int,
    latent_width: int,
    batch_size: int,
    text_tokens: int,
    element_bytes: int,
    attention_backend: str,
    attention_slicing: bool,
) -> dict[str, Any]:
    channels = [int(value) for value in profile["block_out_channels"]]
    heads = [int(value) for value in profile["attention_heads"]]
    transformer_layers = [
        int(value)
        for value in profile["transformer_layers_per_block"]
    ]
    layers_by_level = [
        int(value)
        for value in profile.get(
            "layers_per_block_by_level",
            [int(profile["layers_per_block"])] * len(channels),
        )
    ]
    levels: list[dict[str, Any]] = []
    retained_residual_bytes = 0
    maximum_working_bytes = 0
    maximum_attention_bytes = 0
    for level, channel_count in enumerate(channels):
        level_height = _ceil_div(latent_height, 2**level)
        level_width = _ceil_div(latent_width, 2**level)
        spatial_tokens = level_height * level_width
        feature_bytes = (
            batch_size
            * spatial_tokens
            * channel_count
            * element_bytes
        )
        retained_bytes = feature_bytes * (
            layers_by_level[level] + 1
        )
        convolution_working_bytes = 3 * feature_bytes
        attention_bytes = 0
        if transformer_layers[level] > 0 and heads[level] > 0:
            query_bytes = feature_bytes
            key_value_bytes = (
                batch_size
                * text_tokens
                * channel_count
                * element_bytes
                * 2
            )
            if attention_backend == "eager":
                score_bytes = (
                    batch_size
                    * heads[level]
                    * spatial_tokens
                    * text_tokens
                    * element_bytes
                )
                if attention_slicing:
                    score_bytes = _ceil_div(
                        score_bytes,
                        heads[level],
                    )
                attention_bytes = (
                    query_bytes + key_value_bytes + score_bytes
                )
            else:
                linear_workspace = feature_bytes
                if attention_slicing:
                    linear_workspace = _ceil_div(
                        linear_workspace,
                        heads[level],
                    )
                attention_bytes = (
                    query_bytes
                    + key_value_bytes
                    + linear_workspace
                )
        retained_residual_bytes += retained_bytes
        maximum_working_bytes = max(
            maximum_working_bytes,
            convolution_working_bytes,
        )
        maximum_attention_bytes = max(
            maximum_attention_bytes,
            attention_bytes,
        )
        levels.append(
            {
                "level": level,
                "height": level_height,
                "width": level_width,
                "channels": channel_count,
                "feature_bytes": feature_bytes,
                "retained_residual_bytes": retained_bytes,
                "convolution_working_bytes": (
                    convolution_working_bytes
                ),
                "attention_workspace_bytes": attention_bytes,
                "attention_heads": heads[level],
                "transformer_layers": transformer_layers[level],
            }
        )
    return {
        "levels": levels,
        "retained_residual_bytes": retained_residual_bytes,
        "working_tensor_bytes": maximum_working_bytes,
        "attention_workspace_bytes": maximum_attention_bytes,
        "peak_bytes": (
            retained_residual_bytes
            + maximum_working_bytes
            + maximum_attention_bytes
        ),
        "attention_backend": attention_backend,
        "attention_slicing": attention_slicing,
        "formula": (
            "retained UNet residual features + maximum convolution "
            "working set + maximum cross-attention workspace"
        ),
    }


def _transformer_denoiser_activation(
    *,
    profile: Mapping[str, Any],
    latent_height: int,
    latent_width: int,
    batch_size: int,
    text_tokens: int,
    element_bytes: int,
    attention_backend: str,
    attention_slicing: bool,
) -> dict[str, Any]:
    patch_size = int(profile["patch_size"])
    packing_factor = int(profile.get("packing_factor", 1))
    token_stride = patch_size * packing_factor
    token_height = _ceil_div(latent_height, token_stride)
    token_width = _ceil_div(latent_width, token_stride)
    image_tokens = token_height * token_width
    heads = int(profile["attention_heads"])
    head_dim = int(profile["attention_head_dim"])
    hidden_width = heads * head_dim
    image_state_bytes = (
        batch_size
        * image_tokens
        * hidden_width
        * element_bytes
    )

    attention_pattern = str(
        profile.get("attention_pattern") or "self"
    )
    projected_text_bytes = (
        batch_size
        * text_tokens
        * hidden_width
        * element_bytes
        if attention_pattern in {"cross", "joint"}
        else 0
    )
    retained_residual_bytes = (
        2 * image_state_bytes
        + (
            projected_text_bytes
            if attention_pattern == "joint"
            else 0
        )
    )

    if attention_pattern == "joint":
        qkv_bytes = 3 * (
            image_state_bytes + projected_text_bytes
        )
    elif attention_pattern == "cross":
        qkv_bytes = (
            3 * image_state_bytes + 2 * projected_text_bytes
        )
    else:
        qkv_bytes = 3 * image_state_bytes

    mlp_ratio = float(profile.get("mlp_ratio", 4.0))
    if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
        raise DiffusionEstimateError(
            "denoiser.mlp_ratio must be a finite positive number"
        )
    mlp_bytes = math.ceil(image_state_bytes * mlp_ratio)
    if attention_pattern == "joint":
        mlp_bytes += math.ceil(
            projected_text_bytes * mlp_ratio
        )
    working_tensor_bytes = max(qkv_bytes, mlp_bytes)

    if attention_backend == "eager":
        if attention_pattern == "joint":
            sequence = image_tokens + text_tokens
            attention_workspace_bytes = (
                batch_size
                * heads
                * sequence
                * sequence
                * element_bytes
            )
        else:
            attention_workspace_bytes = (
                batch_size
                * heads
                * image_tokens
                * image_tokens
                * element_bytes
            )
            if attention_pattern == "cross":
                attention_workspace_bytes += (
                    batch_size
                    * heads
                    * image_tokens
                    * text_tokens
                    * element_bytes
                )
        if attention_slicing:
            attention_workspace_bytes = _ceil_div(
                attention_workspace_bytes,
                heads,
            )
    else:
        attention_workspace_bytes = (
            2 * image_state_bytes + projected_text_bytes
        )
        if attention_slicing:
            attention_workspace_bytes = _ceil_div(
                attention_workspace_bytes,
                heads,
            )

    return {
        "architecture": "transformer",
        "model_class": profile.get("model_class"),
        "attention_pattern": attention_pattern,
        "patch_size": patch_size,
        "packing_factor": packing_factor,
        "token_stride": token_stride,
        "token_height": token_height,
        "token_width": token_width,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "hidden_width": hidden_width,
        "attention_heads": heads,
        "attention_head_dim": head_dim,
        "num_layers": int(profile["num_layers"]),
        "num_single_layers": int(
            profile.get("num_single_layers", 0)
        ),
        "image_state_bytes": image_state_bytes,
        "projected_text_bytes": projected_text_bytes,
        "retained_residual_bytes": retained_residual_bytes,
        "working_tensor_bytes": working_tensor_bytes,
        "attention_workspace_bytes": attention_workspace_bytes,
        "peak_bytes": (
            retained_residual_bytes
            + working_tensor_bytes
            + attention_workspace_bytes
        ),
        "attention_backend": attention_backend,
        "attention_slicing": attention_slicing,
        "formula": (
            "patch-token residual state + maximum QKV/MLP working set "
            "+ architecture-specific self/cross/joint attention workspace"
        ),
    }


def _vae_decode_activation(
    *,
    profile: Mapping[str, Any],
    height: int,
    width: int,
    batch_size: int,
    latent_scale: int,
    element_bytes: int,
    slicing: bool,
    tiling: bool,
    tile_size: int,
) -> dict[str, Any]:
    processed_batch = 1 if slicing else batch_size
    processed_height = min(height, tile_size) if tiling else height
    processed_width = min(width, tile_size) if tiling else width
    latent_height = _ceil_div(processed_height, latent_scale)
    latent_width = _ceil_div(processed_width, latent_scale)
    channels = [
        int(value)
        for value in reversed(profile["block_out_channels"])
    ]
    layers_per_block = int(profile["layers_per_block"])
    stages: list[dict[str, int]] = []
    retained_feature_bytes = 0
    maximum_working_bytes = 0
    for stage, channel_count in enumerate(channels):
        stage_height = min(
            processed_height,
            latent_height * 2**stage,
        )
        stage_width = min(
            processed_width,
            latent_width * 2**stage,
        )
        feature_bytes = (
            processed_batch
            * stage_height
            * stage_width
            * channel_count
            * element_bytes
        )
        retained_bytes = feature_bytes * (layers_per_block + 1)
        working_bytes = 2 * feature_bytes
        retained_feature_bytes = max(
            retained_feature_bytes,
            retained_bytes,
        )
        maximum_working_bytes = max(
            maximum_working_bytes,
            working_bytes,
        )
        stages.append(
            {
                "stage": stage,
                "height": stage_height,
                "width": stage_width,
                "channels": channel_count,
                "feature_bytes": feature_bytes,
                "retained_feature_bytes": retained_bytes,
                "working_tensor_bytes": working_bytes,
            }
        )
    return {
        "processed_batch_size": processed_batch,
        "processed_height": processed_height,
        "processed_width": processed_width,
        "slicing": slicing,
        "tiling": tiling,
        "tile_size": tile_size if tiling else None,
        "stages": stages,
        "retained_feature_bytes": retained_feature_bytes,
        "working_tensor_bytes": maximum_working_bytes,
        "peak_bytes": (
            retained_feature_bytes + maximum_working_bytes
        ),
        "formula": (
            "maximum live decoder-stage feature set + maximum "
            "two-feature working set; decoder stages are sequential"
        ),
    }


def _phase(
    name: str,
    components: Mapping[str, int],
) -> dict[str, Any]:
    normalized = {
        str(key): int(value) for key, value in components.items()
    }
    return {
        "phase": name,
        "peak_bytes": sum(normalized.values()),
        "components": normalized,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiffusionEstimateError("expected an object")
    return value


def _generation_dimension(
    requested: int | None,
    configured: Any,
    name: str,
) -> int:
    if requested is not None:
        return _positive_integer(requested, name)
    if (
        isinstance(configured, int)
        and not isinstance(configured, bool)
        and configured > 0
    ):
        return configured
    raise DiffusionEstimateError(
        f"{name} is required because the local denoiser config has no "
        "positive sample_size"
    )


def _positive_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise DiffusionEstimateError(
            f"{name} must be a positive integer"
        )
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise DiffusionEstimateError(
            f"{name} must be a non-negative integer"
        )
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        )
    return number


def _positive_integer_list(
    value: Any,
    name: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DiffusionEstimateError(
            f"{name} must be a non-empty integer array"
        )
    return [
        _positive_integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _nonnegative_integer_list(
    value: Any,
    name: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DiffusionEstimateError(
            f"{name} must be a non-empty integer array"
        )
    return [
        _nonnegative_integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "expected a positive integer"
        )
    return parsed


def _nonnegative_integer_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
        )
    return parsed


def _nonnegative_float_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "expected a finite non-negative number"
        )
    return parsed


def _format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TiB"


def _print_report(report: Mapping[str, Any]) -> None:
    model = _mapping(report["model_profile"])
    timeline = _mapping(report["memory_timeline"])
    print("FakeGPU diffusion generation estimate")
    print(f"  model profile: {model['id']} ({model['display_name']})")
    print(f"  validation status: {report['validation_status']}")
    print(f"  peak phase: {timeline['peak_phase']}")
    print(f"  estimated peak: {_format_bytes(int(timeline['peak_bytes']))}")
    for phase in timeline["phases"]:
        print(
            f"  {phase['phase']}: "
            f"{_format_bytes(int(phase['peak_bytes']))}"
        )
    fit = report.get("fit")
    if isinstance(fit, Mapping):
        print(
            f"  target: {fit['target_profile']} "
            f"({'fits' if fit['fits'] else 'does not fit'})"
        )
    print(
        "  scope: analytical component and tensor-shape model; "
        "not a real-GPU measurement"
    )


__all__ = [
    "DiffusionEstimateError",
    "INSPECTION_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "estimate_diffusion_generation",
    "inspect_diffusion_pipeline",
    "load_diffusion_profiles",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
