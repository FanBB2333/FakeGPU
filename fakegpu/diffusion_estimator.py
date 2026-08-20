from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    usage_error,
)
from ._diffusion_activation import (
    _denoiser_activation,
    _phase,
    _text_encoder_activation,
    _vae_decode_activation,
)
from ._diffusion_pipeline import (
    INSPECTION_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    _CHECKPOINT_VARIANTS,
    _runtime_checkpoint_bytes,
    inspect_diffusion_pipeline,
    load_diffusion_profiles,
)
from ._diffusion_types import (
    DiffusionEstimateError,
    _mapping,
    _nonnegative_integer,
    _nonnegative_number,
    _positive_integer,
)
from .profile_catalog import get_profile
from .structured_io import emit_json


SCHEMA_VERSION = "fakegpu.diffusion_generation_estimate.v1"
SUPPORTED_DTYPES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}
SUPPORTED_ATTENTION_BACKENDS = {"eager", "sdpa"}
SUPPORTED_OFFLOAD_MODES = {"none", "model"}
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
