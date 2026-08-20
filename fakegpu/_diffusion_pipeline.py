"""Reading a diffusion pipeline on disk, and the profiles it is matched to.

Split out of ``diffusion_estimator`` unchanged. One half inspects a local
``diffusers`` checkout — which components it has, which checkpoint variant
each one loads, and the denoiser, text-encoder, and VAE shapes read from
their configs. The other half loads and validates a
``fakegpu.diffusion_profiles.v1`` catalog, which supplies the same shapes
for a pipeline that is not on disk.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._diffusion_types import (
    DiffusionEstimateError,
    _mapping,
    _nonnegative_integer,
    _nonnegative_integer_list,
    _nonnegative_number,
    _positive_integer,
    _positive_integer_list,
)
from .llm_estimator import _DTYPE_BYTES as _SAFETENSORS_DTYPE_BYTES
from .llm_estimator import inspect_safetensors_checkpoint


INSPECTION_SCHEMA_VERSION = "fakegpu.diffusion_pipeline_inspection.v1"


PROFILE_SCHEMA_VERSION = "fakegpu.diffusion_profiles.v1"


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


def _validate_local_profile(
    profile_id: str,
    profile: Mapping[str, Any],
) -> None:
    _validate_profile(profile_id, profile, allow_local=True)


def _validate_profile(
    profile_id: str,
    profile: Mapping[str, Any],
    *,
    allow_local: bool = False,
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
    allowed_statuses = {"reference", "synthetic"}
    if allow_local:
        allowed_statuses.add("local-inspected")
    if profile["status"] not in allowed_statuses:
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
            if (
                allow_local
                and name in {
                    "default_generation.height",
                    "default_generation.width",
                }
                and value is None
            ):
                continue
            if allow_local and name == "conditioning.parameter_count":
                _nonnegative_integer(value, name)
            else:
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
        defaults.get("height") is not None
        and defaults.get("width") is not None
        and (
            int(defaults["height"]) % scale != 0
            or int(defaults["width"]) % scale != 0
        )
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
    _validate_local_profile(root.name, profile)
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
