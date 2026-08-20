"""Activation-memory model for one diffusion phase at a time.

Split out of ``diffusion_estimator`` unchanged: how many transient bytes a
text encoder, a UNet or transformer denoiser step, and the VAE decode each
hold live, given the shapes read from the pipeline and the attention
backend in use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._diffusion_types import DiffusionEstimateError, _ceil_div


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
