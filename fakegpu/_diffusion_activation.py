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
    return {
        "phase": name,
        "peak_bytes": sum(components.values()),
        "components": dict(components),
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
    model = (
        _unet_denoiser_activation
        if profile["architecture"] == "unet"
        else _transformer_denoiser_activation
    )
    return model(
        profile=profile,
        latent_height=latent_height,
        latent_width=latent_width,
        batch_size=batch_size,
        text_tokens=text_tokens,
        element_bytes=element_bytes,
        attention_backend=attention_backend,
        attention_slicing=attention_slicing,
    )


def _unet_materialized_attention(
    *,
    batch_size: int,
    spatial_tokens: int,
    text_tokens: int,
    channels: int,
    heads: int,
    element_bytes: int,
    only_cross_attention: bool,
    upcast_attention: bool,
    slicing: bool,
) -> dict[str, Any]:
    """Live buffers in standard Diffusers AttnProcessor/SlicedAttnProcessor.

    Scores and softmax probabilities coexist. Slicing replaces the processor,
    including when the requested backend is SDPA, and chunks the flattened
    batch/head axis using ``heads // 2`` for native ``slice_size='auto'``.
    """
    head_dim = channels // heads
    if head_dim == 0:
        raise DiffusionEstimateError(
            "standard UNet attention requires at least one channel per head"
        )
    score_heads = heads // 2 if slicing else batch_size * heads
    if score_heads == 0:
        raise DiffusionEstimateError(
            "Diffusers automatic attention slicing requires at least two heads"
        )
    projection_channels = heads * head_dim
    feature_bytes = batch_size * spatial_tokens * projection_channels * element_bytes
    score_element_bytes = 4 if upcast_attention else element_bytes
    candidates = {}
    kinds = ("cross",) if only_cross_attention else ("self", "cross")
    for kind in kinds:
        key_tokens = spatial_tokens if kind == "self" else text_tokens
        qkv_bytes = feature_bytes + 2 * batch_size * key_tokens * projection_channels * element_bytes
        score_bytes = score_heads * spatial_tokens * key_tokens * score_element_bytes
        # get_attention_scores casts Q/K while the caller retains the original
        # projections. Slice views only cast the active flattened head rows.
        cast_bytes = (
            score_heads * (spatial_tokens + key_tokens) * head_dim * 4
            if upcast_attention and element_bytes != 4
            else 0
        )
        if slicing:
            # The full output buffer is allocated before the slice loop. On
            # subsequent iterations the previous bmm slice remains referenced
            # until the next get_attention_scores call returns.
            previous_slice_bytes = score_heads * spatial_tokens * head_dim * element_bytes
            output_bytes = feature_bytes
            score_peak = output_bytes + previous_slice_bytes + cast_bytes + 2 * score_bytes
            # During bmm, probabilities and the newly computed output slice
            # coexist with the preallocated full output.
            probability_bytes = score_heads * spatial_tokens * key_tokens * element_bytes
            value_peak = output_bytes + probability_bytes + previous_slice_bytes
            projection_peak = 2 * feature_bytes + previous_slice_bytes
        else:
            previous_slice_bytes = 0
            output_bytes = 0
            score_peak = cast_bytes + 2 * score_bytes
            probability_bytes = batch_size * heads * spatial_tokens * key_tokens * element_bytes
            value_peak = probability_bytes + 2 * feature_bytes
            projection_peak = value_peak
        candidates[kind] = {
            "query_tokens": spatial_tokens,
            "key_tokens": key_tokens,
            "score_head_rows": score_heads,
            "projection_channels": projection_channels,
            "score_element_bytes": score_element_bytes,
            "qkv_bytes": qkv_bytes,
            "qk_upcast_bytes": cast_bytes,
            "score_buffer_bytes": score_bytes,
            "simultaneous_score_buffer_bytes": 2 * score_bytes,
            "sliced_output_bytes": output_bytes,
            "previous_slice_output_bytes": previous_slice_bytes,
            "peak_bytes": qkv_bytes + max(score_peak, value_peak, projection_peak),
        }
    return {
        "processor": "SlicedAttnProcessor(auto)" if slicing else "AttnProcessor",
        "self_attention": not only_cross_attention,
        "upcast_attention": upcast_attention,
        "candidates": candidates,
        "peak_bytes": max(value["peak_bytes"] for value in candidates.values()),
    }


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
    maximum_feed_forward_bytes = 0
    uses_geglu = profile.get("model_class") == "UNet2DConditionModel"
    materialized_attention = attention_backend == "eager" or attention_slicing
    standard_attention = uses_geglu and profile.get("standard_transformer_attention", True)
    use_attention_buffers = materialized_attention and standard_attention
    only_cross_by_level = profile.get("only_cross_attention_by_level", [False] * len(channels))
    upcast_attention = bool(profile.get("upcast_attention", False))
    if use_attention_buffers and (
        not isinstance(only_cross_by_level, (list, tuple))
        or len(only_cross_by_level) != len(channels)
        or any(type(value) is not bool for value in only_cross_by_level)
    ):
        raise DiffusionEstimateError("only_cross_attention_by_level requires one boolean per level")
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
        feed_forward_bytes = 0
        attention_bytes = 0
        attention_buffers = None
        if transformer_layers[level] > 0 and heads[level] > 0:
            if uses_geglu:
                # Diffusers' standard conditional UNet uses a 4x GEGLU
                # expansion. Its two chunk views retain the 8F projection
                # while the 4F GELU result and 4F product are both live.
                feed_forward_bytes = (8 + 4 + 4) * feature_bytes
            if use_attention_buffers:
                attention_buffers = _unet_materialized_attention(
                    batch_size=batch_size,
                    spatial_tokens=spatial_tokens,
                    text_tokens=text_tokens,
                    channels=channel_count,
                    heads=heads[level],
                    element_bytes=element_bytes,
                    only_cross_attention=only_cross_by_level[level],
                    upcast_attention=upcast_attention,
                    slicing=attention_slicing,
                )
                attention_bytes = attention_buffers["peak_bytes"]
            else:
                workspace = (
                    batch_size
                    * heads[level]
                    * spatial_tokens
                    * text_tokens
                    * element_bytes
                    if attention_backend == "eager"
                    else feature_bytes
                )
                if attention_slicing:
                    workspace = _ceil_div(workspace, heads[level])
                attention_bytes = (
                    feature_bytes
                    + 2 * batch_size * text_tokens * channel_count * element_bytes
                    + workspace
                )
        retained_residual_bytes += retained_bytes
        maximum_working_bytes = max(
            maximum_working_bytes,
            convolution_working_bytes,
            feed_forward_bytes,
        )
        maximum_feed_forward_bytes = max(
            maximum_feed_forward_bytes,
            feed_forward_bytes,
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
                "feed_forward_working_bytes": feed_forward_bytes,
                "attention_workspace_bytes": attention_bytes,
                "attention_heads": heads[level],
                "transformer_layers": transformer_layers[level],
            }
        )
        if attention_buffers is not None:
            levels[-1]["attention_buffers"] = attention_buffers
    mid_attention = None
    if use_attention_buffers and profile.get("mid_block_attention_heads"):
        mid_attention = _unet_materialized_attention(
            batch_size=batch_size,
            spatial_tokens=levels[-1]["height"] * levels[-1]["width"],
            text_tokens=text_tokens,
            channels=channels[-1],
            heads=int(profile["mid_block_attention_heads"]),
            element_bytes=element_bytes,
            only_cross_attention=False,
            upcast_attention=upcast_attention,
            slicing=attention_slicing,
        )
        maximum_attention_bytes = max(maximum_attention_bytes, mid_attention["peak_bytes"])
    report = {
        "levels": levels,
        "retained_residual_bytes": retained_residual_bytes,
        "working_tensor_bytes": maximum_working_bytes,
        "feed_forward_working_bytes": maximum_feed_forward_bytes,
        "feed_forward_activation": "geglu" if uses_geglu else None,
        "feed_forward_note": (
            "Standard Diffusers UNet2DConditionModel GEGLU: 8F projection "
            "+ 4F GELU output + 4F gated output; convolution and "
            "feed-forward operations execute sequentially."
            if uses_geglu
            else "Feed-forward workspace is not modeled for this UNet "
            "class; the prior convolution/attention approximation is retained."
        ),
        "attention_workspace_bytes": maximum_attention_bytes,
        "peak_bytes": (
            retained_residual_bytes
            + maximum_working_bytes
            + maximum_attention_bytes
        ),
        "attention_backend": attention_backend,
        "attention_slicing": attention_slicing,
        "formula": (
            "retained UNet residual features + maximum sequential "
            "convolution/feed-forward working set + maximum "
            "cross-attention workspace"
        ),
    }
    if materialized_attention:
        report["attention_model_scope"] = (
            "Standard Diffusers UNet self/cross attention execute sequentially; "
            "each materialized score operation retains two score buffers. "
            "Automatic slicing replaces SDPA and chunks the flattened batch/head axis. "
            "Native CUDA workspace and allocator accuracy remain unvalidated."
            if use_attention_buffers
            else "The prior cross-attention approximation is retained for this "
            "nonstandard UNet; materialized self-attention and custom processors are not modeled."
        )
        if use_attention_buffers:
            report["formula"] = (
                "retained UNet residual features + maximum sequential "
                "convolution/feed-forward working set + maximum sequential "
                "self/cross materialized-attention buffer set"
            )
        if mid_attention is not None:
            report["mid_block_attention_buffers"] = mid_attention
    return report


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
    packing_factor = profile["packing_factor"]
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

    attention_pattern = profile["attention_pattern"]
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
        # The previous block upsamples before this block reduces its channels.
        # Its convolution therefore runs at this resolution with the previous
        # channel count, which can exceed the current block's output feature.
        upsample_channels = channels[stage - 1] if stage else 0
        working_feature_bytes = max(
            feature_bytes,
            processed_batch * stage_height * stage_width * upsample_channels * element_bytes,
        )
        retained_bytes = working_feature_bytes * (layers_per_block + 1)
        working_bytes = 2 * working_feature_bytes
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
                "upsample_convolution_channels": upsample_channels,
                "working_feature_bytes": working_feature_bytes,
                "retained_feature_bytes": retained_bytes,
                "working_tensor_bytes": working_bytes,
            }
        )
    result = {
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
            "maximum live decoder-stage/upsample-convolution feature set + maximum "
            "two-feature working set; decoder stages are sequential"
        ),
    }
    if slicing or tiling:
        result["output_buffers"] = _vae_decode_output_buffers(
            profile=profile, height=height, width=width, batch_size=batch_size,
            latent_scale=latent_scale, element_bytes=element_bytes,
            slicing=slicing, tiling=tiling, tile_size=tile_size,
        )
    return result


def _vae_decode_output_buffers(
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
    """Model native AutoencoderKL output lifetimes during decode and cat.

    Native tiled_decode retains every overlapping tile, then all concatenated
    rows, until its final concatenation completes. Sliced decode retains each
    completed image until a separate batch concatenation finishes.
    """
    slicing_active = slicing and batch_size > 1
    tiling_active = tiling and (height > tile_size or width > tile_size)
    processed_batch = 1 if slicing_active else batch_size
    image_bytes = processed_batch * height * width * 3 * element_bytes
    full_batch_bytes = batch_size * height * width * 3 * element_bytes
    previous_slices_bytes = full_batch_bytes - image_bytes
    if tiling_active:
        # Native stride truncates in latent pixels, while row_limit truncates
        # in output pixels. Require alignment so both describe the same image.
        if tile_size % (4 * latent_scale):
            raise DiffusionEstimateError(
                "active VAE tiling requires vae_tile_size divisible by four "
                "times the latent scale for native 25% overlap alignment"
            )
        tile_latent_size = tile_size // latent_scale
        latent_stride = tile_latent_size * 3 // 4
        row_limit = tile_size * 3 // 4
        heights = [
            min(tile_latent_size, height // latent_scale - start) * latent_scale
            for start in range(0, height // latent_scale, latent_stride)
        ]
        widths = [
            min(tile_latent_size, width // latent_scale - start) * latent_scale
            for start in range(0, width // latent_scale, latent_stride)
        ]
    else:
        latent_stride = None
        row_limit = None
        heights = [height]
        widths = [width]

    candidates: dict[str, dict[str, Any]] = {}

    def candidate(name: str, components: dict[str, int]) -> None:
        peak = sum(components.values())
        if name not in candidates or peak > candidates[name]["peak_bytes"]:
            candidates[name] = {"peak_bytes": peak, "components": components}

    tile_shapes: dict[tuple[int, int], dict[str, Any]] = {}
    retained_tiles = 0
    for tile_height in heights:
        for tile_width in widths:
            shape = (tile_height, tile_width)
            if shape not in tile_shapes:
                tile_shapes[shape] = _vae_decode_activation(
                    profile=profile, height=tile_height, width=tile_width,
                    batch_size=processed_batch, latent_scale=latent_scale,
                    element_bytes=element_bytes, slicing=False, tiling=False,
                    tile_size=tile_size,
                )
            activation = tile_shapes[shape]
            tile_bytes = processed_batch * tile_height * tile_width * 3 * element_bytes
            candidate("decoder", {
                "decoded_image": tile_bytes,
                "retained_decoded_outputs": previous_slices_bytes + retained_tiles,
                "concat_buffers": 0,
                "retained_features": activation["retained_feature_bytes"],
                "working_tensors": activation["working_tensor_bytes"],
            })
            retained_tiles += tile_bytes
    if tiling_active:
        candidate("tile_assembly", {
            "decoded_image": image_bytes,
            "retained_decoded_outputs": previous_slices_bytes + retained_tiles,
            "concat_buffers": image_bytes,
            "retained_features": 0,
            "working_tensors": 0,
        })
    if slicing_active:
        candidate("slice_assembly", {
            "decoded_image": full_batch_bytes,
            "retained_decoded_outputs": full_batch_bytes,
            "concat_buffers": 0,
            "retained_features": 0,
            "working_tensors": 0,
        })
    peak_phase = max(candidates, key=lambda name: candidates[name]["peak_bytes"])
    return {
        "native_slicing_active": slicing_active,
        "native_tiling_active": tiling_active,
        "decoded_output_channels": 3,
        "processed_batch_size": processed_batch,
        "slice_count": batch_size if slicing_active else 1,
        "tile_count_per_decode": len(heights) * len(widths),
        "tile_output_heights": heights,
        "tile_output_widths": widths,
        "tile_overlap_factor": 0.25 if tiling_active else None,
        "tile_stride_latent_pixels": latent_stride,
        "tile_row_limit_pixels": row_limit,
        "retained_tile_outputs_bytes": retained_tiles if tiling_active else 0,
        "previous_slices_bytes": previous_slices_bytes,
        "full_batch_output_bytes": full_batch_bytes,
        "candidate_phases": candidates,
        "peak_phase": peak_phase,
        "peak_bytes": candidates[peak_phase]["peak_bytes"],
        "peak_components": candidates[peak_phase]["components"],
        "native_configuration_requirement": (
            "Set native VAE tile_sample_min_size to vae_tile_size and "
            "tile_latent_min_size to vae_tile_size / latent_scale; native "
            "enable_tiling alone retains the model's config.sample_size."
            if tiling else None
        ),
        "formula": (
            "maximum of decoder activations plus previously decoded outputs, "
            "overlapping tiles plus row/final concatenations, and sliced-batch "
            "concatenation; these phases execute sequentially"
        ),
    }
