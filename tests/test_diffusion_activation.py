"""Regression cases for native diffusion activation tensor lifetimes."""

from __future__ import annotations

from fakegpu._diffusion_activation import _unet_denoiser_activation


def _unet(
    *,
    model_class: str = "UNet2DConditionModel",
    batch_size: int = 4,
    element_bytes: int = 2,
    transformer_layers: list[int] | None = None,
) -> dict:
    return _unet_denoiser_activation(
        profile={
            "model_class": model_class,
            "block_out_channels": [320, 640, 1280, 1280],
            "layers_per_block": 2,
            "attention_heads": [8, 8, 8, 0],
            "transformer_layers_per_block": (
                [1, 1, 1, 0]
                if transformer_layers is None
                else transformer_layers
            ),
        },
        latent_height=64,
        latent_width=96,
        batch_size=batch_size,
        text_tokens=77,
        element_bytes=element_bytes,
        attention_backend="sdpa",
        attention_slicing=False,
    )


def test_unet_geglu_retains_projection_gate_and_product() -> None:
    # The observed native SD1.4 batch-2/CFG peak is GEGLU multiplication
    # of two [4, 6144, 1280] tensors, while the double-width projection
    # remains alive through its chunk views.
    report = _unet()
    projection_bytes = 4 * 6144 * 2560 * 2
    gate_bytes = 4 * 6144 * 1280 * 2
    product_bytes = gate_bytes
    peak = projection_bytes + gate_bytes + product_bytes

    assert peak == 251_658_240
    assert report["feed_forward_working_bytes"] == peak
    assert report["working_tensor_bytes"] == peak
    assert report["levels"][0]["feed_forward_working_bytes"] == peak
    assert report["levels"][3]["feed_forward_working_bytes"] == 0
    assert report["feed_forward_activation"] == "geglu"


def test_unet_geglu_scales_with_execution_batch_and_dtype() -> None:
    baseline = _unet(batch_size=2, element_bytes=2)
    cfg_batch = _unet(batch_size=4, element_bytes=2)
    full_precision = _unet(batch_size=2, element_bytes=4)

    assert (
        cfg_batch["feed_forward_working_bytes"]
        == full_precision["feed_forward_working_bytes"]
        == 2 * baseline["feed_forward_working_bytes"]
    )


def test_unet_sequential_transformer_depth_does_not_multiply_workspace() -> None:
    shallow = _unet()
    deep = _unet(transformer_layers=[10, 10, 10, 0])

    assert deep["feed_forward_working_bytes"] == shallow["feed_forward_working_bytes"]
    assert deep["peak_bytes"] == shallow["peak_bytes"]


def test_unrecognized_unet_does_not_assume_geglu() -> None:
    report = _unet(model_class="UNet2DModel")

    assert report["feed_forward_activation"] is None
    assert report["feed_forward_working_bytes"] == 0
    assert "not modeled" in report["feed_forward_note"]
    assert report["working_tensor_bytes"] == max(
        level["convolution_working_bytes"] for level in report["levels"]
    )
