from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from fakegpu import (
    DiffusionEstimateError,
    estimate_diffusion_generation,
    inspect_diffusion_pipeline,
    load_diffusion_profiles,
)
from fakegpu.diffusion_estimator import SCHEMA_VERSION, main


SD15 = "stable-diffusion-v1-5"
SDXL = "stable-diffusion-xl-base-1.0"
PIXART = "pixart-sigma-xl-2-1024-ms"


def _phase(report: dict, name: str) -> dict:
    return next(
        item
        for item in report["memory_timeline"]["phases"]
        if item["phase"] == name
    )


def _write_safetensors(
    path: Path,
    *,
    tensors: dict[str, tuple[str, list[int]]],
) -> None:
    dtype_bytes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }
    offset = 0
    header: dict[str, object] = {}
    for name, (dtype, shape) in tensors.items():
        byte_count = math.prod(shape) * dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset
    )


def _write_component(
    root: Path,
    name: str,
    *,
    config: dict[str, object],
    tensor_shape: list[int],
    filename: str,
) -> None:
    component = root / name
    component.mkdir()
    (component / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    _write_safetensors(
        component / filename,
        tensors={f"{name}.weight": ("F16", tensor_shape)},
    )


def _write_pipeline(
    root: Path,
    *,
    transformer: bool,
    flux: bool = False,
) -> None:
    root.mkdir()
    if transformer:
        pipeline_class = "FluxPipeline" if flux else "PixArtSigmaPipeline"
        denoiser_name = "transformer"
        denoiser_class = (
            "FluxTransformer2DModel"
            if flux
            else "PixArtTransformer2DModel"
        )
        denoiser_config: dict[str, object] = {
            "_class_name": denoiser_class,
            "sample_size": 8 if flux else 16,
            "in_channels": 64 if flux else 4,
            "num_layers": 2 if flux else 3,
            "num_single_layers": 1 if flux else 0,
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "patch_size": 1 if flux else 2,
            (
                "joint_attention_dim"
                if flux
                else "cross_attention_dim"
            ): 8,
        }
        latent_channels = 16 if flux else 4
    else:
        pipeline_class = "StableDiffusionPipeline"
        denoiser_name = "unet"
        denoiser_class = "UNet2DConditionModel"
        denoiser_config = {
            "_class_name": denoiser_class,
            "sample_size": 8,
            "in_channels": 4,
            "block_out_channels": [8, 16],
            "layers_per_block": [1, 2],
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            "attention_head_dim": [2, 4],
            "transformer_layers_per_block": [1, 0],
            "cross_attention_dim": 8,
        }
        latent_channels = 4

    model_index = {
        "_class_name": pipeline_class,
        "text_encoder": ["transformers", "T5EncoderModel"],
        denoiser_name: ["diffusers", denoiser_class],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (root / "model_index.json").write_text(
        json.dumps(model_index),
        encoding="utf-8",
    )
    _write_component(
        root,
        "text_encoder",
        config={
            "_class_name": "T5EncoderModel",
            "d_model": 16,
            "num_layers": 2,
            "n_positions": 512,
        },
        tensor_shape=[4, 4],
        filename="model.safetensors",
    )
    _write_component(
        root,
        denoiser_name,
        config=denoiser_config,
        tensor_shape=[8, 8],
        filename="diffusion_pytorch_model.safetensors",
    )
    _write_component(
        root,
        "vae",
        config={
            "_class_name": "AutoencoderKL",
            "latent_channels": latent_channels,
            "block_out_channels": [4, 8, 16, 16],
            "layers_per_block": 1,
        },
        tensor_shape=[4, 4],
        filename="diffusion_pytorch_model.safetensors",
    )


def test_diffusion_profiles_use_fixed_official_component_metadata() -> None:
    profiles = load_diffusion_profiles()

    assert set(profiles) == {SD15, SDXL, PIXART}
    assert profiles[SD15]["denoiser"]["parameter_count"] == 859520964
    assert profiles[SD15]["source"]["revision"] == (
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    assert profiles[SDXL]["conditioning"]["parameter_count"] == (
        817720320
    )
    assert profiles[SDXL]["denoiser"]["parameter_count"] == (
        2567463684
    )
    assert profiles[SDXL]["default_generation"]["height"] == 1024
    assert profiles[SDXL]["source"]["revision"] == (
        "462165984030d82259a11f4367a4eed129e94a7b"
    )
    assert profiles[PIXART]["denoiser"]["architecture"] == "transformer"
    assert profiles[PIXART]["denoiser"]["parameter_count"] == 610856096
    assert profiles[PIXART]["conditioning"]["parameter_count"] == (
        4762310656
    )
    assert profiles[PIXART]["source"]["revision"] == (
        "e102b3591cc82e97071b8b4cb90d834d0c487207"
    )


def test_diffusion_generation_report_separates_phases_and_fit() -> None:
    report = estimate_diffusion_generation(
        SD15,
        target_profile="a100",
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["validation_status"] == "Modeled"
    assert report["inputs"]["classifier_free_guidance"] is True
    assert report["inputs"]["denoiser_batch_size"] == 2
    assert report["latent"]["height"] == 64
    assert report["latent"]["width"] == 64
    assert report["weights"]["total_parameter_count"] == (
        123060557 + 859520964 + 83653863
    )
    assert report["fit"]["fits"] is True
    assert report["fit"]["target_profile"] == "a100"

    for phase in report["memory_timeline"]["phases"]:
        assert phase["peak_bytes"] == sum(phase["components"].values())
    assert report["memory_timeline"]["peak_bytes"] == max(
        phase["peak_bytes"]
        for phase in report["memory_timeline"]["phases"]
    )


def test_diffusion_optimizations_and_shapes_change_modeled_memory() -> None:
    baseline = estimate_diffusion_generation(SDXL)
    offloaded = estimate_diffusion_generation(SDXL, offload="model")
    batch_four = estimate_diffusion_generation(SDXL, batch_size=4)
    batch_four_sliced = estimate_diffusion_generation(
        SDXL,
        batch_size=4,
        vae_slicing=True,
    )
    high_resolution = estimate_diffusion_generation(
        SDXL,
        height=2048,
        width=2048,
    )
    tiled = estimate_diffusion_generation(
        SDXL,
        height=2048,
        width=2048,
        vae_tiling=True,
        vae_tile_size=512,
    )

    assert offloaded["memory_timeline"]["peak_bytes"] < (
        baseline["memory_timeline"]["peak_bytes"]
    )
    assert batch_four["memory_timeline"]["peak_bytes"] > (
        baseline["memory_timeline"]["peak_bytes"]
    )
    assert _phase(batch_four_sliced, "vae_decode")["peak_bytes"] < (
        _phase(batch_four, "vae_decode")["peak_bytes"]
    )
    assert high_resolution["memory_timeline"]["peak_bytes"] > (
        baseline["memory_timeline"]["peak_bytes"]
    )
    assert _phase(tiled, "vae_decode")["peak_bytes"] < (
        _phase(high_resolution, "vae_decode")["peak_bytes"]
    )
    vae_model = high_resolution["activation_models"]["vae_decode"]
    assert vae_model["retained_feature_bytes"] == max(
        stage["retained_feature_bytes"]
        for stage in vae_model["stages"]
    )
    assert "sequential" in vae_model["formula"]


def test_diffusion_attention_and_guidance_controls_are_modeled() -> None:
    eager = estimate_diffusion_generation(
        SDXL,
        attention_backend="eager",
        offload="model",
    )
    sliced = estimate_diffusion_generation(
        SDXL,
        attention_backend="eager",
        attention_slicing=True,
        offload="model",
    )
    without_cfg = estimate_diffusion_generation(
        SDXL,
        guidance_scale=1.0,
        offload="model",
    )

    assert (
        sliced["activation_models"]["denoiser"][
            "attention_workspace_bytes"
        ]
        < eager["activation_models"]["denoiser"][
            "attention_workspace_bytes"
        ]
    )
    assert without_cfg["inputs"]["denoiser_batch_size"] == 1
    assert _phase(without_cfg, "denoise")["peak_bytes"] < (
        _phase(eager, "denoise")["peak_bytes"]
    )


def test_pixart_uses_patch_transformer_memory_model() -> None:
    sdpa = estimate_diffusion_generation(
        PIXART,
        attention_backend="sdpa",
        offload="model",
    )
    eager = estimate_diffusion_generation(
        PIXART,
        attention_backend="eager",
        offload="model",
    )

    assert sdpa["architecture"] == {
        "denoiser_family": "transformer",
        "model_class": "Transformer2DModel",
        "attention_pattern": "cross",
        "guidance_mode": "classifier_free",
        "configuration_source": "fixed_revision_profile",
    }
    activation = sdpa["activation_models"]["denoiser"]
    assert activation["image_tokens"] == 4096
    assert activation["hidden_width"] == 1152
    assert activation["num_layers"] == 28
    assert sdpa["weights"]["total_checkpoint_bytes"] == (
        sdpa["weights"]["total_parameter_count"] * 4
    )
    assert sdpa["weights"]["total_runtime_bytes"] == (
        sdpa["weights"]["total_parameter_count"] * 2
    )
    assert eager["activation_models"]["denoiser"][
        "attention_workspace_bytes"
    ] > (
        sdpa["activation_models"]["denoiser"][
            "attention_workspace_bytes"
        ]
    )
    assert sdpa["accuracy"]["status"] == "uncalibrated"
    assert sdpa["accuracy"]["prediction_interval_bytes"] is None


def test_local_unet_pipeline_is_inspected_from_json_and_headers(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-unet"
    _write_pipeline(model_dir, transformer=False)

    inspection = inspect_diffusion_pipeline(model_dir)
    assert inspection["architecture"]["family"] == "unet"
    assert inspection["roles"]["denoiser"] == ["unet"]
    assert inspection["evidence"]["tensor_payloads_loaded"] is False
    assert inspection["components"]["unet"]["checkpoint"][
        "parameter_count"
    ] == 64

    report = estimate_diffusion_generation(
        model_dir=model_dir,
        text_tokens=32,
    )
    assert report["model_profile"]["source_kind"] == (
        "local_diffusers_pipeline"
    )
    assert report["inputs"]["height"] == 64
    assert report["inputs"]["width"] == 64
    assert report["architecture"]["denoiser_family"] == "unet"
    assert report["weights"]["parameter_counts"] == {
        "conditioning": 16,
        "denoiser": 64,
        "vae": 16,
    }
    assert report["pipeline_inspection"]["model_dir"] == str(
        model_dir.resolve()
    )


def test_local_profile_validation_allows_missing_size_and_conditioning(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-unet-no-text"
    _write_pipeline(model_dir, transformer=False)
    model_index = json.loads(
        (model_dir / "model_index.json").read_text(encoding="utf-8")
    )
    model_index.pop("text_encoder")
    (model_dir / "model_index.json").write_text(
        json.dumps(model_index),
        encoding="utf-8",
    )
    denoiser_config_path = model_dir / "unet" / "config.json"
    denoiser_config = json.loads(
        denoiser_config_path.read_text(encoding="utf-8")
    )
    denoiser_config.pop("sample_size")
    denoiser_config_path.write_text(
        json.dumps(denoiser_config),
        encoding="utf-8",
    )

    inspection = inspect_diffusion_pipeline(model_dir)

    assert inspection["profile"]["status"] == "local-inspected"
    assert inspection["profile"]["default_generation"]["height"] is None
    assert inspection["profile"]["conditioning"]["parameter_count"] == 0


def test_local_patch_and_joint_transformers_use_distinct_shapes(
    tmp_path: Path,
) -> None:
    pixart_dir = tmp_path / "local-pixart"
    flux_dir = tmp_path / "local-flux"
    _write_pipeline(pixart_dir, transformer=True)
    _write_pipeline(flux_dir, transformer=True, flux=True)

    pixart = estimate_diffusion_generation(
        model_dir=pixart_dir,
        text_tokens=32,
        guidance_scale=4.5,
        offload="model",
    )
    flux = estimate_diffusion_generation(
        model_dir=flux_dir,
        text_tokens=32,
        guidance_scale=7.0,
        offload="model",
    )

    pixart_activation = pixart["activation_models"]["denoiser"]
    flux_activation = flux["activation_models"]["denoiser"]
    assert pixart["architecture"]["attention_pattern"] == "cross"
    assert pixart_activation["patch_size"] == 2
    assert pixart_activation["image_tokens"] == 64
    assert flux["architecture"]["attention_pattern"] == "joint"
    assert flux["architecture"]["guidance_mode"] == "embedded"
    assert flux["inputs"]["classifier_free_guidance"] is False
    assert flux["inputs"]["denoiser_batch_size"] == 1
    assert flux_activation["packing_factor"] == 2
    assert flux_activation["image_tokens"] == 64


def test_diffusion_estimator_rejects_invalid_inputs_and_profiles(
    tmp_path: Path,
) -> None:
    with pytest.raises(DiffusionEstimateError):
        estimate_diffusion_generation("missing")
    with pytest.raises(DiffusionEstimateError):
        estimate_diffusion_generation(SD15, height=513)
    with pytest.raises(DiffusionEstimateError):
        estimate_diffusion_generation(SD15, text_tokens=78)
    with pytest.raises(DiffusionEstimateError):
        estimate_diffusion_generation(SD15, dtype="int8")
    with pytest.raises(DiffusionEstimateError):
        estimate_diffusion_generation(
            SD15,
            runtime_overhead_bytes=-1,
        )

    invalid = tmp_path / "profiles.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": "unknown",
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DiffusionEstimateError):
        load_diffusion_profiles(invalid)


def test_diffusion_cli_lists_profiles_and_writes_json(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(["--list-profiles"]) == 0
    listed = capsys.readouterr().out
    assert SD15 in listed
    assert SDXL in listed

    assert main(["--model-profile", SD15, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_profile"]["id"] == SD15

    output = tmp_path / "estimate.json"
    assert (
        main(
            [
                "--model-profile",
                SDXL,
                "--offload",
                "model",
                "--target-profile",
                "a100",
                "--json",
                str(output),
            ]
        )
        == 0
    )
    assert "Diffusion estimate:" in capsys.readouterr().out
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["fit"]["fits"] is True

    local_model = tmp_path / "cli-local-pipeline"
    _write_pipeline(local_model, transformer=True)
    assert (
        main(
            [
                "--model-dir",
                str(local_model),
                "--text-tokens",
                "32",
                "--json",
            ]
        )
        == 0
    )
    local_report = json.loads(capsys.readouterr().out)
    assert local_report["architecture"]["denoiser_family"] == (
        "transformer"
    )
    assert local_report["model_profile"]["source_kind"] == (
        "local_diffusers_pipeline"
    )

    with pytest.raises(SystemExit):
        main([])
