"""Check output lifetimes observed in native AutoencoderKL CPU probes."""
import pytest

from fakegpu import DiffusionEstimateError, estimate_diffusion_generation
from fakegpu._diffusion_activation import _vae_decode_activation


def _buffers(*, height=56, width=72, batch=1, slicing=False, tiling=False, tile=32):
    return _vae_decode_activation(
        profile={"block_out_channels": [8, 16], "layers_per_block": 1},
        height=height, width=width, batch_size=batch, latent_scale=2,
        element_bytes=4, slicing=slicing, tiling=tiling, tile_size=tile,
    )["output_buffers"]


def test_slicing_keeps_inputs_and_new_batch_output_during_cat():
    buffers = _buffers(height=32, width=40, batch=3, slicing=True)
    assert buffers["previous_slices_bytes"] == 30720
    assert buffers["candidate_phases"]["slice_assembly"]["peak_bytes"] == 92160
    # The tiny native CPU probe records three 15,360-byte image inputs and
    # one 46,080-byte output simultaneously live during aten.cat.
    parts = buffers["candidate_phases"]["slice_assembly"]["components"]
    assert parts["retained_decoded_outputs"] == parts["decoded_image"] == 46080
    assert buffers["peak_phase"] == "decoder"
    assert buffers["peak_bytes"] == 373760


def test_large_sliced_batch_assembly_does_not_overlap_decoder_activations():
    buffers = _buffers(height=32, width=40, batch=64, slicing=True)
    assert buffers["peak_phase"] == "slice_assembly"
    assert buffers["peak_bytes"] == 1966080
    assert buffers["peak_components"]["retained_features"] == 0
    assert buffers["peak_components"]["working_tensors"] == 0


def test_rectangular_tiles_keep_overlaps_and_both_concat_outputs():
    buffers = _buffers(tiling=True)
    assert buffers["tile_count_per_decode"] == 9
    assert buffers["tile_output_heights"] == [32, 32, 8]
    assert buffers["tile_output_widths"] == [32, 32, 24]
    assert buffers["tile_stride_latent_pixels"] == 12
    assert buffers["retained_tile_outputs_bytes"] == 76032
    # Native CPU final cat retained 76,032 bytes of overlapping tile storage,
    # 48,384 bytes of row outputs, and a separate 48,384-byte output image.
    assembly = buffers["candidate_phases"]["tile_assembly"]
    assert assembly["peak_bytes"] == 172800
    assert assembly["components"]["concat_buffers"] == 48384
    assert assembly["components"]["decoded_image"] == 48384
    assert buffers["peak_phase"] == "decoder"
    assert buffers["peak_bytes"] == 320512


def test_slicing_and_tiling_keep_previously_completed_images():
    buffers = _buffers(batch=3, slicing=True, tiling=True)
    assert buffers["slice_count"] == 3
    assert buffers["previous_slices_bytes"] == 96768
    assert buffers["candidate_phases"]["tile_assembly"]["peak_bytes"] == 269568
    assert buffers["candidate_phases"]["slice_assembly"]["peak_bytes"] == 290304
    assert buffers["peak_bytes"] == 417280


def test_large_tile_grid_assembly_can_dominate_decoder_memory():
    buffers = _buffers(height=192, width=224, tiling=True)
    assert buffers["tile_count_per_decode"] == 80
    assert buffers["retained_tile_outputs_bytes"] == 880896
    assert buffers["peak_phase"] == "tile_assembly"
    assert buffers["peak_bytes"] == 1913088
    assert buffers["peak_components"]["retained_features"] == 0


def test_native_tile_threshold_and_rectangular_boundary():
    exact = _buffers(height=32, width=32, tiling=True)
    assert exact["native_tiling_active"] is False
    assert set(exact["candidate_phases"]) == {"decoder"}
    rectangular = _buffers(height=32, width=72, tiling=True)
    assert rectangular["tile_output_heights"] == [32, 8]
    assert rectangular["tile_count_per_decode"] == 6
    assert rectangular["retained_tile_outputs_bytes"] == 42240


def test_native_tile_stride_requires_aligned_overlap():
    with pytest.raises(DiffusionEstimateError, match="native 25% overlap alignment"):
        _buffers(tiling=True, tile=20)


@pytest.mark.parametrize("slicing,tiling", [(True, False), (False, True), (True, True)])
def test_diffusion_phase_uses_one_coherent_decode_or_assembly_peak(slicing, tiling):
    report = estimate_diffusion_generation(
        "stable-diffusion-v1-5", height=768, width=512, batch_size=3,
        dtype="bfloat16", vae_slicing=slicing, vae_tiling=tiling, vae_tile_size=256,
    )
    buffers = report["activation_models"]["vae_decode"]["output_buffers"]
    phase = next(x for x in report["memory_timeline"]["phases"] if x["phase"] == "vae_decode")
    components = phase["components"]
    assert phase["peak_bytes"] == (
        components["resident_weights"] + components["latents"]
        + buffers["peak_bytes"] + components["runtime_overhead"]
    )
    assert all(components[name] == value for name, value in buffers["peak_components"].items())
