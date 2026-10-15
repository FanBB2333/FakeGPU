"""Shape and lifetime regressions established by native CPU attention probes."""

from __future__ import annotations

import pytest

from fakegpu._diffusion_activation import _unet_materialized_attention, _unet_denoiser_activation
from fakegpu._diffusion_pipeline import _local_unet_profile
from fakegpu._diffusion_types import DiffusionEstimateError


def _buffers(**changes):
    inputs = dict(batch_size=1, spatial_tokens=256, text_tokens=5, channels=16,
                  heads=4, element_bytes=4, only_cross_attention=False,
                  upcast_attention=False, slicing=False)
    inputs.update(changes)
    return _unet_materialized_attention(**inputs)


def _profile(**changes):
    config = dict(block_out_channels=[16, 32], layers_per_block=1,
                  down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
                  up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
                  attention_head_dim=[4, 4], transformer_layers_per_block=[1, 1])
    config.update(changes)
    return _local_unet_profile(config=config, model_class='UNet2DConditionModel',
                               stats={'parameter_count': 1, 'checkpoint_bytes': 4, 'dtype_bytes': {'F32': 4}},
                               conditioning_width=12)


def _activation(profile, *, backend='eager', slicing=False):
    return _unet_denoiser_activation(profile=profile, latent_height=16, latent_width=16,
                                    batch_size=1, text_tokens=5, element_bytes=4,
                                    attention_backend=backend, attention_slicing=slicing)


def test_eager_includes_native_self_attention_scores_and_softmax_buffers():
    report = _buffers()
    # Native CPU tiny UNet emits [4, 256, 256] self scores and a separate
    # equally sized softmax result, versus [4, 256, 5] cross scores.
    self_attention = report['candidates']['self']
    cross_attention = report['candidates']['cross']
    assert self_attention['score_buffer_bytes'] == 1_048_576
    assert self_attention['simultaneous_score_buffer_bytes'] == 2_097_152
    assert cross_attention['score_buffer_bytes'] == 20_480
    assert self_attention['qkv_bytes'] == 49_152
    assert report['peak_bytes'] == 2_146_304
    assert report['peak_bytes'] == max(value['peak_bytes'] for value in report['candidates'].values())


def test_auto_slicing_chunks_flattened_heads_independently_of_batch():
    one = _buffers(slicing=True)
    three = _buffers(slicing=True, batch_size=3)
    for report in (one, three):
        assert report['processor'] == 'SlicedAttnProcessor(auto)'
        assert report['candidates']['self']['score_head_rows'] == 2
        assert report['candidates']['self']['score_buffer_bytes'] == 524_288
        assert report['candidates']['self']['previous_slice_output_bytes'] == 8_192
    assert three['candidates']['self']['qkv_bytes'] == 3 * one['candidates']['self']['qkv_bytes']
    assert three['candidates']['self']['sliced_output_bytes'] == 49_152


def test_transformer_inner_width_uses_whole_attention_heads():
    report = _buffers(channels=17)
    assert report['candidates']['self']['projection_channels'] == 16
    assert report['candidates']['self']['qkv_bytes'] == 49_152


def test_slicing_overrides_requested_sdpa_processor():
    eager = _activation(_profile(), backend='eager', slicing=True)
    sdpa = _activation(_profile(), backend='sdpa', slicing=True)
    assert sdpa['attention_workspace_bytes'] == eager['attention_workspace_bytes']
    assert sdpa['levels'][0]['attention_buffers']['processor'] == 'SlicedAttnProcessor(auto)'


def test_upcast_attention_keeps_original_projections_and_float32_copies():
    report = _buffers(element_bytes=2, upcast_attention=True)
    self_attention = report['candidates']['self']
    assert self_attention['qkv_bytes'] == 24_576
    assert self_attention['qk_upcast_bytes'] == 32_768
    assert self_attention['score_element_bytes'] == 4
    assert self_attention['simultaneous_score_buffer_bytes'] == 2_097_152
    sliced = _buffers(element_bytes=2, upcast_attention=True, slicing=True, batch_size=3)
    assert sliced['candidates']['self']['qk_upcast_bytes'] == 16_384


def test_cross_only_down_up_blocks_retain_self_attention_in_standard_mid_block():
    profile = _profile(only_cross_attention=[True, False], upcast_attention=True)
    assert profile['only_cross_attention_by_level'] == [True, False]
    assert profile['mid_block_attention_heads'] == 4
    assert profile['upcast_attention'] is True
    report = _activation(profile)
    assert set(report['levels'][0]['attention_buffers']['candidates']) == {'cross'}
    assert report['mid_block_attention_buffers']['self_attention'] is True
    assert report['mid_block_attention_buffers']['candidates']['self']['query_tokens'] == 64


def test_default_inspection_and_sdpa_report_do_not_gain_new_metadata():
    profile = _profile()
    assert 'only_cross_attention_by_level' not in profile
    assert 'upcast_attention' not in profile
    assert 'standard_transformer_attention' not in profile
    report = _activation(profile, backend='sdpa')
    assert 'attention_model_scope' not in report
    assert all('attention_buffers' not in level for level in report['levels'])


def test_nonstandard_attention_keeps_explicitly_unmodeled_approximation():
    report = _activation(_profile(dual_cross_attention=True))
    assert 'not modeled' in report['attention_model_scope']
    assert all('attention_buffers' not in level for level in report['levels'])


@pytest.mark.parametrize('config', [dict(only_cross_attention=[False]), dict(upcast_attention='true')])
def test_invalid_attention_config_is_not_silently_reinterpreted(config):
    with pytest.raises(DiffusionEstimateError):
        _profile(**config)


def test_native_auto_slicing_does_not_invent_nonzero_one_head_chunk():
    with pytest.raises(DiffusionEstimateError, match='at least two heads'):
        _buffers(heads=1, slicing=True)
