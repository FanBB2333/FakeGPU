"""Inference formulas must not silently treat recurrent layers as GQA layers."""

import json

import pytest

from fakegpu.llm_estimator import estimate_decoder_inference
from fakegpu.serving_plan import estimate_serving_plan
from test_llm_estimator import _write_model


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit_layers", [False, True])
def test_hybrid_recurrent_decoder_requires_its_own_state_model(tmp_path, nested, explicit_layers):
    directory = tmp_path / "hybrid"
    config = {
        "model_type": "qwen3_5_text",
        "linear_num_key_heads": 1, "linear_num_value_heads": 1,
        "linear_key_head_dim": 4, "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
    }
    if explicit_layers:
        config["layer_types"] = ["linear_attention", "full_attention"]
    _write_model(directory, config_overrides=config)
    if nested:
        path = directory / "config.json"
        text = json.loads(path.read_text())
        path.write_text(json.dumps({"model_type":"qwen3_5", "text_config":text}))
    with pytest.raises(ValueError, match="recurrent|linear attention|hybrid"):
        estimate_decoder_inference(directory, prompt_tokens=128)
    with pytest.raises(ValueError, match="recurrent|linear attention|hybrid"):
        estimate_serving_plan(directory, active_sequences=2, max_batch_size=2, prompt_tokens=128)


@pytest.mark.parametrize("layers", [
    ["full_attention", "linear_attention"],
    ["full_attention"],
    [{"type": "full_attention"}, "full_attention"],
])
def test_declared_layers_cannot_be_silently_ignored(tmp_path, layers):
    directory = tmp_path / "model"
    _write_model(directory, config_overrides={"layer_types":layers})
    with pytest.raises(ValueError, match="architecture-specific"):
        estimate_decoder_inference(directory, prompt_tokens=8)


def test_attention_only_explicit_layers_preserve_inference_values(tmp_path):
    original, declared = tmp_path / "original", tmp_path / "declared"
    _write_model(original)
    _write_model(declared, config_overrides={"layer_types":["full_attention"]*2})
    before = estimate_decoder_inference(original, prompt_tokens=8)
    after = estimate_decoder_inference(declared, prompt_tokens=8)
    assert after["memory"] == before["memory"]
    assert after["compute"] == before["compute"]
