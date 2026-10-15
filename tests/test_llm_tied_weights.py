import json

import pytest

from test_llm_estimator import _write_model, _write_safetensors

from fakegpu.llm_estimator import estimate_decoder_inference
from fakegpu.serving_plan import estimate_serving_plan


def model(tmp_path, *, overrides=None, header_change=None, sharded=False):
    root = tmp_path / "model"
    config = {"model_type": "qwen3", "tie_word_embeddings": True}
    config.update(overrides or {})
    _write_model(root, config_overrides=config)
    header = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 8]},
        "lm_head.weight": {"dtype": "BF16", "shape": [32, 8]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8, 8]},
    }
    if header_change:
        header_change(header)
    if sharded:
        (root / "model.safetensors").unlink()
        weight_map = {}
        for index, (name, value) in enumerate(header.items()):
            filename = f"model-{index:05}.safetensors"
            _write_safetensors(root, filename=filename, header={name: value})
            weight_map[name] = filename
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    else:
        _write_safetensors(root, filename="model.safetensors", header=header)
    return root


@pytest.mark.parametrize("dtype,element_bytes", [("bfloat16", 2), ("float32", 4)])
@pytest.mark.parametrize("sharded", [False, True])
def test_tied_embedding_runtime_storage_preserves_checkpoint_payload(
    tmp_path, dtype, element_bytes, sharded
):
    root = model(tmp_path, sharded=sharded)
    report = estimate_decoder_inference(root, prompt_tokens=8, dtype=dtype)
    assert report["checkpoint"]["parameter_count"] == 576
    assert report["checkpoint"]["checkpoint_bytes"] == 1152
    assert report["memory"]["base_parameter_bytes"] == 320 * element_bytes
    alias = report["weight_storage"]["tied_embedding_storage"]
    assert alias["removed_parameter_count"] == 256
    assert alias["removed_runtime_bytes"] == 256 * element_bytes
    serving = estimate_serving_plan(
        root,
        prompt_tokens=8,
        generated_tokens=2,
        active_sequences=1,
        max_batch_size=1,
        dtype=dtype,
    )
    assert serving["weight_storage"]["base_model_runtime_bytes"] == 320 * element_bytes


@pytest.mark.parametrize(
    "overrides,header_change",
    [
        ({"tie_word_embeddings": False}, None),
        ({"torchscript": True}, None),
        ({"model_type": "unknown"}, None),
        ({}, lambda h: h.pop("lm_head.weight")),
        ({}, lambda h: h["lm_head.weight"].update(shape=[31, 8])),
        ({}, lambda h: h["lm_head.weight"].update(dtype="F32")),
        ({"quantization_config": {"quant_method": "gptq"}}, None),
    ],
)
def test_unconfirmed_tied_storage_is_not_subtracted(tmp_path, overrides, header_change):
    root = model(tmp_path, overrides=overrides, header_change=header_change)
    report = estimate_decoder_inference(root, prompt_tokens=8)
    assert "tied_embedding_storage" not in report["weight_storage"]
    checkpoint = report["checkpoint"]
    expected = (
        checkpoint["checkpoint_bytes"]
        if report["weight_storage"]["quantization"]["enabled"]
        else checkpoint["parameter_count"] * 2
    )
    assert report["memory"]["base_parameter_bytes"] == expected
