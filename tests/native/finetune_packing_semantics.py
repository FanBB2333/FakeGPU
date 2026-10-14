"""Audit Qwen3.5 segment isolation with tiny, randomly initialized real models.

This checks kernel semantics, not checkpoint accuracy or training budgets. A
block mask and reset positions must isolate both logits and input gradients.
The dense attention model is a positive control for the test construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("huggingface", "llamafactory", "axolotl"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.cuda_device if args.device == "cuda" else "",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", AXOLOTL_DO_NOT_TRACK="1",
        OMP_NUM_THREADS="1",
    )
    import torch
    import transformers
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    if args.device == "cuda":
        from fakegpu._calibration_cuda import _reject_simulated_cuda_environment

        _reject_simulated_cuda_environment()
        if not torch.cuda.is_available():
            raise RuntimeError("packing kernel audit requires a real CUDA device")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.pad_token_id = 0
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    perturbed = [10, 11, 12, 13, 5, 6, 7, 8]
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=args.device)
    allowed = torch.zeros(8, 8, dtype=torch.bool, device=args.device)
    allowed[:4, :4] = torch.ones(4, 4, dtype=torch.bool, device=args.device).tril()
    allowed[4:, 4:] = torch.ones(4, 4, dtype=torch.bool, device=args.device).tril()
    mask = torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)[None, None]

    def model_for(layers):
        torch.manual_seed(42)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=32, linear_value_head_dim=32,
            linear_conv_kernel_dim=4, layer_types=layers,
            pad_token_id=0, eos_token_id=2, use_cache=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2],
                             "mrope_interleaved": True},
        )
        config._attn_implementation = "eager"
        return Qwen3_5ForCausalLM(config).to(device=args.device, dtype=dtype).eval()

    def inputs(ids, model):
        if args.framework == "llamafactory":
            from llamafactory.data import SFTDataCollatorWith4DAttentionMask
            from llamafactory.data.template import TEMPLATES

            collator = SFTDataCollatorWith4DAttentionMask(
                tokenizer=tokenizer, template=TEMPLATES["qwen3_nothink"], model=model,
                pad_to_multiple_of=8, block_diag_attn=True, neat_packing=True,
                attn_implementation="eager", compute_dtype=dtype,
            )
            result = collator([{
                "input_ids": ids, "attention_mask": [1] * 4 + [2] * 4,
                "labels": [-100] + ids[1:4] + [-100] + ids[5:],
                "position_ids": positions.cpu().tolist()[0],
            }])
            result = {name: value.to(args.device) for name, value in result.items()}
            assert torch.equal(result["attention_mask"], mask)
            assert torch.equal(result["position_ids"], positions)
            return result
        return {"input_ids": torch.tensor([ids], device=args.device),
                "attention_mask": mask, "position_ids": positions}

    def check(model):
        with torch.no_grad():
            baseline = model(**inputs(original, model)).logits[:, 4:].float()
            changed = model(**inputs(perturbed, model)).logits[:, 4:].float()
        batch = inputs(original, model)
        embeddings = model.get_input_embeddings()(batch.pop("input_ids")).detach().requires_grad_(True)
        batch.pop("labels", None)
        logits = model(inputs_embeds=embeddings, **batch).logits[:, 4:].float()
        gradient, = torch.autograd.grad(logits.square().mean(), embeddings)
        logit_delta = (baseline - changed).abs().max().item()
        prefix_gradient = gradient[:, :4].abs().max().item()
        return {"suffix_logits_max_delta": logit_delta,
                "suffix_gradient_into_prefix_max": prefix_gradient,
                "isolated": logit_delta <= 1e-6 and prefix_gradient <= 1e-10}

    report = {
        "schema_version": "fakegpu.finetune_packing_semantics_audit.v1",
        "framework": args.framework, "device": args.device, "dtype": str(dtype),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Tiny random real Qwen3.5 models; kernel isolation only, no checkpoint or native training budget claim.",
    }
    try:
        report["dense_control"] = check(model_for(["full_attention", "full_attention"]))
        report["native_hybrid"] = check(model_for(["linear_attention", "full_attention"]))
        assert report["dense_control"]["isolated"]
        assert not report["native_hybrid"]["isolated"]
        if args.framework == "axolotl":
            from axolotl.monkeypatch.models.qwen3_5.modeling import patch_qwen3_5_modeling_packing

            patch_qwen3_5_modeling_packing()
            try:
                report["axolotl_patched_hybrid"] = check(model_for(["linear_attention", "full_attention"]))
            except Exception as exc:
                report["axolotl_patched_hybrid"] = {
                    "isolated": False, "available": False,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                }
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
