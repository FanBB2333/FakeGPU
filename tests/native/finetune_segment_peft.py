"""Real PEFT direct-forward and checkpoint recomputation segment regression.

Run in the pinned native Transformers environments; no pretrained weights needed.
"""

from copy import deepcopy
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    import peft
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    from fakegpu._finetune_segments import isolated_qwen35_segments

    torch.set_num_threads(1)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        pad_token_id=0,
        eos_token_id=2,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
        "use_cache": False,
    }
    allowed = torch.zeros(6, 6, dtype=torch.bool)
    allowed[:3, :3] = allowed[3:, 3:] = torch.ones(3, 3, dtype=torch.bool).tril()
    inputs["attention_mask"] = torch.where(
        allowed, 0.0, torch.finfo(torch.float32).min
    )[None, None]
    cases = []
    for backend in ("eager", "sdpa"):
        for mode in ("none", "nonreentrant", "reentrant"):
            torch.manual_seed(42)
            native = Qwen3_5ForCausalLM(deepcopy(config))
            native.set_attn_implementation(backend)
            model = get_peft_model(
                native,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0,
                    target_modules=["q_proj", "v_proj"],
                ),
            )
            reference = deepcopy(model)
            if mode != "none":
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": mode == "reentrant"}
                )
                model.enable_input_require_grads()
            original = native.forward.__func__
            with isolated_qwen35_segments(native):
                logits = model(**inputs).logits
                logits.float().square().sum().backward()
                changed = inputs["input_ids"].clone()
                changed[0, :3] += 20
                with torch.no_grad():
                    baseline = model(**inputs).logits[:, 3:]
                    perturbed = model(**{**inputs, "input_ids": changed}).logits[:, 3:]
                assert torch.equal(baseline, perturbed)
            assert (
                "forward" not in native.__dict__ and native.forward.__func__ is original
            )
            independent = []
            for start in (0, 3):
                result = reference(
                    input_ids=inputs["input_ids"][:, start : start + 3], use_cache=False
                ).logits
                independent.append(result)
            separate = torch.cat(independent, dim=1)
            separate.float().square().sum().backward()
            delta = (logits - separate).abs().max().item()
            gradient = 0
            for (name, value), (other_name, other) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                assert name == other_name
                if value.requires_grad:
                    assert value.grad is not None and other.grad is not None
                    gradient = max(
                        gradient,
                        (value.grad - other.grad).abs().max().item()
                        / max(other.grad.abs().max().item(), 1e-6),
                    )
            assert delta < 2e-5 and gradient < 0.001
            cases.append(
                {
                    "backend": backend,
                    "checkpointing": mode,
                    "logits_max_delta": delta,
                    "trainable_gradient_max_relative_delta": gradient,
                    "suffix_perturbation_delta": 0,
                    "instance_forward_restored": True,
                }
            )
    result = {
        "passed": True,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": "cpu",
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
