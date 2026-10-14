"""Compare isolated packed Qwen3.5 with independent native segment executions.

Real tiny models exercise both installed Transformers interfaces, all decoder
layers, parameter/input gradients, checkpoint recomputation and serialization.
This is a kernel semantics audit, not a training-time calibration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--collator-framework", choices=("huggingface", "llamafactory", "axolotl"))
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.collator_framework) != bool(args.tokenizer):
        parser.error("collator audits require both --collator-framework and --tokenizer")
    os.environ.update(CUDA_VISIBLE_DEVICES=args.cuda_device if args.device == "cuda" else "",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    import torch
    import transformers
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from fakegpu._finetune_segments import isolated_qwen35_segments
    from fakegpu._calibration_cuda import _reject_simulated_cuda_environment

    _reject_simulated_cuda_environment()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tolerance = 0.02 if args.device == "cuda" else 2e-5
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        pad_token_id=0, eos_token_id=2, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": .5, "mrope_section": [3, 3, 2],
                         "mrope_interleaved": True},
    )
    spans = [[(0, 1), (1, 4), (4, 8)], [(0, 2), (2, 6)]]
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 0, 0]], device=args.device)
    positions = torch.zeros_like(ids)
    allowed = torch.zeros(2, 8, 8, dtype=torch.bool, device=args.device)
    labels = ids.clone()
    for row, segments in enumerate(spans):
        for start, end in segments:
            positions[row, start:end] = torch.arange(end - start, device=args.device)
            allowed[row, start:end, start:end] = torch.ones(end-start, end-start, device=args.device, dtype=torch.bool).tril()
            labels[row, start] = -100
        labels[row, segments[-1][1]:] = -100
    mask = torch.where(allowed, 0., torch.finfo(dtype).min).to(dtype)[:, None]
    inputs = dict(input_ids=ids, position_ids=positions, attention_mask=mask,
                  labels=labels, use_cache=False)
    report = dict(schema_version="fakegpu.finetune_segment_reference_audit.v1",
                  device=args.device, dtype=str(dtype), torch=torch.__version__,
                  transformers=transformers.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cases=[])

    if args.collator_framework:
        from fakegpu._finetune_packed_batches import IsolatedPackingCollator, describe_packed_batch

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        tokenizer.pad_token_id = 0
        if args.collator_framework == "huggingface":
            native = transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
        elif args.collator_framework == "llamafactory":
            from llamafactory.data import SFTDataCollatorWith4DAttentionMask
            from llamafactory.data.template import TEMPLATES

            native = SFTDataCollatorWith4DAttentionMask(
                tokenizer, template=TEMPLATES["qwen3_nothink"], model=None,
                pad_to_multiple_of=8, block_diag_attn=True, neat_packing=True,
                attn_implementation="eager", compute_dtype=dtype,
            )
        else:
            from axolotl.utils.collators.batching import V2BatchSamplerDataCollatorForSeq2Seq

            native = V2BatchSamplerDataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
        features = []
        for row, segments in enumerate(spans):
            if args.collator_framework == "axolotl":
                features.append([
                    {"input_ids":ids[row,start:end].cpu().tolist(),
                     "labels":labels[row,start:end].cpu().tolist(),
                     "attention_mask":[1]*(end-start), "position_ids":list(range(end-start))}
                    for start,end in segments
                ])
            else:
                size = segments[-1][1]
                features.append({
                    "input_ids":ids[row,:size].cpu().tolist(),
                    "labels":labels[row,:size].cpu().tolist(),
                    "position_ids":positions[row,:size].cpu().tolist(),
                    "attention_mask":[index+1 for index,(start,end) in enumerate(segments) for _ in range(end-start)],
                })
        collated = IsolatedPackingCollator(native, framework=args.collator_framework, mask_dtype=dtype)(features)
        collated = {key:value.to(args.device) if isinstance(value,torch.Tensor) else value for key,value in collated.items()}
        assert set(collated) == set(inputs)
        assert all(torch.equal(value,inputs[key]) if isinstance(value,torch.Tensor) else value == inputs[key] for key,value in collated.items())
        report["packed_collation"] = {"framework":args.collator_framework,
                                      "native_class":f"{type(native).__module__}.{type(native).__qualname__}",
                                      "batch":describe_packed_batch(collated)}
        inputs = collated

    def new_model(backend):
        torch.manual_seed(42)
        cfg = deepcopy(config)
        cfg._attn_implementation = backend
        return Qwen3_5ForCausalLM(cfg).to(device=args.device, dtype=dtype).train()

    def captures(model):
        outputs, hooks = [], []
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(lambda module, inputs, out: outputs.append(out.detach().clone())))
        return outputs, hooks

    try:
        for backend in ("eager", "sdpa"):
            for checkpointing in (None, False, True):
                model = new_model(backend)
                reference = new_model(backend)
                if checkpointing is not None:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointing})
                    model.enable_input_require_grads()
                original_methods = [layer.forward.__func__ for layer in model.model.layers]
                packed_layers, hooks = captures(model)
                with isolated_qwen35_segments(model):
                    result = model(**inputs)
                    assert torch.isfinite(result.loss)
                    layer_outputs = packed_layers[:4]
                    objective = sum(result.logits[row, start:end].float().square().sum()
                                    for row, segments in enumerate(spans) for start, end in segments)
                    objective.backward()
                for hook in hooks:
                    hook.remove()
                assert [layer.forward.__func__ for layer in model.model.layers] == original_methods
                logit_deltas, layer_deltas, reference_objectives = [], [], []
                for row, segments in enumerate(spans):
                    for start, end in segments:
                        native_layers, native_hooks = captures(reference)
                        separate = reference(input_ids=ids[row:row+1, start:end], use_cache=False)
                        for hook in native_hooks:
                            hook.remove()
                        logit_deltas.append((result.logits[row:row+1, start:end].float() - separate.logits.float()).abs().max().item())
                        layer_deltas.extend((packed[row:row+1, start:end].float() - native.float()).abs().max().item()
                                            for packed, native in zip(layer_outputs, native_layers))
                        reference_objectives.append(separate.logits.float().square().sum())
                sum(reference_objectives).backward()
                grad_deltas, relative_grad_deltas = [], []
                for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
                    assert name == other_name
                    assert parameter.grad is not None and other.grad is not None
                    assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all()
                    delta = (parameter.grad.float() - other.grad.float()).abs().max().item()
                    grad_deltas.append(delta)
                    relative_grad_deltas.append(delta / max(other.grad.float().abs().max().item(), 1e-6))
                case = dict(backend=backend, checkpointing=checkpointing,
                            logits_max_delta=max(logit_deltas), layer_max_delta=max(layer_deltas),
                            parameter_gradient_max_delta=max(grad_deltas),
                            parameter_gradient_max_relative_delta=max(relative_grad_deltas))
                assert case["logits_max_delta"] <= tolerance, case
                assert case["layer_max_delta"] <= tolerance, case
                assert case["parameter_gradient_max_relative_delta"] <= (0.06 if args.device == "cuda" else 0.001), case
                model.gradient_checkpointing_disable()
                with isolated_qwen35_segments(model):
                    changed = ids.clone()
                    changed[0, :4] += 20
                    with torch.no_grad():
                        baseline = model(**inputs).logits[0, 4:].float()
                        perturbed = model(**{**inputs, "input_ids": changed}).logits[0, 4:].float()
                    embedded = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                    embedded_inputs = {key: value for key, value in inputs.items() if key not in {"input_ids", "labels"}}
                    logits = model(inputs_embeds=embedded, **embedded_inputs).logits[0, 4:].float()
                    gradient, = torch.autograd.grad(logits.square().sum(), embedded)
                case["suffix_logits_max_delta"] = (baseline - perturbed).abs().max().item()
                case["suffix_gradient_into_prefix_max"] = gradient[0, :4].abs().max().item()
                assert case["suffix_logits_max_delta"] == case["suffix_gradient_into_prefix_max"] == 0, case
                with tempfile.TemporaryDirectory(prefix="fakegpu-segments-") as directory:
                    model.save_pretrained(directory)
                    loaded = Qwen3_5ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation=backend)
                    original_buffers = dict(model.named_buffers())
                    case["loaded_buffer_dtype_differences"] = {
                        name: {"original": str(original_buffers[name].dtype), "loaded": str(value.dtype)}
                        for name, value in loaded.named_buffers()
                        if name in original_buffers and value.dtype != original_buffers[name].dtype
                    }
                    # The tiny fixture casts every floating buffer to BF16,
                    # including rotary inv_freq. Native loading initializes
                    # nonpersistent buffers in FP32: align the same execution
                    # dtype before asserting exact round-trip equivalence.
                    loaded = loaded.to(device=args.device, dtype=dtype)
                    assert set(loaded.state_dict()) == set(model.state_dict())
                    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in model.state_dict().items())
                    with isolated_qwen35_segments(loaded), torch.no_grad():
                        restored = loaded(**inputs).logits[0, 4:].float()
                    case["restored_logits_max_delta"] = (baseline - restored).abs().max().item()
                    assert case["restored_logits_max_delta"] == 0, case
                try:
                    with isolated_qwen35_segments(model):
                        invalid = mask.clone()
                        invalid[0, 0, 5, 0] = 0
                        model(**{**inputs, "attention_mask": invalid})
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid cross-segment mask accepted")
                assert [layer.forward.__func__ for layer in model.model.layers] == original_methods
                assert not model._forward_pre_hooks
                report["cases"].append(case)
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
