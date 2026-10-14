#!/usr/bin/env python3
"""Native local MoE and two-rank CPU distributed semantics, with explicit scope."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def moe(args):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from fakegpu.llm_estimator import (
        estimate_decoder_inference,
        inspect_safetensors_checkpoint,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("MoE semantic check requires exactly one visible physical GPU")
    torch.set_num_threads(1)
    torch.manual_seed(3201)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=128,
        use_cache=False,
        attention_dropout=0,
    )
    base = Qwen3MoeForCausalLM(config)
    checkpoint = args.work_dir / "checkpoint"
    checkpoint.mkdir(parents=True)
    base.save_pretrained(checkpoint, safe_serialization=True)
    parameters = sum(p.numel() for p in base.parameters())
    header = inspect_safetensors_checkpoint(checkpoint)
    assert header["parameter_count"] == parameters
    estimate = estimate_decoder_inference(
        checkpoint,
        prompt_tokens=12,
        generated_tokens=3,
        dtype="float32",
        attention_implementation="sdpa",
    )
    assert estimate["model"]["model_kind"] == "mixture_of_experts"
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for attention in ("eager", "sdpa"):
            model = deepcopy(base).to(device=device, dtype=dtype).train()
            model.set_attn_implementation(attention)
            block = model.model.layers[0].mlp
            hidden = torch.linspace(
                -0.7, 0.8, 2 * 9 * 64, device=device, dtype=dtype
            ).reshape(2, 9, 64)
            native_input = hidden.detach().requires_grad_()
            output, logits = block(native_input)
            output.float().square().sum().backward()
            native_grad = native_input.grad.detach().clone()
            native_parameter_grads = {
                n: p.grad.detach().clone()
                for n, p in block.named_parameters()
                if p.grad is not None
            }
            block.zero_grad(set_to_none=True)
            reference_input = hidden.detach().requires_grad_()
            flat = reference_input.reshape(-1, 64)
            router = F.linear(flat, block.gate.weight)
            probabilities = torch.softmax(router.float(), dim=-1)
            selected = probabilities.argsort(dim=-1, descending=True)[:, :2]
            weights = probabilities.gather(1, selected)
            weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype)
            # Independent token/slot ordering and explicit SwiGLU, without calling experts.
            reference_tokens = []
            for token in range(flat.shape[0]):
                result = torch.zeros(64, device=device, dtype=dtype)
                for slot in range(2):
                    expert = block.experts[int(selected[token, slot])]
                    x = flat[token : token + 1]
                    projected = F.linear(
                        F.silu(F.linear(x, expert.gate_proj.weight))
                        * F.linear(x, expert.up_proj.weight),
                        expert.down_proj.weight,
                    )[0]
                    result = result + projected * weights[token, slot]
                reference_tokens.append(result)
            reference = torch.stack(reference_tokens).reshape_as(output)
            reference.float().square().sum().backward()
            output_error = float((output - reference).abs().max())
            gradient_error = float((native_grad - reference_input.grad).abs().max())
            parameter_error = max(
                float((p.grad - native_parameter_grads[n]).abs().max())
                for n, p in block.named_parameters()
                if n in native_parameter_grads
            )
            assert torch.equal(logits.detach(), router.detach())
            assert output_error <= (1e-7 if dtype == torch.float32 else 1e-4)
            assert gradient_error <= (1e-7 if dtype == torch.float32 else 2e-5)
            assert parameter_error <= (1e-6 if dtype == torch.float32 else 5e-4)
            model.zero_grad(set_to_none=True)
            ids = torch.arange(24, device=device).reshape(2, 12) % 128
            result = model(
                input_ids=ids, labels=ids, use_cache=False, output_router_logits=True
            )
            assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
            result.loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            counts = torch.bincount(selected.flatten(), minlength=4).cpu().tolist()
            assert sum(counts) == 18 * 2
            cases.append(
                {
                    "dtype": str(dtype),
                    "attention": attention,
                    "moe_output_max_absolute_error": output_error,
                    "input_gradient_max_absolute_error": gradient_error,
                    "parameter_gradient_max_absolute_error": parameter_error,
                    "expert_token_assignments": counts,
                    "loss": float(result.loss.detach()),
                    "full_model_gradients_finite": True,
                }
            )
    identity = {
        "device": str(device),
        "physical_gpu_used": device.type == "cuda",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if device.type == "cuda":
        identity.update(
            gpu_name=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        )
    return {
        "schema_version": "fakegpu.local_moe_semantics.v1",
        "environment": identity,
        "parameter_count": parameters,
        "header_parameter_count": header["parameter_count"],
        "estimator_model": estimate["model"],
        "cases": cases,
        "scope": "Tiny random native Qwen3 MoE, local routing/output/gradients only. No expert/tensor/pipeline parallelism, pretrained quality, throughput or memory-accuracy claim.",
    }


def distributed(args):
    import torch
    import torch.distributed as dist
    from torch import nn
    from fakegpu.training_plan import estimate_training_plan

    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    try:
        torch.manual_seed(73)
        model = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 3)).double()
        reference = deepcopy(model)
        inputs, labels = (
            torch.randn(8, 8, dtype=torch.float64),
            torch.randn(8, 3, dtype=torch.float64),
        )
        if args.mode == "ddp":
            from torch.nn.parallel import DistributedDataParallel

            wrapped = DistributedDataParallel(model)
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            fully_shard(model, mesh=init_device_mesh("cpu", (world,)))
            wrapped = model
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3, foreach=False)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, foreach=False
        )
        start, end = rank * 4, (rank + 1) * 4
        loss = (wrapped(inputs[start:end]) - labels[start:end]).square().mean()
        loss.backward()
        reference_loss = (reference(inputs) - labels).square().mean()
        reference_loss.backward()

        def full(tensor):
            return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor

        gradient_error = max(
            float((full(p.grad) - q.grad).abs().max())
            for p, q in zip(model.parameters(), reference.parameters())
        )
        assert gradient_error <= 1e-12
        optimizer.step()
        reference_optimizer.step()
        update_error = max(
            float((full(p) - q).abs().max())
            for p, q in zip(model.parameters(), reference.parameters())
        )
        assert update_error <= 1e-12
        parameter_bytes = sum(
            p.numel() * p.element_size() for p in reference.parameters()
        )
        storage = {}
        for key in ("exp_avg", "exp_avg_sq", "step"):
            storage[key] = sum(
                (v[key].to_local() if hasattr(v[key], "to_local") else v[key]).numel()
                * v[key].element_size()
                for v in optimizer.state.values()
            )
        plan = estimate_training_plan(
            {
                "framework": "fsdp2" if args.mode == "fsdp2" else "fsdp",
                "world_size": world,
                "mixed_precision": "no",
                "sharding_strategy": "replicated"
                if args.mode == "ddp"
                else "full_shard",
            },
            parameter_bytes=parameter_bytes,
            activation_bytes=0,
            parameter_element_bytes=8,
            world_size=world,
            parameter_shapes=[list(p.shape) for p in reference.parameters()]
            if args.mode == "fsdp2"
            else None,
        )
        # This exact byte comparison covers replicated FP64 Adam moments only.
        if args.mode == "ddp":
            assert (
                plan["rank_local_storage"]["optimizer_state_bytes"]
                == storage["exp_avg"] + storage["exp_avg_sq"]
            )
        records = [None] * world
        dist.all_gather_object(
            records,
            {
                "rank": rank,
                "gradient_max_absolute_error": gradient_error,
                "update_max_absolute_error": update_error,
                "optimizer_local_tensor_bytes": storage,
            },
        )
        if args.mode == "fsdp2":
            observed = [r["optimizer_local_tensor_bytes"]["exp_avg"] for r in records]
            assert (
                plan["sharding"]["parameter_layout"]["parameter_bytes_by_rank"]
                == observed
            )
            assert plan["rank_local_storage"]["optimizer_state_bytes"] == 2 * max(
                observed
            )
        return {
            "schema_version": "fakegpu.local_distributed_semantics.v1",
            "mode": args.mode,
            "backend": "gloo",
            "torch": torch.__version__,
            "world_size": world,
            "physical_gpu_used": False,
            "parameter_bytes": parameter_bytes,
            "ranks": records,
            "storage_plan": plan,
            "scope": "Two CPU processes on one host. Mathematical synchronization and AdamW update comparison only; no physical multi-GPU, multi-node or communication-performance validation.",
        }
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("moe", "ddp", "fsdp2"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "moe" or args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = moe(args) if args.mode == "moe" else distributed(args)
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mode": args.mode, "output": str(args.output)}))


if __name__ == "__main__":
    main()
