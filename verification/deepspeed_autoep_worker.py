#!/usr/bin/env python3
"""Validate one deterministic DeepSpeed AutoEP training step."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_INITIAL_W1 = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.5, 0.0], [0.0, 0.5]],
]


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nested_close(actual: object, expected: object, tolerance: float) -> bool:
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            _nested_close(actual_item, expected_item, tolerance)
            for actual_item, expected_item in zip(actual, expected)
        )
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _max_abs_delta(before: Any, after: Any) -> float:
    return float((after.detach().float() - before.detach().float()).abs().max())


def _gather_expert_tensor(torch: Any, dist: Any, local: Any) -> Any:
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args(argv)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.deepspeed_autoep.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "zero_stage": args.zero_stage,
        "precision": args.precision,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]

        stage = "import_deepspeed"
        import deepspeed
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        from deepspeed.utils.logging import logger as deepspeed_logger

        deepspeed_logger.setLevel(logging.WARNING)
        logging.getLogger("deepspeed").setLevel(logging.WARNING)
        if world_size != 2:
            raise AssertionError(f"expected world_size=2, got {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("physical CUDA device does not support BF16")

        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        report.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "deepspeed_version": str(deepspeed.__version__),
                "physical_device_name": torch.cuda.get_device_name(0),
                "physical_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            }
        )

        stage = "init_torch_process_group"
        torch.cuda.set_device(0)
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
            device_id=torch.device("cuda", 0),
        )

        stage = "init_deepspeed_backend"
        deepspeed.init_distributed(
            dist_backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=90),
            dist_init_required=False,
        )

        class TinyExpert(torch.nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                self.w1 = torch.nn.Linear(2, 2, bias=False)
                self.w2 = torch.nn.Linear(2, 2, bias=False)
                self.w3 = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.w1.weight.copy_(torch.eye(2) * scale)
                    self.w2.weight.copy_(torch.eye(2))
                    self.w3.weight.copy_(torch.eye(2))

            def forward(self, inputs: Any) -> Any:
                return self.w2(
                    torch.nn.functional.silu(self.w1(inputs))
                    * self.w3(inputs)
                )

        class TinyMoEBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = torch.nn.Linear(2, 2, bias=False)
                self.experts = torch.nn.ModuleList(
                    [TinyExpert(1.0), TinyExpert(0.5)]
                )
                with torch.no_grad():
                    self.gate.weight.copy_(torch.eye(2))
                self.gate.weight.requires_grad_(False)

            def forward(self, hidden_states: Any) -> Any:
                flattened = hidden_states.reshape(-1, 2)
                selected = self.gate(flattened).argmax(dim=-1)
                outputs = [
                    self.experts[int(expert_index)](token.unsqueeze(0))[0]
                    for token, expert_index in zip(flattened, selected)
                ]
                return torch.stack(outputs).reshape_as(hidden_states)

        class TinyAutoEPModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(
                    num_local_experts=2,
                    num_experts_per_tok=1,
                    scoring_func="softmax",
                    norm_topk_prob=True,
                )
                self.block = TinyMoEBlock()

            def forward(self, inputs: Any) -> Any:
                return self.block(inputs)

        stage = "reference_forward"
        model = TinyAutoEPModel()
        reference_model = copy.deepcopy(model)
        input_values = (
            [[[4.0, 1.0], [3.0, 1.0], [1.0, 4.0]]]
            if rank == 0
            else [[[4.0, 1.0], [1.0, 4.0], [1.0, 3.0]]]
        )
        reference_input = torch.tensor(input_values, dtype=torch.float32)
        with torch.no_grad():
            reference_output = reference_model(reference_input)
            reference_loss = torch.nn.functional.mse_loss(
                reference_output,
                torch.zeros_like(reference_output),
                reduction="mean",
            )

        stage = "initialize_engine"
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": args.zero_stage},
            "zero_allow_untested_optimizer": True,
            "fp16": {"enabled": False},
            "bf16": {"enabled": args.precision == "bf16"},
            "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
            "expert_parallel": {
                "enabled": True,
                "autoep_size": 2,
                "moe_layer_pattern": "^block$",
                "router_pattern": "gate",
                "expert_pattern": "experts",
                "expert_w1": "w1",
                "expert_w2": "w2",
                "expert_w3": "w3",
                "num_experts_attr": "num_local_experts",
                "top_k_attr": "num_experts_per_tok",
                "score_func": "softmax",
                "score_apply": "post",
                "route_norm": True,
                "top_k": 1,
                "use_grouped_mm": False,
            },
            "steps_per_print": 1_000_000,
        }
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
            dist_init_required=False,
        )
        block = engine.module.block
        if not isinstance(block, AutoEPMoELayer):
            raise AssertionError(
                f"AutoEP did not replace the MoE block: {type(block).__name__}"
            )
        initial_local = {
            name: getattr(block.experts, name).detach().clone()
            for name in ("w1", "w2", "w3")
        }
        initial_full_w1 = _gather_expert_tensor(
            torch,
            dist,
            initial_local["w1"],
        )
        report.update(
            {
                "engine_type": type(engine).__name__,
                "optimizer_type": type(optimizer).__name__,
                "effective_zero_stage": int(engine.zero_optimization_stage()),
                "moe_layer_type": type(block).__name__,
                "expert_parallel_size": int(block.ep_size),
                "expert_parallel_rank": int(block.ep_rank),
                "num_global_experts": int(block.num_experts),
                "num_local_experts": int(block.num_local_experts),
                "local_expert_shape": list(block.experts.w1.shape),
            }
        )

        stage = "training_step"
        torch.cuda.reset_peak_memory_stats(0)
        inputs = torch.tensor(input_values, dtype=dtype, device="cuda:0")
        target = torch.zeros_like(inputs)
        output = engine(inputs)
        loss = torch.nn.functional.mse_loss(output, target, reduction="mean")
        engine.backward(loss)
        gradient_norms = {
            name: float(getattr(block.experts, name).grad.detach().float().norm())
            for name in ("w1", "w2", "w3")
        }
        engine.step()
        torch.cuda.synchronize()

        final_local = {
            name: getattr(block.experts, name).detach().clone()
            for name in ("w1", "w2", "w3")
        }
        final_full_w1 = _gather_expert_tensor(
            torch,
            dist,
            final_local["w1"],
        )
        output_values = output.detach().float().cpu().tolist()
        reference_output_values = reference_output.float().cpu().tolist()
        loss_value = float(loss.detach().float().cpu())
        reference_loss_value = float(reference_loss.float().cpu())
        parameter_deltas = {
            name: _max_abs_delta(initial_local[name], final_local[name])
            for name in ("w1", "w2", "w3")
        }
        report.update(
            {
                "input": input_values,
                "output": output_values,
                "reference_output": reference_output_values,
                "loss": loss_value,
                "reference_loss": reference_loss_value,
                "global_steps": int(engine.global_steps),
                "gradient_norms": gradient_norms,
                "parameter_max_abs_deltas": parameter_deltas,
                "initial_full_w1": initial_full_w1.float().cpu().tolist(),
                "final_full_w1": final_full_w1.float().cpu().tolist(),
                "tokens_per_expert": block.tokens_per_expert.cpu().tolist(),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            }
        )

        tolerance = 1e-6 if args.precision == "fp32" else 4e-2
        if not _nested_close(
            initial_full_w1.float().cpu().tolist(),
            EXPECTED_INITIAL_W1,
            tolerance,
        ):
            raise AssertionError("AutoEP initial expert repacking is incorrect")
        if not _nested_close(output_values, reference_output_values, tolerance):
            raise AssertionError(
                f"AutoEP output mismatch: {output_values} != {reference_output_values}"
            )
        if abs(loss_value - reference_loss_value) > tolerance:
            raise AssertionError(
                f"AutoEP loss mismatch: {loss_value} != {reference_loss_value}"
            )
        if int(engine.global_steps) != 1:
            raise AssertionError(
                f"expected one optimizer step, got {engine.global_steps}"
            )
        if min(gradient_norms.values()) <= 0:
            raise AssertionError(f"missing expert gradients: {gradient_norms}")
        if min(parameter_deltas.values()) <= 0:
            raise AssertionError(f"expert parameters did not update: {parameter_deltas}")
        expected_tokens = [2.0, 1.0] if rank == 0 else [1.0, 2.0]
        if not _nested_close(
            block.tokens_per_expert.cpu().tolist(),
            expected_tokens,
            0,
        ):
            raise AssertionError(
                f"unexpected expert utilization: {block.tokens_per_expert}"
            )

        stage = "barrier"
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        dist = None
        report.update({"status": "success", "stage": "complete"})
        _write_report(args.report_dir, rank, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_report(args.report_dir, rank, report)
        print(
            f"rank {rank} failed at {stage}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
