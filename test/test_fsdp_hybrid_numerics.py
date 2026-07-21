#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import os
import traceback
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensor_values(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate two-rank FSDP sharding, reduce-scatter, parameter "
            "reconstruction, and full-state-dict restoration with real CUDA "
            "and fake NCCL."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.hybrid_fsdp_numerics.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "physical_device_index": 0,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            StateDictType,
        )

        if world_size != 2:
            raise AssertionError(f"expected world_size=2, got {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")

        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_version"] = str(torch.version.cuda)
        report["physical_device_name"] = torch.cuda.get_device_name(0)
        report["physical_compute_capability"] = list(
            torch.cuda.get_device_capability(0)
        )

        stage = "set_device"
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        stage = "init_process_group"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=60),
        )

        stage = "construct_fsdp"
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
            )
        fsdp_model = FSDP(
            model,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        flat_parameter = next(fsdp_model.parameters())
        report["unsharded_parameter_numel"] = 2
        report["local_shard_numel"] = int(flat_parameter.numel())
        if flat_parameter.numel() != 1:
            raise AssertionError(
                "FSDP did not shard the two-element parameter evenly: "
                f"local numel={flat_parameter.numel()}"
            )

        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)

        stage = "forward_backward"
        scale = float(rank + 1)
        inputs = torch.tensor(
            [[scale, 2.0 * scale]],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = fsdp_model(inputs).sum()
        loss.backward()
        torch.cuda.synchronize()

        local_gradient = flat_parameter.grad
        if local_gradient is None:
            raise AssertionError("FSDP did not produce a local shard gradient")
        expected_local_gradient = torch.tensor(
            [1.5 if rank == 0 else 3.0],
            dtype=torch.float32,
            device=device,
        )
        report["local_loss"] = float(loss.detach().cpu().item())
        report["local_shard_gradient"] = _tensor_values(local_gradient)
        report["expected_local_shard_gradient"] = _tensor_values(
            expected_local_gradient
        )
        if not torch.allclose(
            local_gradient,
            expected_local_gradient,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP reduce-scatter gradient mismatch: "
                f"{report['local_shard_gradient']} != "
                f"{report['expected_local_shard_gradient']}"
            )

        stage = "optimizer_step"
        optimizer.step()
        torch.cuda.synchronize()
        expected_local_parameter = torch.tensor(
            [0.85 if rank == 0 else -0.3],
            dtype=torch.float32,
            device=device,
        )
        report["local_shard_after_step"] = _tensor_values(flat_parameter)
        report["expected_local_shard_after_step"] = _tensor_values(
            expected_local_parameter
        )
        if not torch.allclose(
            flat_parameter,
            expected_local_parameter,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP local optimizer result mismatch: "
                f"{report['local_shard_after_step']} != "
                f"{report['expected_local_shard_after_step']}"
            )

        stage = "summon_full_parameters"
        expected_full_parameter = torch.tensor(
            [[0.85, -0.3]],
            dtype=torch.float32,
            device=device,
        )
        with FSDP.summon_full_params(
            fsdp_model,
            recurse=True,
            writeback=False,
            rank0_only=False,
        ):
            full_parameter = fsdp_model.module.weight.detach().clone()
        report["full_parameter_after_step"] = full_parameter.cpu().tolist()
        report["expected_full_parameter_after_step"] = (
            expected_full_parameter.cpu().tolist()
        )
        if not torch.allclose(
            full_parameter,
            expected_full_parameter,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP full parameter reconstruction mismatch: "
                f"{report['full_parameter_after_step']}"
            )

        stage = "full_state_dict"
        full_state_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with FSDP.state_dict_type(
                fsdp_model,
                StateDictType.FULL_STATE_DICT,
                full_state_config,
            ):
                state_dict = fsdp_model.state_dict()
        state_weight = state_dict.get("weight")
        if state_weight is None:
            raise AssertionError(
                f"full state dict did not contain weight: {sorted(state_dict)}"
            )
        report["full_state_dict_keys"] = sorted(state_dict)
        report["full_state_dict_weight"] = state_weight.tolist()
        if not torch.allclose(
            state_weight,
            expected_full_parameter.cpu(),
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP full state dict contains unexpected values: "
                f"{report['full_state_dict_weight']}"
            )

        stage = "state_dict_serialization"
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        report["serialized_state_dict_bytes"] = int(buffer.tell())
        buffer.seek(0)
        reloaded_state_dict = torch.load(buffer, map_location="cpu")

        stage = "state_dict_restore"
        restored_module = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            restored_module.weight.zero_()
        restored_fsdp = FSDP(
            restored_module,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with FSDP.state_dict_type(
                restored_fsdp,
                StateDictType.FULL_STATE_DICT,
                full_state_config,
            ):
                incompatible = restored_fsdp.load_state_dict(reloaded_state_dict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(
                "FSDP full state dict restore was incomplete: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        with FSDP.summon_full_params(
            restored_fsdp,
            recurse=True,
            writeback=False,
            rank0_only=False,
        ):
            restored_parameter = restored_fsdp.module.weight.detach().clone()
        report["restored_full_parameter"] = restored_parameter.cpu().tolist()
        if not torch.allclose(
            restored_parameter,
            expected_full_parameter,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                "restored FSDP parameter mismatch: "
                f"{report['restored_full_parameter']}"
            )

        stage = "barrier"
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        dist = None

        report["status"] = "success"
        report["stage"] = "complete"
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
            file=os.sys.stderr,
            flush=True,
        )
        if dist is not None:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
