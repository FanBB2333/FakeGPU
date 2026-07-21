#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


def _write_report(report_dir: Path, rank: int, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank_{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _local_tensor(value: Any) -> Any:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _tensor_values(tensor: Any) -> list[Any]:
    return tensor.detach().float().cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate FSDP2/DTensor parameter sharding and averaged gradients "
            "with real CUDA compute and fake NCCL communication."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.hybrid_fsdp2_numerics.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "precision": args.precision,
        "physical_device_index": 0,
        "status": "starting",
    }
    stage = "import_torch"
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if world_size not in (2, 4):
            raise AssertionError(
                f"expected world_size to be 2 or 4, got {world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is not available")

        precision_dtypes = {
            "fp32": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        parameter_dtype = precision_dtypes[args.precision]
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
            timeout=timedelta(seconds=90),
        )

        stage = "construct_fsdp2"
        model = torch.nn.Linear(
            world_size,
            world_size,
            bias=False,
            device=device,
        )
        with torch.no_grad():
            model.weight.copy_(
                torch.eye(world_size, dtype=torch.float32, device=device)
            )
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp",),
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=parameter_dtype,
            reduce_dtype=torch.float32 if parameter_dtype is not None else None,
            output_dtype=torch.float32 if parameter_dtype is not None else None,
        )
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        parameter = next(model.parameters())
        local_parameter = _local_tensor(parameter)
        expected_local_shape = [1, world_size]
        report["parameter_type"] = type(parameter).__name__
        report["parameter_dtype"] = str(parameter.dtype)
        report["global_parameter_shape"] = list(parameter.shape)
        report["local_shard_shape"] = list(local_parameter.shape)
        if list(parameter.shape) != [world_size, world_size]:
            raise AssertionError(
                "FSDP2 changed the global parameter shape unexpectedly: "
                f"{list(parameter.shape)}"
            )
        if list(local_parameter.shape) != expected_local_shape:
            raise AssertionError(
                "FSDP2 did not shard one matrix row per rank: "
                f"rank={rank}, local_shape={list(local_parameter.shape)}"
            )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        stage = "forward_backward"
        scale = float(rank + 1)
        inputs = (
            torch.arange(
                1,
                world_size + 1,
                dtype=torch.float32,
                device=device,
            )
            * scale
        ).reshape(1, world_size)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

        gradient = parameter.grad
        if gradient is None:
            raise AssertionError("FSDP2 did not produce a sharded gradient")
        local_gradient = _local_tensor(gradient)
        average_scale = (world_size + 1.0) / 2.0
        expected_gradient_row = (
            torch.arange(
                1,
                world_size + 1,
                dtype=torch.float32,
                device=device,
            )
            * average_scale
        ).reshape(1, world_size)
        tolerance = 1e-6 if args.precision == "fp32" else 2e-2
        report["local_loss"] = float(loss.detach().float().cpu().item())
        report["gradient_dtype"] = str(local_gradient.dtype)
        report["local_shard_gradient"] = _tensor_values(local_gradient)
        report["expected_local_shard_gradient"] = _tensor_values(
            expected_gradient_row
        )
        if not torch.allclose(
            local_gradient.float(),
            expected_gradient_row,
            atol=tolerance,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP2 reduce-scatter gradient mismatch: "
                f"{report['local_shard_gradient']} != "
                f"{report['expected_local_shard_gradient']}"
            )

        stage = "optimizer_step"
        optimizer.step()
        torch.cuda.synchronize()
        expected_full_parameter = torch.eye(
            world_size,
            dtype=torch.float32,
            device=device,
        ) - 0.1 * expected_gradient_row.expand(world_size, -1)
        expected_local_parameter = expected_full_parameter[rank : rank + 1]
        local_parameter_after_step = _local_tensor(parameter)
        report["local_shard_after_step"] = _tensor_values(
            local_parameter_after_step
        )
        report["expected_local_shard_after_step"] = _tensor_values(
            expected_local_parameter
        )
        if not torch.allclose(
            local_parameter_after_step.float(),
            expected_local_parameter,
            atol=tolerance,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP2 local optimizer result mismatch: "
                f"{report['local_shard_after_step']}"
            )

        stage = "full_tensor"
        full_parameter = parameter.full_tensor().detach()
        torch.cuda.synchronize()
        report["full_parameter_after_step"] = _tensor_values(full_parameter)
        report["expected_full_parameter_after_step"] = _tensor_values(
            expected_full_parameter
        )
        if not torch.allclose(
            full_parameter.float(),
            expected_full_parameter,
            atol=tolerance,
            rtol=0,
        ):
            raise AssertionError(
                "FSDP2 full parameter reconstruction mismatch: "
                f"{report['full_parameter_after_step']}"
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
