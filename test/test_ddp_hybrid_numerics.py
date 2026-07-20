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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate numerical DDP gradient averaging with real CUDA and fake NCCL."
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.hybrid_ddp_numerics.rank.v1",
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
        from torch.nn.parallel import DistributedDataParallel

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

        stage = "construct_ddp"
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
            )
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[0],
            output_device=0,
        )
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        stage = "forward_backward"
        scale = float(rank + 1)
        inputs = torch.tensor(
            [[scale, 2.0 * scale]],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = ddp_model(inputs).sum()
        loss.backward()
        torch.cuda.synchronize()

        gradient = model.weight.grad
        if gradient is None:
            raise AssertionError("DDP did not produce a gradient")
        gradient_cpu = gradient.detach().cpu()
        expected_gradient = torch.tensor([[1.5, 3.0]], dtype=torch.float32)
        report["local_loss"] = float(loss.detach().cpu().item())
        report["gradient"] = gradient_cpu.tolist()
        report["expected_averaged_gradient"] = expected_gradient.tolist()
        if report["local_loss"] <= 0:
            raise AssertionError(f"loss must be non-zero, got {report['local_loss']}")
        if not torch.allclose(
            gradient_cpu,
            expected_gradient,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                f"averaged gradient mismatch: {gradient_cpu.tolist()} "
                f"!= {expected_gradient.tolist()}"
            )

        stage = "optimizer_step"
        optimizer.step()
        torch.cuda.synchronize()
        parameters = model.weight.detach().clone()
        expected_parameters = torch.tensor(
            [[0.85, -0.3]],
            dtype=torch.float32,
            device=device,
        )
        report["parameters_after_step"] = parameters.cpu().tolist()
        report["expected_parameters_after_step"] = (
            expected_parameters.cpu().tolist()
        )
        if not torch.allclose(
            parameters,
            expected_parameters,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                f"optimizer parameters mismatch: {parameters.cpu().tolist()} "
                f"!= {expected_parameters.cpu().tolist()}"
            )

        stage = "parameter_consistency"
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        torch.cuda.synchronize()
        report["gathered_parameters"] = [
            value.cpu().tolist() for value in gathered
        ]
        if any(
            not torch.allclose(value, expected_parameters, atol=1e-6, rtol=0)
            for value in gathered
        ):
            raise AssertionError(
                f"parameters differ across ranks: {report['gathered_parameters']}"
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
