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
        description=(
            "Validate DDP gradient averaging and common execution options "
            "with real CUDA and fake NCCL."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["basic", "no-sync", "find-unused", "static-graph"],
        default="basic",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {
        "schema_version": "fakegpu.hybrid_ddp_numerics.rank.v1",
        "rank": rank,
        "world_size": world_size,
        "variant": args.variant,
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
        ddp_options: dict[str, Any] = {
            "device_ids": [0],
            "output_device": 0,
        }
        if args.variant == "find-unused":
            class ConditionalModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.branch_a = torch.nn.Linear(2, 1, bias=False)
                    self.branch_b = torch.nn.Linear(2, 1, bias=False)

                def forward(self, inputs: Any, branch: int) -> Any:
                    if branch == 0:
                        return self.branch_a(inputs)
                    return self.branch_b(inputs)

            model = ConditionalModel().to(device)
            with torch.no_grad():
                model.branch_a.weight.copy_(
                    torch.tensor([[1.0, 0.0]], device=device)
                )
                model.branch_b.weight.copy_(
                    torch.tensor([[0.0, 1.0]], device=device)
                )
            ddp_options["find_unused_parameters"] = True
        else:
            model = torch.nn.Linear(2, 1, bias=False, device=device)
            with torch.no_grad():
                model.weight.copy_(
                    torch.tensor([[1.0, 0.0]], device=device)
                )
            if args.variant == "static-graph":
                ddp_options.update(
                    {
                        "static_graph": True,
                        "gradient_as_bucket_view": True,
                        "bucket_cap_mb": 0.001,
                    }
                )

        report["ddp_options"] = {
            key: value
            for key, value in ddp_options.items()
            if key not in {"device_ids", "output_device"}
        }
        ddp_model = DistributedDataParallel(model, **ddp_options)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        scale = float(rank + 1)
        inputs = torch.tensor(
            [[scale, 2.0 * scale]],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)

        stage = "forward_backward"
        if args.variant == "no-sync":
            with ddp_model.no_sync():
                first_loss = ddp_model(inputs).sum()
                first_loss.backward()
            second_loss = ddp_model(inputs * 2.0).sum()
            second_loss.backward()
            loss = first_loss.detach() + second_loss.detach()
            expected_gradients = torch.tensor(
                [[4.5, 9.0]], dtype=torch.float32, device=device
            )
            expected_parameters = torch.tensor(
                [[0.55, -0.9]], dtype=torch.float32, device=device
            )
            training_steps = 1
        elif args.variant == "find-unused":
            loss = ddp_model(inputs, rank).sum()
            loss.backward()
            expected_gradients = torch.tensor(
                [[0.5, 1.0], [1.0, 2.0]],
                dtype=torch.float32,
                device=device,
            )
            expected_parameters = torch.tensor(
                [[0.95, -0.1], [-0.1, 0.8]],
                dtype=torch.float32,
                device=device,
            )
            training_steps = 1
        elif args.variant == "static-graph":
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = ddp_model(inputs).sum()
                loss.backward()
                optimizer.step()
            expected_gradients = torch.tensor(
                [[1.5, 3.0]], dtype=torch.float32, device=device
            )
            expected_parameters = torch.tensor(
                [[0.7, -0.6]], dtype=torch.float32, device=device
            )
            training_steps = 2
        else:
            loss = ddp_model(inputs).sum()
            loss.backward()
            expected_gradients = torch.tensor(
                [[1.5, 3.0]], dtype=torch.float32, device=device
            )
            expected_parameters = torch.tensor(
                [[0.85, -0.3]], dtype=torch.float32, device=device
            )
            training_steps = 1
        torch.cuda.synchronize()

        if args.variant == "find-unused":
            gradients = torch.cat(
                [
                    model.branch_a.weight.grad,
                    model.branch_b.weight.grad,
                ],
                dim=0,
            )
        else:
            gradient = model.weight.grad
            if gradient is None:
                raise AssertionError("DDP did not produce a gradient")
            gradients = gradient
        if gradients is None:
            raise AssertionError("DDP did not produce all expected gradients")

        report["local_loss"] = float(loss.detach().cpu().item())
        report["training_steps"] = training_steps
        report["gradient"] = gradients.detach().cpu().tolist()
        report["expected_averaged_gradient"] = (
            expected_gradients.cpu().tolist()
        )
        if report["local_loss"] <= 0:
            raise AssertionError(f"loss must be non-zero, got {report['local_loss']}")
        if not torch.allclose(
            gradients,
            expected_gradients,
            atol=1e-6,
            rtol=0,
        ):
            raise AssertionError(
                f"averaged gradient mismatch: {report['gradient']} "
                f"!= {report['expected_averaged_gradient']}"
            )

        stage = "optimizer_step"
        if args.variant != "static-graph":
            optimizer.step()
        torch.cuda.synchronize()
        if args.variant == "find-unused":
            parameters = torch.cat(
                [
                    model.branch_a.weight.detach(),
                    model.branch_b.weight.detach(),
                ],
                dim=0,
            )
        else:
            parameters = model.weight.detach().clone()
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
                f"optimizer parameters mismatch: {report['parameters_after_step']} "
                f"!= {report['expected_parameters_after_step']}"
            )

        stage = "parameter_consistency"
        parameter_vector = parameters.reshape(-1)
        gathered = [
            torch.empty_like(parameter_vector) for _ in range(world_size)
        ]
        dist.all_gather(gathered, parameter_vector)
        torch.cuda.synchronize()
        report["gathered_parameter_vectors"] = [
            value.cpu().tolist() for value in gathered
        ]
        if any(
            not torch.allclose(
                value,
                expected_parameters.reshape(-1),
                atol=1e-6,
                rtol=0,
            )
            for value in gathered
        ):
            raise AssertionError(
                "parameters differ across ranks: "
                f"{report['gathered_parameter_vectors']}"
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
