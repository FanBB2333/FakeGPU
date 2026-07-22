#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any


EXPECTED_PROCESS_EXIT_CODE = 86


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "node"


def _report_path(
    report_dir: Path,
    *,
    restart_count: int,
    node_name: str,
    local_rank: int,
) -> Path:
    return report_dir / (
        f"attempt_{restart_count}_{_safe_component(node_name)}_"
        f"local_{local_rank}.json"
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sync_cuda(torch: Any, backend: str) -> None:
    if backend == "nccl":
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Terminate one initial DDP worker and validate torchrun restart."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-this-node", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--survivor-wait-seconds", type=int, default=30)
    args = parser.parse_args(argv)

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.survivor_wait_seconds <= 0:
        parser.error("--survivor-wait-seconds must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    max_restarts = int(os.environ.get("TORCHELASTIC_MAX_RESTARTS", "0"))
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    should_exit = restart_count == 0 and (
        args.fail_this_node or rank == args.fail_rank
    )
    report_path = _report_path(
        args.report_dir,
        restart_count=restart_count,
        node_name=args.node_name,
        local_rank=local_rank,
    )
    report: dict[str, Any] = {
        "schema_version": "fakegpu.elastic_ddp.rank.v1",
        "status": "starting",
        "stage": "import_torch",
        "node_name": args.node_name,
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "group_rank": group_rank,
        "world_size": world_size,
        "backend": args.backend,
        "restart_count": restart_count,
        "max_restarts": max_restarts,
        "run_id": run_id,
        "selected_for_initial_exit": should_exit,
        "initial_process_group_destroyed_before_exit": False,
    }
    dist = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        from torch.nn.parallel import DistributedDataParallel

        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_version"] = str(torch.version.cuda)
        if args.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL elastic validation requires real CUDA")
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            report["physical_device_name"] = torch.cuda.get_device_name(0)
            report["physical_compute_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
        else:
            device = torch.device("cpu")

        report["device"] = str(device)
        report["stage"] = "init_process_group"
        store_prefix = f"fakegpu.elastic_ddp.attempt.{restart_count}"
        report["process_group_store_prefix"] = store_prefix
        rendezvous_store = dist.TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            world_size,
            False,
            timedelta(seconds=args.timeout_seconds),
        )
        process_group_store = dist.PrefixStore(
            store_prefix,
            rendezvous_store,
        )
        dist.init_process_group(
            backend=args.backend,
            store=process_group_store,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=args.timeout_seconds),
        )

        if restart_count == 0:
            report["stage"] = "initial_collective"
            probe = torch.tensor(
                [float(rank + 1)],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(probe)
            _sync_cuda(torch, args.backend)
            expected_probe = float(world_size * (world_size + 1) // 2)
            report["initial_all_reduce_value"] = float(probe.cpu().item())
            report["expected_initial_all_reduce_value"] = expected_probe
            if abs(float(probe.cpu().item()) - expected_probe) > 1e-6:
                raise AssertionError(
                    "initial all-reduce mismatch: "
                    f"{float(probe.cpu().item())} != {expected_probe}"
                )

            report["stage"] = "initial_exit_barrier"
            report["status"] = (
                "expected_process_exit" if should_exit else "waiting_for_restart"
            )
            report["expected_process_exit_code"] = (
                EXPECTED_PROCESS_EXIT_CODE if should_exit else None
            )
            _write_report(report_path, report)
            if args.backend == "nccl":
                dist.barrier(device_ids=[0])
            else:
                dist.barrier()

            if should_exit:
                os._exit(EXPECTED_PROCESS_EXIT_CODE)

            report["stage"] = "waiting_for_elastic_agent"
            _write_report(report_path, report)
            deadline = time.monotonic() + args.survivor_wait_seconds
            while time.monotonic() < deadline:
                time.sleep(0.25)
            raise RuntimeError(
                "torchrun did not terminate the surviving initial worker"
            )

        if restart_count != 1:
            raise AssertionError(
                f"expected restart count 1, got {restart_count}"
            )

        report["stage"] = "construct_ddp"
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
            )
        if args.backend == "nccl":
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[0],
                output_device=0,
            )
        else:
            ddp_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        report["stage"] = "forward_backward"
        scale = float(rank + 1)
        inputs = torch.tensor(
            [[scale, 2.0 * scale]],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = ddp_model(inputs).sum()
        loss.backward()
        _sync_cuda(torch, args.backend)

        average_scale = (world_size + 1) / 2.0
        expected_gradient = torch.tensor(
            [[average_scale, 2.0 * average_scale]],
            dtype=torch.float32,
            device=device,
        )
        gradient = model.weight.grad
        if gradient is None:
            raise AssertionError("DDP did not produce a gradient")
        report["gradient"] = gradient.cpu().tolist()
        report["expected_gradient"] = expected_gradient.cpu().tolist()
        if not torch.allclose(gradient, expected_gradient, atol=1e-6, rtol=0):
            raise AssertionError(
                f"averaged gradient mismatch: {report['gradient']} != "
                f"{report['expected_gradient']}"
            )

        report["stage"] = "optimizer_step"
        optimizer.step()
        _sync_cuda(torch, args.backend)
        expected_parameters = torch.tensor(
            [[1.0 - 0.1 * average_scale, -0.2 * average_scale]],
            dtype=torch.float32,
            device=device,
        )
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
                "optimizer result mismatch: "
                f"{report['parameters_after_step']} != "
                f"{report['expected_parameters_after_step']}"
            )

        report["stage"] = "parameter_consistency"
        parameter_vector = parameters.reshape(-1)
        gathered = [
            torch.empty_like(parameter_vector) for _ in range(world_size)
        ]
        dist.all_gather(gathered, parameter_vector)
        _sync_cuda(torch, args.backend)
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
                "parameters differ across restarted ranks: "
                f"{report['gathered_parameter_vectors']}"
            )

        report["stage"] = "final_barrier"
        if args.backend == "nccl":
            dist.barrier(device_ids=[0])
        else:
            dist.barrier()
        dist.destroy_process_group()
        dist = None
        report["status"] = "success"
        report["stage"] = "complete"
        report["local_loss"] = float(loss.detach().cpu().item())
        _write_report(report_path, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_report(report_path, report)
        if dist is not None:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
