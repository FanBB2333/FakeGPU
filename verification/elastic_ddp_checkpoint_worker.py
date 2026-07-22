#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

from elastic_ddp_worker import (
    EXPECTED_PROCESS_EXIT_CODE,
    _make_tracing_store,
    _safe_component,
    _sync_cuda,
    _write_report,
)


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


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    node_name: str,
    local_rank: int,
) -> Path:
    return checkpoint_dir / (
        f"checkpoint_{_safe_component(node_name)}_local_{local_rank}.pt"
    )


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(torch: Any, path: Path, device: Any) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    return checkpoint


def _barrier(dist: Any, backend: str) -> None:
    if backend == "nccl":
        dist.barrier(device_ids=[0])
    else:
        dist.barrier()


def _momentum_buffer(optimizer: Any, parameter: Any) -> Any:
    state = optimizer.state.get(parameter)
    if not isinstance(state, dict):
        raise AssertionError("optimizer state is missing for model.weight")
    momentum = state.get("momentum_buffer")
    if momentum is None:
        raise AssertionError("optimizer momentum_buffer is missing")
    return momentum


def _assert_tensor_close(
    torch: Any,
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0):
        raise AssertionError(
            f"{label} mismatch: {actual.detach().cpu().tolist()} != "
            f"{expected.detach().cpu().tolist()}"
        )


def _gather_state(
    torch: Any,
    dist: Any,
    *,
    parameters: Any,
    momentum: Any,
    world_size: int,
    backend: str,
) -> list[list[float]]:
    state = torch.cat((parameters.reshape(-1), momentum.reshape(-1)))
    gathered = [torch.empty_like(state) for _ in range(world_size)]
    dist.all_gather(gathered, state)
    _sync_cuda(torch, backend)
    return [item.cpu().tolist() for item in gathered]


def _run_step(
    torch: Any,
    ddp_model: Any,
    model: Any,
    optimizer: Any,
    *,
    rank: int,
    world_size: int,
    backend: str,
) -> dict[str, Any]:
    scale = float(rank + 1)
    inputs = torch.tensor(
        [[scale, 2.0 * scale]],
        dtype=torch.float32,
        device=model.weight.device,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = ddp_model(inputs).sum()
    loss.backward()
    _sync_cuda(torch, backend)

    average_scale = (world_size + 1) / 2.0
    expected_gradient = torch.tensor(
        [[average_scale, 2.0 * average_scale]],
        dtype=torch.float32,
        device=model.weight.device,
    )
    gradient = model.weight.grad
    if gradient is None:
        raise AssertionError("DDP did not produce a gradient")
    _assert_tensor_close(
        torch,
        gradient,
        expected_gradient,
        label="averaged gradient",
    )
    optimizer.step()
    _sync_cuda(torch, backend)
    momentum = _momentum_buffer(optimizer, model.weight)
    return {
        "gradient": gradient.detach().cpu().tolist(),
        "parameters": model.weight.detach().cpu().tolist(),
        "momentum": momentum.detach().cpu().tolist(),
        "loss": float(loss.detach().cpu().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint one DDP step, terminate a worker, and validate "
            "model/optimizer recovery after torchrun restarts the group."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-this-node", action="store_true")
    parser.add_argument("--trace-store", action="store_true")
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
    local_restart_count = int(
        os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    )
    restart_count = local_restart_count
    max_restarts = int(os.environ.get("TORCHELASTIC_MAX_RESTARTS", "0"))
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    report_path = _report_path(
        args.report_dir,
        restart_count=restart_count,
        node_name=args.node_name,
        local_rank=local_rank,
    )
    checkpoint_path = _checkpoint_path(
        args.checkpoint_dir,
        node_name=args.node_name,
        local_rank=local_rank,
    )
    report: dict[str, Any] = {
        "schema_version": "fakegpu.elastic_ddp_checkpoint.rank.v1",
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
        "local_restart_count": local_restart_count,
        "max_restarts": max_restarts,
        "run_id": run_id,
        "selected_for_initial_exit": False,
        "initial_process_group_destroyed_before_exit": False,
        "checkpoint_path": str(checkpoint_path),
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
                raise RuntimeError(
                    "NCCL elastic checkpoint validation requires real CUDA"
                )
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            report["physical_device_name"] = torch.cuda.get_device_name(0)
            report["physical_compute_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
        else:
            device = torch.device("cpu")
        report["device"] = str(device)

        report["stage"] = "synchronize_restart_generation"
        rendezvous_store = dist.TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            world_size,
            False,
            timedelta(seconds=args.timeout_seconds),
        )
        control_store = dist.PrefixStore(
            "fakegpu.elastic_ddp_checkpoint.control."
            f"{_safe_component(run_id)}",
            rendezvous_store,
        )
        restart_arrival_count = int(
            control_store.add(f"restart_arrival/{rank}", 1)
        )
        restart_count = restart_arrival_count - 1
        control_store.set(
            f"local_restart_count/{restart_count}/{rank}",
            str(local_restart_count).encode("ascii"),
        )
        observed_local_restart_counts = [
            int(
                control_store.get(
                    f"local_restart_count/{restart_count}/{peer_rank}"
                )
            )
            for peer_rank in range(world_size)
        ]
        should_exit = restart_count == 0 and (
            args.fail_this_node or rank == args.fail_rank
        )
        report.update(
            {
                "restart_count": restart_count,
                "restart_arrival_count": restart_arrival_count,
                "observed_local_restart_counts": observed_local_restart_counts,
                "selected_for_initial_exit": should_exit,
            }
        )
        report_path = _report_path(
            args.report_dir,
            restart_count=restart_count,
            node_name=args.node_name,
            local_rank=local_rank,
        )

        report["stage"] = "init_process_group"
        store_prefix = (
            f"fakegpu.elastic_ddp_checkpoint.attempt.{restart_count}"
        )
        report["process_group_store_prefix"] = store_prefix
        process_group_store = dist.PrefixStore(
            store_prefix,
            rendezvous_store,
        )
        if args.trace_store:
            store_events: list[dict[str, object]] = []
            report["store_events"] = store_events
            process_group_store = _make_tracing_store(
                dist,
                process_group_store,
                store_events,
            )
        dist.init_process_group(
            backend=args.backend,
            store=process_group_store,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=args.timeout_seconds),
        )

        report["stage"] = "construct_model"
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0]], dtype=torch.float32, device=device
                )
            )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9
        )
        average_scale = (world_size + 1) / 2.0
        expected_gradient = torch.tensor(
            [[average_scale, 2.0 * average_scale]],
            dtype=torch.float32,
            device=device,
        )
        expected_step_one_momentum = expected_gradient
        expected_step_one_parameters = torch.tensor(
            [[1.0 - 0.1 * average_scale, -0.2 * average_scale]],
            dtype=torch.float32,
            device=device,
        )

        if restart_count == 0:
            report["stage"] = "initial_ddp_step"
            if args.backend == "nccl":
                ddp_model = DistributedDataParallel(
                    model, device_ids=[0], output_device=0
                )
            else:
                ddp_model = DistributedDataParallel(model)
            step = _run_step(
                torch,
                ddp_model,
                model,
                optimizer,
                rank=rank,
                world_size=world_size,
                backend=args.backend,
            )
            momentum = _momentum_buffer(optimizer, model.weight)
            _assert_tensor_close(
                torch,
                model.weight,
                expected_step_one_parameters,
                label="step-one parameters",
            )
            _assert_tensor_close(
                torch,
                momentum,
                expected_step_one_momentum,
                label="step-one momentum",
            )
            report.update(
                {
                    "completed_steps": 1,
                    "gradient_after_step_one": step["gradient"],
                    "parameters_after_step_one": step["parameters"],
                    "momentum_after_step_one": step["momentum"],
                    "local_loss_step_one": step["loss"],
                    "gathered_state_after_step_one": _gather_state(
                        torch,
                        dist,
                        parameters=model.weight.detach(),
                        momentum=momentum.detach(),
                        world_size=world_size,
                        backend=args.backend,
                    ),
                }
            )

            report["stage"] = "save_checkpoint"
            checkpoint = {
                "schema_version": "fakegpu.elastic_ddp_checkpoint.v1",
                "completed_steps": 1,
                "world_size": world_size,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            _atomic_torch_save(torch, checkpoint, checkpoint_path)
            report["saved_checkpoint"] = _checkpoint_metadata(
                checkpoint_path
            )

            report["stage"] = "initial_exit_barrier"
            report["status"] = (
                "expected_process_exit"
                if should_exit
                else "waiting_for_restart"
            )
            report["expected_process_exit_code"] = (
                EXPECTED_PROCESS_EXIT_CODE if should_exit else None
            )
            _write_report(report_path, report)
            _barrier(dist, args.backend)
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

        report["stage"] = "load_checkpoint"
        checkpoint = _torch_load(torch, checkpoint_path, device)
        if checkpoint.get("schema_version") != (
            "fakegpu.elastic_ddp_checkpoint.v1"
        ):
            raise AssertionError("unexpected checkpoint schema")
        if int(checkpoint.get("completed_steps", -1)) != 1:
            raise AssertionError("checkpoint completed_steps is not 1")
        if int(checkpoint.get("world_size", -1)) != world_size:
            raise AssertionError("checkpoint world_size does not match restart")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        momentum = _momentum_buffer(optimizer, model.weight)
        _assert_tensor_close(
            torch,
            model.weight,
            expected_step_one_parameters,
            label="restored parameters",
        )
        _assert_tensor_close(
            torch,
            momentum,
            expected_step_one_momentum,
            label="restored momentum",
        )
        report.update(
            {
                "checkpoint_loaded": True,
                "loaded_checkpoint": _checkpoint_metadata(checkpoint_path),
                "restored_completed_steps": int(
                    checkpoint["completed_steps"]
                ),
                "restored_parameters": model.weight.detach().cpu().tolist(),
                "restored_momentum": momentum.detach().cpu().tolist(),
                "gathered_restored_state": _gather_state(
                    torch,
                    dist,
                    parameters=model.weight.detach(),
                    momentum=momentum.detach(),
                    world_size=world_size,
                    backend=args.backend,
                ),
            }
        )

        report["stage"] = "resumed_ddp_step"
        if args.backend == "nccl":
            ddp_model = DistributedDataParallel(
                model, device_ids=[0], output_device=0
            )
        else:
            ddp_model = DistributedDataParallel(model)
        step = _run_step(
            torch,
            ddp_model,
            model,
            optimizer,
            rank=rank,
            world_size=world_size,
            backend=args.backend,
        )
        momentum = _momentum_buffer(optimizer, model.weight)
        expected_final_momentum = (
            0.9 * expected_step_one_momentum + expected_gradient
        )
        expected_final_parameters = (
            expected_step_one_parameters - 0.1 * expected_final_momentum
        )
        _assert_tensor_close(
            torch,
            model.weight,
            expected_final_parameters,
            label="resumed parameters",
        )
        _assert_tensor_close(
            torch,
            momentum,
            expected_final_momentum,
            label="resumed momentum",
        )
        report.update(
            {
                "completed_steps": 2,
                "gradient_after_step_two": step["gradient"],
                "parameters_after_step_two": step["parameters"],
                "momentum_after_step_two": step["momentum"],
                "local_loss_step_two": step["loss"],
                "gathered_state_after_step_two": _gather_state(
                    torch,
                    dist,
                    parameters=model.weight.detach(),
                    momentum=momentum.detach(),
                    world_size=world_size,
                    backend=args.backend,
                ),
            }
        )

        report["stage"] = "final_barrier"
        _barrier(dist, args.backend)
        dist.destroy_process_group()
        dist = None
        report["status"] = "success"
        report["stage"] = "complete"
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
