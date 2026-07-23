#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

from elastic_ddp_checkpoint_worker import (
    _assert_tensor_close,
    _atomic_torch_save,
    _barrier,
    _checkpoint_metadata,
    _checkpoint_path,
    _report_path,
    _torch_load,
)
from elastic_ddp_training_state_validation import (
    EXPECTED_GATHERED_PENDING_GRADIENTS,
    EXPECTED_PENDING_GRADIENTS,
    EXPECTED_SAMPLE_IDS,
    EXPECTED_STEP_ONE_EXP_AVG,
    EXPECTED_STEP_ONE_EXP_AVG_SQ,
    EXPECTED_STEP_ONE_GRADIENT,
    EXPECTED_STEP_ONE_PARAMETERS,
    EXPECTED_STEP_TWO_EXP_AVG,
    EXPECTED_STEP_TWO_EXP_AVG_SQ,
    EXPECTED_STEP_TWO_GRADIENT,
    EXPECTED_STEP_TWO_PARAMETERS,
)
from elastic_ddp_worker import (
    EXPECTED_PROCESS_EXIT_CODE,
    _make_tracing_store,
    _safe_component,
    _sync_cuda,
    _write_report,
)


DATA_LOADER_SEED_BASE = 2026072300


class _WorkerRandomDataset:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        worker = torch.utils.data.get_worker_info()
        return {
            "sample_id": torch.tensor(index + 1, dtype=torch.int64),
            "worker_random": torch.rand((), dtype=torch.float32),
            "worker_id": torch.tensor(
                -1 if worker is None else worker.id,
                dtype=torch.int64,
            ),
        }


def _optimizer_state(optimizer: Any, parameter: Any) -> tuple[Any, Any, int]:
    state = optimizer.state.get(parameter)
    if not isinstance(state, dict):
        raise AssertionError("AdamW state is missing for model.weight")
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    step = state.get("step")
    if exp_avg is None or exp_avg_sq is None or step is None:
        raise AssertionError("AdamW moment or step state is missing")
    if hasattr(step, "item"):
        step_value = int(float(step.item()))
    else:
        step_value = int(step)
    return exp_avg, exp_avg_sq, step_value


def _gather_vectors(
    torch: Any,
    dist: Any,
    vector: Any,
    *,
    world_size: int,
    backend: str,
) -> list[list[float]]:
    flattened = vector.reshape(-1)
    gathered = [torch.empty_like(flattened) for _ in range(world_size)]
    dist.all_gather(gathered, flattened)
    _sync_cuda(torch, backend)
    return [item.cpu().tolist() for item in gathered]


def _gather_optimizer_state(
    torch: Any,
    dist: Any,
    model: Any,
    optimizer: Any,
    *,
    world_size: int,
    backend: str,
) -> list[list[float]]:
    exp_avg, exp_avg_sq, _ = _optimizer_state(optimizer, model.weight)
    vector = torch.cat(
        (
            model.weight.detach().reshape(-1),
            exp_avg.detach().reshape(-1),
            exp_avg_sq.detach().reshape(-1),
        )
    )
    return _gather_vectors(
        torch,
        dist,
        vector,
        world_size=world_size,
        backend=backend,
    )


def _backward_micro_step(
    torch: Any,
    ddp_model: Any,
    *,
    scale: float,
    synchronize: bool,
    backend: str,
) -> float:
    context = nullcontext() if synchronize else ddp_model.no_sync()
    with context:
        inputs = torch.tensor(
            [[scale, 2.0 * scale]],
            dtype=torch.float32,
            device=next(ddp_model.parameters()).device,
        )
        loss = ddp_model(inputs).sum() / 2.0
        loss.backward()
    _sync_cuda(torch, backend)
    return float(loss.detach().cpu().item())


def _tensor(
    torch: Any,
    values: list[list[float]],
    *,
    device: Any,
) -> Any:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _distributed_data_iterator(
    torch: Any,
    *,
    logical_rank: int,
    world_size: int,
    epoch: int,
    consumed_samples: int,
    loader_seed: int,
    num_workers: int,
    prefetch_factor: int,
) -> tuple[Any, list[dict[str, Any]]]:
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dataset = _WorkerRandomDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=logical_rank,
        shuffle=False,
        drop_last=False,
    )
    sampler.set_epoch(epoch)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(loader_seed)
    loader_options: dict[str, Any] = {
        "batch_size": 1,
        "sampler": sampler,
        "num_workers": num_workers,
        "generator": loader_generator,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options.update(
            {
                "prefetch_factor": prefetch_factor,
                "multiprocessing_context": "spawn",
            }
        )
    else:
        torch.manual_seed(loader_seed)
    iterator = iter(
        DataLoader(dataset, **loader_options)
    )
    skipped: list[dict[str, Any]] = []
    for _ in range(consumed_samples):
        skipped.append(_next_data_sample(iterator))
    return iterator, skipped


def _next_data_sample(iterator: Any) -> dict[str, Any]:
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise AssertionError("distributed sampler exhausted unexpectedly") from exc
    if not isinstance(batch, dict) or set(batch) != {
        "sample_id",
        "worker_random",
        "worker_id",
    }:
        raise AssertionError(f"unexpected DataLoader batch: {batch!r}")
    return {
        "sample_id": int(batch["sample_id"].reshape(-1)[0].item()),
        "worker_random": float(
            batch["worker_random"].reshape(-1)[0].item()
        ),
        "worker_id": int(batch["worker_id"].reshape(-1)[0].item()),
    }


def _data_worker_pids(iterator: Any) -> list[int]:
    workers = getattr(iterator, "_workers", None)
    if not isinstance(workers, (list, tuple)):
        return []
    return [int(worker.pid) for worker in workers if worker.pid is not None]


def _shutdown_data_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _data_samples_equal(
    actual: object,
    expected: object,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    if len(actual) != len(expected):
        return False
    for actual_item, expected_item in zip(actual, expected):
        if not isinstance(actual_item, dict) or not isinstance(
            expected_item, dict
        ):
            return False
        if int(actual_item.get("sample_id", -1)) != int(
            expected_item.get("sample_id", -1)
        ):
            return False
        if int(actual_item.get("worker_id", -2)) != int(
            expected_item.get("worker_id", -3)
        ):
            return False
        if (
            abs(
                float(actual_item.get("worker_random", -1.0))
                - float(expected_item.get("worker_random", -2.0))
            )
            > tolerance
        ):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint AdamW, StepLR, RNG, and a partially accumulated DDP "
            "gradient, then validate continued training after torchrun restart."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-this-node", action="store_true")
    parser.add_argument("--trace-store", action="store_true")
    parser.add_argument(
        "--resume-rank-shift",
        type=int,
        default=0,
        help=(
            "Load rank-local state from (current rank + shift) modulo world "
            "size; validation uses 1 to exercise rank remapping."
        ),
    )
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--survivor-wait-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.survivor_wait_seconds <= 0:
        parser.error("--survivor-wait-seconds must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not 0 <= args.resume_rank_shift < world_size:
        parser.error("--resume-rank-shift must be in [0, world size)")
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
        "schema_version": "fakegpu.elastic_ddp_training_state.rank.v3",
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
        "optimizer": "adamw",
        "scheduler": "step_lr",
        "gradient_accumulation_steps": 2,
        "checkpoint_replication_mode": "all_rank_states_per_host",
        "resume_rank_shift": args.resume_rank_shift,
        "dataloader_workers": args.dataloader_workers,
        "dataloader_prefetch_factor": (
            args.prefetch_factor if args.dataloader_workers > 0 else None
        ),
        "dataloader_persistent_workers": args.dataloader_workers > 0,
        "dataloader_start_method": (
            "spawn" if args.dataloader_workers > 0 else None
        ),
    }
    dist = None
    data_iterator = None

    try:
        import torch
        import torch.distributed as dist  # type: ignore[no-redef]
        from torch.nn.parallel import DistributedDataParallel

        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_version"] = str(torch.version.cuda)
        if args.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "NCCL elastic training-state validation requires real CUDA"
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
            "fakegpu.elastic_ddp_training_state.control."
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
            f"fakegpu.elastic_ddp_training_state.attempt.{restart_count}"
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

        report["stage"] = "construct_training_state"
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0]], dtype=torch.float32, device=device
                )
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.1,
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(20260723 + rank)

        if restart_count == 0:
            if args.backend == "nccl":
                ddp_model = DistributedDataParallel(
                    model, device_ids=[0], output_device=0
                )
            else:
                ddp_model = DistributedDataParallel(model)

            loader_seed = DATA_LOADER_SEED_BASE + rank
            data_iterator, skipped_data_samples = _distributed_data_iterator(
                torch,
                logical_rank=rank,
                world_size=world_size,
                epoch=0,
                consumed_samples=0,
                loader_seed=loader_seed,
                num_workers=args.dataloader_workers,
                prefetch_factor=args.prefetch_factor,
            )
            if skipped_data_samples:
                raise AssertionError("fresh sampler unexpectedly skipped samples")
            committed_data_samples: list[dict[str, Any]] = []

            report["stage"] = "optimizer_step_one"
            optimizer.zero_grad(set_to_none=True)
            sample_one_a = _next_data_sample(data_iterator)
            committed_data_samples.append(sample_one_a)
            loss_one_a = _backward_micro_step(
                torch,
                ddp_model,
                scale=float(sample_one_a["sample_id"]),
                synchronize=False,
                backend=args.backend,
            )
            sample_one_b = _next_data_sample(data_iterator)
            committed_data_samples.append(sample_one_b)
            loss_one_b = _backward_micro_step(
                torch,
                ddp_model,
                scale=float(sample_one_b["sample_id"]),
                synchronize=True,
                backend=args.backend,
            )
            gradient = model.weight.grad
            if gradient is None:
                raise AssertionError("step-one accumulated gradient is missing")
            _assert_tensor_close(
                torch,
                gradient,
                _tensor(
                    torch, EXPECTED_STEP_ONE_GRADIENT, device=device
                ),
                label="step-one accumulated gradient",
            )
            optimizer.step()
            scheduler.step()
            _sync_cuda(torch, args.backend)
            exp_avg, exp_avg_sq, optimizer_step = _optimizer_state(
                optimizer, model.weight
            )
            for actual, expected, label in (
                (
                    model.weight,
                    EXPECTED_STEP_ONE_PARAMETERS,
                    "step-one parameters",
                ),
                (
                    exp_avg,
                    EXPECTED_STEP_ONE_EXP_AVG,
                    "step-one exp_avg",
                ),
                (
                    exp_avg_sq,
                    EXPECTED_STEP_ONE_EXP_AVG_SQ,
                    "step-one exp_avg_sq",
                ),
            ):
                _assert_tensor_close(
                    torch,
                    actual,
                    _tensor(torch, expected, device=device),
                    label=label,
                )
            if optimizer_step != 1:
                raise AssertionError(
                    f"expected AdamW step 1, got {optimizer_step}"
                )
            if abs(float(optimizer.param_groups[0]["lr"]) - 0.005) > 1e-12:
                raise AssertionError("StepLR did not produce learning rate 0.005")

            report["stage"] = "partial_accumulation_step_two"
            optimizer.zero_grad(set_to_none=True)
            sample_two_a = _next_data_sample(data_iterator)
            committed_data_samples.append(sample_two_a)
            loss_two_a = _backward_micro_step(
                torch,
                ddp_model,
                scale=float(sample_two_a["sample_id"]),
                synchronize=False,
                backend=args.backend,
            )
            pending_gradient = model.weight.grad
            if pending_gradient is None:
                raise AssertionError("pending accumulated gradient is missing")
            _assert_tensor_close(
                torch,
                pending_gradient,
                _tensor(
                    torch,
                    EXPECTED_PENDING_GRADIENTS[rank],
                    device=device,
                ),
                label="pending accumulated gradient",
            )
            random_before_checkpoint = float(
                torch.rand((), generator=rng).item()
            )
            rng_state = rng.get_state()
            probe_rng = torch.Generator(device="cpu")
            probe_rng.set_state(rng_state)
            expected_next_random = float(
                torch.rand((), generator=probe_rng).item()
            )
            committed_sample_ids = [
                int(item["sample_id"]) for item in committed_data_samples
            ]
            if committed_sample_ids != EXPECTED_SAMPLE_IDS[rank][:3]:
                raise AssertionError(
                    "initial DistributedSampler sequence is invalid: "
                    f"{committed_sample_ids}"
                )
            staged_prefetched_sample = _next_data_sample(data_iterator)
            if int(staged_prefetched_sample["sample_id"]) != (
                EXPECTED_SAMPLE_IDS[rank][3]
            ):
                raise AssertionError(
                    "staged prefetched batch has an invalid sample ID"
                )
            dataloader_worker_pids = _data_worker_pids(data_iterator)
            if args.dataloader_workers > 0 and len(dataloader_worker_pids) != (
                args.dataloader_workers
            ):
                raise AssertionError(
                    "DataLoader did not start the configured worker count"
                )
            gathered_pending = _gather_vectors(
                torch,
                dist,
                pending_gradient.detach(),
                world_size=world_size,
                backend=args.backend,
            )
            expected_gathered_pending = _tensor(
                torch,
                EXPECTED_GATHERED_PENDING_GRADIENTS,
                device=device,
            )
            gathered_pending_tensor = torch.tensor(
                gathered_pending,
                dtype=torch.float32,
                device=device,
            )
            _assert_tensor_close(
                torch,
                gathered_pending_tensor,
                expected_gathered_pending,
                label="cross-rank pending gradients",
            )
            gathered_state = _gather_optimizer_state(
                torch,
                dist,
                model,
                optimizer,
                world_size=world_size,
                backend=args.backend,
            )
            local_rank_state = {
                "saved_rank": rank,
                "gradients": {
                    "weight": pending_gradient.detach().cpu().clone()
                },
                "rng_state": rng_state.cpu().clone(),
                "expected_next_random": expected_next_random,
                "data_state": {
                    "sampler": "DistributedSampler",
                    "sampler_epoch": 0,
                    "samples_consumed": len(committed_data_samples),
                    "consumed_sample_ids": committed_sample_ids,
                    "committed_data_samples": committed_data_samples,
                    "next_sample_id": EXPECTED_SAMPLE_IDS[rank][3],
                    "staged_prefetched_sample": staged_prefetched_sample,
                    "loader_batches_yielded": 4,
                    "loader_seed": loader_seed,
                    "dataloader_workers": args.dataloader_workers,
                    "prefetch_factor": (
                        args.prefetch_factor
                        if args.dataloader_workers > 0
                        else None
                    ),
                    "persistent_workers": args.dataloader_workers > 0,
                    "multiprocessing_context": (
                        "spawn" if args.dataloader_workers > 0 else None
                    ),
                },
            }
            gathered_rank_states: list[Any] = [None] * world_size
            dist.all_gather_object(gathered_rank_states, local_rank_state)
            _sync_cuda(torch, args.backend)
            rank_state_bundle: dict[str, dict[str, Any]] = {}
            for state in gathered_rank_states:
                if not isinstance(state, dict):
                    raise AssertionError("gathered rank state is not a dictionary")
                saved_rank = int(state.get("saved_rank", -1))
                if saved_rank in {int(key) for key in rank_state_bundle}:
                    raise AssertionError("rank state bundle contains duplicates")
                rank_state_bundle[str(saved_rank)] = state
            replicated_rank_state_ids = sorted(
                int(key) for key in rank_state_bundle
            )
            if replicated_rank_state_ids != list(range(world_size)):
                raise AssertionError(
                    "rank state bundle is incomplete: "
                    f"{replicated_rank_state_ids}"
                )
            report.update(
                {
                    "completed_optimizer_steps": 1,
                    "accumulation_micro_step": 1,
                    "gradient_after_step_one": gradient.detach().cpu().tolist(),
                    "parameters_after_step_one": model.weight.detach().cpu().tolist(),
                    "exp_avg_after_step_one": exp_avg.detach().cpu().tolist(),
                    "exp_avg_sq_after_step_one": exp_avg_sq.detach().cpu().tolist(),
                    "optimizer_state_step": optimizer_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "scheduler_last_epoch": int(scheduler.last_epoch),
                    "pending_gradient": pending_gradient.detach().cpu().tolist(),
                    "gathered_pending_gradients": gathered_pending,
                    "gathered_state_after_step_one": gathered_state,
                    "rng_seed": 20260723 + rank,
                    "random_before_checkpoint": random_before_checkpoint,
                    "expected_next_random": expected_next_random,
                    "sampler": "DistributedSampler",
                    "sampler_epoch": 0,
                    "samples_consumed": len(committed_data_samples),
                    "consumed_sample_ids": committed_sample_ids,
                    "committed_data_samples": committed_data_samples,
                    "next_sample_id": EXPECTED_SAMPLE_IDS[rank][3],
                    "staged_prefetched_sample": staged_prefetched_sample,
                    "loader_batches_yielded": 4,
                    "loader_seed": loader_seed,
                    "dataloader_worker_pids": dataloader_worker_pids,
                    "replicated_rank_state_ids": replicated_rank_state_ids,
                    "rank_state_replica_count": len(rank_state_bundle),
                    "local_losses_step_one": [loss_one_a, loss_one_b],
                    "local_loss_partial_step_two": loss_two_a,
                }
            )

            report["stage"] = "save_training_state_checkpoint"
            checkpoint = {
                "schema_version": "fakegpu.elastic_ddp_training_state.v3",
                "storage_owner_rank": rank,
                "world_size": world_size,
                "completed_optimizer_steps": 1,
                "accumulation_micro_step": 1,
                "gradient_accumulation_steps": 2,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "checkpoint_replication_mode": "all_rank_states_per_host",
                "rank_states": rank_state_bundle,
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

        report["stage"] = "load_training_state_checkpoint"
        checkpoint = _torch_load(torch, checkpoint_path, device)
        if checkpoint.get("schema_version") != (
            "fakegpu.elastic_ddp_training_state.v3"
        ):
            raise AssertionError("unexpected training-state checkpoint schema")
        if int(checkpoint.get("world_size", -1)) != world_size:
            raise AssertionError("checkpoint world size does not match restart")
        if int(checkpoint.get("completed_optimizer_steps", -1)) != 1:
            raise AssertionError("checkpoint optimizer step is not 1")
        if int(checkpoint.get("accumulation_micro_step", -1)) != 1:
            raise AssertionError("checkpoint accumulation micro-step is not 1")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        rank_states = checkpoint.get("rank_states")
        if not isinstance(rank_states, dict):
            raise AssertionError("checkpoint rank-state bundle is missing")
        replicated_rank_state_ids = sorted(int(key) for key in rank_states)
        if replicated_rank_state_ids != list(range(world_size)):
            raise AssertionError(
                "checkpoint rank-state bundle is incomplete: "
                f"{replicated_rank_state_ids}"
            )
        resume_source_rank = (rank + args.resume_rank_shift) % world_size
        rank_state = rank_states.get(str(resume_source_rank))
        if not isinstance(rank_state, dict):
            raise AssertionError(
                f"checkpoint state for rank {resume_source_rank} is missing"
            )
        if int(rank_state.get("saved_rank", -1)) != resume_source_rank:
            raise AssertionError("checkpoint rank-state identity is invalid")
        if args.backend == "nccl":
            ddp_model = DistributedDataParallel(
                model, device_ids=[0], output_device=0
            )
        else:
            ddp_model = DistributedDataParallel(model)
        restored_gradient = rank_state["gradients"]["weight"].to(device)
        model.weight.grad = restored_gradient.detach().clone()
        rng.set_state(rank_state["rng_state"].cpu())
        restored_next_random = float(torch.rand((), generator=rng).item())
        expected_next_random = float(rank_state["expected_next_random"])
        if abs(restored_next_random - expected_next_random) > 1e-12:
            raise AssertionError(
                "restored RNG state did not reproduce the next value"
            )
        exp_avg, exp_avg_sq, optimizer_step = _optimizer_state(
            optimizer, model.weight
        )
        for actual, expected, label in (
            (
                model.weight,
                EXPECTED_STEP_ONE_PARAMETERS,
                "restored parameters",
            ),
            (exp_avg, EXPECTED_STEP_ONE_EXP_AVG, "restored exp_avg"),
            (
                exp_avg_sq,
                EXPECTED_STEP_ONE_EXP_AVG_SQ,
                "restored exp_avg_sq",
            ),
            (
                model.weight.grad,
                EXPECTED_PENDING_GRADIENTS[resume_source_rank],
                "restored pending gradient",
            ),
        ):
            _assert_tensor_close(
                torch,
                actual,
                _tensor(torch, expected, device=device),
                label=label,
            )
        if optimizer_step != 1:
            raise AssertionError("restored AdamW step is not 1")
        if abs(float(optimizer.param_groups[0]["lr"]) - 0.005) > 1e-12:
            raise AssertionError("restored learning rate is not 0.005")
        gathered_restored_pending = _gather_vectors(
            torch,
            dist,
            model.weight.grad.detach(),
            world_size=world_size,
            backend=args.backend,
        )
        gathered_restored_state = _gather_optimizer_state(
            torch,
            dist,
            model,
            optimizer,
            world_size=world_size,
            backend=args.backend,
        )
        data_state = rank_state.get("data_state")
        if not isinstance(data_state, dict):
            raise AssertionError("checkpoint sampler state is missing")
        restored_sampler_epoch = int(data_state.get("sampler_epoch", -1))
        restored_samples_consumed = int(
            data_state.get("samples_consumed", -1)
        )
        restored_loader_seed = int(data_state.get("loader_seed", -1))
        restored_dataloader_workers = int(
            data_state.get("dataloader_workers", -1)
        )
        restored_prefetch_factor = data_state.get("prefetch_factor")
        if restored_dataloader_workers != args.dataloader_workers:
            raise AssertionError(
                "checkpoint DataLoader worker count differs from restart"
            )
        if restored_dataloader_workers > 0 and int(
            restored_prefetch_factor
        ) != args.prefetch_factor:
            raise AssertionError(
                "checkpoint prefetch factor differs from restart"
            )
        data_iterator, skipped_data_samples = _distributed_data_iterator(
            torch,
            logical_rank=resume_source_rank,
            world_size=world_size,
            epoch=restored_sampler_epoch,
            consumed_samples=restored_samples_consumed,
            loader_seed=restored_loader_seed,
            num_workers=restored_dataloader_workers,
            prefetch_factor=args.prefetch_factor,
        )
        expected_committed_data_samples = data_state.get(
            "committed_data_samples"
        )
        if not _data_samples_equal(
            skipped_data_samples,
            expected_committed_data_samples,
        ):
            raise AssertionError(
                "restored DataLoader did not reproduce committed batches"
            )
        replayed_prefetched_sample = _next_data_sample(data_iterator)
        if not _data_samples_equal(
            [replayed_prefetched_sample],
            [data_state.get("staged_prefetched_sample")],
        ):
            raise AssertionError(
                "restored DataLoader did not replay the staged prefetched batch"
            )
        resumed_sample_id = int(replayed_prefetched_sample["sample_id"])
        if resumed_sample_id != int(data_state.get("next_sample_id", -1)):
            raise AssertionError(
                "restored sampler did not produce the expected next sample"
            )
        restored_dataloader_worker_pids = _data_worker_pids(data_iterator)
        if restored_dataloader_workers > 0 and len(
            restored_dataloader_worker_pids
        ) != restored_dataloader_workers:
            raise AssertionError(
                "restored DataLoader did not start the configured workers"
            )
        report.update(
            {
                "checkpoint_loaded": True,
                "loaded_checkpoint": _checkpoint_metadata(checkpoint_path),
                "checkpoint_storage_owner_rank": int(
                    checkpoint.get("storage_owner_rank", -1)
                ),
                "replicated_rank_state_ids": replicated_rank_state_ids,
                "rank_state_replica_count": len(rank_states),
                "resume_source_rank": resume_source_rank,
                "rank_remapped": resume_source_rank != rank,
                "restored_optimizer_steps": int(
                    checkpoint["completed_optimizer_steps"]
                ),
                "restored_accumulation_micro_step": int(
                    checkpoint["accumulation_micro_step"]
                ),
                "restored_parameters": model.weight.detach().cpu().tolist(),
                "restored_exp_avg": exp_avg.detach().cpu().tolist(),
                "restored_exp_avg_sq": exp_avg_sq.detach().cpu().tolist(),
                "restored_pending_gradient": model.weight.grad.detach().cpu().tolist(),
                "gathered_restored_pending_gradients": gathered_restored_pending,
                "gathered_restored_state": gathered_restored_state,
                "restored_next_random": restored_next_random,
                "restored_sampler": data_state.get("sampler"),
                "restored_sampler_epoch": restored_sampler_epoch,
                "restored_samples_consumed": restored_samples_consumed,
                "restored_consumed_sample_ids": [
                    int(item["sample_id"]) for item in skipped_data_samples
                ],
                "restored_committed_data_samples": skipped_data_samples,
                "restored_loader_seed": restored_loader_seed,
                "restored_dataloader_workers": restored_dataloader_workers,
                "restored_prefetch_factor": restored_prefetch_factor,
                "restored_persistent_workers": data_state.get(
                    "persistent_workers"
                ),
                "restored_multiprocessing_context": data_state.get(
                    "multiprocessing_context"
                ),
                "replayed_prefetched_sample": replayed_prefetched_sample,
                "dataloader_worker_pids": restored_dataloader_worker_pids,
                "resumed_sample_id": resumed_sample_id,
            }
        )

        report["stage"] = "finish_accumulated_step_two"
        loss_two_b = _backward_micro_step(
            torch,
            ddp_model,
            scale=float(resumed_sample_id),
            synchronize=True,
            backend=args.backend,
        )
        final_gradient = model.weight.grad
        if final_gradient is None:
            raise AssertionError("resumed accumulated gradient is missing")
        _assert_tensor_close(
            torch,
            final_gradient,
            _tensor(torch, EXPECTED_STEP_TWO_GRADIENT, device=device),
            label="resumed accumulated gradient",
        )
        optimizer.step()
        scheduler.step()
        _sync_cuda(torch, args.backend)
        exp_avg, exp_avg_sq, optimizer_step = _optimizer_state(
            optimizer, model.weight
        )
        for actual, expected, label in (
            (
                model.weight,
                EXPECTED_STEP_TWO_PARAMETERS,
                "step-two parameters",
            ),
            (exp_avg, EXPECTED_STEP_TWO_EXP_AVG, "step-two exp_avg"),
            (
                exp_avg_sq,
                EXPECTED_STEP_TWO_EXP_AVG_SQ,
                "step-two exp_avg_sq",
            ),
        ):
            _assert_tensor_close(
                torch,
                actual,
                _tensor(torch, expected, device=device),
                label=label,
            )
        if optimizer_step != 2:
            raise AssertionError(f"expected AdamW step 2, got {optimizer_step}")
        if abs(float(optimizer.param_groups[0]["lr"]) - 0.0025) > 1e-12:
            raise AssertionError("StepLR did not produce learning rate 0.0025")
        gathered_final_state = _gather_optimizer_state(
            torch,
            dist,
            model,
            optimizer,
            world_size=world_size,
            backend=args.backend,
        )
        report.update(
            {
                "completed_optimizer_steps": 2,
                "accumulation_micro_step": 0,
                "gradient_after_step_two": final_gradient.detach().cpu().tolist(),
                "parameters_after_step_two": model.weight.detach().cpu().tolist(),
                "exp_avg_after_step_two": exp_avg.detach().cpu().tolist(),
                "exp_avg_sq_after_step_two": exp_avg_sq.detach().cpu().tolist(),
                "optimizer_state_step": optimizer_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "scheduler_last_epoch": int(scheduler.last_epoch),
                "local_loss_remaining_step_two": loss_two_b,
                "gathered_state_after_step_two": gathered_final_state,
            }
        )

        report["stage"] = "final_barrier"
        if data_iterator is not None:
            _shutdown_data_iterator(data_iterator)
            data_iterator = None
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
        if data_iterator is not None:
            try:
                _shutdown_data_iterator(data_iterator)
            except Exception:
                pass
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
