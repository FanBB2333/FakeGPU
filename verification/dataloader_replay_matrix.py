#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


SCHEMA_VERSION = "fakegpu.dataloader_replay_matrix.v1"
SCENARIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
RNG_FIELDS = ("torch_random", "python_random", "numpy_random")


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    world_size: int
    workers: int
    prefetch_factor: int
    epoch: int
    batch_size: int
    committed_batches: int
    staged_batches: int


DEFAULT_SCENARIOS = (
    ReplayScenario(
        name="single-worker-prefetch1",
        world_size=2,
        workers=1,
        prefetch_factor=1,
        epoch=0,
        batch_size=1,
        committed_batches=3,
        staged_batches=2,
    ),
    ReplayScenario(
        name="two-worker-epoch0",
        world_size=2,
        workers=2,
        prefetch_factor=2,
        epoch=0,
        batch_size=2,
        committed_batches=2,
        staged_batches=2,
    ),
    ReplayScenario(
        name="two-worker-epoch3",
        world_size=2,
        workers=2,
        prefetch_factor=2,
        epoch=3,
        batch_size=2,
        committed_batches=2,
        staged_batches=2,
    ),
    ReplayScenario(
        name="deep-prefetch-world4",
        world_size=4,
        workers=2,
        prefetch_factor=4,
        epoch=5,
        batch_size=1,
        committed_batches=5,
        staged_batches=3,
    ),
    ReplayScenario(
        name="four-worker-batched",
        world_size=2,
        workers=4,
        prefetch_factor=2,
        epoch=7,
        batch_size=3,
        committed_batches=2,
        staged_batches=2,
    ),
)


class _MultiRngDataset:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, object]:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            raise AssertionError("replay matrix requires a DataLoader worker")
        return {
            "sample_id": index + 1,
            "worker_id": worker.id,
            "worker_seed": worker.seed,
            "torch_random": torch.rand((), dtype=torch.float64),
            "python_random": random.random(),
            "numpy_random": float(np.random.random()),
        }


def parse_scenario(value: str) -> ReplayScenario:
    fields: dict[str, str] = {}
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        key, separator, raw_value = item.partition("=")
        if separator != "=" or not key.strip() or not raw_value.strip():
            raise argparse.ArgumentTypeError(
                "scenario fields must use key=value separated by semicolons"
            )
        normalized = key.strip().lower().replace("-", "_")
        if normalized in fields:
            raise argparse.ArgumentTypeError(
                f"duplicate scenario field: {normalized}"
            )
        fields[normalized] = raw_value.strip()

    required = {
        "name",
        "world_size",
        "workers",
        "prefetch_factor",
        "epoch",
        "batch_size",
        "committed_batches",
        "staged_batches",
    }
    unknown = sorted(set(fields) - required)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown scenario fields: {', '.join(unknown)}"
        )
    missing = sorted(required - set(fields))
    if missing:
        raise argparse.ArgumentTypeError(
            f"scenario is missing fields: {', '.join(missing)}"
        )
    name = fields["name"]
    if not SCENARIO_NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "scenario name may contain only letters, digits, underscore, dot, "
            "and dash"
        )
    try:
        values = {
            key: int(fields[key])
            for key in required
            if key != "name"
        }
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "scenario numeric fields must be integers"
        ) from exc
    for key in (
        "world_size",
        "workers",
        "prefetch_factor",
        "batch_size",
        "staged_batches",
    ):
        if values[key] <= 0:
            raise argparse.ArgumentTypeError(
                f"scenario {key.replace('_', '-')} must be positive"
            )
    for key in ("epoch", "committed_batches"):
        if values[key] < 0:
            raise argparse.ArgumentTypeError(
                f"scenario {key.replace('_', '-')} must be non-negative"
            )
    if values["committed_batches"] + values["staged_batches"] < values[
        "workers"
    ]:
        raise argparse.ArgumentTypeError(
            "scenario must retrieve at least one batch from every worker"
        )
    return ReplayScenario(name=name, **values)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _loader_seed(
    loader_seed_base: int,
    *,
    scenario_name: str,
    rank: int,
) -> int:
    name_digest = hashlib.sha256(scenario_name.encode("utf-8")).digest()
    scenario_offset = int.from_bytes(name_digest[:4], "big") * 16
    return loader_seed_base + scenario_offset + rank


def _batch_to_record(batch: object) -> dict[str, list[int | float]]:
    if not isinstance(batch, dict):
        raise AssertionError(f"unexpected DataLoader batch: {batch!r}")
    expected_keys = {
        "sample_id",
        "worker_id",
        "worker_seed",
        *RNG_FIELDS,
    }
    if set(batch) != expected_keys:
        raise AssertionError(f"unexpected DataLoader fields: {sorted(batch)}")

    def values(name: str) -> list[object]:
        value = batch[name]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, np.ndarray):
            return value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    record: dict[str, list[int | float]] = {
        "sample_ids": [int(item) for item in values("sample_id")],
        "worker_ids": [int(item) for item in values("worker_id")],
        "worker_seeds": [int(item) for item in values("worker_seed")],
    }
    for field in RNG_FIELDS:
        record[field] = [float(item) for item in values(field)]
    lengths = {len(items) for items in record.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise AssertionError(f"DataLoader batch lengths are invalid: {record}")
    return record


def _worker_pids(iterator: object) -> list[int]:
    workers = getattr(iterator, "_workers", None)
    if not isinstance(workers, (list, tuple)):
        return []
    return [int(worker.pid) for worker in workers if worker.pid is not None]


def _prefetch_state(iterator: object) -> dict[str, int]:
    state: dict[str, int] = {}
    for field in ("_tasks_outstanding", "_send_idx", "_rcvd_idx"):
        value = getattr(iterator, field, None)
        if isinstance(value, int):
            state[field.removeprefix("_")] = value
    return state


def _shutdown_iterator(iterator: object) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _make_iterator(
    scenario: ReplayScenario,
    *,
    dataset_size: int,
    rank: int,
    sampler_seed: int,
    loader_seed: int,
) -> tuple[object, list[int]]:
    dataset = _MultiRngDataset(dataset_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=scenario.world_size,
        rank=rank,
        shuffle=True,
        seed=sampler_seed,
        drop_last=False,
    )
    sampler.set_epoch(scenario.epoch)
    expected_sample_ids = [int(index) + 1 for index in sampler]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(loader_seed)
    loader = DataLoader(
        dataset,
        batch_size=scenario.batch_size,
        sampler=sampler,
        num_workers=scenario.workers,
        generator=generator,
        persistent_workers=True,
        prefetch_factor=scenario.prefetch_factor,
        multiprocessing_context="spawn",
    )
    return iter(loader), expected_sample_ids


def _collect_batches(iterator: object, count: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for _ in range(count):
        try:
            batch = next(iterator)  # type: ignore[arg-type]
        except StopIteration as exc:
            raise AssertionError("DataLoader exhausted before replay cursor") from exc
        batches.append(_batch_to_record(batch))
    return batches


def _drain_iterator(iterator: object) -> int:
    drained = 0
    while True:
        try:
            next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            return drained
        drained += 1


def _flatten_sample_ids(batches: list[dict[str, Any]]) -> list[int]:
    return [
        int(sample_id)
        for batch in batches
        for sample_id in batch["sample_ids"]
    ]


def _validate_batches(
    batches: list[dict[str, Any]],
    *,
    scenario: ReplayScenario,
) -> int:
    worker_seed_bases: set[int] = set()
    observed_workers: set[int] = set()
    for batch in batches:
        for worker_id, worker_seed in zip(
            batch["worker_ids"],
            batch["worker_seeds"],
        ):
            worker_id = int(worker_id)
            worker_seed = int(worker_seed)
            if not 0 <= worker_id < scenario.workers:
                raise AssertionError(f"invalid DataLoader worker ID: {worker_id}")
            observed_workers.add(worker_id)
            worker_seed_bases.add(worker_seed - worker_id)
        for field in RNG_FIELDS:
            if any(
                not 0.0 <= float(value) < 1.0
                for value in batch[field]
            ):
                raise AssertionError(f"invalid {field} values: {batch[field]}")
    if observed_workers != set(range(scenario.workers)):
        raise AssertionError(
            f"not every DataLoader worker produced a batch: {observed_workers}"
        )
    if len(worker_seed_bases) != 1:
        raise AssertionError(
            f"DataLoader workers did not share one base seed: {worker_seed_bases}"
        )
    return next(iter(worker_seed_bases))


def _rng_projection(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "worker_ids": batch["worker_ids"],
            "worker_seeds": batch["worker_seeds"],
            **{field: batch[field] for field in RNG_FIELDS},
        }
        for batch in batches
    ]


def _scenario_epoch_key(scenario: ReplayScenario) -> tuple[int, ...]:
    return (
        scenario.world_size,
        scenario.workers,
        scenario.prefetch_factor,
        scenario.batch_size,
        scenario.committed_batches,
        scenario.staged_batches,
    )


def _runtime_metadata() -> dict[str, object]:
    return {
        "hostname": platform.node(),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "numpy_version": str(np.__version__),
    }


def _build_matrix_report(
    case_reports: list[dict[str, Any]],
    scenarios: Sequence[ReplayScenario],
    *,
    dataset_size: int,
    sampler_seed: int,
    loader_seed_base: int,
    isolated_scenario_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(case_reports) != len(scenarios):
        raise AssertionError("DataLoader case reports do not match scenarios")
    deterministic_cases: list[dict[str, Any]] = []
    sample_order_cases: list[dict[str, Any]] = []
    rng_cases: list[dict[str, Any]] = []
    for case, scenario in zip(case_reports, scenarios):
        if case.get("scenario") != asdict(scenario):
            raise AssertionError(
                f"DataLoader scenario report changed: {case.get('scenario')}"
            )
        ranks = case.get("ranks")
        if not isinstance(ranks, list):
            raise AssertionError("DataLoader rank reports are missing")
        deterministic_ranks = [
            {
                key: rank[key]
                for key in (
                    "rank",
                    "loader_seed",
                    "worker_seed_base",
                    "expected_sample_ids",
                    "committed_batches",
                    "staged_batches",
                )
            }
            for rank in ranks
        ]
        deterministic_cases.append(
            {"scenario": asdict(scenario), "ranks": deterministic_ranks}
        )
        sample_order_cases.append(
            {
                "scenario": asdict(scenario),
                "ranks": [
                    {
                        "rank": rank["rank"],
                        "expected_sample_ids": rank["expected_sample_ids"],
                    }
                    for rank in ranks
                ],
            }
        )
        rng_cases.append(
            {
                "scenario": asdict(scenario),
                "ranks": [
                    {
                        "rank": rank["rank"],
                        "committed_batches": _rng_projection(
                            rank["committed_batches"]
                        ),
                        "staged_batches": _rng_projection(
                            rank["staged_batches"]
                        ),
                    }
                    for rank in ranks
                ],
            }
        )

    epoch_comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(scenarios):
        for right_index in range(left_index + 1, len(scenarios)):
            right = scenarios[right_index]
            if (
                left.epoch == right.epoch
                or _scenario_epoch_key(left) != _scenario_epoch_key(right)
            ):
                continue
            left_orders = [
                rank["expected_sample_ids"]
                for rank in case_reports[left_index]["ranks"]
            ]
            right_orders = [
                rank["expected_sample_ids"]
                for rank in case_reports[right_index]["ranks"]
            ]
            changed = all(
                left_order != right_order
                for left_order, right_order in zip(left_orders, right_orders)
            )
            if not changed:
                raise AssertionError(
                    f"sampler order did not change between {left.name} and "
                    f"{right.name}"
                )
            epoch_comparisons.append(
                {
                    "left": left.name,
                    "left_epoch": left.epoch,
                    "right": right.name,
                    "right_epoch": right.epoch,
                    "all_rank_orders_changed": True,
                }
            )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "runtime": _runtime_metadata(),
        "dataset_size": dataset_size,
        "sampler_seed": sampler_seed,
        "loader_seed_base": loader_seed_base,
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
        "rng_sources": ["torch", "python", "numpy"],
        "scenario_count": len(case_reports),
        "rank_case_count": sum(
            scenario.world_size for scenario in scenarios
        ),
        "worker_processes_started": sum(
            scenario.world_size * scenario.workers * 2
            for scenario in scenarios
        ),
        "all_worker_pids_replaced": True,
        "sampler_partitions_exact": True,
        "epoch_variation_verified": bool(epoch_comparisons),
        "epoch_comparisons": epoch_comparisons,
        "scenario_process_isolation": (
            isolated_scenario_processes is not None
        ),
        "matrix_digest": _digest(deterministic_cases),
        "sample_order_digest": _digest(sample_order_cases),
        "rng_digest": _digest(rng_cases),
        "cases": case_reports,
    }
    if isolated_scenario_processes is not None:
        report["isolated_scenario_processes"] = isolated_scenario_processes
    return report


def run_replay_matrix(
    scenarios: Sequence[ReplayScenario] = DEFAULT_SCENARIOS,
    *,
    dataset_size: int = 48,
    sampler_seed: int = 20260724,
    loader_seed_base: int = 2026072400,
) -> dict[str, Any]:
    if dataset_size <= 0:
        raise ValueError("dataset size must be positive")
    if not scenarios:
        raise ValueError("at least one replay scenario is required")
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("replay scenario names must be unique")

    case_reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        if dataset_size % scenario.world_size != 0:
            raise ValueError(
                f"dataset size must be divisible by world size for {scenario.name}"
            )
        batches_per_rank = math.ceil(
            (dataset_size // scenario.world_size) / scenario.batch_size
        )
        requested_batches = (
            scenario.committed_batches + scenario.staged_batches
        )
        if requested_batches > batches_per_rank:
            raise ValueError(
                f"scenario {scenario.name} requests {requested_batches} batches "
                f"but each rank has only {batches_per_rank}"
            )

        rank_reports: list[dict[str, Any]] = []
        deterministic_ranks: list[dict[str, Any]] = []
        for rank in range(scenario.world_size):
            loader_seed = _loader_seed(
                loader_seed_base,
                scenario_name=scenario.name,
                rank=rank,
            )
            initial_iterator, expected_sample_ids = _make_iterator(
                scenario,
                dataset_size=dataset_size,
                rank=rank,
                sampler_seed=sampler_seed,
                loader_seed=loader_seed,
            )
            initial_pids = _worker_pids(initial_iterator)
            try:
                initial_batches = _collect_batches(
                    initial_iterator,
                    requested_batches,
                )
                initial_prefetch_state = _prefetch_state(initial_iterator)
                initial_cleanup_batches = _drain_iterator(initial_iterator)
            finally:
                _shutdown_iterator(initial_iterator)
            if (
                len(initial_pids) != scenario.workers
                or len(set(initial_pids)) != scenario.workers
                or any(pid <= 0 for pid in initial_pids)
            ):
                raise AssertionError(
                    f"initial DataLoader worker PIDs are invalid: {initial_pids}"
                )
            initial_worker_seed_base = _validate_batches(
                initial_batches,
                scenario=scenario,
            )
            observed_sample_ids = _flatten_sample_ids(initial_batches)
            if observed_sample_ids != expected_sample_ids[
                : len(observed_sample_ids)
            ]:
                raise AssertionError(
                    f"shuffled sampler prefix differs for {scenario.name} rank "
                    f"{rank}: {observed_sample_ids}"
                )

            replay_iterator, replay_expected_sample_ids = _make_iterator(
                scenario,
                dataset_size=dataset_size,
                rank=rank,
                sampler_seed=sampler_seed,
                loader_seed=loader_seed,
            )
            replay_pids = _worker_pids(replay_iterator)
            try:
                replay_batches = _collect_batches(
                    replay_iterator,
                    requested_batches,
                )
                replay_prefetch_state = _prefetch_state(replay_iterator)
                replay_cleanup_batches = _drain_iterator(replay_iterator)
            finally:
                _shutdown_iterator(replay_iterator)
            if replay_expected_sample_ids != expected_sample_ids:
                raise AssertionError("reconstructed DistributedSampler changed")
            if initial_batches != replay_batches:
                raise AssertionError(
                    f"DataLoader replay differs for {scenario.name} rank {rank}"
                )
            if (
                len(replay_pids) != scenario.workers
                or len(set(replay_pids)) != scenario.workers
                or any(pid <= 0 for pid in replay_pids)
            ):
                raise AssertionError(
                    f"replayed DataLoader worker PIDs are invalid: {replay_pids}"
                )
            if set(initial_pids) & set(replay_pids):
                raise AssertionError(
                    f"DataLoader worker PIDs were reused: {initial_pids}, "
                    f"{replay_pids}"
                )
            replay_worker_seed_base = _validate_batches(
                replay_batches,
                scenario=scenario,
            )
            if replay_worker_seed_base != initial_worker_seed_base:
                raise AssertionError("DataLoader worker base seed changed")
            expected_cleanup_batches = batches_per_rank - requested_batches
            if (
                initial_cleanup_batches != expected_cleanup_batches
                or replay_cleanup_batches != expected_cleanup_batches
            ):
                raise AssertionError(
                    "DataLoader cleanup did not drain the remaining epoch"
                )

            committed = initial_batches[: scenario.committed_batches]
            staged = initial_batches[scenario.committed_batches :]
            deterministic_rank = {
                "rank": rank,
                "loader_seed": loader_seed,
                "worker_seed_base": initial_worker_seed_base,
                "expected_sample_ids": expected_sample_ids,
                "committed_batches": committed,
                "staged_batches": staged,
            }
            rank_reports.append(
                {
                    **deterministic_rank,
                    "status": "success",
                    "replay_exact": True,
                    "worker_pids_replaced": True,
                    "initial_worker_pids": initial_pids,
                    "replayed_worker_pids": replay_pids,
                    "initial_prefetch_state": initial_prefetch_state,
                    "replayed_prefetch_state": replay_prefetch_state,
                    "cleanup_batches_drained": expected_cleanup_batches,
                    "sequence_digest": _digest(
                        {
                            "committed_batches": committed,
                            "staged_batches": staged,
                        }
                    ),
                }
            )
            deterministic_ranks.append(deterministic_rank)

        all_expected_ids = [
            sample_id
            for rank_report in rank_reports
            for sample_id in rank_report["expected_sample_ids"]
        ]
        if sorted(all_expected_ids) != list(range(1, dataset_size + 1)):
            raise AssertionError(
                f"DistributedSampler partitions overlap or omit samples in "
                f"{scenario.name}"
            )
        scenario_payload = asdict(scenario)
        deterministic_case = {
            "scenario": scenario_payload,
            "ranks": deterministic_ranks,
        }
        case_reports.append(
            {
                "status": "success",
                "scenario": scenario_payload,
                "sampler": "DistributedSampler",
                "shuffle": True,
                "sampler_seed": sampler_seed,
                "sampler_partition_exact": True,
                "ranks": rank_reports,
                "case_digest": _digest(deterministic_case),
            }
        )
    return _build_matrix_report(
        case_reports,
        scenarios,
        dataset_size=dataset_size,
        sampler_seed=sampler_seed,
        loader_seed_base=loader_seed_base,
    )


def _scenario_argument(scenario: ReplayScenario) -> str:
    return ";".join(
        (
            f"name={scenario.name}",
            f"world-size={scenario.world_size}",
            f"workers={scenario.workers}",
            f"prefetch-factor={scenario.prefetch_factor}",
            f"epoch={scenario.epoch}",
            f"batch-size={scenario.batch_size}",
            f"committed-batches={scenario.committed_batches}",
            f"staged-batches={scenario.staged_batches}",
        )
    )


def run_isolated_replay_matrix(
    scenarios: Sequence[ReplayScenario] = DEFAULT_SCENARIOS,
    *,
    dataset_size: int = 48,
    sampler_seed: int = 20260724,
    loader_seed_base: int = 2026072400,
    scenario_timeout: float = 300.0,
) -> dict[str, Any]:
    if scenario_timeout <= 0:
        raise ValueError("scenario timeout must be positive")
    case_reports: list[dict[str, Any]] = []
    isolated_processes: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="fakegpu-dataloader-replay-"
    ) as raw:
        temporary_root = Path(raw)
        for index, scenario in enumerate(scenarios):
            output_path = temporary_root / f"scenario-{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--in-process",
                f"--dataset-size={dataset_size}",
                f"--sampler-seed={sampler_seed}",
                f"--loader-seed-base={loader_seed_base}",
                f"--scenario={_scenario_argument(scenario)}",
                f"--output={output_path}",
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=scenario_timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"isolated DataLoader scenario {scenario.name} failed with "
                    f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            child_report = json.loads(output_path.read_text(encoding="utf-8"))
            child_cases = child_report.get("cases")
            if (
                child_report.get("status") != "success"
                or not isinstance(child_cases, list)
                or len(child_cases) != 1
                or child_cases[0].get("scenario") != asdict(scenario)
            ):
                raise AssertionError(
                    f"isolated DataLoader scenario report is invalid: "
                    f"{child_report}"
                )
            case_reports.append(child_cases[0])
            isolated_processes.append(
                {
                    "scenario": scenario.name,
                    "runtime": child_report["runtime"],
                }
            )
    return _build_matrix_report(
        case_reports,
        scenarios,
        dataset_size=dataset_size,
        sampler_seed=sampler_seed,
        loader_seed_base=loader_seed_base,
        isolated_scenario_processes=isolated_processes,
    )


def validate_replay_matrix_reports(
    reports: Sequence[dict[str, Any]],
    *,
    require_default_matrix: bool = True,
) -> dict[str, Any]:
    if not reports:
        raise AssertionError("at least one DataLoader replay report is required")
    expected_scenarios = [asdict(scenario) for scenario in DEFAULT_SCENARIOS]
    expected_worker_processes = sum(
        scenario.world_size * scenario.workers * 2
        for scenario in DEFAULT_SCENARIOS
    )
    expected_rank_cases = sum(
        scenario.world_size for scenario in DEFAULT_SCENARIOS
    )
    digest_fields = ("matrix_digest", "sample_order_digest", "rng_digest")
    for report in reports:
        if (
            report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "success"
        ):
            raise AssertionError(f"DataLoader replay report failed: {report}")
        if (
            report.get("multiprocessing_context") != "spawn"
            or report.get("persistent_workers") is not True
            or report.get("rng_sources") != ["torch", "python", "numpy"]
            or report.get("all_worker_pids_replaced") is not True
            or report.get("sampler_partitions_exact") is not True
            or (
                require_default_matrix
                and report.get("epoch_variation_verified") is not True
            )
            or (
                require_default_matrix
                and report.get("scenario_process_isolation") is not True
            )
        ):
            raise AssertionError(
                f"DataLoader replay coverage is incomplete: {report}"
            )
        for field in digest_fields:
            value = report.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise AssertionError(f"invalid DataLoader {field}: {value}")
        cases = report.get("cases")
        if not isinstance(cases, list) or not cases:
            raise AssertionError("DataLoader replay cases are missing")
        if require_default_matrix:
            if (
                int(report.get("scenario_count", -1))
                != len(DEFAULT_SCENARIOS)
                or int(report.get("rank_case_count", -1))
                != expected_rank_cases
                or int(report.get("worker_processes_started", -1))
                != expected_worker_processes
                or [case.get("scenario") for case in cases]
                != expected_scenarios
            ):
                raise AssertionError(
                    f"maintained DataLoader matrix differs: {report}"
                )
        for case in cases:
            scenario = case.get("scenario")
            ranks = case.get("ranks")
            if (
                case.get("status") != "success"
                or case.get("sampler") != "DistributedSampler"
                or case.get("shuffle") is not True
                or case.get("sampler_partition_exact") is not True
                or not isinstance(scenario, dict)
                or not isinstance(ranks, list)
                or len(ranks) != int(scenario.get("world_size", -1))
            ):
                raise AssertionError(f"invalid DataLoader replay case: {case}")
            worker_count = int(scenario.get("workers", -1))
            for rank in ranks:
                initial_pids = rank.get("initial_worker_pids")
                replayed_pids = rank.get("replayed_worker_pids")
                if (
                    rank.get("status") != "success"
                    or rank.get("replay_exact") is not True
                    or rank.get("worker_pids_replaced") is not True
                    or not isinstance(initial_pids, list)
                    or not isinstance(replayed_pids, list)
                    or len(initial_pids) != worker_count
                    or len(replayed_pids) != worker_count
                    or set(int(pid) for pid in initial_pids)
                    & set(int(pid) for pid in replayed_pids)
                ):
                    raise AssertionError(
                        f"invalid DataLoader rank replay: {rank}"
                    )

    digest_summary: dict[str, str] = {}
    for field in digest_fields:
        values = {str(report[field]) for report in reports}
        if len(values) != 1:
            raise AssertionError(
                f"cross-runtime DataLoader {field} differs: {sorted(values)}"
            )
        digest_summary[field] = next(iter(values))
    return {
        "schema_version": "fakegpu.dataloader_replay_matrix_validation.v1",
        "status": "success",
        "report_count": len(reports),
        "scenario_count": int(reports[0]["scenario_count"]),
        "rank_case_count": int(reports[0]["rank_case_count"]),
        "worker_processes_started_per_report": int(
            reports[0]["worker_processes_started"]
        ),
        **digest_summary,
        "runtimes": [report["runtime"] for report in reports],
        "reports": list(reports),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic reconstruction of shuffled persistent "
            "multi-worker DataLoaders across a parameterized replay matrix."
        )
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-size", type=int, default=48)
    parser.add_argument("--sampler-seed", type=int, default=20260724)
    parser.add_argument("--loader-seed-base", type=int, default=2026072400)
    parser.add_argument("--scenario-timeout", type=float, default=300.0)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        default=[],
        help=(
            "Repeatable scenario fields: name, world-size, workers, "
            "prefetch-factor, epoch, batch-size, committed-batches, and "
            "staged-batches. Defaults to the maintained matrix."
        ),
    )
    args = parser.parse_args(argv)
    scenarios = args.scenario or DEFAULT_SCENARIOS
    try:
        if args.in_process or len(scenarios) == 1:
            report = run_replay_matrix(
                scenarios,
                dataset_size=args.dataset_size,
                sampler_seed=args.sampler_seed,
                loader_seed_base=args.loader_seed_base,
            )
        else:
            report = run_isolated_replay_matrix(
                scenarios,
                dataset_size=args.dataset_size,
                sampler_seed=args.sampler_seed,
                loader_seed_base=args.loader_seed_base,
                scenario_timeout=args.scenario_timeout,
            )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if args.output is not None:
            _write_report(args.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1
    if args.output is not None:
        _write_report(args.output, report)
    summary = {
        key: report[key]
        for key in (
            "schema_version",
            "status",
            "scenario_count",
            "rank_case_count",
            "worker_processes_started",
            "scenario_process_isolation",
            "matrix_digest",
            "sample_order_digest",
            "rng_digest",
        )
    }
    if args.output is not None:
        summary["report"] = str(args.output.resolve())
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
