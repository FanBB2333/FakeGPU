from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict

import pytest

from verification.dataloader_replay_matrix import (
    DEFAULT_SCENARIOS,
    ReplayScenario,
    parse_scenario,
    validate_replay_matrix_reports,
)


def _report(*, runtime_name: str = "runtime-a") -> dict[str, object]:
    cases: list[dict[str, object]] = []
    pid = 1000
    for scenario in DEFAULT_SCENARIOS:
        ranks: list[dict[str, object]] = []
        for rank in range(scenario.world_size):
            initial_pids = list(range(pid, pid + scenario.workers))
            pid += scenario.workers
            replayed_pids = list(range(pid, pid + scenario.workers))
            pid += scenario.workers
            ranks.append(
                {
                    "status": "success",
                    "rank": rank,
                    "replay_exact": True,
                    "worker_pids_replaced": True,
                    "initial_worker_pids": initial_pids,
                    "replayed_worker_pids": replayed_pids,
                }
            )
        cases.append(
            {
                "status": "success",
                "scenario": asdict(scenario),
                "sampler": "DistributedSampler",
                "shuffle": True,
                "sampler_partition_exact": True,
                "ranks": ranks,
            }
        )
    return {
        "schema_version": "fakegpu.dataloader_replay_matrix.v1",
        "status": "success",
        "runtime": {"hostname": runtime_name},
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
        "rng_sources": ["torch", "python", "numpy"],
        "scenario_count": len(DEFAULT_SCENARIOS),
        "rank_case_count": sum(
            scenario.world_size for scenario in DEFAULT_SCENARIOS
        ),
        "worker_processes_started": sum(
            scenario.world_size * scenario.workers * 2
            for scenario in DEFAULT_SCENARIOS
        ),
        "all_worker_pids_replaced": True,
        "sampler_partitions_exact": True,
        "epoch_variation_verified": True,
        "matrix_digest": "a" * 64,
        "sample_order_digest": "b" * 64,
        "rng_digest": "c" * 64,
        "cases": cases,
    }


def test_parse_scenario_accepts_repeatable_matrix_fields() -> None:
    scenario = parse_scenario(
        "name=custom;world-size=4;workers=2;prefetch-factor=3;epoch=5;"
        "batch-size=2;committed-batches=2;staged-batches=2"
    )

    assert scenario == ReplayScenario(
        name="custom",
        world_size=4,
        workers=2,
        prefetch_factor=3,
        epoch=5,
        batch_size=2,
        committed_batches=2,
        staged_batches=2,
    )


@pytest.mark.parametrize(
    "value, message",
    [
        (
            "name=missing;workers=1",
            "missing fields",
        ),
        (
            "name=unknown;world-size=1;workers=1;prefetch-factor=1;"
            "epoch=0;batch-size=1;committed-batches=0;staged-batches=1;"
            "extra=1",
            "unknown scenario fields",
        ),
        (
            "name=negative;world-size=1;workers=1;prefetch-factor=1;"
            "epoch=-1;batch-size=1;committed-batches=0;staged-batches=1",
            "must be non-negative",
        ),
        (
            "name=workers;world-size=1;workers=3;prefetch-factor=1;"
            "epoch=0;batch-size=1;committed-batches=1;staged-batches=1",
            "at least one batch from every worker",
        ),
    ],
)
def test_parse_scenario_rejects_invalid_fields(
    value: str,
    message: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse_scenario(value)


def test_validate_replay_matrix_accepts_cross_runtime_match() -> None:
    first = _report(runtime_name="torch-2.8")
    second = _report(runtime_name="torch-2.12")

    summary = validate_replay_matrix_reports([first, second])

    assert summary["status"] == "success"
    assert summary["report_count"] == 2
    assert summary["scenario_count"] == 5
    assert summary["rank_case_count"] == 12
    assert summary["worker_processes_started_per_report"] == 52
    assert summary["matrix_digest"] == "a" * 64


def test_validate_replay_matrix_rejects_cross_runtime_rng_drift() -> None:
    first = _report(runtime_name="torch-2.8")
    second = _report(runtime_name="torch-2.12")
    second["rng_digest"] = "d" * 64

    with pytest.raises(AssertionError, match="rng_digest differs"):
        validate_replay_matrix_reports([first, second])


def test_validate_replay_matrix_rejects_worker_pid_reuse() -> None:
    invalid = deepcopy(_report())
    cases = invalid["cases"]
    assert isinstance(cases, list)
    ranks = cases[0]["ranks"]
    assert isinstance(ranks, list)
    ranks[0]["replayed_worker_pids"] = ranks[0]["initial_worker_pids"]

    with pytest.raises(AssertionError, match="invalid DataLoader rank replay"):
        validate_replay_matrix_reports([invalid])
