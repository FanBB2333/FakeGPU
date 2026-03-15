#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BIN = REPO_ROOT / "build" / "fakegpu_topology_probe"
FAST_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_valid.yaml"
SLOW_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_slow_interconnect.yaml"


def run_probe(config_path: Path) -> dict:
    completed = subprocess.run(
        [
            str(PROBE_BIN),
            "--cluster-config",
            str(config_path),
            "--collective",
            "allreduce",
            "--bytes-per-rank",
            str(1024 * 1024),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise AssertionError(f"topology probe failed for {config_path}")
    return json.loads(completed.stdout)


def sum_scope_time(report: dict, scope: str) -> float:
    return sum(
        float(link["estimated_time_us"])
        for link in report["links"]
        if link["scope"] == scope
    )


def main() -> int:
    if not PROBE_BIN.exists():
        print(f"missing topology probe: {PROBE_BIN}", file=sys.stderr)
        return 2

    fast = run_probe(FAST_CONFIG)
    slow = run_probe(SLOW_CONFIG)

    assert fast["cluster"]["world_size"] == 4
    assert slow["cluster"]["world_size"] == 4
    assert fast["cluster"]["node_count"] == 2
    assert slow["cluster"]["node_count"] == 2
    assert fast["collective"]["type"] == "allreduce"
    assert slow["collective"]["type"] == "allreduce"
    assert len(fast["links"]) >= 4
    assert len(slow["links"]) >= 4

    fast_total = float(fast["collective"]["estimated_time_us"])
    slow_total = float(slow["collective"]["estimated_time_us"])
    if slow_total <= fast_total:
        raise AssertionError(
            f"expected slow interconnect estimate to exceed fast config: fast={fast_total}, slow={slow_total}"
        )

    fast_inter = sum_scope_time(fast, "inter_node")
    slow_inter = sum_scope_time(slow, "inter_node")
    if slow_inter <= fast_inter:
        raise AssertionError(
            f"expected inter-node link estimate to increase: fast={fast_inter}, slow={slow_inter}"
        )

    fast_intra = sum_scope_time(fast, "intra_node")
    slow_intra = sum_scope_time(slow, "intra_node")
    if abs(fast_intra - slow_intra) > 1e-6:
        raise AssertionError(
            f"expected intra-node estimate to stay constant: fast={fast_intra}, slow={slow_intra}"
        )

    print("topology model test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
