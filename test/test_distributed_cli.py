from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fakegpu.distributed_cli import (
    _write_local_cluster_config,
    parse_ranks,
    parse_size,
    parse_tcp_endpoint,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4096", 4096),
        ("64KiB", 64 * 1024),
        ("4MiB", 4 * 1024 * 1024),
        ("2GB", 2_000_000_000),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


def test_parse_rank_ranges() -> None:
    assert parse_ranks("0-2,4,6-7") == [0, 1, 2, 4, 6, 7]


@pytest.mark.parametrize("text", ["", "0,,1", "2-1", "0,0", "rank0"])
def test_reject_invalid_rank_ranges(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_ranks(text)


def test_parse_tcp_endpoint() -> None:
    assert parse_tcp_endpoint("127.0.0.1:29591") == ("127.0.0.1", 29591)
    assert parse_tcp_endpoint("coordinator.internal:65535") == (
        "coordinator.internal",
        65535,
    )


def test_generated_cluster_assigns_ranks_to_logical_nodes(tmp_path: Path) -> None:
    path = tmp_path / "cluster.yaml"
    _write_local_cluster_config(
        path,
        nodes=2,
        ranks_per_node=2,
        profile="a100",
        interconnect_bandwidth_gbps=25.0,
        interconnect_latency_us=50.0,
    )

    text = path.read_text(encoding="utf-8")
    assert "name: local-tcp-simulation" in text
    assert "ranks: [0, 1]" in text
    assert "ranks: [2, 3]" in text
    assert text.count("profile: a100") == 4
    assert "type: tcp" in text
    assert "bandwidth_gbps: 25" in text
