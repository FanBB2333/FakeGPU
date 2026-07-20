from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fakegpu.distributed_cli import (
    _coordinator_env,
    _write_local_cluster_config,
    bandwidth_main,
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


def test_coordinator_env_sets_explicit_report_paths(tmp_path: Path) -> None:
    json_report = tmp_path / "cluster.json"
    markdown_report = tmp_path / "project-report.md"

    env = _coordinator_env(
        endpoint="127.0.0.1:29591",
        cluster_config=None,
        cluster_report=json_report,
        cluster_markdown_report=markdown_report,
    )

    assert env["FAKEGPU_CLUSTER_REPORT_PATH"] == str(json_report.resolve())
    assert env["FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH"] == str(
        markdown_report.resolve()
    )


def test_coordinator_env_uses_default_markdown_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH",
        str(tmp_path / "inherited.md"),
    )

    env = _coordinator_env(
        endpoint="127.0.0.1:29591",
        cluster_config=None,
        cluster_report=tmp_path / "cluster.json",
    )

    assert "FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH" not in env


def test_bandwidth_connect_rejects_coordinator_report_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        bandwidth_main(
            [
                "--connect",
                "127.0.0.1:29591",
                "--world-size",
                "2",
                "--ranks",
                "0",
                "--session",
                "test-session",
                "--cluster-report",
                str(tmp_path / "cluster.json"),
            ]
        )

    assert error.value.code == 2
