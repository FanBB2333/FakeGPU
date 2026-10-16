from __future__ import annotations

from fakegpu.distributed_cli import _print_bandwidth_summary


def test_bandwidth_summary_does_not_print_pass_for_error(capsys) -> None:
    _print_bandwidth_summary(
        {
            "status": "error",
            "endpoint": "127.0.0.1:1234",
            "world_size": 2,
            "local_ranks": [0],
            "payload_bytes_per_rank": 1024,
            "rank_reports": [
                {"rank": 0, "algorithmic_bandwidth_gbps": 0.0}
            ],
        }
    )

    output = capsys.readouterr().out
    assert "FakeGPU TCP bandwidth benchmark: FAIL" in output
    assert "benchmark: PASS" not in output


def test_local_topology_takes_the_node_fabric_from_the_platform(tmp_path) -> None:
    import pytest

    from fakegpu.distributed_cli import _intra_node_fabric, _write_local_cluster_config
    from fakegpu.platforms import PlatformError

    board = _intra_node_fabric("a100", 8, "auto")
    assert board == {"platform": "hgx-a100-8", "type": "nvlink", "bandwidth_gbps": 2400.0, "latency_us": 3.0}
    # DGX-1 rings run over the double NVLinks although some pairs have none.
    assert _intra_node_fabric("v100", 8, "auto")["bandwidth_gbps"] == 400.0
    # No PCIe peer access on the RTX 4090: staged through host at x8 Gen4.
    assert _intra_node_fabric("rtx4090", 2, "auto")["type"] == "pcie"
    assert _intra_node_fabric("rtx4090", 2, "auto")["bandwidth_gbps"] == 126.032
    assert _intra_node_fabric("a100", 16, "auto")["platform"] == "generic"
    with pytest.raises(PlatformError):
        _intra_node_fabric("a100", 16, "hgx-a100-8")

    path = tmp_path / "cluster.yaml"
    _write_local_cluster_config(
        path,
        nodes=2,
        ranks_per_node=8,
        profile="a100",
        intra_node=board,
        interconnect_bandwidth_gbps=25.0,
        interconnect_latency_us=50.0,
    )
    assert "  intra_node:\n    type: nvlink\n    bandwidth_gbps: 2400\n    latency_us: 3\n" in path.read_text()
