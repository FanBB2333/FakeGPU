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
