from __future__ import annotations

import argparse
import base64
from pathlib import Path

import pytest

from verification.run_physical_multihost import (
    NodeSpec,
    _encoded_remote_command,
    _shell_command,
    _write_markdown,
    parse_node_spec,
)


def test_parse_node_spec_supports_posix_and_wsl() -> None:
    posix = parse_node_spec(
        "name=gpu-a;ssh=gpu-a;repo=/srv/fakegpu;"
        "python=/opt/python; shell=posix"
    )
    assert posix == NodeSpec(
        name="gpu-a",
        ssh="gpu-a",
        repo="/srv/fakegpu",
        python="/opt/python",
        shell="posix",
    )

    wsl = parse_node_spec(
        "ssh=user@windows;repo=/home/user/fakegpu;"
        "python=/opt/torch/bin/python;shell=wsl"
    )
    assert wsl.name == "user@windows"
    assert wsl.shell == "wsl"


@pytest.mark.parametrize(
    "value",
    [
        "ssh=gpu-a;repo=relative;python=/opt/python",
        "ssh=gpu-a;repo=/srv/fakegpu;python=python3",
        "ssh=gpu-a;repo=/srv/fakegpu;python=/opt/python;shell=powershell",
        "ssh=gpu-a;repo=/srv/fakegpu",
        "ssh=gpu-a;ssh=gpu-b;repo=/srv/fakegpu;python=/opt/python",
        "ssh=gpu-a;repo=/srv/fakegpu;python=/opt/python;extra=value",
    ],
)
def test_parse_node_spec_rejects_invalid_fields(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_node_spec(value)


def test_encoded_remote_command_roundtrips_shell_text() -> None:
    command = "printf '%s\\n' 'spaces and $shell characters'"
    decoder = _encoded_remote_command(command)
    encoded = decoder.split()[2]
    assert base64.b64decode(encoded).decode("utf-8") == command


def test_shell_command_quotes_environment_and_paths() -> None:
    node = NodeSpec(
        name="gpu",
        ssh="gpu",
        repo="/srv/fake gpu",
        python="/opt/python",
    )
    command = _shell_command(
        node,
        [node.python, "-c", "print('ok')"],
        env={"VALUE": "with spaces"},
        create_dir="/tmp/report dir",
    )
    assert "cd '/srv/fake gpu'" in command
    assert "mkdir -p '/tmp/report dir'" in command
    assert "'VALUE=with spaces'" in command


def test_physical_report_markdown_contains_node_pair_table(tmp_path: Path) -> None:
    report = {
        "status": "success",
        "session": "test",
        "git_commit": "abc123",
        "coordinator_endpoint": "10.0.0.1:29591",
        "nodes": [
            {
                "name": "gpu-a",
                "gpu_name": "GPU A",
                "compute_capability": [8, 6],
                "torch_version": "2.0",
                "torch_cuda_version": "12.0",
            },
            {
                "name": "gpu-b",
                "gpu_name": "GPU B",
                "compute_capability": [12, 0],
                "torch_version": "2.1",
                "torch_cuda_version": "13.0",
            },
        ],
        "cases": {
            "ddp": [{}, {}],
            "ddp_options": {},
            "fsdp": [{}, {}],
            "fsdp2": [{}, {}],
            "fsdp2_mixed": {"fp16": [{}, {}], "bf16": [{}, {}]},
            "collective_mismatch": [
                {"mismatch_result": 5},
                {"mismatch_result": 5},
            ],
            "missing_peer": {},
        },
        "cluster_summary": {
            "collectives": {
                "all_reduce": {"calls": 2, "bytes": 32},
                "broadcast": {"calls": 0, "bytes": 0},
            },
            "node_pairs": [
                {
                    "node_a": "gpu-a",
                    "node_b": "gpu-b",
                    "total_bytes": 64,
                    "peak_combined_bytes_per_operation": 32,
                    "operations": 2,
                    "collective_operations": 2,
                    "point_to_point_operations": 0,
                }
            ],
        },
    }
    path = tmp_path / "report.md"
    _write_markdown(path, report)
    markdown = path.read_text(encoding="utf-8")
    assert "| `gpu-a` | `gpu-b` | 64 | 32 | 2 | 2 | 0 |" in markdown
    assert "DDP options" in markdown
    assert "Hybrid FSDP" in markdown
    assert "Hybrid FSDP2" in markdown
    assert "FSDP2 mixed precision" in markdown
    assert "Collective mismatch" in markdown
