"""Tests for the AI researcher preflight runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run_preflight(
    args: list[str],
    *,
    report_dir: Path,
    device_args: list[str] | None = None,
    preflight_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fakegpu",
            "preflight",
            "--runtime",
            "fakecuda",
            *(device_args or ["--profile", "a100-1g", "--device-count", "1"]),
            "--stage",
            "forward",
            "--report-dir",
            str(report_dir),
            *(preflight_args or []),
            "--",
            *args,
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )


def test_preflight_fakecuda_pass_generates_json_markdown_and_logs(tmp_path: Path) -> None:
    script = tmp_path / "pass_probe.py"
    script.write_text(
        "\n".join(
            [
                "import fakegpu",
                "import torch",
                "with fakegpu.stage('forward'):",
                "    x = torch.empty((1024, 1024), device='cuda', dtype=torch.float32)",
                "    print('peak', torch.cuda.max_memory_allocated())",
                "    del x",
            ]
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "preflight-pass"
    completed = _run_preflight([sys.executable, str(script)], report_dir=report_dir)

    assert completed.returncode == 0, completed.stderr
    assert (report_dir / "preflight_report.md").is_file()
    markdown = (report_dir / "preflight_report.md").read_text(encoding="utf-8")
    assert "## Stage Peaks" in markdown
    assert "## Current Memory By Category" in markdown
    assert "## Largest Allocations" in markdown
    assert (report_dir / "preflight_stdout.log").read_text(encoding="utf-8").startswith("peak ")
    assert "FakeGPU Report Summary" in (report_dir / "preflight_stderr.log").read_text(encoding="utf-8")

    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "preflight.v1"
    assert report["status"] == "PASS_FIT"
    assert report["runtime"] == "fakecuda"
    assert report["stage"] == "forward"
    assert report["tracking_confidence"] == "C2_torch_tensor_lifetime"
    assert report["command"] == [sys.executable, str(script)]
    assert report["logs"]["stdout"].endswith("preflight_stdout.log")

    devices = report["devices"]
    assert len(devices) == 1
    assert devices[0]["total_memory"] == 1024**3
    assert devices[0]["peak_memory"] >= 4 * 1024**2
    assert devices[0]["headroom_bytes"] > 0
    assert devices[0]["allocation_count"] >= 1
    assert devices[0]["tracking_confidence"] == "C2_torch_tensor_lifetime"
    assert isinstance(devices[0]["current_bytes_by_category"], dict)
    assert devices[0]["peak_by_stage"]["forward"] >= devices[0]["peak_memory"]
    assert devices[0]["largest_allocations"]
    largest = devices[0]["largest_allocations"][0]
    assert largest["bytes"] >= 4 * 1024**2
    assert largest["dtype"] == "torch.float32"
    assert largest["shape"] == [1024, 1024]
    assert largest["stage"] == "forward"
    assert largest["category"] in {"activation", "temporary", "tensor"}


def test_preflight_allocation_stacks_generates_json_and_markdown_origin(tmp_path: Path) -> None:
    script = tmp_path / "stack_probe.py"
    script.write_text(
        "\n".join(
            [
                "import fakegpu",
                "import torch",
                "def allocate_from_probe():",
                "    return torch.empty((256, 256), device='cuda', dtype=torch.float32)",
                "with fakegpu.stage('forward'):",
                "    tensor = allocate_from_probe()",
                "    print('peak', torch.cuda.max_memory_allocated())",
            ]
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "preflight-stacks"
    completed = _run_preflight(
        [sys.executable, str(script)],
        report_dir=report_dir,
        preflight_args=["--allocation-stacks"],
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    largest = report["devices"][0]["largest_allocations"][0]
    assert largest["stack"]
    assert any(frame["function"] == "allocate_from_probe" for frame in largest["stack"])

    markdown = (report_dir / "preflight_report.md").read_text(encoding="utf-8")
    assert "Origin" in markdown
    assert "allocate_from_probe" in markdown


def test_preflight_devices_spec_infers_device_count(tmp_path: Path) -> None:
    script = tmp_path / "device_probe.py"
    script.write_text(
        "\n".join(
            [
                "import torch",
                "print('count', torch.cuda.device_count())",
            ]
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "preflight-devices"
    completed = _run_preflight(
        [sys.executable, str(script)],
        report_dir=report_dir,
        device_args=["--devices", "a100-1g:1"],
    )

    assert completed.returncode == 0, completed.stderr
    assert (report_dir / "preflight_stdout.log").read_text(encoding="utf-8").strip() == "count 1"
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert len(report["devices"]) == 1
    assert report["devices"][0]["profile_id"] == "a100-1g"


def test_preflight_fakecuda_oom_returns_failure_report(tmp_path: Path) -> None:
    script = tmp_path / "oom_probe.py"
    script.write_text(
        "\n".join(
            [
                "import fakegpu",
                "import torch",
                "with fakegpu.stage('forward'):",
                "    torch.empty((300_000_000,), device='cuda', dtype=torch.float32)",
            ]
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "preflight-oom"
    completed = _run_preflight([sys.executable, str(script)], report_dir=report_dir)

    assert completed.returncode == 2, completed.stderr
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL_OOM"
    assert report["stage"] == "forward"
    assert report["exit_code"] != 0
    assert report["errors"]
    assert "out of memory" in report["errors"][0]["message"].lower()
    assert (report_dir / "preflight_stderr.log").is_file()
