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
    assert "## Summary" in markdown
    assert "## Stage Peaks" in markdown
    assert "## Current Memory By Category" in markdown
    assert "## Largest Allocations" in markdown
    assert "## Confidence" in markdown
    assert "## Suggested Next Steps" in markdown
    assert "completed `forward` without tracked OOM" in markdown
    assert "attach `preflight_report.json`" in markdown.lower()
    assert (report_dir / "preflight_stdout.log").read_text(encoding="utf-8").startswith("peak ")
    assert "FakeGPU Report Summary" in (report_dir / "preflight_stderr.log").read_text(encoding="utf-8")

    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "preflight.v1"
    assert report["status"] == "PASS_FIT"
    assert report["runtime"] == "fakecuda"
    assert report["stage"] == "forward"
    assert report["tracking_confidence"] == "C2_torch_tensor_lifetime"
    assert report["command"] == [sys.executable, str(script)]
    assert report["target_profiles"] == [{"profile_id": "a100-1g", "count": 1}]
    assert report["calibration_gpu"] is None
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

    schema_path = ROOT / "preflight_report.schema.json"
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "target_profiles" in schema["required"]
    assert "calibration_gpu" in schema["required"]

    checked = subprocess.run(
        [
            sys.executable,
            "verification/check_preflight_report.py",
            "--path",
            str(report_dir / "preflight_report.json"),
            "--expect-status",
            "PASS_FIT",
            "--expect-runtime",
            "fakecuda",
            "--expect-device",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 0, checked.stderr


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
    markdown = (report_dir / "preflight_report.md").read_text(encoding="utf-8")
    assert "## Summary" in markdown
    assert "## Failure Reason" in markdown
    assert "out of memory" in markdown.lower()
    assert "reduce batch size" in markdown.lower()
    assert (report_dir / "preflight_stderr.log").is_file()


def test_preflight_memory_safety_factor_can_fail_estimated_oom(tmp_path: Path) -> None:
    script = tmp_path / "factor_probe.py"
    script.write_text(
        "\n".join(
            [
                "import fakegpu",
                "import torch",
                "with fakegpu.stage('forward'):",
                "    x = torch.empty((1024, 1024), device='cuda', dtype=torch.float32)",
                "    print('peak', torch.cuda.max_memory_allocated())",
            ]
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "preflight-safety-factor"
    completed = _run_preflight(
        [sys.executable, str(script)],
        report_dir=report_dir,
        device_args=["--profile", "test-512m", "--device-count", "1"],
        preflight_args=["--memory-safety-factor", "200"],
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    dev = report["devices"][0]
    assert report["status"] == "FAIL_OOM"
    assert report["memory_safety_factor"] == 200
    assert dev["tracked_peak_memory"] >= 4 * 1024**2
    assert dev["peak_memory"] == dev["estimated_peak_memory"]
    assert dev["peak_memory"] > dev["total_memory"]
    assert any("safety factor" in warning.lower() for warning in report["warnings"])
    markdown = (report_dir / "preflight_report.md").read_text(encoding="utf-8")
    assert "Tracked Peak" in markdown
    assert "Estimated Peak" in markdown


def test_preflight_same_workload_fails_on_small_profile_and_passes_on_large_profile(
    tmp_path: Path,
) -> None:
    elements = "140000000"
    requested_bytes = int(elements) * 4
    workload = [
        sys.executable,
        "verification/preflight_oom_probe.py",
        "--mode",
        "alloc",
        "--elements",
        elements,
    ]

    small_dir = tmp_path / "preflight-matrix-small"
    small = _run_preflight(
        workload,
        report_dir=small_dir,
        device_args=["--profile", "test-512m", "--device-count", "1"],
    )

    assert small.returncode == 2, small.stderr
    small_report = json.loads((small_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert small_report["status"] == "FAIL_OOM"
    assert small_report["target_profiles"] == [{"profile_id": "test-512m", "count": 1}]
    assert small_report["devices"][0]["total_memory"] == 512 * 1024**2
    assert "out of memory" in small_report["errors"][0]["message"].lower()

    large_dir = tmp_path / "preflight-matrix-large"
    large = _run_preflight(
        workload,
        report_dir=large_dir,
        device_args=["--profile", "a100", "--device-count", "1"],
    )

    assert large.returncode == 0, large.stderr
    large_report = json.loads((large_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert large_report["status"] == "PASS_FIT"
    assert large_report["target_profiles"] == [{"profile_id": "a100", "count": 1}]
    assert large_report["devices"][0]["peak_memory"] >= requested_bytes
    assert large_report["devices"][0]["headroom_bytes"] > 0


def test_preflight_strict_treats_skipped_child_output_as_failure(tmp_path: Path) -> None:
    script = tmp_path / "skip_probe.py"
    script.write_text("print('1 skipped in 0.01s')\n", encoding="utf-8")

    report_dir = tmp_path / "preflight-strict-skip"
    completed = _run_preflight(
        [sys.executable, str(script)],
        report_dir=report_dir,
        preflight_args=["--strict"],
    )

    assert completed.returncode == 1, completed.stdout
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL_RUNTIME"
    assert any("skipped" in warning.lower() for warning in report["warnings"])
    assert report["errors"][0]["type"] == "SkippedTest"
    assert "strict" in report["errors"][0]["message"].lower()
