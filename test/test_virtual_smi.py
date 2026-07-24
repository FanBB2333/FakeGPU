from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fakegpu.smi as smi_module
from fakegpu.smi import SmiStatePublisher, main, render_table


ROOT = Path(__file__).resolve().parents[1]


def _snapshot() -> dict:
    return {
        "tracking_confidence": "C2_torch_tensor_lifetime",
        "stage": "forward",
        "devices": [
            {
                "index": 0,
                "name": "Fake NVIDIA Test GPU",
                "profile_id": "test",
                "total_memory": 8 * 2**30,
                "current_memory": 3 * 2**30,
                "peak_memory": 4 * 2**30,
            }
        ],
    }


def test_publisher_and_virtual_smi_include_process_memory(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "state.json"
    publisher = SmiStatePublisher(
        path,
        _snapshot,
        runtime_overhead_bytes=256 * 2**20,
    )
    state = publisher.publish_once(running=True)
    assert state["devices"][0]["reported_memory"] == 3328 * 2**20
    assert state["devices"][0]["reported_peak_memory"] == 4352 * 2**20
    assert state["stage"] == "forward"
    assert main(["--state", str(path)]) == 0
    output = capsys.readouterr().out
    assert "FakeGPU-SMI" in output
    assert "3328 MiB / 8192 MiB" in output
    assert "4352 MiB" in output
    assert "3072 MiB" in output
    assert "4096 MiB" in output
    assert "forward" in output
    assert "C2_torch_tensor_lifetime" in output
    assert "test" in output


def test_render_table_marks_exited_process() -> None:
    state = {
        "hostname": "host",
        "pid": 42,
        "process_name": "python model.py",
        "running": False,
        "devices": [
            {
                "index": 0,
                "name": "Fake GPU",
                "total_memory": 1024,
                "tracked_memory": 512,
                "reported_memory": 768,
            }
        ],
    }
    rendered = render_table([state])
    assert "(exited)" in rendered
    assert "python model.py" in rendered


def test_virtual_smi_rejects_unknown_schema(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": "unknown"}), encoding="utf-8")
    assert main(["--state", str(path)]) == 1
    assert "unsupported schema" in capsys.readouterr().out


def test_virtual_smi_rejects_non_object_state(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    assert main(["--state", str(path)]) == 1
    assert "state root must be an object" in capsys.readouterr().out


def test_render_table_distinguishes_hosts_profiles_and_stages() -> None:
    states = []
    for host, pid, profile, stage in (
        ("host-a", 10, "a100", "forward"),
        ("host-b", 20, "h100", "backward"),
    ):
        states.append(
            {
                "hostname": host,
                "pid": pid,
                "process_name": "python train.py",
                "running": True,
                "tracking_confidence": "C2_torch_tensor_lifetime",
                "stage": stage,
                "devices": [
                    {
                        "index": 0,
                        "name": f"Fake {profile.upper()}",
                        "profile_id": profile,
                        "total_memory": 8 * 2**30,
                        "tracked_memory": 2**30,
                        "peak_tracked_memory": 2 * 2**30,
                        "reported_memory": 2**30,
                        "reported_peak_memory": 2 * 2**30,
                    }
                ],
            }
        )

    rendered = render_table(states)
    assert "| host-a | 0 | a100 |" in rendered
    assert "| host-b | 0 | h100 |" in rendered
    assert "forward" in rendered
    assert "backward" in rendered
    assert "2048 MiB" in rendered


def test_virtual_smi_loop_emits_bounded_ndjson(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    assert (
        main(
            [
                "--state-dir",
                str(tmp_path),
                "--json",
                "--loop",
                "0.001",
                "--count",
                "2",
            ]
        )
        == 0
    )
    samples = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(samples) == 2
    assert all(sample["errors"] == [] for sample in samples)
    assert all(sample["states"][0]["stage"] == "forward" for sample in samples)


def test_virtual_smi_loop_rediscovers_state_directory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "late-state.json"

    def publish_during_wait(_seconds: float) -> None:
        SmiStatePublisher(path, _snapshot).publish_once(running=True)

    monkeypatch.setattr(smi_module.time, "sleep", publish_during_wait)
    assert main(["--state-dir", str(tmp_path), "--loop", "1", "--count", "2"]) == 0
    output = capsys.readouterr().out
    assert output.count("No published FakeCUDA processes found.") == 1
    assert "forward" in output


def test_publisher_uses_environment_stage_and_counts_overhead_without_total(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.json"
    monkeypatch.setenv("FAKEGPU_PREFLIGHT_STAGE", "optimizer_step")
    publisher = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "current_memory": 2**20,
                    "peak_memory": 2 * 2**20,
                    "total_memory": 0,
                }
            ]
        },
        runtime_overhead_bytes=3 * 2**20,
    )

    state = publisher.publish_once(running=True)
    assert state["stage"] == "optimizer_step"
    assert state["devices"][0]["reported_memory"] == 4 * 2**20
    assert state["devices"][0]["reported_peak_memory"] == 5 * 2**20


def test_fakecuda_runtime_publishes_profile_stage_and_peak(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime-state.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env["FAKEGPU_SMI_STATE_PATH"] = str(state_path)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    code = "\n".join(
        [
            "import json",
            "import fakegpu",
            "import torch",
            "fakegpu.init(runtime='fakecuda', profile='test-512m', device_count=1)",
            "with fakegpu.stage('forward'):",
            "    tensor = torch.empty((1024, 1024), device='cuda', dtype=torch.float32)",
            "    from fakegpu import torch_patch",
            "    state = torch_patch._smi_publisher.publish_once(running=True)",
            "    print(json.dumps(state, sort_keys=True))",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads(completed.stdout.strip().splitlines()[-1])
    assert state["stage"] == "forward"
    assert state["tracking_confidence"] == "C3_torch_dispatch_lifetime"
    assert state["devices"][0]["profile_id"] == "test-512m"
    assert state["devices"][0]["tracked_memory"] >= 4 * 2**20
    assert state["devices"][0]["peak_tracked_memory"] >= 4 * 2**20
    assert state["devices"][0]["reported_peak_memory"] >= 4 * 2**20


def test_virtual_smi_count_requires_loop(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--state", "state.json", "--count", "2"])
    assert exc_info.value.code == 2
    assert "--count requires --loop" in capsys.readouterr().err
