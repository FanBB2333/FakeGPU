from __future__ import annotations

import json
from pathlib import Path

from fakegpu.smi import SmiStatePublisher, main, render_table


def _snapshot() -> dict:
    return {
        "tracking_confidence": "C2_torch_tensor_lifetime",
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


def test_publisher_and_virtual_smi_include_process_memory(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    publisher = SmiStatePublisher(
        path,
        _snapshot,
        runtime_overhead_bytes=256 * 2**20,
    )
    state = publisher.publish_once(running=True)
    assert state["devices"][0]["reported_memory"] == 3328 * 2**20
    assert main(["--state", str(path)]) == 0
    output = capsys.readouterr().out
    assert "FakeGPU-SMI" in output
    assert "3328 MiB /     8192 MiB" in output
    assert "3072 MiB" in output


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
