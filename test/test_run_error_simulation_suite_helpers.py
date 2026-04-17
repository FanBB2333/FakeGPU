"""Regression checks for run_error_simulation_suite.py helper behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "test" / "run_error_simulation_suite.py"

spec = importlib.util.spec_from_file_location("error_sim_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class _Completed:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


def test_run_test_file_sets_dummy_xonsh_history(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    path = tmp_path / "test_error_dummy.py"
    path.write_text("def test_dummy():\n    pass\n", encoding="utf-8")

    results = runner.run_test_file(path)

    assert len(results) == 1
    env = captured["kwargs"]["env"]
    assert env["XONSH_HISTORY_BACKEND"] == "dummy"


def test_extract_peak_vram_by_gpu_parses_per_device_peaks() -> None:
    summary = """
======================================================
             FakeGPU Report Summary
======================================================
 Device 0: NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
   Memory: 9.2 MB / 80.0 GB peak (0.0%)
   Alloc: 4 calls | Free: 4 calls
------------------------------------------------------
 Device 1: NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
   Memory: 512.0 KB / 80.0 GB peak (0.0%)
   Alloc: 1 calls | Free: 1 calls
------------------------------------------------------
======================================================
""".strip()

    peaks = runner._extract_peak_vram_by_gpu(summary)

    assert peaks == {
        0: int(9.2 * 1024**2),
        1: 512 * 1024,
    }


def test_extract_peak_vram_by_gpu_from_rendered_html_aggregates_maxima() -> None:
    html = """
    <div class="terminal">
      FakeGPU Report Summary
      Device 0: A100
      Memory: <span class="hl">8.0 MB</span> / 80.0 GB peak (0.0%)
      Device 1: A100
      Memory: <span class="hl">256.0 KB</span> / 80.0 GB peak (0.0%)
    </div>
    <div class="terminal">
      FakeGPU Report Summary
      Device 0: A100
      Memory: <span class="hl">9.2 MB</span> / 80.0 GB peak (0.0%)
      Device 1: A100
      Memory: <span class="hl">1.5 MB</span> / 80.0 GB peak (0.0%)
    </div>
    """

    peaks = runner._extract_peak_vram_by_gpu_from_rendered_html(html)

    assert peaks == {
        0: int(9.2 * 1024**2),
        1: int(1.5 * 1024**2),
    }


def test_build_peak_vram_summary_section_renders_cards() -> None:
    section = runner._build_peak_vram_summary_section({
        0: int(9.2 * 1024**2),
        1: 512 * 1024,
    })

    assert "Peak VRAM by GPU" in section
    assert "GPU 0" in section
    assert "9.2 MB" in section
    assert "GPU 1" in section
    assert "512.0 KB" in section
