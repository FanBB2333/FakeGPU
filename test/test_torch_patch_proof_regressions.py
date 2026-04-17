"""Regression checks for torch_patch profile/device-count proof experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_snippet(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_a100_1g_profile_matches_device_count_and_memory_limit() -> None:
    code = textwrap.dedent(
        """
        import json
        import fakegpu

        fakegpu.init(runtime="fakecuda", profile="a100-1g", device_count=2)

        import torch
        import fakegpu.torch_patch as tp

        total_memory = torch.cuda.mem_get_info(0)[1]
        payload = {
            "device_count": torch.cuda.device_count(),
            "profile_count": len(tp._DEVICE_PROFILES),
            "tracker_count": len(tp._memory_tracker._total),
            "total_memory": total_memory,
            "device_name": torch.cuda.get_device_name(0),
        }
        print(json.dumps(payload, sort_keys=True))
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout.strip())
    assert payload["device_count"] == 2
    assert payload["profile_count"] == 2
    assert payload["tracker_count"] == 2
    assert payload["total_memory"] == 1024**3
    assert payload["device_name"] == "NVIDIA A100-SXM4-1GB"
