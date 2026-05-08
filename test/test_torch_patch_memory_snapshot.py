"""Regression tests for fakecuda memory snapshot metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_memory_snapshot_records_stage_peaks_and_largest_allocations() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch
        import fakegpu.torch_patch as tp

        with fakegpu.stage("forward"):
            x = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
            y = x + 1
            z = y.clone()

        snapshot = tp.memory_snapshot()
        print(json.dumps(snapshot, sort_keys=True))
        """
    )

    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    snapshot = json.loads(completed.stdout.strip())
    device = snapshot["devices"][0]
    assert device["tracking_confidence"] == "C2_torch_tensor_lifetime"
    assert device["peak_by_stage"]["forward"] >= 12 * 1024**2
    assert device["peak_memory"] >= 12 * 1024**2
    assert device["allocation_count"] >= 3

    largest = device["largest_allocations"][0]
    assert largest["bytes"] == 4 * 1024**2
    assert largest["shape"] == [1024, 1024]
    assert largest["dtype"] == "torch.float32"
    assert largest["device"] == 0
    assert largest["stage"] == "forward"
    assert largest["category"] in {"activation", "tensor"}


def test_memory_snapshot_classifies_training_state_categories() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch
        import fakegpu.torch_patch as tp

        with fakegpu.stage("model_load"):
            model = torch.nn.Linear(8, 4).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with fakegpu.stage("forward"):
            x = torch.randn(2, 8, device="cuda")
            loss = model(x).sum()

        with fakegpu.stage("backward"):
            loss.backward()
            for param in model.parameters():
                _ = param.grad

        with fakegpu.stage("optimizer_step"):
            optimizer.step()

        snapshot = tp.memory_snapshot()
        print(json.dumps(snapshot, sort_keys=True))
        """
    )

    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    snapshot = json.loads(completed.stdout.strip())
    device = snapshot["devices"][0]
    categories = device["current_bytes_by_category"]
    assert categories["parameter"] >= 144
    assert categories["gradient"] >= 144
    assert categories["optimizer_state"] >= 288

    assert device["peak_by_stage"]["model_load"] >= 144
    assert device["peak_by_stage"]["forward"] > device["peak_by_stage"]["model_load"]
    assert device["peak_by_stage"]["backward"] >= device["peak_by_stage"]["forward"]
    assert device["peak_by_stage"]["optimizer_step"] >= device["peak_by_stage"]["backward"]

    top_categories = {alloc["category"] for alloc in device["largest_allocations"]}
    assert {"parameter", "gradient", "optimizer_state"}.issubset(top_categories)
