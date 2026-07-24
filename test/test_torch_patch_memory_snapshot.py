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
    assert device["tracking_confidence"] == "C3_torch_dispatch_lifetime"
    assert device["peak_by_stage"]["forward"] >= 12 * 1024**2
    assert device["peak_memory"] >= 12 * 1024**2
    assert device["allocation_count"] >= 3

    largest = device["largest_allocations"][0]
    assert largest["bytes"] == 4 * 1024**2
    assert largest["shape"] == [1024, 1024]
    assert largest["dtype"] == "torch.float32"
    assert largest["device"] == 0
    assert largest["stage"] == "forward"
    assert largest["category"] in {"activation", "temporary", "tensor"}


def test_dispatch_tracker_records_allocations_and_storage_aliases() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch
        import fakegpu.torch_patch as tp

        x = torch.ones(4, device="cuda")
        y = torch.ones(4, device="cuda")
        z = x + y
        viewed = z.view(2, 2)
        snapshot = tp.memory_snapshot()
        print(json.dumps({
            "snapshot": snapshot,
            "same_storage": (
                z.untyped_storage().data_ptr()
                == viewed.untyped_storage().data_ptr()
            ),
        }, sort_keys=True))
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

    payload = json.loads(completed.stdout.strip())
    tracking = payload["snapshot"]["dispatch_tracking"]
    assert payload["same_storage"] is True
    assert payload["snapshot"]["tracking_confidence"] == (
        "C3_torch_dispatch_lifetime"
    )
    assert tracking["enabled"] is True
    assert tracking["operator_calls"] >= 2
    assert tracking["new_allocations"] >= 1
    assert tracking["alias_outputs"] >= 1
    assert tracking["operators"]["aten.add.Tensor"]["new_allocations"] == 1
    assert tracking["operators"]["aten.view.default"]["alias_outputs"] == 1


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


def test_memory_snapshot_classifies_buffers_and_temporaries() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch
        import fakegpu.torch_patch as tp

        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(16, 16))

            def forward(self, x):
                return x * self.scale

        with fakegpu.stage("model_load"):
            module = WithBuffer().cuda()

        with fakegpu.stage("forward"):
            tmp = torch.ones(16, 16, device="cuda")
            out = module(tmp)

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
    assert categories["buffer"] >= 16 * 16 * 4
    assert categories["temporary"] >= 16 * 16 * 4

    top_categories = {alloc["category"] for alloc in device["largest_allocations"]}
    assert "buffer" in top_categories
    assert "temporary" in top_categories


def test_memory_snapshot_does_not_double_count_shared_storage_aliases() -> None:
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
            base = torch.empty((4, 4), device="cuda", dtype=torch.float32)
            after_base = tp.memory_snapshot()["devices"][0]

            viewed = base.view(2, 8)
            sliced = base[:, :2]
            reshaped = base.reshape(1, 16)
            base.add_(1)
            after_aliases = tp.memory_snapshot()["devices"][0]

            contiguous = base.t().contiguous()
            after_contiguous = tp.memory_snapshot()["devices"][0]

        payload = {
            "after_base": after_base,
            "after_aliases": after_aliases,
            "after_contiguous": after_contiguous,
            "ptrs": {
                "base": base.untyped_storage().data_ptr(),
                "viewed": viewed.untyped_storage().data_ptr(),
                "sliced": sliced.untyped_storage().data_ptr(),
                "reshaped": reshaped.untyped_storage().data_ptr(),
                "contiguous": contiguous.untyped_storage().data_ptr(),
            },
        }
        print(json.dumps(payload, sort_keys=True))
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

    payload = json.loads(completed.stdout.strip())
    after_base = payload["after_base"]
    after_aliases = payload["after_aliases"]
    after_contiguous = payload["after_contiguous"]
    ptrs = payload["ptrs"]

    assert ptrs["viewed"] == ptrs["base"]
    assert ptrs["sliced"] == ptrs["base"]
    assert ptrs["reshaped"] == ptrs["base"]
    assert ptrs["contiguous"] != ptrs["base"]

    assert after_base["current_memory"] == 4 * 4 * 4
    assert after_base["allocation_count"] == 1
    assert after_aliases["current_memory"] == after_base["current_memory"]
    assert after_aliases["allocation_count"] == after_base["allocation_count"]

    assert after_contiguous["current_memory"] == after_base["current_memory"] + 4 * 4 * 4
    assert after_contiguous["allocation_count"] == after_base["allocation_count"] + 1
    assert after_contiguous["peak_by_stage"]["forward"] == after_contiguous["current_memory"]


def test_memory_snapshot_attributes_allocations_to_logical_devices() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:2")

        import torch
        import fakegpu.torch_patch as tp

        with fakegpu.stage("forward"):
            dev0 = torch.empty((8, 8), device="cuda:0", dtype=torch.float32)
            dev1 = torch.empty((16, 8), device="cuda:1", dtype=torch.float32)

        snapshot = tp.memory_snapshot()
        payload = {
            "snapshot": snapshot,
            "devices": [str(dev0.device), str(dev1.device)],
        }
        print(json.dumps(payload, sort_keys=True))
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

    payload = json.loads(completed.stdout.strip())
    assert payload["devices"] == ["cuda:0", "cuda:1"]

    devices = payload["snapshot"]["devices"]
    assert len(devices) == 2
    dev0, dev1 = devices

    assert dev0["index"] == 0
    assert dev1["index"] == 1
    assert dev0["current_memory"] == 8 * 8 * 4
    assert dev1["current_memory"] == 16 * 8 * 4
    assert dev0["peak_by_stage"]["forward"] == dev0["current_memory"]
    assert dev1["peak_by_stage"]["forward"] == dev1["current_memory"]
    assert dev0["largest_allocations"][0]["device"] == 0
    assert dev1["largest_allocations"][0]["device"] == 1


def test_memory_snapshot_optionally_records_allocation_stack_trace() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_ALLOCATION_STACKS"] = "1"
        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch
        import fakegpu.torch_patch as tp

        def allocate_from_user_code():
            return torch.empty((8, 8), device="cuda", dtype=torch.float32)

        with fakegpu.stage("forward"):
            tensor = allocate_from_user_code()

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
    stack = snapshot["devices"][0]["largest_allocations"][0]["stack"]
    assert stack
    assert all({"file", "line", "function"}.issubset(frame) for frame in stack)
    assert any(frame["function"] == "allocate_from_user_code" for frame in stack)


def test_fakecuda_oom_error_uses_torch_cuda_oom_type_and_capacity_message() -> None:
    code = textwrap.dedent(
        """
        import json
        import os

        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100-1g:1")

        import torch

        try:
            torch.empty((300_000_000,), device="cuda", dtype=torch.float32)
        except Exception as exc:
            payload = {
                "type": type(exc).__name__,
                "is_cuda_oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                "message": str(exc),
            }
            print(json.dumps(payload, sort_keys=True))
            raise SystemExit(0)

        raise SystemExit("expected fakecuda OOM")
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

    payload = json.loads(completed.stdout.strip())
    message = payload["message"].lower()
    assert payload["is_cuda_oom"] is True
    assert payload["type"] == "OutOfMemoryError"
    assert "cuda out of memory" in message
    assert "tried to allocate" in message
    assert "total capacity" in message
    assert "free" in message
