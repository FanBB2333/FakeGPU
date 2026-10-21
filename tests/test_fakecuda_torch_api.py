"""FakeCUDA ``torch.cuda`` and tensor APIs checked against PyTorch's contracts.

Expected values come from the PyTorch release that the test environment
installs (its Python sources) or from running the same call on the CPU
device, which shares the argument parsing and dtype rules with CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PRELUDE = f"""
import sys
import warnings

sys.path.insert(0, {str(ROOT)!r})
warnings.simplefilter("ignore")

import fakegpu
"""


def _run(code: str) -> None:
    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    env["FAKEGPU_HOME"] = tempfile.mkdtemp(prefix="fakegpu-home-")
    env["FAKEGPU_MACHINE"] = "off"
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    env.pop("FAKEGPU_SMI_STATE_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", _PRELUDE + code],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_to_accepts_every_overload_with_a_cuda_target() -> None:
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import torch

# Tensor.to(device=None, dtype=None, non_blocking=False, copy=False) and
# Tensor.to(other, non_blocking=False, copy=False), as on the CPU device.
x = torch.ones(2, 3)
for call in (
    lambda t, d: t.to(d, torch.float16),
    lambda t, d: t.to(d, torch.float16, True),
    lambda t, d: t.to(torch.device(d), torch.float16, False, True),
):
    assert call(x, "cpu").dtype == torch.float16
    moved = call(x, "cuda:1")
    assert (moved.device, moved.dtype) == (torch.device("cuda", 1), torch.float16)

other = torch.ones(1, device="cuda:1").to(torch.bfloat16)
moved = x.to(other)
assert (moved.device, moved.dtype) == (torch.device("cuda", 1), torch.bfloat16)
assert x.to(torch.ones(1, dtype=torch.float16)).dtype == torch.float16

# nn.Module.to converts floating-point parameters and buffers; integer
# buffers keep their dtype.
for case, call in enumerate((
    lambda m: m.to("cuda", torch.float16),
    lambda m: m.to(device="cuda", dtype=torch.float16),
    lambda m: m.to("cuda").to("cuda", torch.float16),
    lambda m: m.to(other.to(torch.float16)),
)):
    module = call(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)))
    assert module[0].weight.is_cuda, case
    assert module[0].weight.dtype == torch.float16, case
    assert module[1].running_mean.dtype == torch.float16, case
    assert module[1].num_batches_tracked.dtype == torch.int64, case

try:
    torch.nn.Linear(2, 2).to("cuda", torch.int64)
except TypeError as exc:
    assert "only accepts floating point or complex dtypes" in str(exc), exc
else:
    raise AssertionError("an integer dtype must be rejected as on real devices")

# Only the converted copy is resident: 1024 x 1024 FP16 weights plus bias.
torch.cuda.set_device(0)
base = torch.cuda.memory_allocated(0)
layer = torch.nn.Linear(1024, 1024).to("cuda:0", torch.float16)
assert torch.cuda.memory_allocated(0) - base == (1024 * 1024 + 1024) * 2
print("to overloads passed")
"""
    )


def test_torch_autocast_alias_autocasts_cuda_ops() -> None:
    # torch.autocast is the documented spelling; on CUDA, linear and matmul
    # run in the autocast dtype (CPU autocast applies the same op lists).
    _run(
        """
import torch

assert torch.autocast is torch.amp.autocast
reference = torch.nn.Linear(4, 4)
with torch.autocast("cpu", dtype=torch.float16):
    assert reference(torch.ones(2, 4)).dtype == torch.float16

fakegpu.init(runtime="fakecuda", devices="a100")
layer = torch.nn.Linear(4, 4).cuda()
x = torch.ones(2, 4, device="cuda")
for ctx in (
    lambda: torch.autocast("cuda", dtype=torch.float16),
    lambda: torch.autocast(device_type="cuda"),
    lambda: torch.amp.autocast("cuda"),
):
    with ctx():
        assert layer(x).dtype == torch.float16
        assert (x @ x.T).dtype == torch.float16
assert layer(x).dtype == torch.float32

try:
    torch.autocast()
except TypeError:
    pass
else:
    raise AssertionError("device_type is a required argument of torch.autocast")
print("torch.autocast passed")
"""
    )


def test_bf16_probe_follows_torch_cuda_is_bf16_supported() -> None:
    # torch/cuda/__init__.py (2.14): native bf16 from compute capability 8.0
    # of the current device; otherwise True unless including_emulation=False.
    # is_tf32_supported() is is_bf16_supported(including_emulation=False).
    _run(
        """
import inspect

import torch

real_signature = inspect.signature(torch.cuda.is_bf16_supported)
fakegpu.init(runtime="fakecuda", devices="a100,v100")
assert inspect.signature(torch.cuda.is_bf16_supported).parameters.keys() == (
    real_signature.parameters.keys()
)

expected = {0: (True, True, True), 1: (True, False, False)}
for index, (emulated, native, tf32) in expected.items():
    torch.cuda.set_device(index)
    assert torch.cuda.is_bf16_supported() is emulated, index
    assert torch.cuda.is_bf16_supported(including_emulation=False) is native, index
    assert torch.cuda.is_tf32_supported() is tf32, index

# torch.amp.autocast only rejects bfloat16 when it is enabled and
# is_bf16_supported() is false.
torch.cuda.set_device(1)
for spelling in (torch.autocast, torch.amp.autocast):
    with spelling("cuda", dtype=torch.bfloat16):
        assert (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).dtype == torch.bfloat16
    with spelling("cuda", dtype=torch.bfloat16, enabled=False):
        pass

import accelerate.utils

assert accelerate.utils.is_bf16_available(True) is True
print("bf16 probe passed")
"""
    )
