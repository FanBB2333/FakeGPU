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


def test_device_selection_and_queries_follow_torch_cuda() -> None:
    # torch/cuda/__init__.py (2.14): set_device, device and device_of ignore a
    # negative index; get_device_name/capability read get_device_properties,
    # which raises AssertionError("Invalid device id") out of range;
    # _get_device_index rejects non-CUDA devices and, for set_device, a CUDA
    # device without an index.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100,v100")
import torch


def raises(kind, message, call):
    try:
        call()
    except kind as exc:
        assert message in str(exc), exc
    else:
        raise AssertionError(f"expected {kind.__name__}: {message}")


for query in (
    torch.cuda.get_device_properties,
    torch.cuda.get_device_name,
    torch.cuda.get_device_capability,
):
    raises(AssertionError, "Invalid device id", lambda: query(2))
    raises(AssertionError, "Invalid device id", lambda: query(-1))
    raises(ValueError, "Expected a cuda device, but got: cpu", lambda: query("cpu"))
assert torch.cuda.get_device_capability("cuda:1") == (7, 0)
raises(AssertionError, "Invalid peer device id", lambda: torch.cuda.can_device_access_peer(0, 2))

torch.cuda.set_device(1)
torch.cuda.set_device(-1)
assert torch.cuda.current_device() == 1
raises(ValueError, "Expected a torch.device with a specified index", lambda: torch.cuda.set_device("cuda"))
raises(ValueError, "Expected a cuda device, but got: cpu", lambda: torch.cuda.set_device("cpu"))
# c10/cuda/CUDAException.cpp: C10_CUDA_CHECK raises c10::AcceleratorError.
for call in (
    lambda: torch.cuda.set_device(2),
    lambda: torch.cuda.synchronize(2),
    lambda: torch.ones(1, device="cuda:2"),
):
    raises(torch.AcceleratorError, "CUDA error: invalid device ordinal", call)

for no_op in (-1, None):
    with torch.cuda.device(no_op):
        assert torch.cuda.current_device() == 1
with torch.cuda.device_of(torch.ones(1)):
    assert torch.cuda.current_device() == 1
with torch.cuda.device_of(torch.ones(1, device="cuda:0")):
    assert torch.cuda.current_device() == 0
assert torch.cuda.current_device() == 1

# The previous device is taken when the block is entered.
selector = torch.cuda.device(0)
torch.cuda.set_device(1)
with selector:
    assert torch.cuda.current_device() == 0
assert torch.cuda.current_device() == 1

assert torch.cuda._exchange_device(0) == 1
assert torch.cuda.current_device() == 0
with torch.cuda._DeviceGuard(1):
    assert torch.cuda.current_device() == 1
assert torch.cuda.current_device() == 0
print("device selection passed")
"""
    )


def test_device_properties_expose_torch_fields_from_the_profile() -> None:
    # torch/csrc/cuda/Module.cpp (v2.14.1) binds these cudaDeviceProp fields;
    # clock_rate and memory_clock_rate are in kHz, and CUDA builds bind
    # gcnArchName to the device name. torch.cuda._device_limits consumes them.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100,h100")
import torch
from torch.cuda._device_limits import GPULimits

from fakegpu.profile_catalog import get_profile

for index, profile_id in enumerate(("a100", "h100")):
    profile = get_profile(profile_id)
    properties = torch.cuda.get_device_properties(index)
    assert properties.shared_memory_per_block == profile.shared_mem_per_block
    assert properties.shared_memory_per_block_optin == profile.shared_mem_per_block_optin
    assert properties.shared_memory_per_multiprocessor == profile.shared_mem_per_sm
    assert properties.L2_cache_size == profile.l2_cache_bytes
    assert properties.memory_bus_width == profile.memory_bus_width_bits
    assert properties.clock_rate == profile.core_clock_mhz * 1000
    assert properties.memory_clock_rate == profile.memory_clock_mhz * 1000
    assert properties.gcnArchName == properties.name == profile.torch_name
    # DDR transfers twice per clock: bus bytes x 2 x memory clock.
    bandwidth = GPULimits(torch.device("cuda", index)).get_memory_bandwidth_Bps()
    assert abs(bandwidth / 1e9 - profile.memory_bandwidth_gbps) < 1, (profile_id, bandwidth)
print("device properties passed")
"""
    )


def test_streams_and_events_follow_torch_cuda() -> None:
    # c10/cuda/CUDAEvent.h (v2.14.1) elapsed_time raises ValueError unless
    # both events time and were recorded; torch.cuda.Event takes ``external``;
    # torch.Stream has device, device_type and priority, and entering it or
    # torch.cuda.set_stream selects the stream's device
    # (THPStream_enter, THCPModule_setStream_wrap); torch.cuda.stream(None)
    # is a no-op.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import torch


def raises(message, call):
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), exc
    else:
        raise AssertionError(message)


start, end = torch.cuda.Event(), torch.cuda.Event()
start.record()
end.record()
raises("enable_timing=True", lambda: start.elapsed_time(end))
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True, external=False)
assert start.device is None
raises("must be recorded", lambda: start.elapsed_time(end))
start.record()
end.record(torch.cuda.current_stream())
assert start.device == torch.device("cuda", 0)
assert isinstance(start.elapsed_time(end), float)

side = torch.cuda.Stream(device=1, priority=-1)
assert (side.device, side.device_type, side.priority) == (torch.device("cuda", 1), 1, -1)
assert side.record_event().device == torch.device("cuda", 1)

with torch.cuda.stream(None):
    assert torch.cuda.current_stream() == torch.cuda.default_stream()
for enter in (lambda: torch.cuda.stream(side), lambda: side, lambda: torch.cuda.StreamContext(side)):
    with enter():
        assert torch.cuda.current_device() == 1
        assert torch.cuda.current_stream() is side
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(1) is torch.cuda.default_stream(1)
torch.cuda.set_stream(side)
assert torch.cuda.current_device() == 1
print("streams and events passed")
"""
    )


def test_module_moves_keep_parameter_objects() -> None:
    # nn.Module._apply assigns param.data when the new tensor has a
    # compatible shallow-copy type (CPU and CUDA dense tensors do), so a
    # parameter keeps its identity across .cuda()/.to("cuda") and an
    # optimizer built first, as Accelerate's prepare() expects, keeps working.
    # Moving to the device a tensor is already on leaves it unchanged.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import torch

model = torch.nn.Linear(256, 256, bias=False)
weight = model.weight
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
before = torch.cuda.memory_allocated(0)
model.cuda()
assert model.weight is weight and weight.is_cuda
assert torch.cuda.memory_allocated(0) - before == 256 * 256 * 4

model.to("cuda:0")
assert model.weight is weight
assert torch.cuda.memory_allocated(0) - before == 256 * 256 * 4

start = weight.detach().cpu().clone()
model(torch.ones(4, 256, device="cuda")).sum().backward()
optimizer.step()
assert not torch.equal(start, weight.detach().cpu())
# Weight, gradient and AdamW's two moment buffers, all FP32 on the device.
assert len(optimizer.state[weight]) == 3
assert torch.cuda.memory_allocated(0) - before == 4 * 256 * 256 * 4

tied = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 8, bias=False))
tied[1].weight = tied[0].weight
tied.to("cuda:1")
assert tied[1].weight is tied[0].weight and tied[0].weight.device == torch.device("cuda", 1)
print("in-place module moves passed")
"""
    )


def test_cuda_autocast_state_and_custom_fwd() -> None:
    # autocast.__enter__ sets torch.is_autocast_enabled(device) for the block,
    # and torch.amp.custom_fwd(cast_inputs=...) reads that flag, casts the
    # inputs and runs the forward with autocast disabled
    # (torch/amp/autocast_mode.py). The CPU device shows the same rules.
    _run(
        """
import torch


def make_op(device_type):
    class Op(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type=device_type, cast_inputs=torch.float32)
        def forward(ctx, a, b):
            return a @ b

        @staticmethod
        @torch.amp.custom_bwd(device_type=device_type)
        def backward(ctx, grad):
            return grad, grad

    return Op


def run(device_type):
    a = torch.ones(2, 2, device=device_type, dtype=torch.float16)
    with torch.autocast(device_type, dtype=torch.float16):
        state = (torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
        result = make_op(device_type).apply(a, a)
    return state, result.dtype, torch.is_autocast_enabled(device_type)


expected = run("cpu")
assert expected == ((True, torch.float16), torch.float32, False), expected
fakegpu.init(runtime="fakecuda", devices="a100")
observed = run("cuda")
assert observed == expected, observed
print("autocast state passed")
"""
    )


def test_fork_rng_keeps_torch_signature() -> None:
    # torch/random.py: fork_rng(devices=None, enabled=True, _caller="fork_rng",
    # _devices_kw="devices", device_type=None); torch.jit.verify passes _caller.
    _run(
        """
import inspect

import torch

def parameters(function):
    return [(p.name, p.default, p.kind) for p in inspect.signature(function).parameters.values()]


real = parameters(torch.random.fork_rng)
fakegpu.init(runtime="fakecuda", devices="a100")
assert parameters(torch.random.fork_rng) == real, parameters(torch.random.fork_rng)
state = torch.random.get_rng_state()
with torch.random.fork_rng([0], _caller="torch.jit.verify"):
    torch.rand(4)
assert torch.equal(state, torch.random.get_rng_state())
print("fork_rng passed")
"""
    )
