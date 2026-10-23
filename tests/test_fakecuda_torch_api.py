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
import importlib.util
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

if importlib.util.find_spec("accelerate") is not None:
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
    # gcnArchName to the device name. torch.cuda._device_limits (PyTorch
    # 2.10+) consumes them. Older CUDA builds bind fewer fields: 2.9.1 has no
    # clock_rate, memory_clock_rate, memory_bus_width, max_threads_per_block
    # or regs_per_block, so reading one raises AttributeError there.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100,h100")
import importlib.util
import torch

from fakegpu.profile_catalog import get_profile

bound = getattr(torch._C, "_CudaDeviceProperties", None)
fields = None if bound is None else {name for name in dir(bound) if not name.startswith("_")}

for index, profile_id in enumerate(("a100", "h100")):
    profile = get_profile(profile_id)
    properties = torch.cuda.get_device_properties(index)
    expected = {
        "shared_memory_per_block": profile.shared_mem_per_block,
        "shared_memory_per_block_optin": profile.shared_mem_per_block_optin,
        "shared_memory_per_multiprocessor": profile.shared_mem_per_sm,
        "L2_cache_size": profile.l2_cache_bytes,
        "memory_bus_width": profile.memory_bus_width_bits,
        "clock_rate": profile.core_clock_mhz * 1000,
        "memory_clock_rate": profile.memory_clock_mhz * 1000,
        "gcnArchName": profile.torch_name,
    }
    for name, value in expected.items():
        if fields is None or name in fields:
            assert getattr(properties, name) == value, (profile_id, name)
        else:
            assert not hasattr(properties, name), (profile_id, name)
    if fields is not None:
        assert set(vars(properties)) <= fields, sorted(set(vars(properties)) - fields)
    assert properties.name == profile.torch_name
    if importlib.util.find_spec("torch.cuda._device_limits") is not None:
        from torch.cuda._device_limits import GPULimits

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


def test_mps_is_hidden_so_frameworks_pick_the_simulated_gpus() -> None:
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:1")
import importlib.util
import torch

# A CUDA machine has no MPS backend; Accelerate and transformers check MPS
# before CUDA, so on Apple Silicon they would otherwise skip the fake GPUs.
assert not torch.backends.mps.is_available() and not torch.backends.mps.is_built()
if importlib.util.find_spec("accelerate") is not None:
    from accelerate import Accelerator

    assert Accelerator().device.type == "cuda", Accelerator().device
if importlib.util.find_spec("transformers") is not None:
    from transformers import TrainingArguments

    device = TrainingArguments(output_dir="out", report_to=[]).device
    assert device.type == "cuda", device
"""
    )


def test_cuda_generator_behaves_like_cuda_generator_impl() -> None:
    # Measured on an Ampere GPU (torch 2.9.1): torch.Generator("cuda") is a
    # torch._C.Generator on device "cuda" with seed 67280421310721 and a
    # 16-byte uint8 state [seed u64 | offset i64]; manual_seed returns the
    # generator and resets the offset; random kernels advance the offset in
    # steps of 4 (aten/src/ATen/cuda/CUDAGeneratorImpl.cpp, Generator.cpp).
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import pickle
import torch


def raises(kind, message, call):
    try:
        call()
    except kind as exc:
        assert str(exc).startswith(message), exc
    else:
        raise AssertionError(message)


g = torch.Generator("cuda")
assert isinstance(g, torch.Generator) and isinstance(g, torch._C.Generator)
assert isinstance(torch.default_generator, torch.Generator)
assert isinstance(torch.Generator(), torch.Generator)
assert type(torch.Generator()) is torch._C.Generator
assert type(g).__name__ == "Generator"
assert g.device == torch.device("cuda") and g.device.index is None
assert torch.Generator(device="cuda:1").device == torch.device("cuda", 1)
assert torch.Generator(torch.device("cuda")).device == torch.device("cuda")
assert torch.Generator(0).device == torch.device("cuda", 0)
assert g.initial_seed() == 67280421310721 and g.get_offset() == 0
state = g.get_state()
assert (state.dtype, tuple(state.shape), state.device.type) == (torch.uint8, (16,), "cpu")
assert state.tolist() == list((67280421310721).to_bytes(8, "little")) + [0] * 8

assert g.manual_seed(5) is g and g.initial_seed() == 5
assert g.manual_seed(-1).initial_seed() == 2**64 - 1
raises(ValueError, "Overflow when unpacking long long", lambda: g.manual_seed(2**64))
raises(RuntimeError, "manual_seed expected a long, but got float", lambda: g.manual_seed(1.5))
seed = g.seed()
assert seed == g.initial_seed() and g.get_offset() == 0 and seed < 2**53

g.manual_seed(0)
torch.randn(10, generator=g, device="cuda")
assert g.get_offset() == 4
assert g.get_state().tolist() == [0] * 8 + [4] + [0] * 7
torch.empty(5, device="cuda").uniform_(generator=g)
assert g.get_offset() == 8
saved = g.get_state()
a = torch.randn(4, generator=g, device="cuda")
assert g.set_state(saved) is g
assert torch.equal(a, torch.randn(4, generator=g, device="cuda"))
assert not torch.equal(a, torch.randn(4, generator=g, device="cuda"))

raises(RuntimeError, "RNG state is wrong size", lambda: g.set_state(torch.Generator().get_state()))
raises(TypeError, "RNG state must be a torch.ByteTensor", lambda: g.set_state(torch.zeros(16)))
raises(TypeError, "expected a torch.ByteTensor, but got list", lambda: g.set_state([0] * 16))
g.set_state(torch.tensor(list((7).to_bytes(8, "little")), dtype=torch.uint8))
assert (g.initial_seed(), g.get_offset()) == (7, 0)
bad = torch.tensor(list((7).to_bytes(8, "little")) + list((3).to_bytes(8, "little")), dtype=torch.uint8)
raises(RuntimeError, "offset must be a multiple of 4", lambda: g.set_state(bad))
assert g.set_offset(8) is g and g.get_offset() == 8
raises(RuntimeError, "offset must be a multiple of 4", lambda: g.set_offset(3))
assert g.manual_seed(1).get_offset() == 0
assert g.clone_state().device == torch.device("cuda")

g.manual_seed(11)
torch.randn(3, generator=g, device="cuda")
copy = pickle.loads(pickle.dumps(g))
assert (copy.device, copy.initial_seed(), copy.get_offset()) == (torch.device("cuda"), 11, 4)
assert pickle.loads(pickle.dumps(torch.Generator().manual_seed(3))).initial_seed() == 3

# A generator must be on the device of the tensor it fills.
raises(RuntimeError, "Expected a 'cpu' device type for generator but found 'cuda'",
       lambda: torch.randn(3, generator=g))
raises(RuntimeError, "Expected a 'cuda' device type for generator but found 'cpu'",
       lambda: torch.randn(3, generator=torch.Generator(), device="cuda"))
raises(RuntimeError, "Expected a 'cuda' device type for generator but found 'cpu'",
       lambda: torch.empty(3, device="cuda").uniform_(generator=torch.Generator()))

# diffusers.utils.torch_utils.randn_tensor with a CUDA generator.
generator = torch.Generator("cuda").manual_seed(0)
rand_device = "cpu" if generator.device.type != "cuda" else torch.device("cuda")
latents = torch.randn((1, 4, 8, 8), generator=generator, device=rand_device, dtype=torch.float16)
assert (latents.device, latents.dtype, generator.get_offset()) == (torch.device("cuda", 0), torch.float16, 4)
again = torch.randn((1, 4, 8, 8), generator=torch.Generator("cuda").manual_seed(0), device="cuda", dtype=torch.float16)
assert torch.equal(latents, again)
print("generator passed")
"""
    )


def test_cuda_rng_is_per_device_and_separate_from_the_cpu_generator() -> None:
    # Measured on an Ampere GPU: torch.cuda.manual_seed leaves the CPU
    # generator alone, torch.manual_seed seeds CUDA too, the CUDA state is
    # 16 bytes, dropout in eval mode draws nothing, and fork_rng restores
    # the CUDA offset. torch.cuda.random's functions are torch's own.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import torch


def offset(index=0):
    return torch.cuda.default_generators[index].get_offset()


assert torch.cuda.default_generators == ()
torch.manual_seed(0)
cpu = torch.rand(3)
torch.manual_seed(0)
torch.cuda.manual_seed(7)
assert torch.equal(cpu, torch.rand(3))
assert torch.cuda.initial_seed() == 7 and torch.initial_seed() == 0
assert len(torch.cuda.default_generators) == 2
assert [str(g.device) for g in torch.cuda.default_generators] == ["cuda:0", "cuda:1"]
torch.cuda.manual_seed_all(9)
assert [g.initial_seed() for g in torch.cuda.default_generators] == [9, 9]
torch.manual_seed(42)
assert torch.cuda.initial_seed() == 42 and offset() == 0
torch.cuda.seed()
assert torch.cuda.initial_seed() != 42 and torch.initial_seed() == 42

state = torch.cuda.get_rng_state()
assert (state.dtype, tuple(state.shape)) == (torch.uint8, (16,))
assert len(torch.cuda.get_rng_state_all()) == 2
try:
    torch.cuda.set_rng_state(torch.get_rng_state())
except RuntimeError as exc:
    assert "RNG state is wrong size" in str(exc)
else:
    raise AssertionError("a CPU-sized state was accepted")

# Each device has its own generator.
torch.cuda.manual_seed(3)
first = torch.randn(5, device="cuda")
torch.cuda.manual_seed(3)
assert torch.equal(first, torch.randn(5, device="cuda"))
assert offset(0) == 4 and offset(1) == 0
with torch.cuda.device(1):
    torch.cuda.manual_seed(3)
assert torch.equal(first, torch.randn(5, device="cuda:1"))
assert offset(1) == 4
torch.manual_seed(3)
first = torch.randn(5, device="cuda")
torch.manual_seed(3)
assert torch.equal(first, torch.randn(5, device="cuda"))

# Random operations on fake CUDA tensors draw from the CUDA generator.
torch.cuda.manual_seed(0)
x = torch.ones(100, device="cuda")
cpu_state = torch.get_rng_state()
torch.nn.functional.dropout(x, 0.5)
assert offset() == 4
torch.nn.functional.dropout(x, 0.5, training=False)
assert offset() == 4
x.uniform_()
torch.nn.init.kaiming_uniform_(torch.empty(4, 4, device="cuda"))
assert offset() == 12
q = torch.randn(1, 2, 4, 8, device="cuda")
torch.nn.functional.scaled_dot_product_attention(q, q, q)
assert offset() == 16
assert torch.equal(cpu_state, torch.get_rng_state())

torch.cuda.manual_seed(1)
with torch.random.fork_rng(devices=[0]):
    torch.randn(5, device="cuda")
assert offset() == 0
print("cuda rng passed")
"""
    )


def test_cuda_initializes_on_first_use() -> None:
    # Measured on an Ampere GPU, each call in a fresh process: the first
    # list leaves torch.cuda.is_initialized() False, the second sets it.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:1")
import fakegpu._upstream as upstream
import torch

keep = {
    "is_available": lambda: torch.cuda.is_available(),
    "device_count": lambda: torch.cuda.device_count(),
    "version": lambda: torch.version.cuda,
    "get_arch_list": lambda: torch.cuda.get_arch_list(),
    "get_gencode_flags": lambda: torch.cuda.get_gencode_flags(),
    "cudnn_version": lambda: torch.backends.cudnn.version(),
    "nccl_version": lambda: torch.cuda.nccl.version(),
    "manual_seed": lambda: torch.cuda.manual_seed(5),
    "set_rng_state": lambda: torch.cuda.set_rng_state(torch.zeros(16, dtype=torch.uint8)),
    "manual_seed_all": lambda: torch.cuda.manual_seed_all(9),
    "seed": lambda: torch.cuda.seed(),
    "default_generators": lambda: len(torch.cuda.default_generators),
    "generator": lambda: torch.Generator("cuda"),
    "stream": lambda: torch.cuda.Stream(),
    "event": lambda: torch.cuda.Event(),
    "priority_range": lambda: torch.cuda.Stream.priority_range(),
    "memory_allocated": lambda: torch.cuda.memory_allocated(),
    "memory_stats": lambda: torch.cuda.memory_stats(),
    "empty_cache": lambda: torch.cuda.empty_cache(),
    "pin_memory": lambda: torch.zeros(1).pin_memory(),
    "is_pinned": lambda: torch.zeros(1).is_pinned(),
    "grad_scaler": lambda: torch.amp.GradScaler("cuda"),
}
assert not torch.cuda.is_initialized()
for name, call in keep.items():
    call()
    assert not torch.cuda.is_initialized(), name
assert len(torch.cuda.default_generators) == 0

initialize = {
    "init": lambda: torch.cuda.init(),
    "lazy_init": lambda: torch.cuda._lazy_init(),
    "current_device": lambda: torch.cuda.current_device(),
    "set_device": lambda: torch.cuda.set_device(0),
    "device": lambda: torch.cuda.device(0).__enter__(),
    "get_device_name": lambda: torch.cuda.get_device_name(),
    "get_device_capability": lambda: torch.cuda.get_device_capability(),
    "get_device_properties": lambda: torch.cuda.get_device_properties(0),
    "is_bf16_supported": lambda: torch.cuda.is_bf16_supported(),
    "empty": lambda: torch.empty(1, device="cuda"),
    "tensor_cuda": lambda: torch.zeros(1).cuda(),
    "tensor_to": lambda: torch.zeros(1).to("cuda"),
    "module_cuda": lambda: torch.nn.Linear(1, 1).cuda(),
    "current_stream": lambda: torch.cuda.current_stream(),
    "default_stream": lambda: torch.cuda.default_stream(),
    "synchronize": lambda: torch.cuda.synchronize(),
    "get_rng_state": lambda: torch.cuda.get_rng_state(),
    "initial_seed": lambda: torch.cuda.initial_seed(),
    "mem_get_info": lambda: torch.cuda.mem_get_info(),
}
first = True
for name, call in initialize.items():
    upstream._INITIALIZED = False
    torch.cuda._initialized = False
    call()
    assert torch.cuda.is_initialized() and torch.cuda._initialized, name
    if first:
        # Queued before initialization: manual_seed(5), set_rng_state(zeros),
        # manual_seed_all(9), seed(). torch.cuda._lazy_init runs the plain
        # calls first, then the latest of manual_seed and manual_seed_all.
        assert len(torch.cuda.default_generators) == 1
        state = torch.cuda.default_generators[0].get_state().tolist()
        assert state == [9] + [0] * 15, state
        first = False
print("lazy init passed")
"""
    )


_VERSION_CHECK = """
fakegpu.init(runtime="fakecuda", devices="a100,b200")
import importlib.metadata
import re
import torch

pins = {}
for requirement in importlib.metadata.requires("torch") or []:
    match = re.match(r"([A-Za-z0-9_.-]+)(\\[[^]]*\\])?\\s*\\(?==\\s*([0-9.]+)", requirement)
    if match:
        pins[match.group(1)] = match.group(3)
cuda = next((v for k, v in pins.items() if k.startswith(("nvidia-cuda-runtime", "cuda-toolkit"))), None)
cudnn = next((v for k, v in pins.items() if k.startswith("nvidia-cudnn")), None)
nccl = next((v for k, v in pins.items() if k.startswith("nvidia-nccl")), None)
expected_cuda, expected_cudnn = EXPECTED
if cuda:
    major, minor = cuda.split(".")[:2]
    parts = [int(p) for p in cudnn.split(".")[:3]]
    expected_cuda = expected_cuda or f"{major}.{minor}"
    expected_cudnn = expected_cudnn or parts[0] * (10000 if parts[0] >= 9 else 1000) + parts[1] * 100 + parts[2]
    assert torch.cuda.nccl.version() == tuple(int(p) for p in nccl.split(".")[:3])
if expected_cuda:
    assert torch.version.cuda == expected_cuda, (torch.version.cuda, expected_cuda)
    assert torch.backends.cudnn.version() == expected_cudnn, torch.backends.cudnn.version()
assert isinstance(torch.backends.cudnn.version(), int) and torch.backends.cudnn.is_available()
assert len(torch.cuda.nccl.version()) == 3

arch = torch.cuda.get_arch_list()
assert "sm_80" in arch and "sm_100" in arch, arch
assert torch._C._cuda_getArchFlags().split() == arch
assert torch.cuda.get_gencode_flags() == " ".join(
    f"-gencode compute=compute_{a.split('_')[1]},code={a.split('_')[0]}_{a.split('_')[1]}" for a in arch
)
print("versions passed")
"""


def test_cuda_versions_follow_the_installed_torch_wheel() -> None:
    # torch 2.9.1+cu128 on an Ampere GPU reports torch.version.cuda "12.8",
    # cudnn.version() 91002 (CUDNN_VERSION = 9*10000 + 10*100 + 2) and
    # nccl.version() (2, 27, 5): the wheel's requires_dist pins of
    # nvidia-cuda-runtime-cu12 12.8.90, nvidia-cudnn-cu12 9.10.2.21 and
    # nvidia-nccl-cu12 2.27.5. get_gencode_flags derives from
    # get_arch_list (torch/cuda/__init__.py).
    import torch

    if torch.version.cuda is not None:
        return  # a CUDA build reports its own versions
    _run(_VERSION_CHECK.replace("EXPECTED", "(None, None)"))
    os.environ["FAKEGPU_CUDA_VERSION"] = "12.6"
    os.environ["FAKEGPU_CUDNN_VERSION"] = "9.5.1"
    try:
        _run(_VERSION_CHECK.replace("EXPECTED", '("12.6", 90501)'))
    finally:
        del os.environ["FAKEGPU_CUDA_VERSION"]
        del os.environ["FAKEGPU_CUDNN_VERSION"]


def test_cuda_stack_falls_back_to_the_pypi_wheel_of_the_torch_minor() -> None:
    # A CPU-only wheel pins no CUDA packages; the default PyPI wheel of the
    # same minor decides (torch 2.9.1 requires nvidia-cuda-runtime-cu12
    # 12.8.90, nvidia-cudnn-cu12 9.10.2.21, nvidia-nccl-cu12 2.27.5).
    from types import SimpleNamespace

    from fakegpu import torch_patch

    fake_torch = SimpleNamespace(__version__="2.9.1+cpu")
    original_pins = torch_patch._installed_torch_pins
    original_native = torch_patch._native_torch_cuda
    torch_patch._installed_torch_pins = lambda: {}
    torch_patch._native_torch_cuda = None
    try:
        assert torch_patch._emulated_cuda_stack(fake_torch) == torch_patch._CudaStack(
            cuda="12.8", cudnn=91002, nccl=(2, 27, 5)
        )
        # cudnn_version.h of cuDNN 8: CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + PATCH.
        fake_torch.__version__ = "2.3.1"
        assert torch_patch._emulated_cuda_stack(fake_torch).cudnn == 8902
    finally:
        torch_patch._installed_torch_pins = original_pins
        torch_patch._native_torch_cuda = original_native


def test_grad_scaler_keeps_torch_scale_and_growth() -> None:
    # Measured on an Ampere GPU: GradScaler("cuda") is enabled with scale
    # 65536.0, growth 2.0, backoff 0.5, interval 2000 and a float32 scale
    # tensor on cuda:0; an inf gradient skips the step and halves the scale;
    # a second unscale_ raises. Growth after growth_interval clean steps is
    # torch's _amp_update_scale_.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:1")
import torch

assert torch.amp.GradScaler is torch.amp.grad_scaler.GradScaler
scaler = torch.amp.GradScaler("cuda")
assert scaler.is_enabled() and scaler.get_scale() == 65536.0
assert (scaler.get_growth_factor(), scaler.get_backoff_factor(), scaler.get_growth_interval()) == (2.0, 0.5, 2000)
assert scaler._growth_tracker is None and scaler._scale is None
legacy = torch.cuda.amp.GradScaler()
assert legacy.is_enabled() and legacy.get_scale() == 65536.0

weight = torch.nn.Parameter(torch.ones(2, device="cuda"))
opt = torch.optim.SGD([weight], lr=0.5)
scaler = torch.amp.GradScaler("cuda", init_scale=8.0, growth_interval=2)
history = []
for step, value in enumerate([1.0, 1.0, float("inf"), 1.0, 1.0, 1.0]):
    opt.zero_grad()
    scaled = scaler.scale((weight * value).sum())
    if step == 0:
        assert (scaled.dtype, scaled.device) == (torch.float32, torch.device("cuda", 0))
        scale = scaler._scale
        assert (scale.dtype, scale.device, tuple(scale.shape)) == (torch.float32, torch.device("cuda", 0), ())
    scaled.backward()
    before = weight.detach().clone()
    if step == 1:
        scaler.unscale_(opt)
        assert torch.equal(weight.grad, torch.ones(2, device="cuda"))
        try:
            scaler.unscale_(opt)
        except RuntimeError as exc:
            assert "unscale_() has already been called on this optimizer since the last update()." in str(exc)
        else:
            raise AssertionError("second unscale_ accepted")
    scaler.step(opt)
    stepped = not torch.equal(before, weight.detach())
    scaler.update()
    history.append((stepped, scaler.get_scale(), int(scaler._growth_tracker.item())))
assert history == [
    (True, 8.0, 1), (True, 16.0, 0), (False, 8.0, 0),
    (True, 8.0, 1), (True, 16.0, 0), (True, 16.0, 1),
], history
scaler.update(1024.0)
assert scaler.get_scale() == 1024.0
print("grad scaler passed")
"""
    )


def test_to_returns_self_only_when_nothing_changes() -> None:
    # Measured on an Ampere GPU: Tensor.to and .cuda return the tensor itself
    # for the same device, dtype and memory format without copy=True
    # (to_will_alias in aten/src/ATen/native/TensorConversions.cpp); every
    # other result, a move between devices included, is a new tensor.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:2")
import torch

t = torch.ones(2, device="cuda")
p = torch.nn.Parameter(torch.ones(2, device="cuda"))
for same in (
    lambda: t.to("cuda"), lambda: t.to("cuda:0"), lambda: t.to(torch.device("cuda", 0)),
    lambda: t.to(0), lambda: t.to(torch.float32), lambda: t.to("cuda", torch.float32),
    lambda: t.to(device="cuda", dtype=torch.float32), lambda: t.to("cuda", non_blocking=True),
    lambda: t.to(torch.ones(1, device="cuda")), lambda: t.to(memory_format=torch.preserve_format),
    lambda: t.to(memory_format=torch.contiguous_format), lambda: t.cuda(), lambda: t.cuda(0),
    lambda: t.cuda(non_blocking=True), lambda: t.cuda("cuda:0"), lambda: t.to(),
):
    assert same() is t
assert p.to("cuda") is p and p.cuda() is p
assert t.to("cuda", copy=True) is not t and t.to(torch.float16) is not t

host = torch.ones(2)
device = host.cuda()
device.add_(1)
assert host.tolist() == [1.0, 1.0]
for back in (t.cpu(), t.to("cpu")):
    back.add_(5)
assert t.tolist() == [1.0, 1.0]
other = t.to("cuda:1")
other.add_(1)
assert t.tolist() == [1.0, 1.0] and other.device == torch.device("cuda", 1)

leaf = torch.ones(2, requires_grad=True)
leaf.cuda().sum().backward()
assert leaf.grad.tolist() == [1.0, 1.0]
print("to identity passed")
"""
    )


def test_pin_memory_returns_a_pinned_copy() -> None:
    # Measured on an Ampere GPU: pin_memory() returns a new pinned tensor
    # (views of it are pinned, clones are not, pinning it again returns it),
    # keeps requires_grad, dtype and strides; pinning a CUDA tensor raises
    # (aten/src/ATen/native/Memory.cpp _pin_memory).
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:1")
import torch

source = torch.ones(3)
pinned = source.pin_memory()
assert pinned is not source and pinned.is_pinned() and not source.is_pinned()
assert pinned.device.type == "cpu" and torch.equal(pinned, source)
assert pinned.data_ptr() != source.data_ptr()
assert pinned.pin_memory() is pinned and pinned[1:].is_pinned()
assert not pinned.clone().is_pinned() and pinned.to("cpu") is pinned
assert torch.ones(2, requires_grad=True).pin_memory().requires_grad
assert torch.ones(2, dtype=torch.float16).pin_memory().dtype == torch.float16
assert not torch.ones(3, 4).t().pin_memory().is_contiguous()
assert type(torch.nn.Parameter(torch.ones(2)).pin_memory()) is torch.Tensor
assert torch.ones(2, 2).to_sparse().pin_memory().is_pinned()
assert pinned.cuda(non_blocking=True).device == torch.device("cuda", 0)

cuda = torch.ones(2, device="cuda")
assert not cuda.is_pinned()
try:
    cuda.pin_memory()
except RuntimeError as exc:
    assert str(exc) == "cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned", exc
else:
    raise AssertionError("pinned a CUDA tensor")

from torch.utils.data import DataLoader

loader = DataLoader([torch.ones(2), torch.ones(2)], batch_size=2, pin_memory=True)
assert next(iter(loader)).is_pinned()
print("pin memory passed")
"""
    )


def test_stream_priorities_clamp_to_priority_range() -> None:
    # Measured on an Ampere GPU: priority_range() is (0, -3) (the driver
    # reports (0, -5); c10/cuda/CUDAStream.h keeps four pools) and a stream's
    # priority is clamped into it.
    _run(
        """
fakegpu.init(runtime="fakecuda", devices="a100:1")
import torch

assert torch.cuda.Stream.priority_range() == (0, -3)
assert torch.cuda.Stream().priority_range() == (0, -3)
requested = (1, 5, 0, -1, -2, -3, -4, -5, -10)
assert [torch.cuda.Stream(priority=p).priority for p in requested] == [0, 0, 0, -1, -2, -3, -3, -3, -3]
assert torch.cuda.default_stream().priority == 0 and torch.cuda.current_stream().priority == 0
try:
    torch.cuda.Stream(priority=1.5)
except TypeError as exc:
    assert "'float' object cannot be interpreted as an integer" in str(exc)
else:
    raise AssertionError("float priority accepted")
print("stream priority passed")
"""
    )
