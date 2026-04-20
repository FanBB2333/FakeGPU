# ============================================================================
# Vendored upstream: FakeCudaTensor backend from pytorch-fakegpu
# ============================================================================
#
# Source : https://github.com/FanBB2333/pytorch-fakegpu
#          (fork of PyTorch release/2.11, file: torch/fakegpu.py)
# Vendored: 2026-04-17
#
# This file is a verbatim copy of the upstream FakeCudaTensor implementation.
# Do NOT modify it — apply enhancements in torch_patch.py instead, so that
# re-syncing from upstream remains a simple file replacement.
#
# The upstream module provides:
#   - FakeCudaTensor subclass with __torch_function__ protocol
#   - tensor.device == cuda:N  /  tensor.is_cuda == True
#   - DataParallel / DistributedDataParallel / torch.distributed stubs
#   - wrap_tensor() / unwrap_tensor() / enable()
# ============================================================================

from __future__ import annotations

import functools
import os
import weakref
from collections import OrderedDict
from typing import Any

import torch

_ENABLED = False
_CURRENT_DEVICE = 0
_NUM_DEVICES = int(
    os.environ.get("TORCH_FAKEGPU_DEVICE_COUNT", os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
)
_DEVICE_NAME = os.environ.get("TORCH_FAKEGPU_DEVICE_NAME", "FakeGPU CUDA Device")
_REGISTERED_PARAMETERS: "weakref.WeakSet[FakeCudaTensor]" = weakref.WeakSet()

_ORIG_TORCH_LOAD = None
_ORIG_TENSOR_TO = None
_ORIG_TENSOR_CPU = None
_ORIG_MODULE_TO = None
_ORIG_DEVICE_PROP = None
_ORIG_IS_CUDA_PROP = None
_DIST_INITIALIZED = False
_DIST_BACKEND = "nccl"
_DIST_WORLD_SIZE = 1
_DIST_RANK = 0
_ORIG_DIST_INIT: Any = None
_ORIG_DIST_DESTROY: Any = None
_DEFAULT_STREAMS = {}
_CURRENT_STREAMS = {}
_NEXT_STREAM_ID = 1


def _normalize_cuda_device(device: Any, *, allow_none: bool = False) -> torch.device | None:
    if device is None:
        if allow_none:
            return None
        return torch.device(f"cuda:{_CURRENT_DEVICE}")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.device(f"cuda:{_CURRENT_DEVICE if device.index is None else device.index}")
    return None


def _normalize_device_index(device: Any) -> int:
    normalized = _normalize_cuda_device(device)
    if normalized is None:
        return _CURRENT_DEVICE
    index = 0 if normalized.index is None else int(normalized.index)
    if index < 0 or index >= _NUM_DEVICES:
        raise RuntimeError(f"Invalid fake CUDA device index {index}")
    return index


class _FakeDeviceProperties:
    def __init__(self, index: int):
        self.name = _DEVICE_NAME
        self.major = 8
        self.minor = 0
        self.total_memory = 80 * 1024**3
        self.multi_processor_count = 108
        self.is_multi_gpu_board = False
        self.is_integrated = False
        self.max_threads_per_multi_processor = 2048
        self.max_threads_per_block = 1024
        self.regs_per_block = 65536
        self.regs_per_multiprocessor = 65536
        self.warp_size = 32
        self.gcnArchName = ""
        self.index = index


class _FakeDeviceCtx:
    def __init__(self, device: Any):
        self.device_index = _normalize_device_index(device)
        self.prev_index = _CURRENT_DEVICE

    def __enter__(self):
        set_device(self.device_index)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_device(self.prev_index)
        return None


class _FakeDeviceOfCtx:
    def __init__(self, obj: Any):
        self.device_index = _find_device_index(obj)
        self.prev_index = _CURRENT_DEVICE

    def __enter__(self):
        if self.device_index is not None:
            set_device(self.device_index)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_device(self.prev_index)
        return None


def _next_stream_id() -> int:
    global _NEXT_STREAM_ID
    stream_id = _NEXT_STREAM_ID
    _NEXT_STREAM_ID += 1
    return stream_id


class _FakeStream:
    def __init__(
        self,
        device: Any = None,
        priority: int = 0,
        stream_id: int | None = None,
        **kwargs: Any,
    ):
        self.device_index = _normalize_device_index(device) if device is not None else _CURRENT_DEVICE
        self.stream_id = _next_stream_id() if stream_id is None else int(stream_id)
        self.cuda_stream = self.stream_id

    def synchronize(self) -> None:
        return None

    def wait_stream(self, stream: Any) -> None:
        return None

    def wait_event(self, event: Any) -> None:
        return None

    def record_event(self, event: Any = None) -> Any:
        return _FakeEvent() if event is None else event

    def query(self) -> bool:
        return True

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeEvent:
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self._time = 0.0

    def record(self, stream: Any = None) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def wait(self, stream: Any = None) -> None:
        return None

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return 0.0


class _FakeStreamCtx:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.prev_stream = None

    def __enter__(self):
        self.prev_stream = _current_stream(device=self.stream.device_index)
        _set_stream(self.stream)
        return self.stream

    def __exit__(self, *args: Any) -> None:
        if self.prev_stream is not None:
            _set_stream(self.prev_stream)
        return None


class _FakeDataParallel(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Any = None,
        output_device: Any = None,
        dim: int = 0,
    ) -> None:
        super().__init__()
        self.module = module
        self.dim = dim
        self.device_ids = (
            list(range(device_count())) if device_ids is None else [_normalize_device_index(d) for d in device_ids]
        )
        self.output_device = (
            self.device_ids[0] if output_device is None else _normalize_device_index(output_device)
        )
        self.src_device_obj = torch.device(f"cuda:{self.device_ids[0] if self.device_ids else 0}")

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)


def _fake_data_parallel(
    module: torch.nn.Module,
    inputs: Any,
    device_ids: Any = None,
    output_device: Any = None,
    dim: int = 0,
    module_kwargs: Any | None = None,
) -> Any:
    if not isinstance(inputs, tuple):
        inputs = () if inputs is None else (inputs,)
    if module_kwargs is None:
        module_kwargs = {}
    return module(*inputs, **module_kwargs)


class _FakeDistributedDataParallel(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Any = None,
        output_device: Any = None,
        dim: int = 0,
        process_group: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.module = module
        self.dim = dim
        self.device_ids = (
            None if device_ids is None else [_normalize_device_index(d) for d in device_ids]
        )
        self.output_device = (
            None if output_device is None else _normalize_device_index(output_device)
        )
        self.process_group = process_group

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)


class _FakeAsyncWork:
    def wait(self, timeout: Any = None) -> bool:
        return True


def _split_sizes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _fake_comm_broadcast(
    tensor: torch.Tensor,
    devices: Any = None,
    *,
    out: Any = None,
) -> tuple[torch.Tensor, ...]:
    cpu_tensor = unwrap_tensor(tensor)
    if out is not None:
        for target in out:
            unwrap_tensor(target).copy_(cpu_tensor)
        return tuple(out)
    device_indices = [_normalize_device_index(device) for device in devices]
    return tuple(wrap_tensor(cpu_tensor.clone(), device_index=device_index) for device_index in device_indices)


def _fake_comm_broadcast_coalesced(
    tensors: Any,
    devices: Any,
    buffer_size: int = 10485760,
) -> list[list[torch.Tensor]]:
    device_indices = [_normalize_device_index(device) for device in devices]
    return [
        [wrap_tensor(unwrap_tensor(tensor).clone(), device_index=device_index) for tensor in tensors]
        for device_index in device_indices
    ]


def _fake_comm_scatter(
    tensor: torch.Tensor,
    devices: Any = None,
    chunk_sizes: Any = None,
    dim: int = 0,
    streams: Any = None,
    *,
    out: Any = None,
) -> tuple[torch.Tensor, ...]:
    cpu_tensor = unwrap_tensor(tensor)
    if out is not None:
        chunk_sizes = [unwrap_tensor(item).size(dim) for item in out]
        chunks = list(torch.split(cpu_tensor, chunk_sizes, dim=dim))
        for target, chunk in zip(out, chunks):
            unwrap_tensor(target).copy_(chunk)
        return tuple(out)

    device_indices = [_normalize_device_index(device) for device in devices]
    if chunk_sizes is None:
        chunk_sizes = _split_sizes(cpu_tensor.size(dim), len(device_indices))
    chunks = list(torch.split(cpu_tensor, chunk_sizes, dim=dim))
    return tuple(
        wrap_tensor(chunk, device_index=device_index)
        for chunk, device_index in zip(chunks, device_indices)
    )


def _fake_comm_gather(
    tensors: Any,
    dim: int = 0,
    destination: Any = None,
    *,
    out: Any = None,
) -> torch.Tensor:
    cpu_tensors = [unwrap_tensor(tensor) for tensor in tensors]
    result = torch.cat(cpu_tensors, dim=dim)
    if out is not None:
        unwrap_tensor(out).copy_(result)
        return out
    if destination == "cpu" or destination == torch.device("cpu") or destination == -1:
        return result
    return wrap_tensor(result, device_index=_normalize_device_index(destination))


def _fake_comm_reduce_add(inputs: Any, destination: Any = None) -> torch.Tensor:
    cpu_tensors = [unwrap_tensor(tensor) for tensor in inputs]
    result = cpu_tensors[0].clone()
    for other in cpu_tensors[1:]:
        result.add_(other)
    return wrap_tensor(result, device_index=_normalize_device_index(destination))


def _fake_comm_reduce_add_coalesced(
    inputs: Any,
    destination: Any = None,
    buffer_size: int = 10485760,
) -> tuple[torch.Tensor, ...]:
    outputs = []
    for tensor_group in zip(*inputs):
        outputs.append(_fake_comm_reduce_add(tensor_group, destination=destination))
    return tuple(outputs)


def _dist_is_available() -> bool:
    return True


def _dist_is_nccl_available() -> bool:
    return True


def _dist_is_initialized() -> bool:
    return _DIST_INITIALIZED


def _dist_init_process_group(
    backend: str | None = None,
    init_method: Any = None,
    timeout: Any = None,
    world_size: int = -1,
    rank: int = -1,
    store: Any = None,
    group_name: str = "",
    pg_options: Any = None,
    device_id: Any = None,
) -> None:
    global _DIST_INITIALIZED, _DIST_BACKEND, _DIST_WORLD_SIZE, _DIST_RANK
    _DIST_INITIALIZED = True
    _DIST_BACKEND = "nccl" if backend is None else str(backend)
    _DIST_WORLD_SIZE = 1 if world_size in (-1, None) else int(world_size)
    _DIST_RANK = 0 if rank in (-1, None) else int(rank)

    # Create a real ProcessGroup using PyTorch's ``fake`` distributed backend.
    # FSDP and other distributed modules call ``_get_default_group()`` which
    # requires a real C++ ProcessGroup instance — our global-state-only
    # approach is not sufficient.  The ``fake`` backend (PyTorch 2.1+)
    # provides an in-process ProcessGroup that satisfies all type checks
    # without needing NCCL or gloo.
    if _ORIG_DIST_INIT is not None:
        import os

        _env_set: list[str] = []
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            _env_set.append("MASTER_ADDR")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            _env_set.append("MASTER_PORT")
        try:
            _ORIG_DIST_INIT(
                backend="fake",
                rank=_DIST_RANK,
                world_size=_DIST_WORLD_SIZE,
            )
        except Exception:
            pass  # Fall back to global-state-only mode
        finally:
            for key in _env_set:
                os.environ.pop(key, None)
    return None


def _dist_destroy_process_group(group: Any = None) -> None:
    global _DIST_INITIALIZED, _DIST_WORLD_SIZE, _DIST_RANK
    if _ORIG_DIST_DESTROY is not None:
        try:
            _ORIG_DIST_DESTROY(group)
        except Exception:
            pass
    _DIST_INITIALIZED = False
    _DIST_WORLD_SIZE = 1
    _DIST_RANK = 0
    return None


def _dist_get_backend(group: Any = None) -> str | None:
    return _DIST_BACKEND if _DIST_INITIALIZED else None


def _dist_get_world_size(group: Any = None) -> int:
    return _DIST_WORLD_SIZE if _DIST_INITIALIZED else 1


def _dist_get_rank(group: Any = None) -> int:
    return _DIST_RANK if _DIST_INITIALIZED else 0


def _dist_barrier(group: Any = None, async_op: bool = False, device_ids: Any = None) -> Any:
    return _FakeAsyncWork() if async_op else None


def _dist_all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None, async_op: bool = False) -> Any:
    return _FakeAsyncWork() if async_op else None


def _dist_broadcast(
    tensor: torch.Tensor,
    src: int | None = None,
    group: Any = None,
    async_op: bool = False,
    group_src: Any = None,
) -> Any:
    return _FakeAsyncWork() if async_op else None


def _dist_all_gather(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    source = unwrap_tensor(tensor)
    for target in tensor_list:
        unwrap_tensor(target).copy_(source)
    return _FakeAsyncWork() if async_op else None


def _copy_collective_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    dst = unwrap_tensor(destination)
    src = unwrap_tensor(source)
    if dst.numel() == src.numel():
        dst.copy_(src.reshape_as(dst))
        return
    if dst.numel() < src.numel():
        # reduce_scatter: output is smaller than input — take first chunk
        dst.copy_(src.reshape(-1)[: dst.numel()].reshape_as(dst))
    else:
        # all_gather: output is larger than input — replicate input to fill
        flat_dst = dst.reshape(-1)
        flat_src = src.reshape(-1)
        src_numel = flat_src.numel()
        for offset in range(0, flat_dst.numel(), src_numel):
            end = min(offset + src_numel, flat_dst.numel())
            flat_dst[offset:end].copy_(flat_src[: end - offset])


def _dist_all_gather_object(object_list: list[Any], obj: Any, group: Any = None) -> None:
    for index in range(len(object_list)):
        object_list[index] = obj
    return None


def _dist_all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    _copy_collective_tensor(output_tensor, input_tensor)
    return _FakeAsyncWork() if async_op else None


def _dist_reduce(
    tensor: torch.Tensor,
    dst: int,
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
    group_dst: Any = None,
) -> Any:
    return _FakeAsyncWork() if async_op else None


def _dist_gather(
    tensor: torch.Tensor,
    gather_list: list[torch.Tensor] | None = None,
    dst: int = 0,
    group: Any = None,
    async_op: bool = False,
    group_dst: Any = None,
) -> Any:
    if gather_list:
        unwrap_tensor(gather_list[0]).copy_(unwrap_tensor(tensor))
    return _FakeAsyncWork() if async_op else None


def _dist_scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = None,
    src: int = 0,
    group: Any = None,
    async_op: bool = False,
    group_src: Any = None,
) -> Any:
    if scatter_list:
        unwrap_tensor(tensor).copy_(unwrap_tensor(scatter_list[0]))
    return _FakeAsyncWork() if async_op else None


def _dist_reduce_scatter(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if input_list:
        _copy_collective_tensor(output, input_list[0])
    return _FakeAsyncWork() if async_op else None


def _dist_reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    _copy_collective_tensor(output, input)
    return _FakeAsyncWork() if async_op else None


def _dist_all_to_all(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Any = None,
    async_op: bool = False,
) -> Any:
    for output, source in zip(output_tensor_list, input_tensor_list):
        unwrap_tensor(output).copy_(unwrap_tensor(source))
    return _FakeAsyncWork() if async_op else None


def _dist_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Any = None,
    input_split_sizes: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    unwrap_tensor(output).copy_(unwrap_tensor(input))
    return _FakeAsyncWork() if async_op else None


def _dist_broadcast_object_list(
    object_list: list[Any],
    src: int = 0,
    group: Any = None,
    device: Any = None,
    group_src: Any = None,
) -> None:
    return None


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, device_index=None):
        out = torch.Tensor._make_subclass(cls, raw_data, raw_data.requires_grad if raw_data is not None else False)
        out.raw_data = raw_data
        out.device_index = _CURRENT_DEVICE if device_index is None else int(device_index)
        return out

    def __init__(self, size, dtype, raw_data=None, device_index=None):
        return None

    @property
    def device(self):
        return torch.device(f"cuda:{self.device_index}")

    @property
    def is_cuda(self):
        return True

    @property
    def grad(self):
        raw = getattr(self.raw_data, "grad", None)
        if raw is None:
            self.__dict__.pop("_fakecuda_grad", None)
            return None
        cached = self.__dict__.get("_fakecuda_grad")
        if cached is not None and getattr(cached, "raw_data", None) is raw:
            return cached
        wrapped = wrap_tensor(raw.detach(), device_index=self.device_index)
        self.__dict__["_fakecuda_grad"] = wrapped
        return wrapped

    @grad.setter
    def grad(self, value) -> None:
        if value is None:
            if getattr(self, "raw_data", None) is not None:
                self.raw_data.grad = None
            self.__dict__.pop("_fakecuda_grad", None)
            return
        wrapped = value if isinstance(value, FakeCudaTensor) else wrap_tensor(value, device_index=self.device_index)
        self.__dict__["_fakecuda_grad"] = wrapped
        if getattr(self, "raw_data", None) is not None:
            self.raw_data.grad = unwrap_tensor(wrapped)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        device_index = _infer_device_index(args, kwargs)
        cpu_args = _tree_map(unwrap_tensor, args)
        cpu_kwargs = _tree_map(unwrap_tensor, kwargs)
        result = func(*cpu_args, **cpu_kwargs)
        return _tree_map(lambda obj: _wrap_function_result(obj, device_index), result)

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        self.raw_data.backward(
            gradient=unwrap_tensor(gradient),
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )
        sync_parameter_grads()

    def record_stream(self, stream) -> None:
        """No-op for FakeCudaTensor — FSDP calls this during post-backward."""
        pass


def wrap_tensor(t: torch.Tensor, device_index: int | None = None) -> FakeCudaTensor:
    if isinstance(t, FakeCudaTensor):
        return t
    return FakeCudaTensor(
        tuple(t.shape),
        t.dtype,
        t,
        device_index=_CURRENT_DEVICE if device_index is None else int(device_index),
    )


def unwrap_tensor(obj):
    if isinstance(obj, FakeCudaTensor):
        return obj.raw_data
    return obj


def register_parameter(t: FakeCudaTensor) -> None:
    _REGISTERED_PARAMETERS.add(t)


def sync_parameter_grads() -> None:
    for param in list(_REGISTERED_PARAMETERS):
        raw = getattr(param, "raw_data", None)
        if raw is None:
            continue
        if raw.grad is None:
            param.grad = None
            continue
        param.grad = wrap_tensor(raw.grad.detach(), device_index=getattr(param, "device_index", _CURRENT_DEVICE))


def _tree_map(fn, obj):
    # torch.Size is a tuple subclass but should be preserved as-is.
    # Its elements are plain ints (no tensors to unwrap/wrap).
    if isinstance(obj, torch.Size):
        return obj
    if isinstance(obj, tuple):
        return tuple(_tree_map(fn, item) for item in obj)
    if isinstance(obj, list):
        return [_tree_map(fn, item) for item in obj]
    if isinstance(obj, OrderedDict):
        return OrderedDict((key, _tree_map(fn, value)) for key, value in obj.items())
    if isinstance(obj, dict):
        return obj.__class__((key, _tree_map(fn, value)) for key, value in obj.items())
    return fn(obj)


def _wrap_function_result(obj, device_index):
    if isinstance(obj, torch.Tensor):
        return wrap_tensor(obj, device_index=device_index)
    return obj


def _infer_device_index(args, kwargs) -> int:
    for value in list(args) + list(kwargs.values()):
        index = _find_device_index(value)
        if index is not None:
            return index
    target = kwargs.get("device")
    normalized = _normalize_cuda_device(target, allow_none=True)
    if normalized is not None:
        return _normalize_device_index(normalized)
    return _CURRENT_DEVICE


def _find_device_index(value):
    if isinstance(value, FakeCudaTensor):
        return value.device_index
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
        return 0 if value.device.index is None else int(value.device.index)
    if isinstance(value, tuple):
        for item in value:
            index = _find_device_index(item)
            if index is not None:
                return index
        return None
    if isinstance(value, list):
        for item in value:
            index = _find_device_index(item)
            if index is not None:
                return index
        return None
    if isinstance(value, dict):
        for item in value.values():
            index = _find_device_index(item)
            if index is not None:
                return index
        return None
    return None


def _normalize_to_args(args, kwargs):
    target = kwargs.get("device")
    consume_first_arg = False
    if target is None and args:
        candidate = args[0]
        if isinstance(candidate, torch.Tensor):
            target = candidate.device
            consume_first_arg = True
        elif isinstance(candidate, (int, str, torch.device)):
            target = candidate
            consume_first_arg = True
    normalized = _normalize_cuda_device(target, allow_none=True)
    return target, normalized, consume_first_arg


def _factory_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        target = kwargs.get("device")
        normalized = _normalize_cuda_device(target, allow_none=True)
        if normalized is None:
            return fn(*args, **kwargs)
        cpu_args = _tree_map(unwrap_tensor, args)
        cpu_kwargs = _tree_map(unwrap_tensor, kwargs)
        kwargs = dict(kwargs)
        cpu_kwargs["device"] = torch.device("cpu")
        result = fn(*cpu_args, **cpu_kwargs)
        return _tree_map(lambda obj: _wrap_function_result(obj, normalized.index or 0), result)

    return wrapper


def _install_legacy_cuda_types() -> None:
    _LEGACY_TYPES = {
        "FloatTensor": torch.float32,
        "DoubleTensor": torch.float64,
        "HalfTensor": torch.float16,
        "BFloat16Tensor": torch.bfloat16,
        "IntTensor": torch.int32,
        "LongTensor": torch.int64,
        "ShortTensor": torch.int16,
        "ByteTensor": torch.uint8,
        "CharTensor": torch.int8,
        "BoolTensor": torch.bool,
    }

    def _make_legacy_factory(dtype: Any) -> type:
        _dtype = dtype

        class _LegacyCudaTensor:
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                kwargs = dict(kwargs)
                kwargs["device"] = torch.device(f"cuda:{_CURRENT_DEVICE}")
                if args and not isinstance(args[0], int):
                    return torch.tensor(args[0], dtype=_dtype, **kwargs)
                return torch.empty(*args, dtype=_dtype, **kwargs)

        return _LegacyCudaTensor

    for name, dtype in _LEGACY_TYPES.items():
        setattr(torch.cuda, name, _make_legacy_factory(dtype))


def _tensor_to(self, *args, **kwargs):
    target, normalized, consume_first_arg = _normalize_to_args(args, kwargs)
    base = self.raw_data if isinstance(self, FakeCudaTensor) else self
    if normalized is not None:
        kwargs = dict(kwargs)
        kwargs["device"] = torch.device("cpu")
        call_args = args[1:] if consume_first_arg else args
        result = _ORIG_TENSOR_TO(base, *call_args, **kwargs)
        return wrap_tensor(result, device_index=normalized.index or 0)
    result = _ORIG_TENSOR_TO(base, *args, **kwargs)
    if isinstance(self, FakeCudaTensor) and isinstance(result, torch.Tensor):
        if target is not None:
            target_device = (
                torch.device(f"cuda:{target}") if isinstance(target, int) else torch.device(target)
            )
            if target_device.type != "cuda":
                return result
        return wrap_tensor(result, device_index=self.device_index)
    return result


def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
    target = torch.device(f"cuda:{_normalize_device_index(device)}")
    kwargs = {"device": target, "non_blocking": non_blocking}
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    return _tensor_to(self, **kwargs)


def _tensor_cpu(self, memory_format=None):
    kwargs = {}
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    base = self.raw_data if isinstance(self, FakeCudaTensor) else self
    return _ORIG_TENSOR_CPU(base, **kwargs)


def _normalize_module_device(device) -> torch.device:
    normalized = _normalize_cuda_device(device, allow_none=False)
    if normalized is None:
        raise RuntimeError(f"Invalid device for fake CUDA module conversion: {device}")
    return normalized


def _module_cuda_impl(module, device=None, _memo=None):
    target = _normalize_module_device(device)
    if _memo is None:
        _memo = {"parameters": {}, "buffers": {}}

    for child in module.children():
        _module_cuda_impl(child, target, _memo)

    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        param_id = id(param)
        wrapped = _memo["parameters"].get(param_id)
        if wrapped is None:
            converted = param.detach().clone().requires_grad_(param.requires_grad)
            wrapped = torch.nn.Parameter(wrap_tensor(converted, device_index=target.index or 0), requires_grad=param.requires_grad)
            wrapped.raw_data = converted
            wrapped.device_index = target.index or 0
            if param.grad is not None:
                wrapped.grad = wrap_tensor(param.grad.detach(), device_index=target.index or 0)
            register_parameter(wrapped)
            _memo["parameters"][param_id] = wrapped
        module._parameters[name] = wrapped

    for name, buf in list(module._buffers.items()):
        if buf is None:
            continue
        buf_id = id(buf)
        converted = _memo["buffers"].get(buf_id)
        if converted is None:
            converted = wrap_tensor(buf.detach().clone(), device_index=target.index or 0)
            converted.raw_data = buf.detach().clone()
            converted.device_index = target.index or 0
            _memo["buffers"][buf_id] = converted
        module._buffers[name] = converted

    return module


def _module_to(self, *args, **kwargs):
    target = kwargs.get("device")
    if target is None and args:
        target = args[0]
    normalized = _normalize_cuda_device(target, allow_none=True)
    if normalized is not None:
        return _module_cuda_impl(self, normalized)
    return _ORIG_MODULE_TO(self, *args, **kwargs)


def _module_cuda(self, device=None):
    return _module_cuda_impl(self, device)


def is_available() -> bool:
    return True


def device_count() -> int:
    return _NUM_DEVICES


def current_device() -> int:
    return _CURRENT_DEVICE


def set_device(device: Any) -> None:
    global _CURRENT_DEVICE
    _CURRENT_DEVICE = _normalize_device_index(device)


def get_device_name(device: Any = None) -> str:
    return _DEVICE_NAME


def get_device_capability(device: Any = None) -> tuple[int, int]:
    return (8, 0)


def get_device_properties(device: Any = None) -> _FakeDeviceProperties:
    return _FakeDeviceProperties(_normalize_device_index(device) if device is not None else _CURRENT_DEVICE)


def synchronize(device: Any = None) -> None:
    return None


def empty_cache() -> None:
    return None


def memory_allocated(device: Any = None) -> int:
    return 0


def memory_reserved(device: Any = None) -> int:
    return 0


def mem_get_info(device: Any = None) -> tuple[int, int]:
    total = 80 * 1024**3
    return (total, total)


def max_memory_allocated(device: Any = None) -> int:
    return 0


def max_memory_reserved(device: Any = None) -> int:
    return 0


def memory_cached(device: Any = None) -> int:
    return 0


def max_memory_cached(device: Any = None) -> int:
    return 0


def reset_peak_memory_stats(device: Any = None) -> None:
    return None


def reset_max_memory_allocated(device: Any = None) -> None:
    return None


def reset_max_memory_cached(device: Any = None) -> None:
    return None


def reset_accumulated_memory_stats(device: Any = None) -> None:
    return None


def memory_stats(device: Any = None) -> dict[str, Any]:
    return {}


def memory_summary(device: Any = None, abbreviated: bool = False) -> str:
    return "FakeGPU: no real CUDA memory to report.\n"


def memory_snapshot(*args: Any, **kwargs: Any) -> list[Any]:
    return []


def get_arch_list() -> list[str]:
    return ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def get_gencode_flags() -> str:
    return ""


def _cpu_default_generator() -> Any:
    return torch.random.default_generator


def manual_seed(seed: int) -> None:
    _cpu_default_generator().manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    _cpu_default_generator().manual_seed(int(seed))


def seed() -> None:
    _cpu_default_generator().seed()


def seed_all() -> None:
    _cpu_default_generator().seed()


def initial_seed() -> int:
    return int(_cpu_default_generator().initial_seed())


def is_bf16_supported(device: Any = None) -> bool:
    return True


def ipc_collect() -> None:
    return None


def can_device_access_peer(device: int, peer_device: int) -> bool:
    return True


def cudart() -> None:
    return None


def _stream(stream: Any) -> _FakeStreamCtx:
    return _FakeStreamCtx(stream)


def _default_stream(device: Any = None) -> _FakeStream:
    device_index = _normalize_device_index(device)
    stream = _DEFAULT_STREAMS.get(device_index)
    if stream is None:
        stream = _FakeStream(device=device_index, stream_id=0)
        _DEFAULT_STREAMS[device_index] = stream
    return stream


def _current_stream(device: Any = None) -> _FakeStream:
    device_index = _normalize_device_index(device)
    return _CURRENT_STREAMS.get(device_index, _default_stream(device_index))


def _set_stream(stream: Any) -> None:
    if stream is None:
        return None
    _CURRENT_STREAMS[_normalize_device_index(stream.device_index)] = stream
    return None


def _map_location_target_device_index(map_location: Any) -> int | None:
    normalized = _normalize_cuda_device(map_location, allow_none=True)
    if normalized is not None:
        return _normalize_device_index(normalized)

    if not isinstance(map_location, dict):
        return None

    cpu_targets = []
    all_cuda_targets = set()

    for source, target in map_location.items():
        normalized_target = _normalize_cuda_device(target, allow_none=True)
        if normalized_target is None:
            continue
        target_index = _normalize_device_index(normalized_target)
        all_cuda_targets.add(target_index)
        if source == "cpu" or (isinstance(source, torch.device) and source.type == "cpu"):
            cpu_targets.append(target_index)

    if cpu_targets:
        if len(set(cpu_targets)) != 1:
            raise RuntimeError("Mixed fake CUDA targets for CPU map_location are not supported")
        return cpu_targets[0]

    if len(all_cuda_targets) == 1:
        return next(iter(all_cuda_targets))

    if len(all_cuda_targets) > 1:
        raise RuntimeError("Mixed fake CUDA map_location dict targets are not supported")

    return None


def _patch_torch_load() -> None:
    global _ORIG_TORCH_LOAD
    if _ORIG_TORCH_LOAD is not None:
        return
    _ORIG_TORCH_LOAD = torch.load

    @functools.wraps(_ORIG_TORCH_LOAD)
    def _fakecuda_load(*args, **kwargs):
        map_location = kwargs.get("map_location")
        if map_location is None and len(args) >= 2:
            map_location = args[1]
        target_device_index = _map_location_target_device_index(map_location)
        if target_device_index is not None:
            if len(args) >= 2:
                args = (args[0], "cpu") + args[2:]
            else:
                kwargs["map_location"] = "cpu"
            loaded = _ORIG_TORCH_LOAD(*args, **kwargs)
            return _move_loaded_to_cuda(loaded, device_index=target_device_index)
        return _ORIG_TORCH_LOAD(*args, **kwargs)

    torch.load = _fakecuda_load


def _move_loaded_to_cuda(obj, device_index: int = 0):
    memo = {}
    return _move_loaded_to_cuda_impl(obj, memo, device_index)


def _move_loaded_to_cuda_impl(obj, memo, device_index: int):
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if isinstance(obj, torch.Tensor):
        moved = obj.to(torch.device(f"cuda:{device_index}"))
        memo[obj_id] = moved
        return moved
    if isinstance(obj, list):
        result = []
        memo[obj_id] = result
        result.extend(_move_loaded_to_cuda_impl(item, memo, device_index) for item in obj)
        return result
    if isinstance(obj, OrderedDict):
        result = OrderedDict()
        memo[obj_id] = result
        for key, value in obj.items():
            result[key] = _move_loaded_to_cuda_impl(value, memo, device_index)
        return result
    if isinstance(obj, dict):
        result = obj.__class__()
        memo[obj_id] = result
        for key, value in obj.items():
            result[key] = _move_loaded_to_cuda_impl(value, memo, device_index)
        return result
    if isinstance(obj, tuple):
        result = obj.__class__(*(_move_loaded_to_cuda_impl(item, memo, device_index) for item in obj)) if hasattr(obj, "_fields") else tuple(_move_loaded_to_cuda_impl(item, memo, device_index) for item in obj)
        memo[obj_id] = result
        return result
    return obj


def enable() -> None:
    global _ENABLED, _ORIG_TENSOR_TO, _ORIG_TENSOR_CPU, _ORIG_MODULE_TO, _ORIG_DEVICE_PROP, _ORIG_IS_CUDA_PROP
    if _ENABLED:
        return

    import torch.cuda
    import torch.cuda.graphs
    import torch.cuda.memory
    import torch.cuda.random
    import torch.distributed
    import torch.nn
    import torch.nn.parallel.comm
    import torch.nn.parallel.distributed
    import torch.nn.parallel

    # Save original dist functions before patching — needed for fake backend
    # delegation so FSDP gets a real ProcessGroup.
    global _ORIG_DIST_INIT, _ORIG_DIST_DESTROY
    _ORIG_DIST_INIT = torch.distributed.init_process_group
    _ORIG_DIST_DESTROY = torch.distributed.destroy_process_group
    import torch.nn.parallel.data_parallel

    _ORIG_TENSOR_TO = torch.Tensor.to
    _ORIG_TENSOR_CPU = torch.Tensor.cpu
    _ORIG_MODULE_TO = torch.nn.Module.to
    _ORIG_DEVICE_PROP = torch.Tensor.device
    _ORIG_IS_CUDA_PROP = torch.Tensor.is_cuda

    torch.Tensor.to = _tensor_to
    torch.Tensor.cuda = _tensor_cuda
    torch.Tensor.cpu = _tensor_cpu
    torch.nn.Module.to = _module_to
    torch.nn.Module.cuda = _module_cuda
    torch.nn.DataParallel = _FakeDataParallel
    torch.nn.parallel.DataParallel = _FakeDataParallel
    torch.nn.parallel.data_parallel.DataParallel = _FakeDataParallel
    torch.nn.parallel.data_parallel.data_parallel = _fake_data_parallel
    torch.nn.parallel.DistributedDataParallel = _FakeDistributedDataParallel
    torch.nn.parallel.distributed.DistributedDataParallel = _FakeDistributedDataParallel
    if hasattr(torch.nn.parallel, "DistributedDataParallelCPU"):
        torch.nn.parallel.DistributedDataParallelCPU = _FakeDistributedDataParallel
    torch.nn.parallel.comm.broadcast = _fake_comm_broadcast
    torch.nn.parallel.comm.broadcast_coalesced = _fake_comm_broadcast_coalesced
    torch.nn.parallel.comm.scatter = _fake_comm_scatter
    torch.nn.parallel.comm.gather = _fake_comm_gather
    torch.nn.parallel.comm.reduce_add = _fake_comm_reduce_add
    torch.nn.parallel.comm.reduce_add_coalesced = _fake_comm_reduce_add_coalesced

    torch.cuda.is_available = is_available
    torch.cuda.device_count = device_count
    torch.cuda.current_device = current_device
    torch.cuda.set_device = set_device
    torch.cuda.get_device_name = get_device_name
    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.synchronize = synchronize
    torch.cuda.empty_cache = empty_cache
    torch.cuda.memory_allocated = memory_allocated
    torch.cuda.memory_reserved = memory_reserved
    torch.cuda.max_memory_allocated = max_memory_allocated
    torch.cuda.max_memory_reserved = max_memory_reserved
    torch.cuda.memory_cached = memory_cached
    torch.cuda.max_memory_cached = max_memory_cached
    torch.cuda.reset_peak_memory_stats = reset_peak_memory_stats
    torch.cuda.reset_max_memory_allocated = reset_max_memory_allocated
    torch.cuda.reset_max_memory_cached = reset_max_memory_cached
    torch.cuda.reset_accumulated_memory_stats = reset_accumulated_memory_stats
    torch.cuda.memory_stats = memory_stats
    torch.cuda.memory_summary = memory_summary
    torch.cuda.memory_snapshot = memory_snapshot
    torch.cuda.mem_get_info = mem_get_info
    torch.cuda.get_arch_list = get_arch_list
    torch.cuda.get_gencode_flags = get_gencode_flags
    torch.cuda.manual_seed = manual_seed
    torch.cuda.manual_seed_all = manual_seed_all
    torch.cuda.seed = seed
    torch.cuda.seed_all = seed_all
    torch.cuda.initial_seed = initial_seed
    torch.cuda.ipc_collect = ipc_collect
    torch.cuda.can_device_access_peer = can_device_access_peer
    torch.cuda.cudart = cudart
    torch.cuda.is_bf16_supported = is_bf16_supported
    torch.cuda.Stream = _FakeStream
    torch.cuda.Event = _FakeEvent
    torch.cuda.stream = _stream
    torch.cuda.device = _FakeDeviceCtx
    torch.cuda.device_of = _FakeDeviceOfCtx
    torch.cuda.set_stream = _set_stream
    torch.cuda.current_stream = _current_stream
    torch.cuda.default_stream = _default_stream
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda._is_compiled = lambda: True
    torch.cuda._lazy_init = lambda: None
    torch.cuda.is_initialized = lambda: True
    torch.cuda.init = lambda: None
    torch.cuda._initialized = True
    torch.cuda._cached_device_count = _NUM_DEVICES
    torch.cuda._exchange_device = lambda device: _CURRENT_DEVICE
    torch.cuda._get_device = lambda device: _normalize_device_index(device)
    if hasattr(torch.cuda, "_maybe_exchange_device"):
        torch.cuda._maybe_exchange_device = lambda device: _CURRENT_DEVICE

    torch.cuda.default_generators = tuple(_cpu_default_generator() for _ in range(_NUM_DEVICES))

    for attr, stub in {
        "_cuda_getCurrentRawStream": lambda device_index=0: 0,
        "_cuda_isCurrentStreamCapturing": lambda: False,
        "_cuda_getDeviceCount": lambda: _NUM_DEVICES,
        "_cuda_getDevice": lambda: _CURRENT_DEVICE,
        "_cuda_setDevice": lambda device: set_device(device),
        "_cuda_init": lambda: None,
        "_cuda_emptyCache": lambda: None,
        "_cuda_memoryStats": lambda device=0: {},
        "_cuda_memorySnapshot": lambda: [],
        "_cuda_resetPeakMemoryStats": lambda device=0: None,
        "_cuda_resetAccumulatedMemoryStats": lambda device=0: None,
        "_cuda_ipc_collect": lambda: None,
        "_cuda_canDeviceAccessPeer": lambda device, peer_device: True,
        "_cuda_getArchFlags": lambda: get_gencode_flags(),
        "_broadcast": lambda tensor, devices: _fake_comm_broadcast(tensor, devices=devices),
        "_broadcast_out": lambda tensor, out: _fake_comm_broadcast(tensor, out=out),
        "_broadcast_coalesced": lambda tensors, devices, buffer_size=10485760: _fake_comm_broadcast_coalesced(
            tensors, devices=devices, buffer_size=buffer_size
        ),
        "_scatter": lambda tensor, devices, chunk_sizes, dim, streams=None: _fake_comm_scatter(
            tensor, devices=devices, chunk_sizes=chunk_sizes, dim=dim, streams=streams
        ),
        "_gather": lambda tensors, dim, destination: _fake_comm_gather(
            tensors, dim=dim, destination=destination
        ),
    }.items():
        setattr(torch._C, attr, stub)

    torch.cuda.graphs._cuda_isCurrentStreamCapturing = lambda: False
    torch.cuda.graphs.is_current_stream_capturing = lambda: False
    torch.cuda.memory.empty_cache = empty_cache
    torch.cuda.memory.memory_allocated = memory_allocated
    torch.cuda.memory.memory_reserved = memory_reserved
    torch.cuda.memory.mem_get_info = mem_get_info
    torch.cuda.memory.max_memory_allocated = max_memory_allocated
    torch.cuda.memory.max_memory_reserved = max_memory_reserved
    torch.cuda.memory.memory_cached = memory_cached
    torch.cuda.memory.max_memory_cached = max_memory_cached
    torch.cuda.memory.memory_stats = memory_stats
    torch.cuda.memory.memory_summary = memory_summary
    torch.cuda.memory.memory_snapshot = memory_snapshot
    torch.cuda.memory.reset_peak_memory_stats = reset_peak_memory_stats
    torch.cuda.memory.reset_max_memory_allocated = reset_max_memory_allocated
    torch.cuda.memory.reset_max_memory_cached = reset_max_memory_cached
    torch.cuda.memory.reset_accumulated_memory_stats = reset_accumulated_memory_stats
    torch.cuda.random.manual_seed = manual_seed
    torch.cuda.random.manual_seed_all = manual_seed_all
    torch.cuda.random.seed = seed
    torch.cuda.random.seed_all = seed_all
    torch.cuda.random.initial_seed = initial_seed
    torch.distributed.is_available = _dist_is_available
    torch.distributed.is_nccl_available = _dist_is_nccl_available
    torch.distributed.is_initialized = _dist_is_initialized
    torch.distributed.init_process_group = _dist_init_process_group
    torch.distributed.destroy_process_group = _dist_destroy_process_group
    torch.distributed.get_backend = _dist_get_backend
    torch.distributed.get_world_size = _dist_get_world_size
    torch.distributed.get_rank = _dist_get_rank
    torch.distributed.barrier = _dist_barrier
    torch.distributed.all_reduce = _dist_all_reduce
    torch.distributed.broadcast = _dist_broadcast
    torch.distributed.all_gather = _dist_all_gather
    torch.distributed.all_gather_into_tensor = _dist_all_gather_into_tensor
    torch.distributed.all_gather_object = _dist_all_gather_object
    torch.distributed.reduce = _dist_reduce
    torch.distributed.gather = _dist_gather
    torch.distributed.scatter = _dist_scatter
    torch.distributed.reduce_scatter = _dist_reduce_scatter
    torch.distributed.reduce_scatter_tensor = _dist_reduce_scatter_tensor
    torch.distributed.all_to_all = _dist_all_to_all
    torch.distributed.all_to_all_single = _dist_all_to_all_single
    torch.distributed.broadcast_object_list = _dist_broadcast_object_list
    torch.distributed._all_gather_base = _dist_all_gather_into_tensor
    torch.distributed._reduce_scatter_base = _dist_reduce_scatter_tensor
    _install_legacy_cuda_types()

    for name in [
        "tensor",
        "as_tensor",
        "zeros",
        "ones",
        "empty",
        "full",
        "rand",
        "randn",
        "randint",
        "arange",
        "linspace",
        "logspace",
        "eye",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "rand_like",
        "randn_like",
        "randint_like",
        "scalar_tensor",
        "sparse_coo_tensor",
    ]:
        original = getattr(torch, name, None)
        if original is not None:
            setattr(torch, name, _factory_wrapper(original))

    _patch_torch_load()
    _ENABLED = True
