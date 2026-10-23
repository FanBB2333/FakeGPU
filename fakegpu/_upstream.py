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

import contextlib
import functools
import inspect
import os
import struct
import sys
import traceback
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
_ORIG_DDP: Any = None
# True while a real gloo process group joins the processes of a
# multi-process launch; see _real_launch().
_DIST_REAL = False
_ORIG_DIST: dict[str, Any] = {}
# Set by torch_patch while it tracks memory. Called with the meta tensor a
# factory would have made, its device and the host allocation error, when
# the host cannot back a new simulated tensor.
_HOST_ALLOCATION_FAILED: Any = None
# Backends that new_group() was asked for, by group.
_GROUP_BACKENDS: dict[Any, str] = {}
_REAL_COLLECTIVES = (
    "barrier", "all_reduce", "broadcast", "all_gather", "all_gather_object", "reduce",
    "gather", "scatter", "all_to_all", "all_to_all_single", "broadcast_object_list",
    "get_rank", "get_world_size", "get_backend", "new_group", "monitored_barrier",
    "_make_nccl_premul_sum",
)
# Factors of the ReduceOps made by _make_nccl_premul_sum, by id(op); the C++
# op keeps its factor where Python cannot read it.
_PREMUL_FACTORS: dict[int, Any] = {}
_DEFAULT_STREAMS = {}
_CURRENT_STREAMS = {}
_NEXT_STREAM_ID = 1
# torch.cuda.memory functions as torch defines them, saved by enable() before
# the stubs below replace them.  fakegpu.torch_patch puts them back over its
# modeled allocator bindings.
_REAL_CUDA_MEMORY_API: dict[str, Any] = {}
_CUDA_MEMORY_API_NAMES = (
    "empty_cache",
    "memory_allocated",
    "memory_reserved",
    "mem_get_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_stats",
    "memory_summary",
    "memory_snapshot",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_accumulated_memory_stats",
    "caching_allocator_alloc",
    "caching_allocator_delete",
)
# Lazy CUDA initialization (torch/cuda/__init__.py: _lazy_init, _lazy_call).
_INITIALIZED = False
_INITIALIZING = False
_QUEUED_CALLS: list[tuple[Any, list[str]]] = []
# _LazySeedTracker: only the latest manual_seed and manual_seed_all run.
_QUEUED_SEED_CALLS: dict[str, tuple[Any, list[str]] | None] = {"seed": None, "seed_all": None}
_QUEUED_SEED_ORDER: list[str] = []
# Run once at initialization, before the queued calls (torch._C._cuda_init()).
_LAZY_INIT_HOOKS: list[Any] = []
# The first host-to-device move of tensors that only torch.load can see.
_MOVING_LOADED_TENSORS = False


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

    def is_capturing(self) -> bool:
        return False

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


# DDP's reducer error for parameters without a gradient (torch/csrc/distributed/c10d/reducer.cpp).
_UNFINISHED_REDUCTION = (
    "Expected to have finished reduction in the prior iteration before starting a new one. "
    "This error indicates that your module has parameters that were not used in producing loss. "
    "You can enable unused parameter detection by passing the keyword argument "
    "`find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by \n"
    "making sure all `forward` function outputs participate in calculating loss. \n"
    "If you already have done the above, then the distributed data parallel module wasn't able to "
    "locate the output tensors in the return value of your module's `forward` function. Please include "
    "the loss function and the structure of the return value of `forward` of your module when reporting "
    "this issue (e.g. list, dict, iterable).\n"
    "Parameter indices which did not receive grad for rank {rank}: {indices}\n"
    " In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL "
    "to print out information about which particular parameters did not receive gradient on this rank as "
    "part of this error"
)


def _ddp_arguments(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """DDP's arguments, bound the way the installed torch binds them.

    A misspelled keyword then fails, and a positional argument lands in the
    same parameter, as on real GPUs.
    """

    if _ORIG_DDP is None:
        return {**dict(zip(("device_ids", "output_device", "dim"), args)), **kwargs}
    try:
        bound = inspect.signature(_ORIG_DDP.__init__).bind(None, module, *args, **kwargs)
    except TypeError as exc:
        raise TypeError(f"DistributedDataParallel.__init__() {exc}") from None
    return dict(bound.arguments)


try:
    from torch.distributed.algorithms.join import Join as _Join
    from torch.distributed.algorithms.join import JoinHook as _JoinHook
    from torch.distributed.algorithms.join import Joinable as _Joinable
except Exception:  # PyTorch built without torch.distributed
    _Join = None

    class _JoinHook:  # type: ignore[no-redef]
        pass

    class _Joinable:  # type: ignore[no-redef]
        def __init__(self) -> None:
            return None


_MIB = 1024 * 1024


def _assign_buckets(
    tensors: list[torch.Tensor], limits: list[int], expect_sparse: list[bool], indices: list[int] | None = None
) -> list[list[int]]:
    """DDP's bucket assignment, computed by the installed torch (``_compute_bucket_assignment_by_size``)."""

    if not tensors:
        return []
    raw = [unwrap_tensor(tensor) for tensor in tensors]
    positions = list(range(len(raw))) if indices is None else list(indices)
    compute = getattr(torch.distributed, "_compute_bucket_assignment_by_size", None)
    if compute is None:
        by_dtype: dict[torch.dtype, list[int]] = {}
        for position, tensor in zip(positions, raw):
            by_dtype.setdefault(tensor.dtype, []).append(position)
        return list(by_dtype.values())
    arguments: list[Any] = [raw, list(limits), list(expect_sparse)]
    if indices is not None:
        arguments.append(list(indices))
    buckets, _ = compute(*arguments)
    return [list(bucket) for bucket in buckets]


class _FakeGradBucket:
    """The bucket DDP hands a communication hook, with ``dist.GradBucket``'s methods.

    ``gradients()`` are views of the current buffer, as in the C++ bucket,
    so they follow ``set_buffer``.
    """

    def __init__(self, index: int, is_last: bool, buffer: torch.Tensor, parameters: list[Any]) -> None:
        self._index = index
        self._is_last = is_last
        self._buffer = buffer
        self._parameters = list(parameters)

    def index(self) -> int:
        return self._index

    def is_last(self) -> bool:
        return self._is_last

    def buffer(self) -> torch.Tensor:
        return self._buffer

    def set_buffer(self, tensor: torch.Tensor) -> None:
        self._buffer = tensor

    def gradients(self) -> list[torch.Tensor]:
        views, offset = [], 0
        for parameter in self._parameters:
            views.append(self._buffer[offset:offset + parameter.numel()].view(parameter.shape))
            offset += parameter.numel()
        return views

    def parameters(self) -> list[Any]:
        return list(self._parameters)


class _FakeDDPJoinHook(_JoinHook):
    """DDP's join hook: a joined rank shadows the collectives of the ranks still training."""

    def __init__(self, ddp: Any, divide_by_initial_world_size: bool) -> None:
        self.ddp = ddp
        ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self) -> None:
        ddp = self.ddp
        ddp._maybe_rebuild_buckets()
        if ddp._syncs_buffers():
            ddp._broadcast_buffers(ddp._find_common_rank(ddp._distributed_rank, False))
        should_sync = ddp._check_global_requires_backward_grad_sync(is_joined_rank=True)
        ddp.require_forward_param_sync = should_sync
        if should_sync and ddp._reduces():
            ddp._reduce_buckets(joined=True)

    def post_hook(self, is_last_joiner: bool) -> None:
        self.ddp._sync_final_model(is_last_joiner)


class _FakeDistributedDataParallel(torch.nn.Module, _Joinable):
    def __init__(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        _Joinable.__init__(self)
        options = _ddp_arguments(module, args, kwargs)
        ignored = set(getattr(module, "_ddp_params_and_buffers_to_ignore", ()))
        self._parameters_to_sync = [
            parameter for name, parameter in module.named_parameters() if name not in ignored
        ]
        if not any(parameter.requires_grad for parameter in self._parameters_to_sync) and not options.get(
            "delay_all_reduce_named_params"
        ):
            raise RuntimeError(
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient."
            )
        device_ids, output_device = options.get("device_ids"), options.get("output_device")
        if device_ids is not None and len(device_ids) > 1:
            raise ValueError("device_ids can only be None or contain a single element.")
        self.module = module
        self.dim = options.get("dim", 0)
        self.device_ids = (
            None if device_ids is None else [_normalize_device_index(d) for d in device_ids]
        )
        self.output_device = (
            None if output_device is None else _normalize_device_index(output_device)
        )
        self.process_group = self._resolve_process_group(options)
        # Like DDP, the device of the first parameter it manages.
        self.device = self._parameters_to_sync[0].device if self._parameters_to_sync else torch.device("cpu")
        self.find_unused_parameters = bool(options.get("find_unused_parameters", False))
        self.static_graph = bool(options.get("static_graph", False))
        self.gradient_as_bucket_view = bool(options.get("gradient_as_bucket_view", False))
        # PyTorch 2.14 adds forward_sync_buffers; broadcast_buffers=False alone
        # also keeps each rank's buffers when DDP is built.
        forward_sync_buffers, broadcast_buffers = options.get("forward_sync_buffers"), options.get("broadcast_buffers")
        if forward_sync_buffers is not None:
            self.broadcast_buffers, init_buffers = bool(forward_sync_buffers), True
        else:
            self.broadcast_buffers = init_buffers = True if broadcast_buffers is None else bool(broadcast_buffers)
        self.require_backward_grad_sync = True
        # Whether the next forward starts from rank 0's buffers: the first one
        # does, later ones when the forward before had gradients synced.
        self.require_forward_param_sync = True
        self.logger = None
        self._comm_hooks: list[tuple[Any, Any]] = []
        self._comm_hook: tuple[Any, Any] | None = None
        self._divide_by_initial_world_size = True
        # The forward's notice to a join context; its value is the number of ranks still training.
        self._join_work: Any = None
        # Trainable parameters, in the order of DDP's reducer.
        self._trainable = [parameter for parameter in self._parameters_to_sync if parameter.requires_grad]
        self._init_buckets(module, options)
        self._local_used = [False] * len(self._trainable)
        self._queued = False
        if _DIST_REAL:
            self._join_processes(options.get("init_sync", True), init_buffers)
        # Trainable parameters whose gradient arrived since the last synced forward.
        self._marked: set[int] = set()
        for index, parameter in enumerate(self._trainable):
            raw = unwrap_tensor(parameter)
            raw.register_hook(self._marker(index))
            raw.register_hook(self._queue_reduction)

    def _resolve_process_group(self, options: dict[str, Any]) -> Any:
        """DDP's group: the given one, the mesh's, or the default group."""

        group, mesh = options.get("process_group"), options.get("device_mesh")
        if group and mesh is not None:
            raise RuntimeError("Cannot specify both process_group and device_mesh arguments.")
        if mesh is not None:
            if mesh.ndim != 1:
                raise RuntimeError(f"Only 1D device mesh is supported, but got {mesh}.")
            self.device_mesh = mesh
            return mesh.get_group(mesh_dim=0)
        if group is not None:
            return group
        c10d = torch.distributed.distributed_c10d
        if c10d.is_initialized():
            return c10d._get_default_group()
        if _DIST_INITIALIZED:
            return None  # torch's fake backend is missing; FakeGPU keeps the job's state
        # Without init_process_group torch raises "Default process group has not been initialized".
        return c10d._get_default_group()

    def _init_buckets(self, module: torch.nn.Module, options: dict[str, Any]) -> None:
        """DDP's gradient buckets: one bucket at first, rebuilt in gradient order (reducer.cpp)."""

        cap_mb, cap_list = options.get("bucket_cap_mb"), options.get("bucket_cap_mb_list")
        if cap_list:
            initial = rebuild = [int(cap * _MIB) for cap in cap_list]
        else:
            cap = int((25 if cap_mb is None else cap_mb) * _MIB)
            first = getattr(torch.distributed, "_DEFAULT_FIRST_BUCKET_BYTES", _MIB) if cap_mb is None else cap
            rebuild = [first, cap]
            if self.static_graph or not self.find_unused_parameters:
                initial = [sys.maxsize]
            else:
                initial = [first, cap] if first < cap else [cap]
        owners: dict[int, Any] = {}
        for submodule in module.modules():
            for parameter in submodule.parameters(recurse=False):
                owners.setdefault(id(parameter), submodule)
        embeddings = (torch.nn.Embedding, torch.nn.EmbeddingBag)
        self._expect_sparse = [
            isinstance(owners.get(id(p)), embeddings) and bool(owners[id(p)].sparse) for p in self._trainable
        ]
        self._rebuild_limits = rebuild
        # Reversed: the last parameters' gradients come first in backward.
        self._buckets = list(reversed(_assign_buckets(self._trainable, initial, self._expect_sparse)))
        self._can_rebuild = self.static_graph or not self.find_unused_parameters
        # The synced backward whose gradient order the buckets are rebuilt in.
        self._record_iteration = 1 if self.static_graph else 0
        self._synced_backwards = 0
        self._ready_order: list[int] = []
        self._pending_order: list[int] | None = None

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if torch.is_grad_enabled():
            self._check_prior_reduction()
        work = _Join.notify_join_context(self) if _Join is not None else None
        if work:
            self._join_work = work
        if torch.is_grad_enabled():
            self._maybe_rebuild_buckets()
        if self._syncs_buffers():
            self._sync_buffers()
        if self._join_enabled():
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)
        output = self.module(*inputs, **kwargs)
        self.require_forward_param_sync = torch.is_grad_enabled() and self.require_backward_grad_sync
        if self.require_forward_param_sync:
            self._marked.clear()
        return output

    def _marker(self, index: int) -> Any:
        def mark(grad: torch.Tensor) -> None:
            if self.require_backward_grad_sync:
                self._marked.add(index)
                if self._recording() and index not in self._ready_order:
                    self._ready_order.append(index)

        return mark

    def _recording(self) -> bool:
        return (
            self._can_rebuild
            and self._pending_order is None
            and self._synced_backwards == self._record_iteration
        )

    def _queue_reduction(self, grad: torch.Tensor) -> None:
        if self.require_backward_grad_sync and not self._queued:
            self._queued = True
            torch.autograd.Variable._execution_engine.queue_callback(self._finish_backward)

    def _finish_backward(self) -> None:
        self._queued = False
        if self._reduces():
            self._reduce_buckets(joined=False)
        if self._recording() and len(self._ready_order) == len(self._trainable):
            self._pending_order = list(self._ready_order)
        self._ready_order = []
        self._synced_backwards += 1
        self._join_work = None

    def _maybe_rebuild_buckets(self) -> None:
        if self._pending_order is None or not self._can_rebuild:
            return
        order, self._pending_order = self._pending_order, None
        self._can_rebuild = False
        self._buckets = _assign_buckets(
            [self._trainable[i] for i in order], self._rebuild_limits, [self._expect_sparse[i] for i in order], order
        )

    def _reduces(self) -> bool:
        """Whether backward reduces gradients: across launched ranks, or through a registered hook."""

        return _DIST_REAL or self._comm_hook is not None

    def _check_prior_reduction(self) -> None:
        """Fail like DDP's reducer when the last backward left a parameter without a gradient.

        Without ``find_unused_parameters`` or ``static_graph``, DDP waits for
        the gradient of every trainable parameter, so on real GPUs the next
        forward raises.
        """

        if not self._marked or self.find_unused_parameters or self.static_graph:
            return
        unmarked = [index for index in range(len(self._trainable)) if index not in self._marked]
        if unmarked:
            rank = torch.distributed.get_rank(self.process_group)
            raise RuntimeError(_UNFINISHED_REDUCTION.format(rank=rank, indices=" ".join(map(str, unmarked))))

    def _buffers_to_sync(self) -> list[torch.Tensor]:
        ignored = set(getattr(self.module, "_ddp_params_and_buffers_to_ignore", ()))
        return [buffer for name, buffer in self.module.named_buffers() if name not in ignored]

    @contextlib.contextmanager
    def no_sync(self) -> Any:
        previous = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = previous

    def _join_processes(self, init_sync: bool, init_buffers: bool) -> None:
        """Start every rank from rank 0's parameters, and from its buffers with ``init_buffers``."""

        initial = [*self._parameters_to_sync, *(self._buffers_to_sync() if init_buffers else [])] if init_sync else []
        source = self._global_rank(0)
        with torch.no_grad():
            for tensor in initial:
                _ORIG_DIST["broadcast"](unwrap_tensor(tensor).detach(), src=source, group=self.process_group)

    def _global_rank(self, group_rank: int) -> int:
        group = self.process_group
        if group is None or group is torch.distributed.GroupMember.WORLD:
            return int(group_rank)
        return torch.distributed.distributed_c10d.get_global_rank(group, int(group_rank))

    @property
    def _distributed_rank(self) -> int:
        return torch.distributed.get_rank(self.process_group)

    def _reduce_buckets(self, joined: bool) -> None:
        """Reduce the gradients bucket by bucket, as DDP's reducer does when backward ends.

        Every rank first sums which parameters got a gradient, so a parameter
        that some rank used gets zeros where it was unused and one that no
        rank used keeps no gradient. A joined rank contributes zeros, and the
        parameters its last backward used, as DDP's join hook does. Without a
        communication hook each bucket is scaled by 1/world size (or by the
        ranks still training, with ``divide_by_initial_world_size=False``) and
        summed; with one, the hook's result becomes the gradients.
        """

        group = self.process_group
        params = [unwrap_tensor(parameter) for parameter in self._trainable]
        if not joined:
            self._local_used = [parameter.grad is not None for parameter in params]
        used = torch.tensor(self._local_used, dtype=torch.int32)
        if _DIST_REAL:
            _ORIG_DIST["all_reduce"](used, op=_sum_op(), group=group)
        counts = used.tolist()
        divisor = None if joined or self._comm_hook is not None else self._divisor()
        with torch.no_grad():
            for index, bucket in enumerate(self._buckets):
                grads = [None if joined else params[i].grad for i in bucket]
                if any(grad is not None and grad.is_sparse for grad in grads):
                    for i, grad in zip(bucket, grads):
                        if grad is not None and _DIST_REAL:
                            _ORIG_DIST["all_reduce"](grad, op=_sum_op(), group=group)
                            grad.div_(divisor or 1)
                    continue
                flat = torch.cat([
                    (grad if grad is not None else torch.zeros_like(params[i])).reshape(-1)
                    for i, grad in zip(bucket, grads)
                ])
                if self._comm_hook is not None:
                    result = self._run_comm_hook(index, bucket, flat)
                else:
                    if divisor is not None:
                        flat.mul_(1.0 / divisor)
                    if _DIST_REAL:
                        _ORIG_DIST["all_reduce"](flat, op=_sum_op(), group=group)
                    result = flat
                if joined:
                    continue
                offset = 0
                for i in bucket:
                    size = params[i].numel()
                    if counts[i]:
                        if params[i].grad is None:
                            params[i].grad = torch.zeros_like(params[i])
                        params[i].grad.copy_(result[offset:offset + size].view_as(params[i]))
                    offset += size

    def _run_comm_hook(self, index: int, bucket: list[int], flat: torch.Tensor) -> torch.Tensor:
        hook, state = self._comm_hook  # type: ignore[misc]
        first = self._trainable[bucket[0]]
        buffer = wrap_tensor(flat, device_index=first.device_index) if isinstance(first, FakeCudaTensor) else flat
        grad_bucket = _FakeGradBucket(
            index, index == len(self._buckets) - 1, buffer, [self._trainable[i] for i in bucket]
        )
        future = hook(state, grad_bucket)
        if not hasattr(future, "wait"):
            raise RuntimeError(f"Unable to cast {future!r} to torch.futures.Future")
        result = future.wait()
        # DDP reads the future's value as one tensor (PythonCommHook::parseHookResult).
        if not isinstance(result, torch.Tensor):
            raise RuntimeError(f"Unable to cast {result!r} to Tensor")
        return unwrap_tensor(result).reshape(-1)

    def _divisor(self) -> int:
        world = (
            _ORIG_DIST["get_world_size"](self.process_group)
            if _DIST_REAL
            else _dist_get_world_size(self.process_group)
        )
        if self._join_enabled() and not self._divide_by_initial_world_size and self._join_work is not None:
            # The forward's join notice summed one per rank still training.
            values = self._join_work.get_future().value()
            return int(unwrap_tensor(values[0]).item())
        return world

    # --- communication hooks ---------------------------------------------------------

    def _log_and_throw(self, err_type: type, err_msg: str) -> None:
        raise err_type(err_msg)

    def _check_comm_hook(self, hook: Any) -> None:
        check = getattr(_ORIG_DDP, "_check_comm_hook", None)
        if check is not None:
            check(self, hook)  # torch's own checks of the hook's signature
        elif not callable(hook):
            raise TypeError("Communication hook must be callable.")

    def register_comm_hook(self, state: object, hook: Any) -> None:
        """Run ``hook(state, bucket)`` on each gradient bucket, as DDP does.

        Its future's tensor becomes the bucket's gradients. In a simulated job
        the hook's collectives use the local tensors only.
        """

        self._check_comm_hook(hook)
        if self._comm_hook is not None:
            raise RuntimeError("register_comm_hook or register_builtin_comm_hook can only be called once.")
        self._comm_hooks.append((hook, state))
        self._comm_hook = (hook, state)

    def _register_builtin_comm_hook(self, comm_hook_type: Any) -> None:
        if self._comm_hook is not None:
            raise RuntimeError("register_builtin_comm_hook or register_comm_hook can only be called once.")
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        hook = {"ALLREDUCE": default_hooks.allreduce_hook, "FP16_COMPRESS": default_hooks.fp16_compress_hook}[
            comm_hook_type.name
        ]
        self._comm_hook = (hook, self.process_group)

    # --- uneven inputs (Join) ----------------------------------------------------------

    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ) -> Any:
        return _Join(
            [self], enable, throw_on_early_termination, divide_by_initial_world_size=divide_by_initial_world_size
        )

    def join_hook(self, **kwargs: Any) -> Any:
        return _FakeDDPJoinHook(self, kwargs.get("divide_by_initial_world_size", True))

    @property
    def join_device(self) -> torch.device:
        return self.device

    @property
    def join_process_group(self) -> Any:
        return self.process_group

    def _join_enabled(self) -> bool:
        return bool(getattr(getattr(self, "_join_config", None), "enable", False))

    def _syncs_buffers(self) -> bool:
        return bool(
            _DIST_REAL and self.broadcast_buffers and self.require_forward_param_sync and self._buffers_to_sync()
        )

    def _sync_buffers(self) -> None:
        # Under a join context rank 0 may have joined; the highest training rank holds the buffers.
        authoritative = self._find_common_rank(self._distributed_rank, True) if self._join_enabled() else 0
        self._broadcast_buffers(authoritative)

    def _broadcast_buffers(self, group_rank: int) -> None:
        source = self._global_rank(group_rank)
        with torch.no_grad():
            for buffer in self._buffers_to_sync():
                _ORIG_DIST["broadcast"](unwrap_tensor(buffer).detach(), src=source, group=self.process_group)

    def _find_common_rank(self, input_rank: int, rank_cond: bool) -> int:
        rank_to_use = torch.tensor([input_rank if rank_cond else -1], device=self.device)
        torch.distributed.all_reduce(rank_to_use, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            raise ValueError(
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue."
            )
        return int(rank_to_use.item())

    def _check_global_requires_backward_grad_sync(self, is_joined_rank: bool) -> Any:
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)
        work = torch.distributed.all_reduce(requires_sync_tensor, group=self.process_group, async_op=True)
        if is_joined_rank:
            work.wait()
            return requires_sync_tensor.item() != 0
        return None

    def _sync_final_model(self, is_last_joiner: bool) -> None:
        """Give every rank the model of the last rank to join."""

        self._authoritative_rank = self._find_common_rank(self._distributed_rank, is_last_joiner)
        if not _DIST_REAL:
            return
        source = self._global_rank(self._authoritative_rank)
        states = [*self._parameters_to_sync, *(self._buffers_to_sync() if self.broadcast_buffers else [])]
        with torch.no_grad():
            for tensor in states:
                _ORIG_DIST["broadcast"](unwrap_tensor(tensor).detach(), src=source, group=self.process_group)


class _FakeAsyncWork:
    """A finished work handle, where torch cannot wrap a future in a real one."""

    def __init__(self, future: Any) -> None:
        self._future = future

    def wait(self, timeout: Any = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True

    def exception(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def get_future(self) -> Any:
        return self._future

    def result(self) -> list[torch.Tensor]:
        return self._future.value()


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
    **kwargs: Any,
) -> None:
    global _DIST_INITIALIZED, _DIST_BACKEND, _DIST_WORLD_SIZE, _DIST_RANK, _DIST_REAL
    if kwargs and _ORIG_DIST_INIT is not None:
        inspect.signature(_ORIG_DIST_INIT).bind_partial(**kwargs)  # an unknown keyword fails as in torch
    launch = _real_launch(world_size, rank, init_method, store)
    if launch is not None and _ORIG_DIST_INIT is not None:
        size, index = launch
        # Newer arguments, e.g. PyTorch 2.14's enable_reconfigure, go to torch as given.
        options: dict[str, Any] = {**kwargs, "backend": "gloo", "world_size": size, "rank": index}
        for name, value in (("init_method", init_method), ("store", store), ("timeout", timeout)):
            if value is not None:
                options[name] = value
        if group_name:
            options["group_name"] = group_name
        # A failed rendezvous must surface, as it would on real GPUs.
        _ORIG_DIST_INIT(**options)
        _DIST_INITIALIZED = _DIST_REAL = True
        _DIST_BACKEND = _backend_name(backend)
        _DIST_WORLD_SIZE, _DIST_RANK = size, index
        return None

    _DIST_INITIALIZED = True
    _DIST_BACKEND = _backend_name(backend)
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


def _backend_name(backend: Any) -> str:
    """What ``get_backend()`` reports, as torch's ``Backend`` names it.

    Without a backend torch reports ``undefined`` and gives the machine's
    accelerator, CUDA here, its default backend (NCCL) only.
    """

    return "undefined" if backend is None else str(backend).lower()


def _real_launch(world_size: Any, rank: Any, init_method: Any, store: Any) -> tuple[int, int] | None:
    """World size and rank when ``init_process_group`` joins real processes.

    Collectives then run over gloo on the CPU tensors behind the simulated
    GPUs, so every rank sees the values NCCL would produce. That needs the
    other ranks to be starting too, which FakeGPU takes as given for workers
    of torchrun or ``accelerate launch`` (``TORCHELASTIC_RUN_ID``), for
    ``multiprocessing`` children such as ``mp.spawn`` workers, and for an
    explicit ``init_method`` or ``store``. Any other process, e.g. one that
    finds ``WORLD_SIZE`` and ``MASTER_ADDR`` in a cluster pod's environment,
    keeps the in-process simulation instead of waiting for peers that never
    come. ``FAKEGPU_DIST_COLLECTIVES=real`` or ``simulate`` overrides this.
    """

    choice = os.environ.get("FAKEGPU_DIST_COLLECTIVES", "").strip().lower()
    if choice == "simulate":
        return None
    size = world_size if world_size not in (-1, None) else os.environ.get("WORLD_SIZE")
    index = rank if rank not in (-1, None) else os.environ.get("RANK")
    try:
        size, index = int(size), int(index)
    except (TypeError, ValueError):
        return None
    explicit = store is not None or (init_method is not None and str(init_method) != "env://")
    rendezvous = explicit or ("MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ)
    if size < 2 or not rendezvous:
        return None
    import multiprocessing

    launched = (
        choice == "real"
        or explicit
        or "TORCHELASTIC_RUN_ID" in os.environ
        or multiprocessing.parent_process() is not None
    )
    return (size, index) if launched else None


def _dist_destroy_process_group(group: Any = None) -> None:
    global _DIST_INITIALIZED, _DIST_WORLD_SIZE, _DIST_RANK, _DIST_REAL
    if group is None:
        _GROUP_BACKENDS.clear()
    else:
        _GROUP_BACKENDS.pop(group, None)
    if _DIST_REAL:
        _ORIG_DIST_DESTROY(group)
        if group is None:
            _DIST_INITIALIZED = _DIST_REAL = False
            _DIST_WORLD_SIZE, _DIST_RANK = 1, 0
        return None
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
    if _DIST_INITIALIZED:
        return _requested_backend(group)
    # Before init_process_group torch raises "Default process group has not been initialized".
    return _ORIG_DIST["get_backend"](group)


def _requested_backend(group: Any) -> str:
    """The backend the script asked for ``group``; subgroups default to the job's."""

    if group is None or group is torch.distributed.GroupMember.WORLD:
        return _DIST_BACKEND
    return _GROUP_BACKENDS.get(group, _DIST_BACKEND)


def _backend_for(group: Any, device_type: str) -> str | None:
    """The backend that ``group`` runs collectives of ``device_type`` tensors on, if any.

    As in torch: ``nccl`` handles CUDA tensors only; no named backend
    (``undefined``) gives the CUDA machine's accelerator its NCCL backend
    only; ``cpu:gloo,cuda:nccl`` maps each device; gloo and other backends
    handle both devices.
    """

    backend = _requested_backend(group).lower().replace(" ", "")
    if ":" in backend:
        devices = dict(item.split(":", 1) for item in backend.split(",") if ":" in item)
        return devices.get(device_type)
    if backend in ("nccl", "undefined"):
        return "nccl" if device_type == "cuda" else None
    return backend


def _nccl_handles(tensor: Any, group: Any) -> bool:
    """Whether NCCL would run a collective on ``tensor``: a CUDA tensor in a group whose backend is NCCL."""

    return isinstance(tensor, FakeCudaTensor) and _backend_for(group, "cuda") == "nccl"


def _check_devices(group: Any, *tensors: Any) -> None:
    """Fail as ProcessGroup::getBackend does when no backend serves the tensors' device.

    The dispatcher picks the CUDA kernel when any tensor is on a GPU;
    otherwise the CPU one, which a plain NCCL group lacks.
    """

    present = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    if not present or any(tensor.device.type == "cuda" for tensor in present):
        return
    device_type = present[0].device.type
    if _backend_for(group, device_type) is None:
        raise RuntimeError(f"No backend type associated with device type {device_type}")


def _check_single_device(group: Any, output: Any, source: Any) -> None:
    """ProcessGroupNCCL's check of the two tensors of an ``*_into_tensor``/``*_tensor`` collective."""

    devices = {output.device.type, source.device.type}
    if "cuda" in devices and len(devices) > 1 and _backend_for(group, "cuda") == "nccl":
        raise ValueError("Tensors must be CUDA and dense")


# c10 ScalarType names, as torch's dtype errors print them.
_SCALAR_TYPE_NAMES = {
    torch.float16: "Half", torch.float32: "Float", torch.float64: "Double", torch.bfloat16: "BFloat16",
    torch.int8: "Char", torch.uint8: "Byte", torch.int16: "Short", torch.int32: "Int", torch.int64: "Long",
    torch.bool: "Bool",
}


def _torch_at_least(major: int, minor: int) -> bool:
    try:
        return torch.torch_version.TorchVersion(torch.__version__) >= (major, minor)
    except Exception:
        return True


_NAMED_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def _as_installed_signature(name: str, patched: Any) -> Any:
    """Make ``patched`` reject the arguments the installed ``torch.distributed.<name>`` lacks.

    The patched functions take the newest PyTorch arguments, such as
    barrier's ``timeout`` and all_gather_object's ``weights_only``. PyTorch
    2.9 has neither, and passing one raises TypeError there.
    """
    original = _ORIG_DIST.get(name)
    if original is None:
        return patched
    try:
        signature = inspect.signature(original)
        named = {
            parameter.name
            for parameter in inspect.signature(patched).parameters.values()
            if parameter.kind in _NAMED_PARAMETER_KINDS
        }
    except (TypeError, ValueError):
        return patched
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return patched
    if not named - set(signature.parameters):
        return patched

    @functools.wraps(patched)
    def checked(*args: Any, **kwargs: Any) -> Any:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{name}() {exc}") from None
        return patched(*args, **kwargs)

    checked.__signature__ = signature  # type: ignore[attr-defined]
    return checked


def _premul_scalar_dtype(dtype: torch.dtype) -> torch.dtype:
    """The C++ type ProcessGroupNCCL reads a PREMUL_SUM factor as (unpackPreMulSum<T>).

    Before PyTorch 2.14 bfloat16 tensors used a float factor.
    """

    if dtype == torch.bfloat16 and not _torch_at_least(2, 14):
        return torch.float32
    return dtype


def _dist_make_nccl_premul_sum(factor: Any) -> Any:
    op = _ORIG_DIST["_make_nccl_premul_sum"](factor)
    key = id(op)
    _PREMUL_FACTORS[key] = factor
    try:
        weakref.finalize(op, _PREMUL_FACTORS.pop, key, None)
    except TypeError:
        pass
    return op


def _remember_premul_factors_of_calls() -> None:
    """Also remember the factor of ``ReduceOp.PREMUL_SUM(factor)`` (FSDP2 in PyTorch 2.14)."""

    red_op_type = type(torch.distributed.ReduceOp.PREMUL_SUM)
    original = red_op_type.__call__

    def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
        op = original(self, *args, **kwargs)
        if args and isinstance(op, torch.distributed.ReduceOp):
            _PREMUL_FACTORS[id(op)] = args[0]
            try:
                weakref.finalize(op, _PREMUL_FACTORS.pop, id(op), None)
            except TypeError:
                pass
        return op

    try:
        red_op_type.__call__ = __call__
    except (AttributeError, TypeError):
        pass


def _premul_factor(op: Any) -> Any:
    """The factor of a ``_make_nccl_premul_sum`` op, or None for any other op."""

    if op is None or not isinstance(op, torch.distributed.ReduceOp):
        return None
    if not op == torch.distributed.ReduceOp.PREMUL_SUM:
        return None
    return _PREMUL_FACTORS.get(id(op))


def _premul_multiplier(factor: Any, dtype: torch.dtype) -> float:
    """The factor NCCL multiplies each input by.

    NCCL reads ``sizeof(dtype)`` bytes of the factor. Before PyTorch 2.14 a
    bfloat16 reduction passed a 4-byte float, so NCCL took its low two bytes
    as the bfloat16 factor (zero for 0.5 and other powers of two).
    """

    value = float(unwrap_tensor(factor).reshape(-1)[0]) if isinstance(factor, torch.Tensor) else float(factor)
    if dtype == torch.bfloat16 and _premul_scalar_dtype(dtype) == torch.float32:
        low = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF
        return struct.unpack("<f", struct.pack("<I", low << 16))[0]
    return value


def _check_reduce_op(tensor: Any, op: Any, group: Any) -> None:
    """Reject the reductions torch rejects: invalid ops, and for a CUDA tensor what getNcclReduceOp rejects."""

    if op is None:
        return
    # Converting the op into the collective's options raises torch's TypeError,
    # e.g. for a bare ReduceOp.PREMUL_SUM without its factor.
    torch.distributed.distributed_c10d.AllreduceOptions().reduceOp = op
    if not _nccl_handles(tensor, group):
        return
    reduce_op = torch.distributed.ReduceOp
    for name in ("BAND", "BOR", "BXOR"):
        if op == getattr(reduce_op, name):
            raise ValueError(f"Cannot use ReduceOp.{name} with NCCL")
    if _is_avg(op) and tensor.dtype == torch.bool:
        raise TypeError("Cannot use ReduceOp.AVG with boolean inputs")
    factor = _premul_factor(op)
    if factor is not None:
        if tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
            raise TypeError("PreMulSum Data type must be half, float, bfloat16 or double")
        expected = _premul_scalar_dtype(tensor.dtype)
        if isinstance(factor, torch.Tensor) and factor.dtype != expected:
            raise RuntimeError(
                f"expected scalar type {_SCALAR_TYPE_NAMES.get(expected, expected)} "
                f"but found {_SCALAR_TYPE_NAMES.get(factor.dtype, factor.dtype)}"
            )


def _premultiply(tensor: Any, raw: torch.Tensor, op: Any, group: Any) -> Any:
    """Apply an NCCL PREMUL_SUM's factor to ``raw`` and return the SUM that finishes it.

    Other ops come back unchanged; gloo rejects PREMUL_SUM on a gloo group.
    """

    factor = _premul_factor(op)
    if factor is None or not _nccl_handles(tensor, group):
        return op
    with torch.no_grad():
        raw.mul_(torch.tensor(_premul_multiplier(factor, raw.dtype), dtype=raw.dtype))
    return _sum_op()


def _torch_knows(group: Any) -> bool:
    """Whether torch's own process groups can answer for ``group``.

    A simulated job's subgroups live in torch's ``fake`` backend, which
    knows each one's ranks.
    """

    return _DIST_REAL or (group is not None and torch.distributed.distributed_c10d.is_initialized())


def _dist_get_world_size(group: Any = None) -> int:
    if _torch_knows(group):
        return _ORIG_DIST["get_world_size"](group)
    if _DIST_INITIALIZED:
        return _DIST_WORLD_SIZE
    # Before init_process_group torch raises "Default process group has not been initialized".
    return _ORIG_DIST["get_world_size"](group)


def _dist_get_rank(group: Any = None) -> int:
    if _torch_knows(group):
        return _ORIG_DIST["get_rank"](group)
    if _DIST_INITIALIZED:
        return _DIST_RANK
    return _ORIG_DIST["get_rank"](group)


def _dist_new_group(*args: Any, **kwargs: Any) -> Any:
    # Subgroups use the default group's backend, gloo across launched
    # processes or torch's fake backend in a simulated job; NCCL is not there.
    original = _ORIG_DIST["new_group"]
    arguments = dict(inspect.signature(original).bind(*args, **kwargs).arguments)
    backend = arguments.pop("backend", None)
    arguments.pop("pg_options", None)
    group = original(**arguments)
    if backend is not None and not torch.distributed.distributed_c10d._rank_not_in_group(group):
        _GROUP_BACKENDS[group] = str(backend).lower()
    return group


def _sum_op() -> Any:
    return torch.distributed.ReduceOp.SUM


def _is_avg(op: Any) -> bool:
    return op is not None and op == torch.distributed.ReduceOp.AVG


def _divide(raw: torch.Tensor, count: int) -> None:
    """Finish an AVG in place; integer tensors round toward zero, as NCCL does."""

    if raw.is_floating_point() or raw.is_complex():
        raw.div_(count)
    else:
        raw.div_(count, rounding_mode="trunc")


def _not_in_group(group: Any, name: str) -> bool:
    """Whether this rank is outside ``group``; torch then warns and skips the collective."""

    c10d = torch.distributed.distributed_c10d
    if not c10d._rank_not_in_group(group):
        return False
    c10d._warn_not_in_group(name)
    return True


def _torch_name(new: str, old: str) -> str:
    """The name the installed torch gives an operation in its warnings (renamed after 2.9)."""

    return new if hasattr(torch.distributed.distributed_c10d, new) else old


def _real_all_reduce(raw: torch.Tensor, op: Any, group: Any) -> None:
    """All-reduce a CPU tensor in place; gloo lacks AVG on older PyTorch."""

    if _is_avg(op):
        _ORIG_DIST["all_reduce"](raw, op=_sum_op(), group=group)
        _divide(raw, _ORIG_DIST["get_world_size"](group))
        return
    _ORIG_DIST["all_reduce"](raw, op=_sum_op() if op is None else op, group=group)


def _done(async_op: bool, outputs: list[torch.Tensor] | None = None) -> Any:
    """The handle of a collective FakeGPU already finished, or None without ``async_op``.

    Like NCCL's work, it is completed and its future holds the output tensors.
    """

    if not async_op:
        return None
    future: Any = torch.futures.Future()
    future.set_result(list(outputs or []))
    create = getattr(getattr(torch._C, "_distributed_c10d", None), "_create_work_from_future", None)
    return _FakeAsyncWork(future) if create is None else create(future)


def _dist_barrier(group: Any = None, async_op: bool = False, device_ids: Any = None, timeout: Any = None) -> Any:
    if _not_in_group(group, "barrier"):
        return None
    if _DIST_REAL:
        options = {"timeout": timeout} if timeout is not None else {}
        return _ORIG_DIST["barrier"](group=group, async_op=async_op, **options)
    return _done(async_op)


def _dist_monitored_barrier(group: Any = None, timeout: Any = None, wait_all_ranks: bool = False) -> Any:
    if _not_in_group(group, "monitored_barrier"):
        return None
    if _torch_at_least(2, 14):
        # PyTorch 2.14 accepts any group with a CPU backend.
        if _backend_for(group, "cpu") is None:
            raise ValueError("monitored_barrier requires a group with a CPU-capable backend (e.g. GLOO).")
    elif _requested_backend(group).lower() != "gloo":
        raise ValueError("monitored_barrier is only implemented for GLOO backend.")
    if _DIST_REAL:
        return _ORIG_DIST["monitored_barrier"](group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)
    return None


def _dist_all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None, async_op: bool = False) -> Any:
    if _not_in_group(group, "all_reduce"):
        return None
    _check_devices(group, tensor)
    _check_reduce_op(tensor, op, group)
    raw = unwrap_tensor(tensor)
    op = _premultiply(tensor, raw, op, group)
    if _DIST_REAL:
        _real_all_reduce(raw, op, group)
    return _done(async_op, [tensor])


def _dist_broadcast(
    tensor: torch.Tensor,
    src: int | None = None,
    group: Any = None,
    async_op: bool = False,
    group_src: Any = None,
) -> Any:
    if _not_in_group(group, "broadcast"):
        return None
    _check_devices(group, tensor)
    if _DIST_REAL:
        options = {"group_src": group_src} if group_src is not None else {"src": src}
        return _ORIG_DIST["broadcast"](unwrap_tensor(tensor), group=group, async_op=async_op, **options)
    return _done(async_op, [tensor])


def _dist_all_gather(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, "all_gather"):
        return None
    _check_devices(group, tensor, *tensor_list)
    if _DIST_REAL:
        return _ORIG_DIST["all_gather"](
            [unwrap_tensor(item) for item in tensor_list], unwrap_tensor(tensor), group=group, async_op=async_op
        )
    for target in tensor_list:
        # Other ranks may contribute tensors of other sizes.
        _copy_collective_tensor(target, tensor)
    return _done(async_op, tensor_list)


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


def _weights_only(weights_only: bool) -> dict[str, bool]:
    # PyTorch 2.14 added the argument; pass it only when set, for older versions.
    return {"weights_only": True} if weights_only else {}


def _dist_all_gather_object(object_list: list[Any], obj: Any, group: Any = None, weights_only: bool = False) -> None:
    if _DIST_REAL:
        return _ORIG_DIST["all_gather_object"](object_list, obj, group=group, **_weights_only(weights_only))
    for index in range(len(object_list)):
        object_list[index] = obj
    return None


def _dist_all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, _torch_name("all_gather_single", "all_gather_into_tensor")):
        return None
    _check_devices(group, output_tensor, input_tensor)
    _check_single_device(group, output_tensor, input_tensor)
    if _nccl_handles(output_tensor, group):
        # ProcessGroupNCCL::all_gather_single checks these.
        if input_tensor.dtype != output_tensor.dtype:
            raise TypeError("output tensor must have the same type as input tensor")
        if input_tensor.numel() * _dist_get_world_size(group) != output_tensor.numel():
            raise ValueError("output tensor size must be equal to world_size times input tensor size")
    if _DIST_REAL:
        source = unwrap_tensor(input_tensor).contiguous()
        parts = [torch.empty_like(source) for _ in range(_ORIG_DIST["get_world_size"](group))]
        _ORIG_DIST["all_gather"](parts, source, group=group)
        target = unwrap_tensor(output_tensor)
        # Rank-major order covers both the concatenated and the stacked layout.
        target.copy_(torch.cat([part.reshape(-1) for part in parts]).reshape(target.shape))
        return _done(async_op, [output_tensor])
    _copy_collective_tensor(output_tensor, input_tensor)
    return _done(async_op, [output_tensor])


def _dist_reduce(
    tensor: torch.Tensor,
    dst: int | None = None,
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
    group_dst: Any = None,
) -> Any:
    if _not_in_group(group, "reduce"):
        return None
    _check_devices(group, tensor)
    _check_reduce_op(tensor, op, group)
    c10d = torch.distributed.distributed_c10d
    if group_dst is not None:
        dst = group_dst if group is None else c10d.get_global_rank(group, group_dst)
    if dst is None:
        dst = 0
    raw = unwrap_tensor(tensor)
    if _DIST_REAL:
        # Only the destination ends with the reduction; scale a copy elsewhere.
        is_dst = c10d.get_rank() == dst
        source = raw if is_dst else raw.clone()
        op = _premultiply(tensor, source, op, group)
        if _is_avg(op):
            _ORIG_DIST["reduce"](source, dst=dst, op=_sum_op(), group=group)
            if is_dst:
                _divide(raw, _ORIG_DIST["get_world_size"](group))
            return _done(async_op, [tensor])
        work = _ORIG_DIST["reduce"](source, dst=dst, op=_sum_op() if op is None else op, group=group,
                                    async_op=async_op)
        return work if source is raw else _done(async_op, [tensor])
    if _dist_get_rank() == dst:
        _premultiply(tensor, raw, op, group)
    return _done(async_op, [tensor])


def _group_root(group: Any, rank: Any, group_rank: Any) -> int:
    """The group rank of a collective's root, given as a global rank or a group rank."""

    if group_rank is not None:
        return int(group_rank)
    rank = 0 if rank is None else int(rank)
    if group is None or group is torch.distributed.GroupMember.WORLD:
        return rank
    return torch.distributed.distributed_c10d.get_group_rank(group, rank)


def _dist_gather(
    tensor: torch.Tensor,
    gather_list: list[torch.Tensor] | None = None,
    dst: int | None = None,
    group: Any = None,
    async_op: bool = False,
    group_dst: Any = None,
) -> Any:
    if _not_in_group(group, "gather"):
        return None
    _check_devices(group, tensor, *(gather_list or []))
    if _DIST_REAL:
        options = {"group_dst": group_dst} if group_dst is not None else {"dst": 0 if dst is None else dst}
        targets = [unwrap_tensor(item) for item in gather_list] if gather_list else None
        return _ORIG_DIST["gather"](unwrap_tensor(tensor), targets, group=group, async_op=async_op, **options)
    # torch's checks (_validate_output_list_for_rank); every rank sends this rank's tensor.
    if _group_root(group, dst, group_dst) == _dist_get_rank(group):
        if not gather_list:
            raise ValueError("Argument ``gather_list`` must be specified on destination rank.")
        for target in gather_list:
            _copy_collective_tensor(target, tensor)
    elif gather_list:
        raise ValueError("Argument ``gather_list`` must NOT be specified on non-destination ranks.")
    return _done(async_op, gather_list or [])


def _dist_scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = None,
    src: int | None = None,
    group: Any = None,
    async_op: bool = False,
    group_src: Any = None,
) -> Any:
    if _not_in_group(group, "scatter"):
        return None
    _check_devices(group, tensor, *(scatter_list or []))
    if _DIST_REAL:
        options = {"group_src": group_src} if group_src is not None else {"src": 0 if src is None else src}
        sources = [unwrap_tensor(item) for item in scatter_list] if scatter_list else None
        return _ORIG_DIST["scatter"](unwrap_tensor(tensor), sources, group=group, async_op=async_op, **options)
    rank = _dist_get_rank(group)
    if _group_root(group, src, group_src) == rank:
        if not scatter_list:
            raise ValueError("Argument ``scatter_list`` must be specified on source rank.")
        # The source keeps its own slot.
        unwrap_tensor(tensor).copy_(unwrap_tensor(scatter_list[rank]))
    elif scatter_list:
        raise ValueError("Argument ``scatter_list`` must NOT be specified on non-source ranks.")
    return _done(async_op, [tensor])


def _dist_reduce_scatter(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, "reduce_scatter"):
        return None
    _check_devices(group, output, *input_list)
    _check_reduce_op(output, op, group)
    if _DIST_REAL:
        stacked = torch.stack([unwrap_tensor(item).reshape(-1) for item in input_list])
        op = _premultiply(output, stacked, op, group)
        _real_all_reduce(stacked, op, group)
        target = unwrap_tensor(output)
        target.copy_(stacked[_ORIG_DIST["get_rank"](group)].reshape(target.shape))
        return _done(async_op, [output])
    if input_list:
        _copy_collective_tensor(output, input_list[0])
        _premultiply(output, unwrap_tensor(output), op, group)
    return _done(async_op, [output])


def _dist_reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, _torch_name("reduce_scatter_single", "reduce_scatter_tensor")):
        return None
    _check_devices(group, output, input)
    _check_single_device(group, output, input)
    if _nccl_handles(output, group):
        # ProcessGroupNCCL::reduce_scatter_single checks these.
        if input.dtype != output.dtype:
            raise TypeError("input tensor must be the same type as the output tensor.")
        if input.numel() != output.numel() * _dist_get_world_size(group):
            raise ValueError("input tensor must be the same size as output size times world size")
    _check_reduce_op(output, op, group)
    if _DIST_REAL:
        reduced = unwrap_tensor(input).clone()
        op = _premultiply(output, reduced, op, group)
        _real_all_reduce(reduced, op, group)
        target = unwrap_tensor(output)
        size = _ORIG_DIST["get_world_size"](group)
        target.copy_(reduced.reshape(size, -1)[_ORIG_DIST["get_rank"](group)].reshape(target.shape))
        return _done(async_op, [output])
    _copy_collective_tensor(output, input)
    _premultiply(output, unwrap_tensor(output), op, group)
    return _done(async_op, [output])


def _dist_all_to_all(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, "all_to_all"):
        return None
    _check_devices(group, *output_tensor_list, *input_tensor_list)
    if _DIST_REAL:
        return _ORIG_DIST["all_to_all"](
            [unwrap_tensor(item) for item in output_tensor_list],
            [unwrap_tensor(item) for item in input_tensor_list],
            group=group,
            async_op=async_op,
        )
    for output, source in zip(output_tensor_list, input_tensor_list):
        _copy_collective_tensor(output, source)
    return _done(async_op, output_tensor_list)


def _dist_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Any = None,
    input_split_sizes: Any = None,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    if _not_in_group(group, "all_to_all_single"):
        return None
    _check_devices(group, output, input)
    if _DIST_REAL:
        return _ORIG_DIST["all_to_all_single"](
            unwrap_tensor(output),
            unwrap_tensor(input),
            output_split_sizes,
            input_split_sizes,
            group=group,
            async_op=async_op,
        )
    # With uneven splits, e.g. MoE token dispatch, output and input differ in size.
    _copy_collective_tensor(output, input)
    return _done(async_op, [output])


def _dist_broadcast_object_list(
    object_list: list[Any],
    src: int | None = None,
    group: Any = None,
    device: Any = None,
    group_src: Any = None,
    weights_only: bool = False,
) -> None:
    if _not_in_group(group, "broadcast_object_list"):
        return None
    if _DIST_REAL:
        options = {"group_src": group_src} if group_src is not None else {"src": 0 if src is None else src}
        # The objects travel through gloo on the CPU, whatever device is named.
        return _ORIG_DIST["broadcast_object_list"](
            object_list, group=group, **options, **_weights_only(weights_only)
        )
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
        try:
            result = fn(*cpu_args, **cpu_kwargs)
        except RuntimeError as exc:
            # FakeCUDA keeps tensor data in host memory. A host with less
            # memory than the request fails here, before the simulated GPU
            # could raise the OutOfMemoryError a real GPU raises.
            if _HOST_ALLOCATION_FAILED is None or "can't allocate memory" not in str(exc):
                raise
            planned = fn(*cpu_args, **dict(cpu_kwargs, device=torch.device("meta")))
            _HOST_ALLOCATION_FAILED(planned, normalized.index or 0, exc)
            raise
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
    # aten::to returns ``self`` when dtype, device and memory format already
    # match and no copy is requested (to_will_alias in
    # aten/src/ATen/native/TensorConversions.cpp); any other result is a new
    # tensor, so a move between devices never shares storage. The host
    # tensor computes the result: an alias of it means nothing changed.
    target, normalized, consume_first_arg = _normalize_to_args(args, kwargs)
    is_fake = isinstance(self, FakeCudaTensor)
    base = self.raw_data if is_fake else self
    if normalized is not None:
        index = normalized.index or 0
        kwargs = dict(kwargs)
        kwargs["device"] = torch.device("cpu")
        call_args = args[1:] if consume_first_arg else args
        result = _ORIG_TENSOR_TO(base, *call_args, **kwargs)
        if result is base:
            if is_fake and self.device_index == index:
                return self
            if is_fake or not _MOVING_LOADED_TENSORS:
                result = base.clone()
        return wrap_tensor(result, device_index=index)
    result = _ORIG_TENSOR_TO(base, *args, **kwargs)
    if is_fake and isinstance(result, torch.Tensor):
        if target is not None:
            target_device = (
                torch.device(f"cuda:{target}") if isinstance(target, int) else torch.device(target)
            )
            if target_device.type != "cuda":
                return base.clone() if result is base else result
        if result is base:
            return self
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
    if isinstance(self, FakeCudaTensor):
        # A device-to-host copy is a new tensor, never a view of the device one.
        result = _ORIG_TENSOR_CPU(self.raw_data, **kwargs)
        return self.raw_data.clone() if result is self.raw_data else result
    return _ORIG_TENSOR_CPU(self, **kwargs)


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
    _lazy_init()
    return _CURRENT_DEVICE


def is_initialized() -> bool:
    """``torch.cuda.is_initialized``: False until the first CUDA use."""
    return _INITIALIZED


def _lazy_init() -> None:
    """``torch.cuda._lazy_init``: initialize once, then run the queued calls.

    As in torch/cuda/__init__.py, the queued seed calls run after the other
    queued calls, a call that re-enters returns at once, and a failing call
    raises ``DeferredCudaCallError`` and leaves CUDA uninitialized.
    """
    global _INITIALIZED, _INITIALIZING
    if _INITIALIZED or _INITIALIZING:
        return
    _INITIALIZING = True
    try:
        for hook in list(_LAZY_INIT_HOOKS):
            hook()
        queued = list(_QUEUED_CALLS)
        queued.extend(
            _QUEUED_SEED_CALLS[kind]
            for kind in _QUEUED_SEED_ORDER
            if _QUEUED_SEED_CALLS[kind] is not None
        )
        _QUEUED_CALLS.clear()
        _QUEUED_SEED_CALLS.update(seed=None, seed_all=None)
        _QUEUED_SEED_ORDER.clear()
        for queued_call, orig_traceback in queued:
            try:
                queued_call()
            except Exception as e:
                msg = (
                    f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                    f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                )
                error = getattr(torch.cuda, "DeferredCudaCallError", RuntimeError)
                raise error(msg) from e
    finally:
        _INITIALIZING = False
    _INITIALIZED = True
    torch.cuda._initialized = True


def _rebind_imported(replacements: dict[Any, Any]) -> None:
    """Rebind names torch modules imported from torch.cuda (e.g.
    ``from . import _lazy_init`` in torch.cuda.memory) to their replacements."""
    import sys

    by_identity = {id(original): (original, fake) for original, fake in replacements.items()}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("torch"):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            entry = by_identity.get(id(value))
            if entry is not None and entry[0] is value:
                namespace[name] = entry[1]


def _lazy_call(callable: Any, **kwargs: Any) -> None:
    """``torch.cuda._lazy_call``: run now if initialized, else at initialization."""
    if _INITIALIZED:
        callable()
        return
    entry = (callable, traceback.format_stack())
    if kwargs.get("seed_all", False):
        kind = "seed_all"
    elif kwargs.get("seed", False):
        kind = "seed"
    else:
        _QUEUED_CALLS.append(entry)
        return
    _QUEUED_SEED_CALLS[kind] = entry
    # The latest of manual_seed and manual_seed_all runs last.
    other = "seed" if kind == "seed_all" else "seed_all"
    _QUEUED_SEED_ORDER[:] = [other, kind]


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
    # The host tensors torch.load just built are visible to nobody else, so
    # the simulated device takes their storage instead of a second host copy.
    global _MOVING_LOADED_TENSORS
    memo = {}
    previous, _MOVING_LOADED_TENSORS = _MOVING_LOADED_TENSORS, True
    try:
        return _move_loaded_to_cuda_impl(obj, memo, device_index)
    finally:
        _MOVING_LOADED_TENSORS = previous


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

    for name in _CUDA_MEMORY_API_NAMES:
        function = getattr(torch.cuda.memory, name, None)
        if function is not None and name not in _REAL_CUDA_MEMORY_API:
            _REAL_CUDA_MEMORY_API[name] = function

    # Save original dist functions before patching — needed for fake backend
    # delegation so FSDP gets a real ProcessGroup.
    global _ORIG_DIST_INIT, _ORIG_DIST_DESTROY, _ORIG_DDP
    _ORIG_DIST_INIT = torch.distributed.init_process_group
    _ORIG_DIST_DESTROY = torch.distributed.destroy_process_group
    _ORIG_DDP = torch.nn.parallel.distributed.DistributedDataParallel
    _ORIG_DIST.update(
        (name, getattr(torch.distributed, name))
        for name in _REAL_COLLECTIVES
        if hasattr(torch.distributed, name)
    )
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
    lazy_replacements = {
        "_lazy_init": _lazy_init,
        "_lazy_call": _lazy_call,
        "is_initialized": is_initialized,
    }
    _rebind_imported(
        {getattr(torch.cuda, name): fake for name, fake in lazy_replacements.items()}
    )
    for name, fake in lazy_replacements.items():
        setattr(torch.cuda, name, fake)
    torch.cuda.init = _lazy_init
    torch.cuda._initialized = _INITIALIZED
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
        "_cuda_memorySnapshot": lambda arg=None: {
            "segments": [],
            "device_traces": [[] for _ in range(_NUM_DEVICES)],
            "allocator_settings": {},
            "external_annotations": [],
        },
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
    if "new_group" in _ORIG_DIST:
        torch.distributed.new_group = _dist_new_group
    if "monitored_barrier" in _ORIG_DIST:
        torch.distributed.monitored_barrier = _dist_monitored_barrier
    for name in _ORIG_DIST:
        patched = getattr(torch.distributed, name, None)
        if patched is not None and patched is not _ORIG_DIST[name]:
            setattr(torch.distributed, name, _as_installed_signature(name, patched))
    if "_make_nccl_premul_sum" in _ORIG_DIST:
        torch.distributed._make_nccl_premul_sum = _dist_make_nccl_premul_sum
        _remember_premul_factors_of_calls()
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
