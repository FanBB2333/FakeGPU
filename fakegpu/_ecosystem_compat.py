from __future__ import annotations

import copy
import dataclasses
import functools
import re
from typing import Any


def _patch_accelerate_utils() -> None:
    """Patch ``accelerate`` detection helpers so the library treats FakeGPU as real CUDA.

    When ``fakegpu.patch_torch()`` has already set ``torch.cuda.is_available()``
    to ``True``, accelerate's ``is_cuda_available()`` normally picks that up.
    However some helpers cache their result at import time or check extra
    conditions (like ``torch.cuda.device_count() > 0``).  We explicitly patch
    them here so that accelerate **always** sees CUDA as available regardless
    of import order.
    """
    try:
        import accelerate.utils.imports as _aui
    except ImportError:
        return

    _aui.is_cuda_available = lambda: True
    if hasattr(_aui, "is_bf16_available"):
        # Accelerate answers bf16 on CUDA with torch.cuda.is_bf16_supported()
        # and calls this as is_bf16_available(True).
        def _is_bf16_available(ignore_tpu: bool = False) -> bool:
            import torch

            return bool(torch.cuda.is_bf16_supported())

        _aui.is_bf16_available = _is_bf16_available

    # Re-export to top-level accelerate.utils if the names are there
    try:
        import accelerate.utils as _au

        _au.is_cuda_available = _aui.is_cuda_available
        if hasattr(_au, "is_bf16_available"):
            _au.is_bf16_available = _aui.is_bf16_available
    except (ImportError, AttributeError):
        pass


def _patch_upstream_fakecuda_tensor_compat(upstream: Any, torch_mod: Any) -> None:
    fake_tensor_cls = upstream.FakeCudaTensor

    # torch.utils.checkpoint groups non-CPU tensors by ``tensor.get_device()``.
    # The subclass stores CPU data internally, so inheriting Tensor.get_device
    # leaks the backing CPU value (-1) and checkpoint tries to enter cuda:-1.
    fake_tensor_cls.get_device = lambda self: int(self.device_index)

    def _patched_set_(self, source, storage_offset=0, size=None, stride=None):
        raw_source = upstream.unwrap_tensor(source)
        with torch_mod.no_grad():
            if size is None and stride is None and storage_offset == 0:
                torch_mod.Tensor.set_(self, raw_source)
            else:
                if size is None:
                    size = tuple(raw_source.shape)
                if stride is None:
                    stride = tuple(raw_source.stride())
                torch_mod.Tensor.set_(self, raw_source, storage_offset, size, stride)

        # FSDP mutates FlatParameter storage via ``set_()``. Keep the
        # CPU-side shadow tensor in sync so subsequent unwraps see the new
        # storage instead of the previously freed one.
        self.raw_data = self.as_subclass(torch_mod.Tensor)

        if isinstance(source, fake_tensor_cls):
            self.device_index = source.device_index
        return self

    fake_tensor_cls.set_ = _patched_set_
    fake_tensor_cls.record_stream = lambda self, stream: None
    fake_tensor_cls.is_cpu = property(lambda self: False)

    # Array API consumers (including scikit-learn) use torch.asarray, which
    # does not dispatch through FakeCudaTensor.__torch_function__.
    original_asarray = torch_mod.asarray

    @functools.wraps(original_asarray)
    def _asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False):
        from torch.overrides import _get_current_function_mode_stack
        from torch.utils._device import DeviceContext

        source = obj.device if isinstance(obj, torch_mod.Tensor) else torch_mod.device("cpu")
        # Like torch.get_default_device, inspect the active device context, but
        # preserve the input device when no default was explicitly selected.
        if device is None:
            device = next((
                mode.device for mode in reversed(_get_current_function_mode_stack())
                if isinstance(mode, DeviceContext)
            ), source)
        normalized = upstream._normalize_cuda_device(device, allow_none=True)
        target = normalized if normalized is not None else torch_mod.device(device)
        changes_device = source.type != target.type or (
            source.type == "cuda" and source.index != target.index
        )
        if copy is False and changes_device:
            raise ValueError("cannot share memory between different devices")
        result = original_asarray(
            upstream.unwrap_tensor(obj),
            dtype=dtype,
            device="cpu" if normalized is not None else target,
            copy=True if changes_device else copy,
            requires_grad=requires_grad,
        )
        return (
            upstream.wrap_tensor(result, device_index=normalized.index)
            if normalized is not None else result
        )

    torch_mod.asarray = _asarray


@functools.lru_cache(maxsize=None)
def _shard_metadata_types() -> tuple[Any, Any]:
    """Return (ShardMetadata, ShardedTensorMetadata), or Nones without them."""
    try:
        from torch.distributed._shard.metadata import ShardMetadata
        from torch.distributed._shard.sharded_tensor.metadata import (
            ShardedTensorMetadata,
        )
    except Exception:
        return None, None
    return ShardMetadata, ShardedTensorMetadata


def _patch_upstream_all_gather_object(upstream: Any, torch_mod: Any) -> None:
    def _clone_gathered_object_for_rank(obj: Any, rank: int) -> Any:
        ShardMetadata, ShardedTensorMetadata = _shard_metadata_types()

        if ShardMetadata is not None and isinstance(obj, ShardMetadata):
            shard_offsets = list(obj.shard_offsets)
            if shard_offsets:
                shard_offsets[0] = int(obj.shard_sizes[0]) * rank
            placement = re.sub(
                r"rank:\d+/", f"rank:{rank}/", str(obj.placement), count=1
            )
            return dataclasses.replace(
                obj,
                shard_offsets=shard_offsets,
                placement=placement,
            )

        if ShardedTensorMetadata is not None and isinstance(obj, ShardedTensorMetadata):
            return dataclasses.replace(
                obj,
                shards_metadata=[
                    _clone_gathered_object_for_rank(shard, rank)
                    for shard in obj.shards_metadata
                ],
            )

        if isinstance(obj, list):
            return [_clone_gathered_object_for_rank(item, rank) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_clone_gathered_object_for_rank(item, rank) for item in obj)
        if isinstance(obj, dict):
            return obj.__class__(
                (key, _clone_gathered_object_for_rank(value, rank))
                for key, value in obj.items()
            )
        return copy.deepcopy(obj)

    def _patched_all_gather_object(
        object_list: list[Any], obj: Any, group: Any = None, weights_only: bool = False
    ) -> None:
        if upstream._DIST_REAL:
            # Other processes hold the other ranks' objects.
            return upstream._ORIG_DIST["all_gather_object"](
                object_list, obj, group=group, **upstream._weights_only(weights_only)
            )
        for index in range(len(object_list)):
            object_list[index] = _clone_gathered_object_for_rank(obj, index)
        return None

    upstream._dist_all_gather_object = _patched_all_gather_object
    torch_mod.distributed.all_gather_object = _patched_all_gather_object


def _patch_fsdp_device_handling() -> None:
    """Patch FSDP internal device resolution for FakeGPU compatibility.

    On macOS, ``torch.device(0)`` resolves to ``mps:0`` via the C++
    accelerator lookup, which cannot be overridden from Python.  This causes
    FSDP's device_id resolution and ``_FSDPDeviceHandle.from_device()`` to
    produce MPS-backed handles while model parameters report ``cuda:0``
    (from FakeCudaTensor), leading to device mismatch errors.

    We fix this by:
    1. Wrapping ``_FSDPDeviceHandle.from_device`` to remap any non-cuda
       device to cuda before creating the handle.
    2. Wrapping ``_get_device_from_device_id`` to fix integer device_id
       resolution (``torch.device(0)`` → ``mps:0`` → remapped to ``cuda:0``).
    """
    try:
        from torch.distributed.fsdp._common_utils import _FSDPDeviceHandle
    except ImportError:
        return  # FSDP not available

    _orig_from_device = _FSDPDeviceHandle.from_device.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_device(cls, device):
        import torch as _torch

        # Remap non-cuda device types to cuda for FakeGPU
        if device.type not in ("cuda", "cpu", "meta"):
            device = _torch.device(
                "cuda", device.index if device.index is not None else 0
            )
        return _orig_from_device(cls, device)

    _FSDPDeviceHandle.from_device = _patched_from_device

    # Also patch _get_device_from_device_id for the consistency check in
    # _get_compute_device (compares device_from_device_id vs param.device).
    try:
        import torch.distributed.fsdp._init_utils as _fsdp_init
        import torch.distributed.fsdp.fully_sharded_data_parallel as _fsdp_mod
    except ImportError:
        return

    _orig_get_device = getattr(_fsdp_init, "_get_device_from_device_id", None)
    if _orig_get_device is not None:

        @functools.wraps(_orig_get_device)
        def _patched_get_device(device_id, rank, device_handle):
            result = _orig_get_device(device_id, rank, device_handle)
            if result is not None and result.type not in ("cuda", "cpu", "meta"):
                import torch as _torch

                result = _torch.device(
                    "cuda", result.index if result.index is not None else 0
                )
            return result

        _fsdp_init._get_device_from_device_id = _patched_get_device
        # Also patch the direct import in the FSDP module
        if hasattr(_fsdp_mod, "_get_device_from_device_id"):
            _fsdp_mod._get_device_from_device_id = _patched_get_device

    # A module still on the CPU computes on torch.device(current_device()),
    # which is mps:N on Apple Silicon, so FSDP(cpu_module, device_id=cuda:0)
    # failed its consistency check. Resolve the device, then check as FSDP does.
    _orig_get_compute_device = getattr(_fsdp_init, "_get_compute_device", None)
    if _orig_get_compute_device is not None:

        @functools.wraps(_orig_get_compute_device)
        def _patched_get_compute_device(module, ignored_params, device_from_device_id, rank, device_handle):
            compute_device = _orig_get_compute_device(module, ignored_params, None, rank, device_handle)
            if compute_device.type not in ("cuda", "cpu", "meta"):
                import torch as _torch

                compute_device = _torch.device("cuda", compute_device.index or 0)
            if device_from_device_id is not None and compute_device != device_from_device_id:
                raise ValueError(
                    f"Inconsistent compute device and `device_id` on rank {rank}: "
                    f"{compute_device} vs {device_from_device_id}"
                )
            return compute_device

        _fsdp_init._get_compute_device = _patched_get_compute_device


def _patch_default_accelerator(torch_mod: Any) -> None:
    """Report CUDA as the machine's accelerator, as on a CUDA build with GPUs.

    ``torch._C._get_accelerator()`` picks the device of FSDP2's default mesh,
    of FSDP1's device handle and of ``barrier``. On Apple Silicon it returns
    ``mps`` even while FakeCUDA hides the MPS backend, so ``fully_shard``
    without a mesh failed on the missing ``torch.mps.current_device``.
    """

    def _get_accelerator(check: bool = False) -> Any:
        return torch_mod.device("cuda")

    torch_mod._C._get_accelerator = _get_accelerator


def _patch_fsdp2_post_backward(upstream: Any) -> None:
    """Keep FSDP2's post-backward hook on the autograd graph FakeCUDA builds.

    FakeCUDA records autograd on the CPU tensors behind each simulated GPU
    tensor, while ``torch.autograd.Function.apply`` records it on the tensors
    it receives. ``RegisterPostBackwardFunction`` therefore cut the graph
    between FSDP2 modules: the earlier module's parameters got no gradient and
    the later module reduced its gradients only in the final callback. Apply
    it to the CPU tensors and hand the simulated GPU tensors on, so the
    gradients and their reduce-scatter happen during backward, as on a GPU.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
            RegisterPostBackwardFunction,
        )
    except ImportError:
        return
    fake_tensor_cls = upstream.FakeCudaTensor
    # A bound classmethod, a C descriptor on newer PyTorch.
    original = RegisterPostBackwardFunction.apply

    def _apply(cls: Any, param_group: Any, *inputs: Any) -> Any:
        devices = [
            tensor.device_index if isinstance(tensor, fake_tensor_cls) else None
            for tensor in inputs
        ]
        if all(device is None for device in devices):
            return original(param_group, *inputs)
        outputs = original(param_group, *(upstream.unwrap_tensor(t) for t in inputs))
        return tuple(
            output if device is None else upstream.wrap_tensor(output, device_index=device)
            for output, device in zip(outputs, devices)
        )

    RegisterPostBackwardFunction.apply = classmethod(_apply)


def _patch_fsdp_state_moves(upstream: Any, torch_mod: Any) -> None:
    """Move CPU parameters and buffers onto the simulated GPU when FSDP does.

    FSDP1 with ``device_id`` and FSDP2's ``fully_shard`` move a module that is
    still on the CPU with ``tensor.data = tensor.to(device)``, which leaves a
    simulated GPU tensor's CPU storage in a plain CPU tensor. Move those tensors
    the way ``Module.to("cuda")`` does under FakeCUDA, in place, and leave the
    rest to FSDP.
    """
    fake_tensor_cls = upstream.FakeCudaTensor

    def _move(tensors: list[Any], device: Any) -> None:
        holder = torch_mod.nn.Module()
        for index, tensor in enumerate(tensors):
            if isinstance(tensor, torch_mod.nn.Parameter):
                holder._parameters[f"p{index}"] = tensor
            else:
                holder._buffers[f"b{index}"] = tensor
        # FakeCUDA's Module.to swaps each moved tensor in place.
        holder.to(device)

    def _wrap(original: Any) -> Any:
        @functools.wraps(original)
        def _move_states_to_device(params: Any, buffers: Any, device: Any) -> Any:
            if device is not None and torch_mod.device(device).type == "cuda":
                on_cpu = [t for t in (*params, *buffers) if not isinstance(t, fake_tensor_cls) and t.device.type == "cpu"]
                if on_cpu:
                    _move(on_cpu, device)
                params = [t for t in params if not isinstance(t, fake_tensor_cls)]
                buffers = [t for t in buffers if not isinstance(t, fake_tensor_cls)]
            return original(params, buffers, device)

        _move_states_to_device.__fakegpu_patched__ = True
        return _move_states_to_device

    import importlib

    for name in (
        "torch.distributed.fsdp._init_utils",
        "torch.distributed.fsdp._fully_shard._fsdp_init",
        "torch.distributed.fsdp._fully_shard._fully_shard",
        "torch.distributed._composable.replicate_with_fsdp",
    ):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        original = getattr(module, "_move_states_to_device", None)
        if original is not None and not getattr(original, "__fakegpu_patched__", False):
            module._move_states_to_device = _wrap(original)


def _patch_fsdp_simulated_gpu(upstream: Any, torch_mod: Any) -> None:
    """Let FSDP1 and FSDP2 place, shard and reduce on the simulated GPUs."""

    _patch_default_accelerator(torch_mod)
    _patch_fsdp2_post_backward(upstream)
    _patch_fsdp_state_moves(upstream, torch_mod)


def _patch_fsdp_runtime_compat(fake_tensor_cls: type) -> None:
    try:
        import torch
        import torch.distributed.fsdp._runtime_utils as _fsdp_runtime
    except ImportError:
        return

    _orig_register = getattr(_fsdp_runtime, "_register_post_backward_hook", None)
    if _orig_register is None:
        return

    @functools.wraps(_orig_register)
    def _patched_register_post_backward_hook(state, handle):
        if not handle:
            return _orig_register(state, handle)

        flat_param = handle.flat_param
        if not isinstance(flat_param, fake_tensor_cls):
            return _orig_register(state, handle)

        if not torch.is_grad_enabled():
            return

        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            return

        # FakeCudaTensor aliases do not expose the expected AccumulateGrad via
        # ``expand_as(...).grad_fn.next_functions`` after FSDP's internal
        # storage rebinding. Use the post-accumulate hook instead to avoid
        # asserting during forward.
        hook = functools.partial(_fsdp_runtime._post_backward_hook, state, handle)
        hook_handle = flat_param.register_post_accumulate_grad_hook(hook)
        flat_param._post_backward_hook_state = (None, hook_handle)
    _fsdp_runtime._register_post_backward_hook = _patched_register_post_backward_hook


__all__ = [
    "_patch_accelerate_utils",
    "_patch_fsdp_device_handling",
    "_patch_fsdp_simulated_gpu",
    "_patch_fsdp_runtime_compat",
    "_patch_upstream_all_gather_object",
    "_patch_upstream_fakecuda_tensor_compat",
]
