from __future__ import annotations

import functools
from collections import OrderedDict

import torch

from . import ops as _ops  # noqa: F401
from .backend_module import FakeGpuBackendModule, get_current_device
from .device_guard import FakeGpuDeviceGuard
from .hooks import FakeGpuPrivateUse1Hooks
from .tensor_wrapper import register_parameter, unwrap_tensor

_INITIALIZED = False
_ORIG_TORCH_LOAD = None
_ORIG_TENSOR_TO = None
_ORIG_MODULE_TO = None


def _normalize_module_device(device) -> torch.device:
    if device is None:
        return torch.device("fgpu:0")
    if isinstance(device, int):
        return torch.device(f"fgpu:{device}")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "fgpu":
            raise RuntimeError(f"Invalid device for Module.fgpu(): {device}")
        return torch.device(f"fgpu:{device.index or 0}")
    raise TypeError(f"Unsupported device specifier: {device!r}")


def _normalize_fgpu_target(device):
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.type == "fgpu" and device.index is None:
        return torch.device(f"fgpu:{get_current_device()}")
    return device


def _module_fgpu(self, device=None, _memo=None):
    target = _normalize_module_device(device)
    if _memo is None:
        _memo = {"parameters": {}, "buffers": {}}

    for child in self.children():
        _module_fgpu(child, target, _memo)

    for name, param in list(self._parameters.items()):
        if param is None:
            continue
        param_id = id(param)
        wrapped = _memo["parameters"].get(param_id)
        if wrapped is None:
            if hasattr(param, "raw_data") and getattr(param.device, "type", None) == "fgpu":
                wrapped = param
            else:
                converted = param.to(target)
                wrapped = torch.nn.Parameter(converted, requires_grad=param.requires_grad)
                wrapped.raw_data = param.detach().clone().requires_grad_(param.requires_grad)
                wrapped.device_index = target.index or 0
                if param.grad is not None:
                    grad = param.grad.to(target)
                    grad.raw_data = unwrap_tensor(grad)
                    wrapped.grad = grad
            register_parameter(wrapped)
            _memo["parameters"][param_id] = wrapped
        self._parameters[name] = wrapped

    for name, buf in list(self._buffers.items()):
        if buf is None:
            continue
        buf_id = id(buf)
        converted = _memo["buffers"].get(buf_id)
        if converted is None:
            if hasattr(buf, "raw_data") and getattr(buf.device, "type", None) == "fgpu":
                converted = buf
            else:
                converted = buf.to(target)
                converted.raw_data = unwrap_tensor(converted)
                converted.device_index = target.index or 0
            _memo["buffers"][buf_id] = converted
        self._buffers[name] = converted

    return self


def _is_fgpu_map_location(map_location) -> bool:
    if isinstance(map_location, str):
        return map_location == "fgpu" or map_location.startswith("fgpu:")
    if isinstance(map_location, torch.device):
        return map_location.type == "fgpu"
    return False


def _move_loaded_to_fgpu(obj):
    memo = {}
    return _move_loaded_to_fgpu_impl(obj, memo)


def _move_loaded_to_fgpu_impl(obj, memo):
    obj_id = id(obj)
    cached = memo.get(obj_id)
    if cached is not None:
        return cached

    if isinstance(obj, torch.Tensor):
        moved = obj.to("fgpu")
        memo[obj_id] = moved
        return moved

    if isinstance(obj, list):
        result = []
        memo[obj_id] = result
        result.extend(_move_loaded_to_fgpu_impl(item, memo) for item in obj)
        return result

    if isinstance(obj, OrderedDict):
        result = OrderedDict()
        memo[obj_id] = result
        for key, value in obj.items():
            result[key] = _move_loaded_to_fgpu_impl(value, memo)
        return result

    if isinstance(obj, dict):
        result = obj.__class__()
        memo[obj_id] = result
        for key, value in obj.items():
            result[key] = _move_loaded_to_fgpu_impl(value, memo)
        return result

    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            result = obj.__class__(*( _move_loaded_to_fgpu_impl(item, memo) for item in obj))
        else:
            result = tuple(_move_loaded_to_fgpu_impl(item, memo) for item in obj)
        memo[obj_id] = result
        return result

    return obj


def _patch_torch_load() -> None:
    global _ORIG_TORCH_LOAD
    if _ORIG_TORCH_LOAD is not None:
        return

    _ORIG_TORCH_LOAD = torch.load

    @functools.wraps(_ORIG_TORCH_LOAD)
    def _fgpu_torch_load(*args, **kwargs):
        map_location = kwargs.get("map_location")
        if map_location is None and len(args) >= 2:
            map_location = args[1]

        if _is_fgpu_map_location(map_location):
            if len(args) >= 2:
                args = (args[0], "cpu") + args[2:]
            else:
                kwargs["map_location"] = "cpu"
            loaded = _ORIG_TORCH_LOAD(*args, **kwargs)
            return _move_loaded_to_fgpu(loaded)

        return _ORIG_TORCH_LOAD(*args, **kwargs)

    torch.load = _fgpu_torch_load


def _patch_to_methods() -> None:
    global _ORIG_TENSOR_TO, _ORIG_MODULE_TO
    if _ORIG_TENSOR_TO is None:
        _ORIG_TENSOR_TO = torch.Tensor.to

        def _fgpu_tensor_to(self, *args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _normalize_fgpu_target(kwargs["device"])
            if args:
                first = _normalize_fgpu_target(args[0])
                if first is not args[0]:
                    args = (first,) + args[1:]
            return _ORIG_TENSOR_TO(self, *args, **kwargs)

        torch.Tensor.to = _fgpu_tensor_to

    if _ORIG_MODULE_TO is None:
        _ORIG_MODULE_TO = torch.nn.Module.to

        def _fgpu_module_to(self, *args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _normalize_fgpu_target(kwargs["device"])
            if args:
                first = _normalize_fgpu_target(args[0])
                if first is not args[0]:
                    args = (first,) + args[1:]
            return _ORIG_MODULE_TO(self, *args, **kwargs)

        torch.nn.Module.to = _fgpu_module_to


def init_privateuse1() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    torch.utils.rename_privateuse1_backend("fgpu")
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_packed_sequence=True,
        for_storage=False,
    )
    torch.nn.Module.fgpu = _module_fgpu
    _patch_torch_load()
    _patch_to_methods()
    torch._register_device_module("fgpu", FakeGpuBackendModule())
    torch._C._acc.register_python_privateuseone_hook(FakeGpuPrivateUse1Hooks())
    torch._C._acc.register_python_privateuseone_device_guard(FakeGpuDeviceGuard())
    _INITIALIZED = True
