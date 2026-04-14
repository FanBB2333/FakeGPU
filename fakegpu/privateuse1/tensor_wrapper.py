from __future__ import annotations

import weakref

import torch

from .backend_module import get_current_device

_REGISTERED_PARAMETERS: "weakref.WeakSet[FgpuTensor]" = weakref.WeakSet()


class FgpuTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, device_index=None):
        out = torch._C._acc.create_empty_tensor(size, dtype)
        out.__class__ = cls
        out.raw_data = raw_data
        out.device_index = get_current_device() if device_index is None else int(device_index)
        return out

    def __init__(self, size, dtype, raw_data=None, device_index=None):
        # Tensor subclasses created from C++ allocation paths do not reliably
        # call __init__, so the real state is attached in __new__.
        return None

    @property
    def device(self):
        return torch.device(f"fgpu:{self.device_index}")

    @property
    def is_fgpu(self):
        return True

    @property
    def grad(self):
        raw = getattr(self.raw_data, "grad", None)
        if raw is None:
            self.__dict__.pop("_fgpu_grad", None)
            return None
        cached = self.__dict__.get("_fgpu_grad")
        if cached is not None and getattr(cached, "raw_data", None) is raw:
            return cached
        wrapped = wrap_tensor(raw.detach(), device_index=self.device_index)
        self.__dict__["_fgpu_grad"] = wrapped
        return wrapped

    @grad.setter
    def grad(self, value) -> None:
        if value is None:
            if getattr(self, "raw_data", None) is not None:
                self.raw_data.grad = None
            self.__dict__.pop("_fgpu_grad", None)
            return
        wrapped = value if isinstance(value, FgpuTensor) else wrap_tensor(value)
        self.__dict__["_fgpu_grad"] = wrapped
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
        raw_gradient = unwrap_tensor(gradient)
        self.raw_data.backward(
            gradient=raw_gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )
        sync_parameter_grads()


def wrap_tensor(t: torch.Tensor, device_index=None) -> FgpuTensor:
    if isinstance(t, FgpuTensor):
        return t
    out = FgpuTensor(
        tuple(t.shape),
        t.dtype,
        t,
        device_index=get_current_device() if device_index is None else int(device_index),
    )
    return out


def unwrap_tensor(obj):
    if isinstance(obj, FgpuTensor):
        return obj.raw_data
    return obj


def register_parameter(t: FgpuTensor) -> None:
    _REGISTERED_PARAMETERS.add(t)


def sync_parameter_grads() -> None:
    for param in list(_REGISTERED_PARAMETERS):
        raw = getattr(param, "raw_data", None)
        if raw is None:
            continue
        if raw.grad is None:
            param.grad = None
            continue
        grad = wrap_tensor(raw.grad.detach(), device_index=getattr(param, "device_index", get_current_device()))
        param.grad = grad


def _tree_map(fn, obj):
    if isinstance(obj, tuple):
        return tuple(_tree_map(fn, item) for item in obj)
    if isinstance(obj, list):
        return [_tree_map(fn, item) for item in obj]
    if isinstance(obj, dict):
        return {key: _tree_map(fn, value) for key, value in obj.items()}
    return fn(obj)


def _wrap_function_result(obj, device_index):
    if isinstance(obj, torch.Tensor):
        return wrap_tensor(obj, device_index=device_index)
    return obj


def _infer_device_index(args, kwargs) -> int:
    for value in list(args) + list(kwargs.values()):
        index = _find_fgpu_device_index(value)
        if index is not None:
            return index
    return get_current_device()


def _find_fgpu_device_index(value):
    if isinstance(value, FgpuTensor):
        return value.device_index
    if isinstance(value, tuple):
        for item in value:
            index = _find_fgpu_device_index(item)
            if index is not None:
                return index
        return None
    if isinstance(value, list):
        for item in value:
            index = _find_fgpu_device_index(item)
            if index is not None:
                return index
        return None
    if isinstance(value, dict):
        for item in value.values():
            index = _find_fgpu_device_index(item)
            if index is not None:
                return index
        return None
    return None
