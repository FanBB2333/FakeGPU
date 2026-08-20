"""Raise on operations that mix tensors from different fake devices.

Split out of ``torch_patch`` unchanged. Real CUDA refuses to operate across
devices; these wrappers reproduce that refusal for FakeCUDA tensors, over
the torch functions and ``Tensor`` dunders that take more than one tensor.
Set ``FAKEGPU_CROSS_DEVICE_CHECK=0`` to turn the check off.
"""

from __future__ import annotations

import functools
import os
from typing import Any


_CROSS_DEVICE_CHECK = os.environ.get("FAKEGPU_CROSS_DEVICE_CHECK", "1") != "0"


def _is_fake_tensor(tensor: Any) -> bool:
    """Return whether a tensor is owned by PyTorch FakeTensorMode."""
    return getattr(tensor, "fake_mode", None) is not None or (
        type(tensor).__module__ == "torch._subclasses.fake_tensor"
        and type(tensor).__name__ == "FakeTensor"
    )


def _check_same_device(*tensors: Any) -> None:
    """Raise RuntimeError if tensors span multiple fake CUDA devices."""
    if not _CROSS_DEVICE_CHECK:
        return

    first_dev: int | None = None
    for t in tensors:
        # FakeCudaTensor carries its device index as an attribute.
        dev = getattr(t, "device_index", None)
        if dev is None:
            continue  # untracked tensor (e.g. pure CPU) — skip
        if first_dev is None:
            first_dev = dev
        elif dev != first_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, cuda:{first_dev} and cuda:{dev}!"
            )


_BINARY_DUNDER_TORCH_OPS: dict[str, Any] = {
    "__add__": lambda torch_mod, self, other: torch_mod.add(self, other),
    "__radd__": lambda torch_mod, self, other: torch_mod.add(other, self),
    "__sub__": lambda torch_mod, self, other: torch_mod.sub(self, other),
    "__rsub__": lambda torch_mod, self, other: torch_mod.sub(other, self),
    "__mul__": lambda torch_mod, self, other: torch_mod.mul(self, other),
    "__rmul__": lambda torch_mod, self, other: torch_mod.mul(other, self),
    "__truediv__": lambda torch_mod, self, other: torch_mod.true_divide(self, other),
    "__rtruediv__": lambda torch_mod, self, other: torch_mod.true_divide(other, self),
    "__matmul__": lambda torch_mod, self, other: torch_mod.matmul(self, other),
    "__rmatmul__": lambda torch_mod, self, other: torch_mod.matmul(other, self),
}


def _wrap_multi_tensor_op(orig_fn: Any, torch_mod: Any) -> Any:
    """Wrap a torch function to check device consistency of tensor args."""

    @functools.wraps(orig_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tensors = []
        for a in args:
            if isinstance(a, torch_mod.Tensor):
                tensors.append(a)
            elif isinstance(a, (list, tuple)):
                for item in a:
                    if isinstance(item, torch_mod.Tensor):
                        tensors.append(item)
        for v in kwargs.values():
            if isinstance(v, torch_mod.Tensor):
                tensors.append(v)
        if len(tensors) >= 2:
            _check_same_device(*tensors)
        return orig_fn(*args, **kwargs)

    return wrapper


def _wrap_tensor_binary_op(
    orig_fn: Any,
    dunder_name: str,
    torch_mod: Any,
) -> Any:
    """Wrap a Tensor binary method to check cross-device with torch-friendly calls."""

    @functools.wraps(orig_fn)
    def wrapper(self: Any, other: Any) -> Any:
        if isinstance(other, torch_mod.Tensor):
            _check_same_device(self, other)
        torch_op = _BINARY_DUNDER_TORCH_OPS.get(dunder_name)
        if torch_op is not None:
            return torch_op(torch_mod, self, other)
        return orig_fn(self, other)

    return wrapper
