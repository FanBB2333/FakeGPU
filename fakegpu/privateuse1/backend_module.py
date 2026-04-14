from __future__ import annotations

import os

import torch

_NUM_DEVICES = int(os.environ.get("FAKEGPU_DEVICE_COUNT", "8"))
_CURRENT_DEVICE = 0


def get_device_count() -> int:
    return _NUM_DEVICES


def get_current_device() -> int:
    return _CURRENT_DEVICE


def normalize_device_index(device, optional: bool = False) -> int:
    if device is None:
        return get_current_device() if optional else 0
    if isinstance(device, int):
        index = device
    else:
        if isinstance(device, str):
            device = torch.device(device)
        if hasattr(device, "index") and device.index is not None:
            index = device.index() if callable(device.index) else device.index
        else:
            index = get_current_device()
    index = 0 if index is None else int(index)
    if index < 0 or index >= get_device_count():
        raise RuntimeError(f"Invalid fgpu device index {index}, expected 0 <= index < {get_device_count()}")
    return index


def set_current_device(device) -> None:
    global _CURRENT_DEVICE
    _CURRENT_DEVICE = normalize_device_index(device, optional=True)


class _DeviceCtx:
    def __init__(self, device) -> None:
        self.device_index = normalize_device_index(device, optional=True)
        self.prev_device_index = get_current_device()

    def __enter__(self):
        set_current_device(self.device_index)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        set_current_device(self.prev_device_index)
        return None


class _Utils:
    @staticmethod
    def _get_device_index(device, optional: bool = False) -> int:
        return normalize_device_index(device, optional=optional)


class FakeGpuBackendModule:
    _utils = _Utils()

    def is_initialized(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return get_current_device()

    def set_device(self, device) -> None:
        set_current_device(device)

    def device(self, device) -> _DeviceCtx:
        return _DeviceCtx(device)

    def device_count(self) -> int:
        return get_device_count()

    def _is_in_bad_fork(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        return None

    def get_rng_state(self, device="fgpu"):
        raise NotImplementedError("FakeGPU PrivateUse1 does not expose device RNG state yet")

    def set_rng_state(self, new_state, device="fgpu") -> None:
        raise NotImplementedError("FakeGPU PrivateUse1 does not expose device RNG state yet")

    def get_amp_supported_dtype(self):
        return []
