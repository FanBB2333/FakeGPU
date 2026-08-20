from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence

from ._api import InitResult as NativeInitResult
from ._api import _FakeGpuRuntimeConfig
from ._api import init as _native_init
from ._api import is_initialized as _native_is_initialized

RuntimeName = Literal["auto", "native", "fakecuda"]


@dataclass(frozen=True)
class RuntimeInitResult:
    runtime: Literal["native", "fakecuda"]
    backend: str
    native_result: NativeInitResult | None = None

    @property
    def lib_dir(self):
        return self.native_result.lib_dir if self.native_result is not None else None

    @property
    def handles(self):
        return self.native_result.handles if self.native_result is not None else {}


def init(
    *,
    runtime: RuntimeName = "auto",
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    mode: str | None = None,
    oom_policy: str | None = None,
    unsupported_api: str | None = None,
    dist_mode: str | None = None,
    cluster_config: str | os.PathLike[str] | None = None,
    coordinator_addr: str | None = None,
    coordinator_transport: str | None = None,
    profile: str | None = None,
    device_count: int | None = None,
    devices: str | Sequence[str] | None = None,
    update_env: bool = True,
    force: bool = False,
) -> RuntimeInitResult:
    runtime_norm = (runtime or "auto").strip().lower()
    if runtime_norm not in {"auto", "native", "fakecuda"}:
        raise ValueError(
            f"Unsupported fakegpu runtime {runtime!r}. "
            "Expected one of: 'auto', 'native', 'fakecuda'."
        )

    selected_runtime = runtime_norm
    if selected_runtime == "auto":
        # The vendored fakegpu._upstream backend ships with this package, so
        # the fakecuda runtime is always available and "auto" resolves to it.
        selected_runtime = "fakecuda"

    if selected_runtime == "native":
        config = _FakeGpuRuntimeConfig(
            mode=mode,
            oom_policy=oom_policy,
            unsupported_api=unsupported_api,
            dist_mode=dist_mode,
            cluster_config=cluster_config,
            coordinator_addr=coordinator_addr,
            coordinator_transport=coordinator_transport,
            profile=profile,
            device_count=device_count,
            devices=devices,
        )
        return _init_native_runtime(
            config,
            build_dir=build_dir,
            lib_dir=lib_dir,
            update_env=update_env,
            force=force,
        )

    return _init_fakecuda_runtime(
        device_count=device_count,
        profile=profile,
        devices=devices,
        force=force,
    )


def is_initialized() -> bool:
    if _native_is_initialized():
        return True

    from .torch_patch import is_patched

    return is_patched()


def patch_torch(*, num_devices: int | None = None, device_name: str | None = None):
    from .torch_patch import patch

    return patch(num_devices=num_devices, device_name=device_name)


def _init_native_runtime(
    config: _FakeGpuRuntimeConfig,
    *,
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    update_env: bool = True,
    force: bool = False,
) -> RuntimeInitResult:
    native_result = _native_init(
        build_dir=build_dir,
        lib_dir=lib_dir,
        mode=config.mode,
        oom_policy=config.oom_policy,
        unsupported_api=config.unsupported_api,
        dist_mode=config.dist_mode,
        cluster_config=config.cluster_config,
        coordinator_addr=config.coordinator_addr,
        coordinator_transport=config.coordinator_transport,
        profile=config.profile,
        device_count=config.device_count,
        devices=config.devices,
        update_env=update_env,
        force=force,
    )
    return RuntimeInitResult(runtime="native", backend="native", native_result=native_result)


def _init_fakecuda_runtime(
    *,
    device_count: int | None = None,
    profile: str | None = None,
    devices: str | Sequence[str] | None = None,
    force: bool = False,
) -> RuntimeInitResult:
    os.environ["FAKEGPU_RUNTIME"] = "fakecuda"
    if profile is not None:
        os.environ["FAKEGPU_PROFILE"] = str(profile)
        if devices is None:
            os.environ.pop("FAKEGPU_PROFILES", None)
    if devices is not None:
        os.environ["FAKEGPU_PROFILES"] = ",".join(devices) if not isinstance(devices, str) else devices

    effective_device_count = device_count
    if effective_device_count is None:
        effective_device_count = _infer_device_count_from_devices(
            devices if devices is not None else os.environ.get("FAKEGPU_PROFILES")
        )

    patch_result = patch_torch(num_devices=effective_device_count)
    return RuntimeInitResult(runtime="fakecuda", backend=patch_result.backend)


def _infer_device_count_from_devices(devices: str | Sequence[str] | None) -> int | None:
    if devices is None:
        return None
    specs = [devices] if isinstance(devices, str) else list(devices)
    total = 0
    for item in specs:
        for spec in str(item).split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                total += int(parts[1].strip())
            else:
                total += 1
    return total or None
