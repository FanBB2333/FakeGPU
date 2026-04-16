from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from ._api import InitResult as NativeInitResult
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
        selected_runtime = "fakecuda" if _detect_custom_torch_fakegpu_available() else "native"

    if selected_runtime == "native":
        return _init_native_runtime(
            build_dir=build_dir,
            lib_dir=lib_dir,
            mode=mode,
            oom_policy=oom_policy,
            dist_mode=dist_mode,
            cluster_config=cluster_config,
            coordinator_addr=coordinator_addr,
            coordinator_transport=coordinator_transport,
            profile=profile,
            device_count=device_count,
            devices=devices,
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


def init_privateuse1() -> None:
    from .privateuse1 import init_privateuse1 as _init_privateuse1

    _init_privateuse1()


def _detect_custom_torch_fakegpu_available() -> bool:
    if "torch.fakegpu" in sys.modules:
        return True

    for entry in sys.path:
        search_root = Path(entry or os.getcwd())
        torch_dir = search_root / "torch"
        if (torch_dir / "fakegpu.py").is_file():
            return True
        if (torch_dir / "fakegpu" / "__init__.py").is_file():
            return True
    return False


def _init_native_runtime(
    *,
    build_dir: str | os.PathLike[str] | None = None,
    lib_dir: str | os.PathLike[str] | None = None,
    mode: str | None = None,
    oom_policy: str | None = None,
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
    native_result = _native_init(
        build_dir=build_dir,
        lib_dir=lib_dir,
        mode=mode,
        oom_policy=oom_policy,
        dist_mode=dist_mode,
        cluster_config=cluster_config,
        coordinator_addr=coordinator_addr,
        coordinator_transport=coordinator_transport,
        profile=profile,
        device_count=device_count,
        devices=devices,
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
    if profile is not None:
        os.environ["FAKEGPU_PROFILE"] = str(profile)
    if devices is not None:
        os.environ["FAKEGPU_PROFILES"] = ",".join(devices) if not isinstance(devices, str) else devices

    patch_result = patch_torch(num_devices=device_count)
    return RuntimeInitResult(runtime="fakecuda", backend=patch_result.backend)
