#!/usr/bin/env python3
"""Run nanoGPT's train.py under baseline or FakeGPU-backed configurations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _read_profile_memory_bytes(profile_id: str) -> int | None:
    normalized = profile_id.strip().lower()
    if not normalized:
        return None
    profiles_dir = _repo_root() / "profiles"
    candidates = [
        profiles_dir / f"{normalized}.yaml",
        profiles_dir / f"{normalized}.yml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith("memory_bytes:"):
                    _, value = line.split(":", 1)
                    return int(value.strip(), 0)
    return None


def _resolve_profile_memory_bytes(*, profile: str | None, devices: str | None) -> int | None:
    if profile:
        return _read_profile_memory_bytes(profile)
    if not devices:
        return None
    first = devices.split(",", 1)[0].strip()
    if not first:
        return None
    if ":" in first:
        first = first.split(":", 1)[0].strip()
    return _read_profile_memory_bytes(first)


def _has_cli_arg(args: list[str], key: str) -> bool:
    prefix = f"{key}="
    for index, arg in enumerate(args):
        if arg == key:
            return True
        if arg.startswith(prefix):
            return True
        if index > 0 and args[index - 1] == key:
            return True
    return False


def _prepare_train_args(train_args: list[str], *, mode: str, info: dict[str, object]) -> list[str]:
    prepared = list(train_args)
    if mode == "full" and info.get("fakegpu_runtime") == "fakecuda":
        if not _has_cli_arg(prepared, "--dtype"):
            prepared.append("--dtype=float32")
        if not _has_cli_arg(prepared, "--eval_iters"):
            prepared.append("--eval_iters=2")
        if not _has_cli_arg(prepared, "--batch_size"):
            prepared.append("--batch_size=8")
        if not _has_cli_arg(prepared, "--block_size"):
            prepared.append("--block_size=64")
    return prepared


class _FakeCudaMemoryLimiter:
    def __init__(self, limit_bytes: int):
        self.limit_bytes = int(limit_bytes)
        self.used_bytes = 0

    def reserve_bytes(self, size_bytes: int, *, context: str) -> None:
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            return
        next_used = self.used_bytes + size_bytes
        if next_used > self.limit_bytes:
            attempted_mib = size_bytes / 1024**2
            total_mib = self.limit_bytes / 1024**2
            used_mib = self.used_bytes / 1024**2
            raise RuntimeError(
                "CUDA out of memory. "
                f"Tried to allocate {attempted_mib:.2f} MiB during {context}. "
                f"FakeGPU limit is {total_mib:.2f} MiB, {used_mib:.2f} MiB already reserved."
            )
        self.used_bytes = next_used

    def reserve_tensor(self, tensor: Any, *, context: str) -> None:
        size_bytes = None
        try:
            storage = tensor.untyped_storage()
            size_bytes = storage.nbytes()
        except Exception:
            pass
        if size_bytes is None:
            size_bytes = tensor.numel() * tensor.element_size()
        self.reserve_bytes(size_bytes, context=context)

    def mem_get_info(self) -> tuple[int, int]:
        free_bytes = max(0, self.limit_bytes - self.used_bytes)
        return free_bytes, self.limit_bytes


def _module_nbytes(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        try:
            storage = tensor.untyped_storage()
            storage_id = int(storage.data_ptr())
            if storage_id in seen:
                continue
            seen.add(storage_id)
            total += int(storage.nbytes())
        except Exception:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _apply_memory_override(total_bytes: int | None) -> None:
    if total_bytes is None:
        return

    os.environ["FAKEGPU_TOTAL_MEMORY"] = str(total_bytes)
    os.environ["TORCH_FAKEGPU_TOTAL_MEMORY"] = str(total_bytes)

    try:
        from fakegpu import torch_patch

        torch_patch._TOTAL_MEMORY = int(total_bytes)
        print(
            "[WRAPPER] Applied fakegpu.torch_patch total-memory override: "
            f"{total_bytes / 1024**3:.2f} GiB"
        )
    except Exception as exc:
        print(f"[WRAPPER] fakegpu.torch_patch memory override skipped: {exc}")

    try:
        import torch.fakegpu as torch_fakegpu
    except Exception:
        return

    if hasattr(torch_fakegpu, "_FakeDeviceProperties"):
        original_init = torch_fakegpu._FakeDeviceProperties.__init__

        def _patched_init(self, index):
            original_init(self, index)
            self.total_memory = int(total_bytes)

        torch_fakegpu._FakeDeviceProperties.__init__ = _patched_init

    if hasattr(torch_fakegpu, "mem_get_info"):
        torch_fakegpu.mem_get_info = lambda device=None: (int(total_bytes), int(total_bytes))

    print(
        "[WRAPPER] Applied torch.fakegpu total-memory override: "
        f"{total_bytes / 1024**3:.2f} GiB"
    )


def _patch_pin_memory_if_needed(mode: str, info: dict) -> None:
    if mode != "full":
        return
    if info.get("fakegpu_runtime") != "fakecuda":
        return

    import torch

    try:
        torch.zeros(1).pin_memory()
    except Exception as exc:
        torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
        print(
            "[WRAPPER] Patched torch.Tensor.pin_memory() to no-op for fakecuda "
            f"after probe failure: {exc}"
        )


def _install_fakecuda_memory_limit_if_needed(
    mode: str,
    info: dict[str, object],
    *,
    total_memory_bytes: int | None,
) -> None:
    if total_memory_bytes is None:
        return
    if mode != "full":
        return
    if info.get("fakegpu_runtime") != "fakecuda":
        return

    import torch

    limiter = _FakeCudaMemoryLimiter(total_memory_bytes)
    module_transfer_depth = 0

    def _targets_cuda(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is None:
            return True
        if isinstance(device, str):
            return device.startswith("cuda")
        if hasattr(device, "type"):
            return getattr(device, "type", None) == "cuda"
        return False

    orig_tensor_to = torch.Tensor.to
    orig_tensor_cuda = torch.Tensor.cuda
    orig_module_to = torch.nn.Module.to
    orig_module_cuda = torch.nn.Module.cuda

    def limited_tensor_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if module_transfer_depth == 0 and _targets_cuda(args, kwargs):
            limiter.reserve_tensor(self, context="Tensor.to(cuda)")
        return orig_tensor_to(self, *args, **kwargs)

    def limited_tensor_cuda(self: Any, *args: Any, **kwargs: Any) -> Any:
        if module_transfer_depth == 0:
            limiter.reserve_tensor(self, context="Tensor.cuda()")
        return orig_tensor_cuda(self, *args, **kwargs)

    def limited_module_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal module_transfer_depth
        if _targets_cuda(args, kwargs):
            limiter.reserve_bytes(_module_nbytes(self), context=f"{type(self).__name__}.to(cuda)")
            module_transfer_depth += 1
            try:
                return orig_module_to(self, *args, **kwargs)
            finally:
                module_transfer_depth -= 1
        return orig_module_to(self, *args, **kwargs)

    def limited_module_cuda(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal module_transfer_depth
        limiter.reserve_bytes(_module_nbytes(self), context=f"{type(self).__name__}.cuda()")
        module_transfer_depth += 1
        try:
            return orig_module_cuda(self, *args, **kwargs)
        finally:
            module_transfer_depth -= 1

    torch.Tensor.to = limited_tensor_to  # type: ignore[assignment]
    torch.Tensor.cuda = limited_tensor_cuda  # type: ignore[assignment]
    torch.nn.Module.to = limited_module_to  # type: ignore[assignment]
    torch.nn.Module.cuda = limited_module_cuda  # type: ignore[assignment]
    torch.cuda.mem_get_info = lambda device=None: limiter.mem_get_info()
    print(
        "[WRAPPER] Installed fakecuda virtual memory limiter: "
        f"{total_memory_bytes / 1024**3:.2f} GiB"
    )


def setup_fakegpu(
    mode: str,
    *,
    device_count: int | None,
    total_memory_bytes: int | None,
    profile: str | None,
    devices: str | None,
) -> dict:
    """Initialize FakeGPU for the requested mode and return diagnostic info."""

    info: dict[str, object] = {
        "mode": mode,
        "fakegpu_runtime": None,
        "fakegpu_backend": None,
        "fakegpu_init_error": None,
    }

    if mode == "baseline":
        print("[WRAPPER] Mode: baseline - No FakeGPU initialization")
        return info

    _ensure_repo_root_on_path()

    import fakegpu

    runtime = "native" if mode == "partial" else "auto"
    description = {
        "partial": "fakegpu + native PyTorch (C-level interception only)",
        "full": "fakegpu + pytorch-fakegpu / torch patching",
    }[mode]
    print(f"[WRAPPER] Mode: {mode} - {description}")

    try:
        result = fakegpu.init(
            runtime=runtime,
            device_count=device_count,
            profile=profile,
            devices=devices,
        )
        info["fakegpu_runtime"] = result.runtime
        info["fakegpu_backend"] = result.backend
        print(
            f"[WRAPPER] fakegpu.init(runtime={runtime!r}) -> "
            f"runtime={result.runtime}, backend={result.backend}"
        )
    except Exception as exc:
        info["fakegpu_init_error"] = str(exc)
        print(f"[WRAPPER] fakegpu.init(runtime={runtime!r}) failed: {exc}")
        raise

    _apply_memory_override(total_memory_bytes)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="FakeGPU nanoGPT training wrapper")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "partial", "full"],
        help="FakeGPU configuration mode",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=None,
        help="Number of fake GPU devices to expose",
    )
    parser.add_argument(
        "--total-memory-gb",
        type=float,
        default=None,
        help="Total memory per fake device in GiB",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="FakeGPU profile preset ID (for example: a100, a100-1g, h100)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="FakeGPU device preset spec (for example: a100-1g or a100-1g,a100)",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to train.py",
    )

    args = parser.parse_args()
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    total_memory_bytes = None
    if args.total_memory_gb is not None:
        total_memory_bytes = int(args.total_memory_gb * 1024**3)
    else:
        total_memory_bytes = _resolve_profile_memory_bytes(
            profile=args.profile,
            devices=args.devices,
        )

    print("=" * 70)
    print("[WRAPPER] nanoGPT Validation Test")
    print(f"[WRAPPER] Mode: {args.mode}")
    print(f"[WRAPPER] Device count: {args.device_count or 'default'}")
    print(f"[WRAPPER] Profile: {args.profile or 'default'}")
    print(f"[WRAPPER] Devices spec: {args.devices or 'default'}")
    print(
        "[WRAPPER] Total memory/device: "
        f"{args.total_memory_gb if args.total_memory_gb is not None else (f'{total_memory_bytes / 1024**3:.2f}' if total_memory_bytes is not None else 'default')} GiB"
    )
    print(f"[WRAPPER] Train args: {train_args}")
    print("=" * 70)

    info = setup_fakegpu(
        args.mode,
        device_count=args.device_count,
        total_memory_bytes=total_memory_bytes,
        profile=args.profile,
        devices=args.devices,
    )

    train_args = _prepare_train_args(train_args, mode=args.mode, info=info)

    import torch

    _patch_pin_memory_if_needed(args.mode, info)
    _install_fakecuda_memory_limit_if_needed(
        args.mode,
        info,
        total_memory_bytes=total_memory_bytes,
    )

    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"[WRAPPER] PyTorch version: {torch.__version__}")
    print(f"[WRAPPER] torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[WRAPPER] torch.cuda.device_count(): {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(index)
                total_gib = getattr(props, "total_memory", 0) / 1024**3
                print(
                    f"[WRAPPER] Device {index}: "
                    f"{torch.cuda.get_device_name(index)}, Memory: {total_gib:.1f} GiB"
                )
            except Exception as exc:
                print(f"[WRAPPER] Device {index}: error getting info: {exc}")
    print("=" * 70)

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"[WRAPPER] Effective train args: {train_args}")
    sys.argv = ["train.py", *train_args]

    start_time = time.time()
    try:
        exec_globals = {
            "__name__": "__main__",
            "__file__": str(script_dir / "train.py"),
        }
        with open(script_dir / "train.py", "r", encoding="utf-8") as handle:
            code = compile(handle.read(), str(script_dir / "train.py"), "exec")
        exec(code, exec_globals, exec_globals)
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"[WRAPPER] Training completed successfully in {elapsed:.1f}s")
        print("[WRAPPER] Result: PASS")
        print("=" * 70)
        return 0
    except SystemExit as exc:
        elapsed = time.time() - start_time
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
        print("=" * 70)
        print(f"[WRAPPER] Training exited with code {exc.code} in {elapsed:.1f}s")
        print(f"[WRAPPER] Result: {'PASS' if exit_code == 0 else 'FAIL'}")
        print("=" * 70)
        return exit_code
    except Exception as exc:
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"[WRAPPER] Training FAILED with error in {elapsed:.1f}s")
        print(f"[WRAPPER] Error type: {type(exc).__name__}")
        print(f"[WRAPPER] Error message: {exc}")
        print("[WRAPPER] Result: FAIL")
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
