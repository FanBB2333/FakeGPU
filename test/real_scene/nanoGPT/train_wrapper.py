#!/usr/bin/env python3
"""Run nanoGPT's train.py under baseline or FakeGPU-backed configurations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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


def setup_fakegpu(mode: str, *, device_count: int | None, total_memory_bytes: int | None) -> dict:
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
        result = fakegpu.init(runtime=runtime, device_count=device_count)
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

    print("=" * 70)
    print("[WRAPPER] nanoGPT Validation Test")
    print(f"[WRAPPER] Mode: {args.mode}")
    print(f"[WRAPPER] Device count: {args.device_count or 'default'}")
    print(
        "[WRAPPER] Total memory/device: "
        f"{args.total_memory_gb if args.total_memory_gb is not None else 'default'} GiB"
    )
    print(f"[WRAPPER] Train args: {train_args}")
    print("=" * 70)

    info = setup_fakegpu(
        args.mode,
        device_count=args.device_count,
        total_memory_bytes=total_memory_bytes,
    )

    import torch

    _patch_pin_memory_if_needed(args.mode, info)

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
