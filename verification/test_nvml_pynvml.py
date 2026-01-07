#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _expected_device_count(*, device_count: int | None, devices: str | None) -> int | None:
    if device_count is not None:
        return device_count

    if devices:
        total = 0
        for part in devices.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                _name, count_s = part.split(":", 1)
                total += int(count_s.strip())
            else:
                total += 1
        return total

    env_count = os.environ.get("FAKEGPU_DEVICE_COUNT")
    if env_count:
        try:
            return int(env_count)
        except ValueError:
            return None

    return 8


def _to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "linux":
        print("This test is intended to run on Linux (e.g. inside the VM).")
        return 2

    parser = argparse.ArgumentParser(description="Validate FakeGPU NVML (GPU management) via pynvml.")
    parser.add_argument("--build-dir", help="CMake build dir containing FakeGPU shared libraries (e.g. build_linux).")
    parser.add_argument("--lib-dir", help="Directory containing FakeGPU shared libraries (takes precedence over --build-dir).")
    parser.add_argument("--profile", help="GPU preset ID for all devices (e.g. a100, h100).")
    parser.add_argument("--device-count", type=int, help="Number of devices to expose (default: 8).")
    parser.add_argument("--devices", help="Per-device preset spec, e.g. 'a100:4,h100:4' or 't4,l40s'.")
    ns = parser.parse_args(argv)

    import fakegpu

    fakegpu.init(
        build_dir=ns.build_dir,
        lib_dir=ns.lib_dir,
        profile=ns.profile,
        device_count=ns.device_count,
        devices=ns.devices,
        update_env=True,
    )

    try:
        import pynvml  # provided by nvidia-ml-py
    except Exception as exc:
        print(f"error: failed to import pynvml ({exc}). Install: uv pip install nvidia-ml-py", file=sys.stderr)
        return 1

    expected = _expected_device_count(device_count=ns.device_count, devices=ns.devices)

    pynvml.nvmlInit()
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        print(f"NVML device count: {count}")
        if expected is not None and count != expected:
            print(f"error: expected {expected} device(s), got {count}", file=sys.stderr)
            return 1

        for i in range(min(count, 3)):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _to_str(pynvml.nvmlDeviceGetName(handle))
            uuid = _to_str(pynvml.nvmlDeviceGetUUID(handle))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            print()
            print(f"[GPU {i}] {name}")
            print(f"  UUID: {uuid}")
            print(f"  Memory: total={mem.total / (1024**3):.2f}GiB used={mem.used / (1024**3):.2f}GiB free={mem.free / (1024**3):.2f}GiB")

            if "Fake NVIDIA" not in name:
                print(f"error: unexpected device name (not FakeGPU?): {name}", file=sys.stderr)
                return 1

        print("\nOK: NVML GPU management looks good.")
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
