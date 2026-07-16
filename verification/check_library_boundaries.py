#!/usr/bin/env python3
"""Verify that CUDA Driver and Runtime libraries do not leak each other's API."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-dir", default="build")
    ns = parser.parse_args()

    lib_dir = Path(ns.lib_dir)
    if sys.platform == "darwin":
        driver = lib_dir / "libcuda.dylib"
        runtime = lib_dir / "libcudart.dylib"
        nvml = lib_dir / "libnvidia-ml.dylib"
        nm_command = ["nm", "-gU"]
    else:
        driver = lib_dir / "libcuda.so.1"
        runtime = lib_dir / "libcudart.so.12"
        nvml = lib_dir / "libnvidia-ml.so.1"
        nm_command = ["nm", "-D", "--defined-only"]

    driver_symbols = _defined_symbols(driver, nm_command)
    runtime_symbols = _defined_symbols(runtime, nm_command)
    nvml_symbols = _defined_symbols(nvml, nm_command)

    _require(driver_symbols, "cuInit", driver)
    # CUDA 12.x and PyTorch resolve these symbols directly from the libcuda
    # handle.  Missing any required entry can make cublasCreate fail before the
    # first GEMM, even when basic Runtime kernels appear to work.
    for symbol in (
        "cuLinkCreate_v2",
        "cuLinkAddData_v2",
        "cuLinkComplete",
        "cuLinkDestroy",
        "cuLaunchKernelEx",
        "cuModuleGetGlobal_v2",
        "cuKernelGetFunction",
        "cuKernelGetAttribute",
        "cuKernelSetAttribute",
        "cuLibraryGetKernel",
        "cuLibraryLoadData",
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        "cuStreamGetCaptureInfo_v2",
        "cuTensorMapEncodeTiled",
    ):
        _require(driver_symbols, symbol, driver)
    _forbid(driver_symbols, "cudaMalloc", driver)
    _forbid(driver_symbols, "cudaLaunchKernel", driver)
    _require(runtime_symbols, "cudaMalloc", runtime)
    _require(runtime_symbols, "cudaLaunchKernel", runtime)
    _require(nvml_symbols, "nvmlInit", nvml)
    _require(nvml_symbols, "nvmlDeviceGetNvLinkRemoteDeviceType", nvml)
    _require(nvml_symbols, "nvmlDeviceGetNvLinkRemotePciInfo_v2", nvml)
    _forbid(nvml_symbols, "cuInit", nvml)
    _forbid(nvml_symbols, "cudaMalloc", nvml)

    print("OK: CUDA Driver/Runtime library symbol boundaries are separated")
    return 0


def _defined_symbols(path: Path, nm_command: list[str]) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"missing library: {path}")
    completed = subprocess.run(
        [*nm_command, str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    symbols: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts:
            symbol = parts[-1]
            # Mach-O's nm prefixes external C symbols with one underscore.
            if sys.platform == "darwin" and symbol.startswith("_"):
                symbol = symbol[1:]
            symbols.add(symbol)
    return symbols


def _require(symbols: set[str], symbol: str, path: Path) -> None:
    if symbol not in symbols:
        raise SystemExit(f"{path}: required symbol {symbol} is missing")


def _forbid(symbols: set[str], symbol: str, path: Path) -> None:
    if symbol in symbols:
        raise SystemExit(f"{path}: Runtime symbol {symbol} leaked into CUDA Driver library")


if __name__ == "__main__":
    raise SystemExit(main())
