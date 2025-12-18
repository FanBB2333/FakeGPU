#!/bin/bash
# Run nvitop once using FakeGPU libraries
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="${BUILD_DIR}/libcublas.so.12:${BUILD_DIR}/libcudart.so.12:${BUILD_DIR}/libcuda.so.1:${BUILD_DIR}/libnvidia-ml.so.1" \
nvitop --once
