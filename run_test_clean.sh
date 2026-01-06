#!/usr/bin/env bash
set -euo pipefail

# Clean runner for a concise PyTorch + cuBLAS sanity test.
# Uses the repo's standardized `./fgpu` wrapper to set LD_LIBRARY_PATH/LD_PRELOAD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./fgpu python3 test/test_pytorch_with_cublas.py 2>&1 | grep -v "^\[Fake\|^\[Global\|^\[Monitor"
