#!/usr/bin/env bash
set -euo pipefail

# Test Python GPU detection with fake GPU library

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate conda environment
CONDA_ENV=${CONDA_ENV:-patent}
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

BUILD_DIR=${BUILD_DIR:-build}
PYTHON_BIN="${FAKEGPU_PYTHON:-$(command -v python)}"
export FAKEGPU_PYTHON="$PYTHON_BIN"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: build directory $BUILD_DIR not found. Build first with: cmake --build build"
    exit 1
fi

run_fakegpu() {
    "$PROJECT_ROOT/fgpu" --build-dir "$BUILD_DIR" "$@"
}

echo "========================================"
echo "Testing with pynvml (NVML Python API)"
echo "========================================"
echo ""

# Check if pynvml is installed
if ! "$PYTHON_BIN" -c "import pynvml" 2>/dev/null; then
    echo "Installing nvidia-ml-py3..."
    "$PYTHON_BIN" -m pip install nvidia-ml-py3 --quiet
fi

run_fakegpu "$PYTHON_BIN" verification/test_gpu.py

echo ""
echo "========================================"
echo "Report generated:"
echo "========================================"
if [[ -f fake_gpu_report.json ]]; then
    cat fake_gpu_report.json
fi

echo ""
echo "========================================"
echo "Testing with PyTorch (optional)"
echo "========================================"
echo ""

if "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    echo "PyTorch detected, running test..."
    run_fakegpu "$PYTHON_BIN" verification/test_pytorch.py
else
    echo "PyTorch not installed. Skipping PyTorch test."
    echo "To test with PyTorch, install it first: pip install torch"
fi
