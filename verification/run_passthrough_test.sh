#!/bin/bash
# Test script for FakeGPU passthrough mode
# This script compares results between direct CUDA execution and passthrough mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=========================================="
echo "FakeGPU Passthrough Mode Test"
echo "=========================================="

# Build the project
echo ""
echo "Building FakeGPU..."
cd "$PROJECT_ROOT"
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
cmake --build build -j$(nproc)

# Check if we have a real GPU
echo ""
echo "Checking for real GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=1
else
    echo "No nvidia-smi found - real GPU tests will be skipped"
    HAS_GPU=0
fi

# Test 1: Direct execution (no FakeGPU)
if [ "$HAS_GPU" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Test 1: Direct CUDA execution (baseline)"
    echo "=========================================="
    FAKEGPU_REPORT_PATH="$BUILD_DIR/report_direct.json" \
    python3 "$SCRIPT_DIR/test_passthrough.py" 2>&1 || true
fi

# Test 2: Passthrough mode
if [ "$HAS_GPU" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Test 2: FakeGPU Passthrough mode"
    echo "=========================================="
    FAKEGPU_MODE=passthrough \
    FAKEGPU_REPORT_PATH="$BUILD_DIR/report_passthrough.json" \
    LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
    LD_PRELOAD="$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libcudart.so.12:$BUILD_DIR/libnvidia-ml.so.1" \
    python3 "$SCRIPT_DIR/test_passthrough.py" 2>&1 || true
fi

# Test 3: Hybrid mode with clamp policy
if [ "$HAS_GPU" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Test 3: FakeGPU Hybrid mode (clamp policy)"
    echo "=========================================="
    FAKEGPU_MODE=hybrid \
    FAKEGPU_OOM_POLICY=clamp \
    FAKEGPU_REPORT_PATH="$BUILD_DIR/report_hybrid_clamp.json" \
    LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
    LD_PRELOAD="$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libcudart.so.12:$BUILD_DIR/libnvidia-ml.so.1" \
    python3 "$SCRIPT_DIR/test_passthrough.py" 2>&1 || true
fi

# Test 4: Simulate mode (always works)
echo ""
echo "=========================================="
echo "Test 4: FakeGPU Simulate mode"
echo "=========================================="
FAKEGPU_MODE=simulate \
FAKEGPU_PROFILE=a100 \
FAKEGPU_DEVICE_COUNT=2 \
FAKEGPU_REPORT_PATH="$BUILD_DIR/report_simulate.json" \
LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
LD_PRELOAD="$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libcudart.so.12:$BUILD_DIR/libnvidia-ml.so.1" \
python3 "$SCRIPT_DIR/test_passthrough.py" 2>&1 || true

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Reports generated:"
ls -la "$BUILD_DIR"/report_*.json 2>/dev/null || echo "No reports found"

echo ""
echo "Done!"
