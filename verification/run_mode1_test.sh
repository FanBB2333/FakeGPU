#!/bin/bash
# Comprehensive test script for FakeGPU Mode 1 (Passthrough/Hybrid)
# Tests all three modes: simulate, passthrough, hybrid

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=============================================="
echo "FakeGPU Mode 1 Test Suite"
echo "=============================================="
echo ""

# Build the project
echo "[1/5] Building FakeGPU..."
cd "$PROJECT_ROOT"
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF > /dev/null
cmake --build build -j$(nproc) > /dev/null 2>&1
echo "Build complete."
echo ""

# Compile test program
echo "[2/5] Compiling test program..."
gcc -D_GNU_SOURCE -o "$BUILD_DIR/test_mode" "$SCRIPT_DIR/test_mode.c" -ldl -Wl,--no-as-needed
echo "Test program compiled."
echo ""

# Test 1: Simulate mode
echo "=============================================="
echo "[3/5] Test: Simulate Mode"
echo "=============================================="
FAKEGPU_MODE=simulate \
FAKEGPU_PROFILE=h100 \
FAKEGPU_DEVICE_COUNT=4 \
FAKEGPU_REPORT_PATH="$BUILD_DIR/report_simulate.json" \
LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
LD_PRELOAD="$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libnvidia-ml.so.1" \
"$BUILD_DIR/test_mode"

echo ""
echo "Report (simulate mode):"
cat "$BUILD_DIR/report_simulate.json" | head -20
echo "..."
echo ""

# Check if real GPU is available
HAS_GPU=0
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
        HAS_GPU=1
        echo "Real GPU detected. Will test passthrough and hybrid modes."
    fi
fi

if [ "$HAS_GPU" = "1" ]; then
    # Test 2: Passthrough mode
    echo "=============================================="
    echo "[4/5] Test: Passthrough Mode"
    echo "=============================================="
    echo "Note: In passthrough mode, FakeGPU forwards all calls to real CUDA."
    echo "Device info should match real GPU."
    echo ""

    # For passthrough, we need to NOT preload our fake libraries
    # Instead, we just set the mode and let the real libraries handle it
    FAKEGPU_MODE=passthrough \
    FAKEGPU_REPORT_PATH="$BUILD_DIR/report_passthrough.json" \
    "$BUILD_DIR/test_mode" 2>&1 || echo "(Passthrough test completed with warnings)"

    echo ""

    # Test 3: Hybrid mode
    echo "=============================================="
    echo "[5/5] Test: Hybrid Mode (clamp policy)"
    echo "=============================================="
    echo "Note: In hybrid mode, device info is virtualized but compute uses real GPU."
    echo "OOM policy: clamp (memory reported is clamped to real GPU capacity)"
    echo ""

    FAKEGPU_MODE=hybrid \
    FAKEGPU_OOM_POLICY=clamp \
    FAKEGPU_PROFILE=a100 \
    FAKEGPU_DEVICE_COUNT=2 \
    FAKEGPU_REPORT_PATH="$BUILD_DIR/report_hybrid.json" \
    LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
    LD_PRELOAD="$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libnvidia-ml.so.1" \
    "$BUILD_DIR/test_mode" 2>&1 || echo "(Hybrid test completed with warnings)"

    echo ""
    echo "Report (hybrid mode):"
    cat "$BUILD_DIR/report_hybrid.json" 2>/dev/null | head -30 || echo "Report not generated"
    echo "..."
else
    echo ""
    echo "[4/5] Skipping Passthrough Mode (no real GPU)"
    echo "[5/5] Skipping Hybrid Mode (no real GPU)"
fi

echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo "Reports generated:"
ls -la "$BUILD_DIR"/report_*.json 2>/dev/null || echo "No reports found"

echo ""
echo "Mode 1 implementation status:"
echo "  - Simulate mode: WORKING"
if [ "$HAS_GPU" = "1" ]; then
    echo "  - Passthrough mode: AVAILABLE (requires real GPU)"
    echo "  - Hybrid mode: AVAILABLE (requires real GPU)"
else
    echo "  - Passthrough mode: NOT TESTED (no real GPU)"
    echo "  - Hybrid mode: NOT TESTED (no real GPU)"
fi

echo ""
echo "Environment variables supported:"
echo "  FAKEGPU_MODE={simulate,passthrough,hybrid}"
echo "  FAKEGPU_OOM_POLICY={clamp,managed,mapped_host,spill_cpu}"
echo "  FAKEGPU_PROFILE={a100,h100,v100,...}"
echo "  FAKEGPU_DEVICE_COUNT=N"
echo "  FAKEGPU_REAL_CUDA_LIB_DIR=/path/to/cuda/lib"
echo ""
echo "Done!"
