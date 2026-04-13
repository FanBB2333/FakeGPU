#!/usr/bin/env bash
set -euo pipefail

# Analyze PyTorch GPU usage with fake GPU library

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BUILD_DIR=${BUILD_DIR:-build}
PYTHON_BIN="${FAKEGPU_PYTHON:-$(command -v python3 || command -v python)}"
export FAKEGPU_PYTHON="$PYTHON_BIN"
OUTPUT_FILE="${TMPDIR:-/tmp}/pytorch_test_output.txt"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: build directory $BUILD_DIR not found. Build first with: cmake --build build"
    exit 1
fi

run_fakegpu() {
    "$PROJECT_ROOT/fgpu" --build-dir "$BUILD_DIR" "$@"
}

echo "========================================"
echo "PyTorch Fake GPU Analysis"
echo "========================================"
echo ""

# Check if Python and PyTorch are available
if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    echo "PyTorch not installed. Please install with: pip install torch"
    exit 1
fi

echo "Running detailed PyTorch test..."
echo ""

# Run the test and capture output
run_fakegpu "$PYTHON_BIN" verification/test_pytorch_detailed.py > "$OUTPUT_FILE" 2>&1

echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo ""

# Extract test results
grep -E "^Test [0-9]:" "$OUTPUT_FILE" || true
echo ""

# Count successful operations
success_count=$(grep -c "^✓" "$OUTPUT_FILE" || echo "0")
fail_count=$(grep -c "^✗" "$OUTPUT_FILE" || echo "0")

echo "Successful operations: $success_count"
echo "Failed operations: $fail_count"
echo ""

echo "========================================"
echo "Fake GPU API Calls Intercepted"
echo "========================================"
echo ""

# Extract and count fake GPU calls
echo "CUDA Runtime API calls intercepted:"
grep "\[FakeCUDA\]" "$OUTPUT_FILE" | sed 's/\[FakeCUDA\] /  - /' | sort | uniq -c | sort -rn || echo "  None"
echo ""

echo "NVML API calls intercepted:"
grep "\[FakeNVML\]" "$OUTPUT_FILE" | sed 's/\[FakeNVML\] /  - /' | sort | uniq -c | sort -rn || echo "  None"
echo ""

echo "CUDA Driver API calls intercepted:"
grep "\[FakeCUDA-Driver\]" "$OUTPUT_FILE" | sed 's/\[FakeCUDA-Driver\] /  - /' | sort | uniq -c | sort -rn || echo "  None"
echo ""

echo "========================================"
echo "Analysis"
echo "========================================"
echo ""

# Check if device properties show fake GPU
if grep -q "Name:.*Fake NVIDIA" "$OUTPUT_FILE"; then
    echo "✓ PyTorch is using FAKE GPU devices"
    fake_gpu=$(sed -n 's/^.*Name: //p' "$OUTPUT_FILE" | head -1)
    echo "  Device properties show: $fake_gpu"
    echo ""
    echo "  Intercepted operations:"
else
    echo "⚠ PyTorch is using REAL GPU for device detection"
    real_gpu=$(sed -n 's/^.*Name: //p' "$OUTPUT_FILE" | head -1)
    real_gpu=${real_gpu:-Unknown GPU}
    echo "  Device properties show: $real_gpu"
    echo ""
    echo "  However, some CUDA Runtime API operations ARE being intercepted:"
fi

# Check which operations were intercepted
if grep -q "cudaMalloc" "$OUTPUT_FILE"; then
    echo "  ✓ Memory allocation (cudaMalloc/cudaFree)"
fi

if grep -q "cudaLaunchKernel" "$OUTPUT_FILE"; then
    echo "  ✓ Kernel launches (cudaLaunchKernel)"
fi

if grep -q "cudaGetDeviceCount" "$OUTPUT_FILE"; then
    echo "  ✓ Device queries (cudaGetDeviceCount)"
fi

if grep -q "cudaSetDevice" "$OUTPUT_FILE"; then
    echo "  ✓ Device management (cudaSetDevice)"
fi

echo ""
echo "========================================"
echo "Conclusion"
echo "========================================"
echo ""

if grep -q "Name:.*Fake NVIDIA" "$OUTPUT_FILE"; then
    echo "✓ PyTorch is fully using the fake GPU library!"
    echo ""
    echo "This means:"
    echo "- Device detection returns fake GPUs"
    echo "- Memory operations use fake GPU tracking"
    echo "- Kernel launches are intercepted (but not executed)"
    echo ""
    echo "Use case: Testing GPU-dependent code without physical GPUs"
else
    echo "⚠ PyTorch is using REAL GPU for device detection"
    echo ""
    echo "BUT: CUDA Runtime API operations ARE being intercepted by fake GPU!"
    echo ""
    echo "What this means:"
    echo "- Device queries (cudaGetDeviceCount, cudaGetDeviceProperties) use real GPU"
    echo "- Memory operations (cudaMalloc, cudaFree) are intercepted by fake GPU"
    echo "- Kernel launches (cudaLaunchKernel) are intercepted by fake GPU"
    echo "- Device management (cudaSetDevice) is intercepted by fake GPU"
    echo ""
    echo "This is expected on systems with real NVIDIA drivers because:"
    echo "- PyTorch uses CUDA Driver API (cuDeviceGet, etc.) for device detection"
    echo "- Driver API queries go directly to the real driver (not intercepted)"
    echo "- But CUDA Runtime API calls ARE intercepted via FakeGPU preloading"
    echo ""
    echo "Practical implications:"
    echo "- PyTorch sees real GPU properties (name, memory, compute capability)"
    echo "- But memory allocations are tracked by fake GPU"
    echo "- Kernel launches are logged but not executed"
    echo "- Useful for debugging memory usage without actual computation"
    echo ""
    echo "To fully use fake GPU (including device detection):"
    echo "- Test on a system without NVIDIA drivers, OR"
    echo "- Use the fake GPU library in a container without real drivers, OR"
    echo "- Implement CUDA Driver API interception (complex)"
fi

echo ""
echo "Full test output saved to: $OUTPUT_FILE"
echo ""
