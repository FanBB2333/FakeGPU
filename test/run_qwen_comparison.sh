#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Qwen2.5 Model Inference Comparison Test"
echo "=========================================="
echo ""

# Create output directory
mkdir -p test/output

# Test 1: Run with fakeGPU
echo "=========================================="
echo "Test 1: Running with fakeGPU"
echo "=========================================="
echo "This test uses fakeGPU to simulate GPU operations"
echo ""

timeout 300 ./fgpu python3 test/test_load_qwen2_5.py 2>&1 | tee test/output/fakegpu_output.txt

echo ""
echo "FakeGPU test output saved to test/output/fakegpu_output.txt"
echo ""

# Check for success indicators
echo "=========================================="
echo "Test Result"
echo "=========================================="
if grep -q "TEST PASSED" test/output/fakegpu_output.txt; then
    echo "✓ FakeGPU test: PASSED"
    echo ""
    echo "Generated token:"
    grep "Generated token:" test/output/fakegpu_output.txt
else
    echo "✗ FakeGPU test: FAILED"
    echo ""
    echo "Last 30 lines of output:"
    tail -n 30 test/output/fakegpu_output.txt
fi
echo ""

echo "=========================================="
echo "Full output file available at:"
echo "  - test/output/fakegpu_output.txt"
echo "=========================================="
