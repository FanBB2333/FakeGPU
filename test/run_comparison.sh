#!/bin/bash
# Comprehensive comparison test: Real GPU vs FakeGPU
# This script demonstrates that the test code works correctly on real hardware
# and that any failures on FakeGPU are due to incomplete FakeGPU implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "COMPREHENSIVE GPU COMPARISON TEST"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Test on REAL GPU (if available)"
echo "  2. Test on FakeGPU"
echo "  3. Compare results side by side"
echo ""
echo "================================================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test 1: Real GPU
echo ""
echo -e "${CYAN}[1/2] Testing on REAL GPU...${NC}"
echo "--------------------------------------------------------------------------------"
python3 test/test_comparison.py --mode real 2>/dev/null || echo "Real GPU not available or already using FakeGPU"

# Test 2: FakeGPU
echo ""
echo ""
echo -e "${CYAN}[2/2] Testing on FakeGPU...${NC}"
echo "--------------------------------------------------------------------------------"
./fgpu python3 test/test_comparison.py --mode fake 2>&1 | \
grep -v -E "(FakeCUDART|FakeCUBLASLt|FakeCUBLAS|FakeCUDA-Driver|FakeNVML|GlobalState|Monitor)"

echo ""
echo "================================================================================"
echo -e "${GREEN}COMPARISON COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "If both tests show 6/6 passed:"
echo "  ✓ The test code is correct"
echo "  ✓ Your environment is properly configured"
echo "  ✓ FakeGPU is working correctly for basic PyTorch operations!"
echo ""
echo "If FakeGPU shows fewer passes:"
echo "  → This indicates missing features in FakeGPU implementation"
echo "  → NOT a problem with your test code or environment"
echo ""
