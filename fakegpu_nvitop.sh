#!/bin/bash
# FakeGPU nvitop wrapper script
# This script runs nvitop with FakeGPU libraries and handles terminal state

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Check if libraries exist
if [[ ! -f "$BUILD_DIR/libnvidia-ml.so.1" ]]; then
    echo "Error: FakeGPU libraries not found. Please build first with:"
    echo "  cmake -S . -B build && cmake --build build"
    exit 1
fi

# Save terminal state
stty_state=$(stty -g 2>/dev/null) || stty_state=""

# Cleanup function to restore terminal
cleanup() {
    # Reset terminal to sane state
    if [[ -n "$stty_state" ]]; then
        stty "$stty_state" 2>/dev/null || true
    fi
    # Fallback reset
    stty sane 2>/dev/null || true
    # Make cursor visible
    printf '\033[?25h'
}

trap cleanup EXIT INT TERM

# Run nvitop via our Python wrapper (which uses os._exit to avoid GC crash)
LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
LD_PRELOAD="$BUILD_DIR/libcublas.so.12:$BUILD_DIR/libcudart.so.12:$BUILD_DIR/libcuda.so.1:$BUILD_DIR/libnvidia-ml.so.1" \
python3 "${SCRIPT_DIR}/fakegpu_nvitop.py" "$@"

exit_code=$?

# Cleanup is called by trap
exit $exit_code
