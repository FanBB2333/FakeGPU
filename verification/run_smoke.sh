#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test: builds FakeGPU, builds the verification binary, runs it with FakeGPU enabled,
# and shows the generated report.

BUILD_DIR=${BUILD_DIR:-build}
REPORT=${REPORT:-fake_gpu_report.json}

cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

if [[ "$(uname -s)" == "Darwin" ]]; then
    cc verification/verify_preload.c -o "$BUILD_DIR/verify_preload"
else
    cc verification/verify_preload.c -o "$BUILD_DIR/verify_preload" -ldl
fi

./fgpu "$BUILD_DIR/verify_preload"

echo "\nSmoke test finished. Report preview (if present):"
if [[ -f "$REPORT" ]]; then
    head -n 40 "$REPORT"
else
    echo "Report not found at $REPORT"
fi
