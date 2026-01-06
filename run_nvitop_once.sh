#!/bin/bash
# Run nvitop once using FakeGPU libraries
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

"${SCRIPT_DIR}/fgpu" nvitop --once
