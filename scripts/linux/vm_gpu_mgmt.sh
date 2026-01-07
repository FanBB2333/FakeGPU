#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this script is intended to run on Linux (inside your VM)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"

VENV_DIR="${VENV_DIR:-.venv-linux}"
BUILD_DIR="${BUILD_DIR:-build_linux}"
UV_GROUP="${UV_GROUP:-linux_vm_gpu_mgmt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP="${FAKEGPU_VM_BOOTSTRAP:-0}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: missing required command: $cmd" >&2
    return 1
  fi
}

bootstrap_apt() {
  if [[ "$BOOTSTRAP" != "1" ]]; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  echo "Bootstrapping build dependencies via apt-get..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    python3-venv \
    python3-pip
}

bootstrap_apt

require_cmd "$PYTHON_BIN"
require_cmd cmake
require_cmd cc
require_cmd c++

if ! command -v uv >/dev/null 2>&1; then
  if [[ "$BOOTSTRAP" == "1" ]]; then
    echo "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    else
      "$PYTHON_BIN" -m pip install --user -U uv
    fi
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: uv not found in this Linux environment.

Install uv (choose one):
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - python3 -m pip install -U uv
EOF
    exit 1
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR"
fi

VENV_PY="$REPO_ROOT/$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "error: venv python not found: $VENV_PY" >&2
  exit 1
fi

export PATH="$REPO_ROOT/$VENV_DIR/bin:$PATH"

deps="$("$PYTHON_BIN" tools/fakegpu_uv_deps.py "$UV_GROUP" | tr '\n' ' ')"
if [[ -n "$deps" ]]; then
  uv pip install --python "$VENV_PY" $deps
fi

cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

# Sanity: preload works (no torch required).
FAKEGPU_PYTHON="$VENV_PY" BUILD_DIR="$BUILD_DIR" ./ftest smoke

# GPU management: NVML via pynvml + gpustat.
FAKEGPU_PYTHON="$VENV_PY" "$VENV_PY" verification/test_nvml_pynvml.py --build-dir "$BUILD_DIR"

if command -v gpustat >/dev/null 2>&1; then
  echo
  echo "gpustat (FakeGPU-enabled):"
  ./fgpu --build-dir "$BUILD_DIR" gpustat -cp || true
fi
