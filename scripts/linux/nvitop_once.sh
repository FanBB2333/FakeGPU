#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this script is intended to run on Linux." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build_linux}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: missing required command: $cmd" >&2
    return 1
  fi
}

find_conda_base() {
  if [[ -n "${FAKEGPU_CONDA_BASE:-}" ]]; then
    echo "$FAKEGPU_CONDA_BASE"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi

  for d in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" "/opt/conda"; do
    if [[ -x "$d/bin/conda" ]]; then
      echo "$d"
      return 0
    fi
  done

  return 1
}

require_cmd cmake

CONDA_BASE="$(find_conda_base || true)"
if [[ -z "$CONDA_BASE" ]]; then
  cat >&2 <<'EOF'
error: conda not found.

Set FAKEGPU_CONDA_BASE=/path/to/miniconda_or_miniforge, or ensure `conda` is in PATH.
EOF
  exit 1
fi

CONDA_PY="$CONDA_BASE/bin/python"
if [[ ! -x "$CONDA_PY" ]]; then
  echo "error: conda python not found: $CONDA_PY" >&2
  exit 1
fi

if ! "$CONDA_PY" -c "import nvitop" >/dev/null 2>&1; then
  cat >&2 <<EOF
error: nvitop is not importable in conda base ($CONDA_BASE).

Install it first:
  $CONDA_BASE/bin/conda install -n base -c conda-forge nvitop
or:
  $CONDA_PY -m pip install -U nvitop
EOF
  exit 1
fi

cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

exec ./fgpu --build-dir "$BUILD_DIR" "$CONDA_PY" -m nvitop --once "$@"
