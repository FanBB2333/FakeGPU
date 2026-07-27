#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${FAKEGPU_PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$REPO_ROOT/build-cpu-sim}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/test.sh <suite>

Suites:
  python    Run the maintained Python regression suite
  smoke     Build and validate native library loading and reports
  cpu       Validate CPU-backed cuBLAS simulation
  all       Run python, smoke, and cpu
EOF
}

build_native() {
  BUILD_DIR="$BUILD_DIR" "$SCRIPT_DIR/build.sh" --release
}

run_fakegpu() {
  local build_dir="$1"
  shift
  "$PYTHON_BIN" -m fakegpu --build-dir "$build_dir" "$@"
}

run_python() {
  "$PYTHON_BIN" -m pytest -q tests
}

run_smoke() {
  build_native

  "$PYTHON_BIN" scripts/validation/check_library_boundaries.py \
    --lib-dir "$BUILD_DIR"
  "$PYTHON_BIN" -m fakegpu capabilities \
    --source-root "$REPO_ROOT" \
    --build-dir "$BUILD_DIR" \
    --strict
  "$PYTHON_BIN" tests/native/unsupported_api_policy.py \
    --build-dir "$BUILD_DIR"

  local preload_probe="$BUILD_DIR/verify_preload"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    cc tests/native/verify_preload.c -o "$preload_probe"
  else
    cc tests/native/verify_preload.c -o "$preload_probe" -ldl
  fi

  FAKEGPU_REPORT_PATH="$BUILD_DIR/fake_gpu_report.json" \
    run_fakegpu "$BUILD_DIR" "$preload_probe"
  "$PYTHON_BIN" scripts/validation/check_report.py \
    --path "$BUILD_DIR/fake_gpu_report.json"

  FAKEGPU_REPORT_PATH="$BUILD_DIR/fake_gpu_report_memory_types.json" \
    FAKEGPU_SMI_STATE_DIR="$BUILD_DIR/smi-native" \
    FAKEGPU_SMI_DETAIL_LIMIT=1 \
    FAKEGPU_SMI_MAX_STATE_BYTES=65536 \
    FAKEGPU_NVLINK_GROUPS="0,1;2,3" \
    FAKEGPU_NVLINK_BANDWIDTH_GBPS=800 \
    FAKEGPU_FAULT_EVENTS="0:XID_79:critical;1:NVLINK_CRC:error:3" \
    FAKEGPU_MIG_LAYOUT="0:1g.10gb:10240:2;1:2g.20gb:20480" \
    run_fakegpu "$BUILD_DIR" "$PYTHON_BIN" tests/native/memory_types.py
  FAKEGPU_BUILD_DIR="$BUILD_DIR" \
    "$PYTHON_BIN" tests/native/coordinator_smoke.py

  if "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    "$PYTHON_BIN" tests/support/fakecuda_profile_matrix.py
  else
    echo "torch not found; skipped the FakeCUDA profile matrix."
  fi
}

run_cpu() {
  BUILD_DIR="$CPU_BUILD_DIR" "$SCRIPT_DIR/build.sh" --release

  local probe="$CPU_BUILD_DIR/cublas_cpu_sim"
  local smi_state="$CPU_BUILD_DIR/fakegpu_smi_state.json"
  c++ -std=c++17 tests/native/cublas_cpu_sim.cpp -o "$probe" \
    -L "$CPU_BUILD_DIR" -lcublas -lcudart -lcuda -lnvidia-ml

  FAKEGPU_REPORT_PATH="$CPU_BUILD_DIR/fake_gpu_report.json" \
    FAKEGPU_SMI_STATE_PATH="$smi_state" \
    run_fakegpu "$CPU_BUILD_DIR" "$probe"
  "$PYTHON_BIN" scripts/validation/check_report.py \
    --path "$CPU_BUILD_DIR/fake_gpu_report.json" \
    --smi-state "$smi_state" \
    --expect-io \
    --expect-flops \
    --expect-unsupported-api

  if "$PYTHON_BIN" -c \
      "import torch; raise SystemExit(0 if torch.version.cuda else 1)" \
      >/dev/null 2>&1; then
    FAKEGPU_REPORT_PATH="$CPU_BUILD_DIR/fake_gpu_report.json" \
      run_fakegpu "$CPU_BUILD_DIR" "$PYTHON_BIN" tests/native/cpu_sim_matmul.py
  else
    echo "CUDA-enabled torch not found; skipped the PyTorch matmul check."
  fi
}

suite="${1:-}"
case "$suite" in
  python)
    run_python
    ;;
  smoke)
    run_smoke
    ;;
  cpu)
    run_cpu
    ;;
  all)
    run_python
    run_smoke
    run_cpu
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "error: unknown suite: $suite" >&2
    usage >&2
    exit 2
    ;;
esac
