#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
LOGGING="OFF"
CPU_SIMULATION="ON"
EXTRA_CMAKE_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/build.sh [options] [-- <extra CMake arguments>]

Options:
  --build-dir PATH       Build directory (default: build)
  --debug                Debug build with FakeGPU logging enabled
  --release              Release build with FakeGPU logging disabled
  --logging              Enable FakeGPU logging without changing build type
  --no-cpu-simulation    Disable CPU-backed operator simulation
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --debug)
      BUILD_TYPE="Debug"
      LOGGING="ON"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      LOGGING="OFF"
      shift
      ;;
    --logging)
      LOGGING="ON"
      shift
      ;;
    --no-cpu-simulation)
      CPU_SIMULATION="OFF"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CMAKE_ARGS+=("$@")
      break
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CMAKE_ARGS=(
  -S "$REPO_ROOT"
  -B "$BUILD_DIR"
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DENABLE_FAKEGPU_LOGGING=$LOGGING"
  "-DENABLE_FAKEGPU_CPU_SIMULATION=$CPU_SIMULATION"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  CMAKE_ARGS+=(
    "-DCMAKE_C_COMPILER=/usr/bin/clang"
    "-DCMAKE_CXX_COMPILER=/usr/bin/clang++"
  )
fi

if [[ ${#EXTRA_CMAKE_ARGS[@]} -gt 0 ]]; then
  cmake "${CMAKE_ARGS[@]}" "${EXTRA_CMAKE_ARGS[@]}"
else
  cmake "${CMAKE_ARGS[@]}"
fi
cmake --build "$BUILD_DIR" --parallel
