#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is intended to run on macOS." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v colima >/dev/null 2>&1; then
  echo "error: colima not found." >&2
  exit 1
fi

if ! colima status >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: colima is not running.

Start an x86_64 VM first (example):
  colima start --arch x86_64
EOF
  exit 1
fi

arch="$(colima ssh -- bash -lc 'uname -m' | tr -d '\r' | tail -n 1 || true)"
if [[ "$arch" != "x86_64" ]]; then
  echo "error: expected Linux x86_64 in Colima, got: ${arch:-"(unknown)"}" >&2
  exit 1
fi

echo "Colima arch: $arch"
echo "Repo mount: $REPO_ROOT"
echo

remote_repo="'${REPO_ROOT//\'/\'\"\'\"\'}'"
colima ssh -- bash -lc "cd $remote_repo && scripts/linux/nvitop_once.sh"
