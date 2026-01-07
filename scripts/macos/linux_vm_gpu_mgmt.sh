#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is intended to run on macOS." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BACKEND="${FAKEGPU_VM_BACKEND:-auto}" # auto|colima|orb
BOOTSTRAP="${FAKEGPU_VM_BOOTSTRAP:-1}" # 1 installs build deps + uv inside VM

shquote() {
  printf "'%s'" "${1//\'/\'\"\'\"\'}"
}

pick_backend() {
  if [[ "$BACKEND" == "colima" || "$BACKEND" == "orb" ]]; then
    echo "$BACKEND"
    return 0
  fi

  if command -v colima >/dev/null 2>&1; then
    echo "colima"
    return 0
  fi
  if command -v orb >/dev/null 2>&1; then
    echo "orb"
    return 0
  fi

  echo "error: neither colima nor orb is available." >&2
  return 1
}

VM_BACKEND="$(pick_backend)"

vm_exec() {
  local cmd="$1"
  case "$VM_BACKEND" in
    colima)
      if ! colima status >/dev/null 2>&1; then
        cat >&2 <<'EOF'
error: colima is not running.

Start an x86_64 VM first (example):
  colima start --arch x86_64
EOF
        return 1
      fi
      colima ssh -- bash -lc "$cmd"
      ;;
    orb)
      local args=()
      if [[ -n "${FAKEGPU_ORB_MACHINE:-}" ]]; then
        args+=(-m "$FAKEGPU_ORB_MACHINE")
      fi
      if [[ -n "${FAKEGPU_ORB_USER:-}" ]]; then
        args+=(-u "$FAKEGPU_ORB_USER")
      fi
      orb "${args[@]}" bash -lc "$cmd"
      ;;
    *)
      echo "error: unknown backend: $VM_BACKEND" >&2
      return 1
      ;;
  esac
}

remote_repo="$(shquote "$REPO_ROOT")"
remote_cmd="cd $remote_repo && uname -m"
arch="$(vm_exec "$remote_cmd" | tr -d '\r' | tail -n 1 || true)"

if [[ "$arch" != "x86_64" ]]; then
  cat >&2 <<EOF
error: expected Linux x86_64 inside VM, got: ${arch:-"(unknown)"}.

- Colima: restart with 'colima start --arch x86_64'
- OrbStack: pick an x86 machine via 'FAKEGPU_ORB_MACHINE=...'
EOF
  exit 1
fi

echo "Using backend: $VM_BACKEND (arch: $arch)"
echo "Repo mount: $REPO_ROOT"
echo

remote_cmd="cd $remote_repo && FAKEGPU_VM_BOOTSTRAP=$(shquote "$BOOTSTRAP") scripts/linux/vm_gpu_mgmt.sh"
vm_exec "$remote_cmd"
