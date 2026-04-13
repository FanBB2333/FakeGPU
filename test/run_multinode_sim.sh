#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$PROJECT_ROOT/test/output"

NPROC="${1:-2}"
if [[ "$NPROC" != "2" && "$NPROC" != "4" ]]; then
    echo "usage: $0 [2|4]" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

COORDINATOR_BIN="$BUILD_DIR/fakegpu-coordinator"
FGPU_BIN="$PROJECT_ROOT/fgpu"
NCCL_LIB="$BUILD_DIR/libnccl.so.2"
RUN_LOG="$OUTPUT_DIR/ddp_multinode_${NPROC}r_torchrun.log"
COORD_LOG="$OUTPUT_DIR/ddp_multinode_${NPROC}r_coordinator.log"
REPORT_FILE="$OUTPUT_DIR/ddp_multinode_${NPROC}r_gap_report.md"

for path in "$COORDINATOR_BIN" "$FGPU_BIN" "$NCCL_LIB"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required artifact: $path" >&2
        exit 1
    fi
done

if ! command -v torchrun >/dev/null 2>&1; then
    echo "missing required command: torchrun" >&2
    exit 1
fi

TMPDIR="$(mktemp -d -t fakegpu-ddp-step10-XXXXXX)"
SOCKET_PATH="$TMPDIR/coordinator.sock"
REPORT_DIR="$TMPDIR/reports"
MASTER_PORT="$(python3 -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
TORCHRUN_EXIT=0

cleanup() {
    if [[ -n "${COORDINATOR_PID:-}" ]] && kill -0 "$COORDINATOR_PID" >/dev/null 2>&1; then
        python3 -c 'import socket, sys; sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sock.connect(sys.argv[1]); sock.sendall(b"SHUTDOWN\n"); sock.recv(4096); sock.close()' "$SOCKET_PATH" >/dev/null 2>&1 || true
        wait "$COORDINATOR_PID" >/dev/null 2>&1 || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

"$COORDINATOR_BIN" --transport unix --address "$SOCKET_PATH" >"$COORD_LOG" 2>&1 &
COORDINATOR_PID="$!"

for _ in $(seq 1 50); do
    if [[ -S "$SOCKET_PATH" ]]; then
        break
    fi
    sleep 0.1
done

if [[ ! -S "$SOCKET_PATH" ]]; then
    echo "coordinator socket was not created: $SOCKET_PATH" >&2
    exit 1
fi

export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"

set +e
"$FGPU_BIN" \
    --mode simulate \
    --dist-mode simulate \
    --coordinator-transport unix \
    --coordinator-addr "$SOCKET_PATH" \
    --device-count "$NPROC" \
    torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    "$SCRIPT_DIR/test_ddp_multinode.py" \
    --report-dir "$REPORT_DIR" \
    >"$RUN_LOG" 2>&1
TORCHRUN_EXIT="$?"
set -e

REPORT_DIR_ENV="$REPORT_DIR" \
REPORT_FILE_ENV="$REPORT_FILE" \
NPROC_ENV="$NPROC" \
TORCHRUN_EXIT_ENV="$TORCHRUN_EXIT" \
RUN_LOG_ENV="$RUN_LOG" \
COORD_LOG_ENV="$COORD_LOG" \
python3 - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR_ENV"])
report_file = Path(os.environ["REPORT_FILE_ENV"])
nproc = int(os.environ["NPROC_ENV"])
torchrun_exit = int(os.environ["TORCHRUN_EXIT_ENV"])
run_log = Path(os.environ["RUN_LOG_ENV"])
coord_log = Path(os.environ["COORD_LOG_ENV"])

reports = []
for index in range(nproc):
    path = report_dir / f"rank_{index}.json"
    if path.exists():
        reports.append(json.loads(path.read_text(encoding="utf-8")))

missing = [str(index) for index in range(nproc) if not (report_dir / f"rank_{index}.json").exists()]
all_success = len(reports) == nproc and all(item.get("status") == "success" for item in reports)
overall_status = "success" if all_success and torchrun_exit == 0 else "gap"
log_excerpt = []
if run_log.exists():
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Error" in line or "error" in line or "ImportError" in line or "RuntimeError" in line or "undefined symbol" in line:
            log_excerpt.append(line.strip())
    if not log_excerpt:
        log_excerpt = run_log.read_text(encoding="utf-8", errors="replace").splitlines()[:10]

lines = [
    "# Step 10 Torch Distributed Smoke Report",
    "",
    f"- `nproc_per_node`: {nproc}",
    f"- `torchrun_exit_code`: {torchrun_exit}",
    f"- `overall_status`: {overall_status}",
    f"- `torchrun_log`: `{run_log}`",
    f"- `coordinator_log`: `{coord_log}`",
]

if missing:
    lines.extend([
        "",
        "## Missing Rank Reports",
        "",
        f"- missing ranks: {', '.join(missing)}",
    ])

if log_excerpt:
    lines.extend([
        "",
        "## Torchrun Log Excerpt",
        "",
    ])
    lines.extend([f"- {line}" for line in log_excerpt[:10]])

lines.extend([
    "",
    "## Rank Results",
    "",
])

if not reports:
    lines.append("- no per-rank reports were generated")
else:
    for item in sorted(reports, key=lambda value: value.get("rank", -1)):
        lines.append(
            f"- rank {item.get('rank')}: status={item.get('status')} stage={item.get('failed_stage', 'completed')}"
        )
        if item.get("status") != "success":
            lines.append(
                f"  error={item.get('exception_type')}: {item.get('exception_message')}"
            )

lines.extend([
    "",
    "## Gap Summary",
    "",
])

gaps = []
for item in sorted(reports, key=lambda value: value.get("rank", -1)):
    if item.get("status") == "success":
        continue
    gaps.append(
        f"- rank {item.get('rank')} failed at `{item.get('failed_stage')}`: "
        f"{item.get('exception_type')}: {item.get('exception_message')}"
    )

if gaps:
    lines.extend(gaps)
elif log_excerpt:
    lines.append(f"- launcher failed before rank reports were written: {log_excerpt[0]}")
else:
    lines.append("- no framework gap was observed in this run")

report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

cat "$REPORT_FILE"

if [[ -s "$REPORT_FILE" ]]; then
    exit 0
fi

exit 1
