#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$PROJECT_ROOT/test/output"

NPROC="${1:-4}"
if [[ "$NPROC" != "4" ]]; then
    echo "usage: $0 [4]" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

COORDINATOR_BIN="$BUILD_DIR/fakegpu-coordinator"
FGPU_BIN="$PROJECT_ROOT/fgpu"
NCCL_LIB="$BUILD_DIR/libnccl.so.2"
CLUSTER_CONFIG="$PROJECT_ROOT/verification/data/cluster_valid.yaml"
RUN_LOG="$OUTPUT_DIR/ddp_multinode_${NPROC}r_torchrun.log"
COORD_LOG="$OUTPUT_DIR/ddp_multinode_${NPROC}r_coordinator.log"
REPORT_FILE="$OUTPUT_DIR/ddp_multinode_${NPROC}r_validation_report.md"
CLUSTER_REPORT="$OUTPUT_DIR/ddp_multinode_${NPROC}r_cluster_report.json"

for path in "$COORDINATOR_BIN" "$FGPU_BIN" "$NCCL_LIB" "$CLUSTER_CONFIG"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required artifact: $path" >&2
        exit 1
    fi
done

if ! command -v torchrun >/dev/null 2>&1; then
    echo "missing required command: torchrun" >&2
    exit 1
fi

MASTER_PORT="$(python3 -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
TMPDIR="$(mktemp -d -t fakegpu-ddp-step15-XXXXXX)"
SOCKET_PATH="$TMPDIR/coordinator.sock"
RANK_REPORT_DIR="$TMPDIR/reports"
TORCHRUN_EXIT=0
CHECK_REPORT_EXIT=0
COORDINATOR_STOPPED=0

shutdown_coordinator() {
    if [[ "${COORDINATOR_STOPPED:-0}" == "1" ]]; then
        return
    fi
    COORDINATOR_STOPPED=1
    if [[ -n "${COORDINATOR_PID:-}" ]] && kill -0 "$COORDINATOR_PID" >/dev/null 2>&1; then
        python3 -c 'import socket, sys; sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sock.connect(sys.argv[1]); sock.sendall(b"SHUTDOWN\n"); sock.recv(4096); sock.close()' "$SOCKET_PATH" >/dev/null 2>&1 || true
        wait "$COORDINATOR_PID" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    shutdown_coordinator
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
FAKEGPU_CLUSTER_REPORT_PATH="$CLUSTER_REPORT" \
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
    --cluster-config "$CLUSTER_CONFIG" \
    --coordinator-transport unix \
    --coordinator-addr "$SOCKET_PATH" \
    --device-count "$NPROC" \
    torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    "$SCRIPT_DIR/test_ddp_multinode.py" \
    --report-dir "$RANK_REPORT_DIR" \
    --epochs 1 \
    >"$RUN_LOG" 2>&1
TORCHRUN_EXIT="$?"
set -e

shutdown_coordinator

set +e
python3 "$PROJECT_ROOT/verification/check_cluster_report.py" \
    --path "$CLUSTER_REPORT" \
    --expect-world-size "$NPROC" \
    --expect-node-count 2 \
    --expect-collective all_reduce \
    --expect-collective broadcast \
    --min-ranks "$NPROC"
CHECK_REPORT_EXIT="$?"
set -e

REPORT_DIR_ENV="$RANK_REPORT_DIR" \
REPORT_FILE_ENV="$REPORT_FILE" \
CLUSTER_REPORT_ENV="$CLUSTER_REPORT" \
NPROC_ENV="$NPROC" \
TORCHRUN_EXIT_ENV="$TORCHRUN_EXIT" \
CHECK_REPORT_EXIT_ENV="$CHECK_REPORT_EXIT" \
RUN_LOG_ENV="$RUN_LOG" \
COORD_LOG_ENV="$COORD_LOG" \
python3 - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR_ENV"])
report_file = Path(os.environ["REPORT_FILE_ENV"])
cluster_report_path = Path(os.environ["CLUSTER_REPORT_ENV"])
nproc = int(os.environ["NPROC_ENV"])
torchrun_exit = int(os.environ["TORCHRUN_EXIT_ENV"])
check_report_exit = int(os.environ["CHECK_REPORT_EXIT_ENV"])
run_log = Path(os.environ["RUN_LOG_ENV"])
coord_log = Path(os.environ["COORD_LOG_ENV"])

rank_reports = []
for rank in range(nproc):
    path = report_dir / f"rank_{rank}.json"
    if path.exists():
        rank_reports.append(json.loads(path.read_text(encoding="utf-8")))

cluster_report = None
if cluster_report_path.exists():
    cluster_report = json.loads(cluster_report_path.read_text(encoding="utf-8"))

all_success = (
    torchrun_exit == 0
    and check_report_exit == 0
    and len(rank_reports) == nproc
    and all(item.get("status") == "success" for item in rank_reports)
)

lines = [
    "# Step 15 DDP Validation Report",
    "",
    f"- `nproc_per_node`: {nproc}",
    f"- `torchrun_exit_code`: {torchrun_exit}",
    f"- `cluster_report_check_exit_code`: {check_report_exit}",
    f"- `overall_status`: {'success' if all_success else 'gap'}",
    f"- `torchrun_log`: `{run_log}`",
    f"- `coordinator_log`: `{coord_log}`",
    f"- `cluster_report`: `{cluster_report_path}`",
]

lines.extend([
    "",
    "## Rank Results",
    "",
])

if not rank_reports:
    lines.append("- no per-rank reports were generated")
else:
    barrier_successes = 0
    for item in sorted(rank_reports, key=lambda value: value.get("rank", -1)):
        if item.get("barrier_before_training") == "ok":
            barrier_successes += 1
        if item.get("barrier_after_training") == "ok":
            barrier_successes += 1
        lines.append(
            f"- rank {item.get('rank')}: status={item.get('status')} "
            f"epochs={item.get('epochs_completed', 0)} steps={item.get('steps_completed', 0)} "
            f"stage={item.get('failed_stage', 'completed')}"
        )
        if item.get("status") != "success":
            lines.append(
                f"  error={item.get('exception_type')}: {item.get('exception_message')}"
            )
    lines.append("")
    lines.append(f"- framework barrier successes: {barrier_successes}")

lines.extend([
    "",
    "## Cluster Report Summary",
    "",
])

if cluster_report is None:
    lines.append("- cluster report was not generated")
else:
    cluster = cluster_report.get("cluster", {})
    collectives = cluster_report.get("collectives", {})
    lines.append(
        f"- world_size={cluster.get('world_size')} node_count={cluster.get('node_count')} "
        f"communicators={cluster.get('communicators')}"
    )
    for name in ("all_reduce", "broadcast", "barrier"):
        stats = collectives.get(name, {})
        lines.append(f"- {name}: calls={stats.get('calls')} bytes={stats.get('bytes')}")

lines.extend([
    "",
    "## Log Excerpt",
    "",
])

if run_log.exists():
    excerpt = run_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    if excerpt:
        lines.extend([f"- {line}" for line in excerpt])
    else:
        lines.append("- torchrun log was empty")
else:
    lines.append("- torchrun log was not found")

report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

cat "$REPORT_FILE"

if [[ "$TORCHRUN_EXIT" -ne 0 || "$CHECK_REPORT_EXIT" -ne 0 ]]; then
    exit 1
fi

if ! python3 - <<'PY' "$RANK_REPORT_DIR" "$NPROC"
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
nproc = int(sys.argv[2])
for rank in range(nproc):
    path = report_dir / f"rank_{rank}.json"
    if not path.exists():
        raise SystemExit(1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise SystemExit(1)
    if int(payload.get("epochs_completed", 0)) < 1:
        raise SystemExit(1)
    if payload.get("barrier_before_training") != "ok":
        raise SystemExit(1)
    if payload.get("barrier_after_training") != "ok":
        raise SystemExit(1)
raise SystemExit(0)
PY
then
    exit 1
fi

exit 0
