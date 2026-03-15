#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$PROJECT_ROOT/test/output"

NPROC="${1:-2}"
if [[ "$NPROC" != "2" ]]; then
    echo "usage: $0 [2]" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

COORDINATOR_BIN="$BUILD_DIR/fakegpu-coordinator"
NCCL_LIB="$BUILD_DIR/libnccl.so.2"
CLUSTER_CONFIG="$PROJECT_ROOT/verification/data/cluster_hybrid_2r.yaml"
RUN_LOG="$OUTPUT_DIR/hybrid_multinode_${NPROC}r_runner.log"
COORD_LOG="$OUTPUT_DIR/hybrid_multinode_${NPROC}r_coordinator.log"
REPORT_FILE="$OUTPUT_DIR/hybrid_multinode_${NPROC}r_validation_report.md"
CLUSTER_REPORT="$OUTPUT_DIR/hybrid_multinode_${NPROC}r_cluster_report.json"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python3" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

for path in "$COORDINATOR_BIN" "$NCCL_LIB" "$CLUSTER_CONFIG" "$SCRIPT_DIR/test_hybrid_multinode.py"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required artifact: $path" >&2
        exit 1
    fi
done

TMPDIR="$(/usr/bin/mktemp -d -t fakegpu-hybrid-step19-XXXXXX)"
SOCKET_PATH="$TMPDIR/coordinator.sock"
RANK_REPORT_DIR="$TMPDIR/reports"
RUN_EXIT=0
CHECK_REPORT_EXIT=0
COORDINATOR_STOPPED=0

shutdown_coordinator() {
    if [[ "${COORDINATOR_STOPPED:-0}" == "1" ]]; then
        return
    fi
    COORDINATOR_STOPPED=1
    if [[ -n "${COORDINATOR_PID:-}" ]] && kill -0 "$COORDINATOR_PID" >/dev/null 2>&1; then
        "$PYTHON_BIN" -c 'import socket, sys; sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sock.connect(sys.argv[1]); sock.sendall(b"SHUTDOWN\n"); sock.recv(4096); sock.close()' "$SOCKET_PATH" >/dev/null 2>&1 || true
        wait "$COORDINATOR_PID" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    shutdown_coordinator
    /bin/rm -rf "$TMPDIR"
}
trap cleanup EXIT

FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
FAKEGPU_CLUSTER_REPORT_PATH="$CLUSTER_REPORT" \
"$COORDINATOR_BIN" --transport unix --address "$SOCKET_PATH" >"$COORD_LOG" 2>&1 &
COORDINATOR_PID="$!"

for _ in {1..50}; do
    if [[ -S "$SOCKET_PATH" ]]; then
        break
    fi
    /bin/sleep 0.1
done

if [[ ! -S "$SOCKET_PATH" ]]; then
    echo "coordinator socket was not created: $SOCKET_PATH" >&2
    exit 1
fi

set +e
FAKEGPU_MODE=hybrid \
FAKEGPU_DEVICE_COUNT=1 \
FAKEGPU_DIST_MODE=simulate \
FAKEGPU_CLUSTER_CONFIG="$CLUSTER_CONFIG" \
FAKEGPU_COORDINATOR_TRANSPORT=unix \
FAKEGPU_COORDINATOR_ADDR="$SOCKET_PATH" \
"$PYTHON_BIN" "$SCRIPT_DIR/test_hybrid_multinode.py" \
    --report-dir "$RANK_REPORT_DIR" \
    --world-size "$NPROC" \
    --python-bin "$PYTHON_BIN" \
    --nccl-lib "$NCCL_LIB" \
    >"$RUN_LOG" 2>&1
RUN_EXIT="$?"
set -e

shutdown_coordinator

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/verification/check_cluster_report.py" \
    --path "$CLUSTER_REPORT" \
    --expect-world-size "$NPROC" \
    --expect-node-count 2 \
    --expect-collective all_reduce \
    --expect-collective broadcast \
    --expect-links \
    --min-ranks "$NPROC"
CHECK_REPORT_EXIT="$?"
set -e

REPORT_DIR_ENV="$RANK_REPORT_DIR" \
REPORT_FILE_ENV="$REPORT_FILE" \
CLUSTER_REPORT_ENV="$CLUSTER_REPORT" \
NPROC_ENV="$NPROC" \
RUN_EXIT_ENV="$RUN_EXIT" \
CHECK_REPORT_EXIT_ENV="$CHECK_REPORT_EXIT" \
RUN_LOG_ENV="$RUN_LOG" \
COORD_LOG_ENV="$COORD_LOG" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR_ENV"])
report_file = Path(os.environ["REPORT_FILE_ENV"])
cluster_report_path = Path(os.environ["CLUSTER_REPORT_ENV"])
nproc = int(os.environ["NPROC_ENV"])
run_exit = int(os.environ["RUN_EXIT_ENV"])
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
    run_exit == 0
    and check_report_exit == 0
    and len(rank_reports) == nproc
    and all(item.get("status") == "success" for item in rank_reports)
)

lines = [
    "# Step 19 Hybrid Validation Report",
    "",
    f"- `nproc_per_node`: {nproc}",
    f"- `runner_exit_code`: {run_exit}",
    f"- `cluster_report_check_exit_code`: {check_report_exit}",
    f"- `overall_status`: {'success' if all_success else 'gap'}",
    f"- `runner_log`: `{run_log}`",
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
    for item in sorted(rank_reports, key=lambda value: value.get("rank", -1)):
        lines.append(
            f"- rank {item.get('rank')}: status={item.get('status')} "
            f"matmul_checksum={item.get('matmul_checksum')} "
            f"all_reduce={item.get('all_reduce_value', 'n/a')} "
            f"broadcast={item.get('broadcast_value', 'n/a')}"
        )
        lines.append(
            f"- rank {item.get('rank')} detail: "
            f"matmul_max_abs_diff={item.get('matmul_max_abs_diff')} "
            f"comm_init={item.get('comm_init_result')} "
            f"destroy={item.get('destroy_result')}"
        )
        if item.get("status") != "success":
            lines.append(
                f"- rank {item.get('rank')} error: {item.get('exception_type')}: "
                f"{item.get('exception_message')}"
            )

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
    links = cluster_report.get("links", [])
    lines.append(
        f"- world_size={cluster.get('world_size')} node_count={cluster.get('node_count')} "
        f"communicators={cluster.get('communicators')}"
    )
    for name in ("all_reduce", "broadcast"):
        stats = collectives.get(name, {})
        lines.append(f"- {name}: calls={stats.get('calls')} bytes={stats.get('bytes')}")
    lines.append(f"- links: {len(links)}")

lines.extend([
    "",
    "## Log Excerpt",
    "",
])

if run_log.exists():
    excerpt = [line.rstrip() for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]]
    if excerpt:
        lines.extend([f"- {line}" for line in excerpt])
    else:
        lines.append("- runner log was empty")
else:
    lines.append("- runner log was not found")

report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

/bin/cat "$REPORT_FILE"

if [[ "$RUN_EXIT" -ne 0 || "$CHECK_REPORT_EXIT" -ne 0 ]]; then
    exit 1
fi

"$PYTHON_BIN" - <<'PY' "$RANK_REPORT_DIR" "$NPROC"
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
nproc = int(sys.argv[2])

for rank in range(nproc):
    payload = json.loads((report_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise SystemExit(f"rank {rank} did not succeed: {payload}")
    if float(payload.get("matmul_max_abs_diff", 1.0)) > 1e-5:
        raise SystemExit(f"rank {rank} matmul diff too high: {payload}")
    expected = float(nproc * (nproc + 1) // 2)
    if abs(float(payload.get("all_reduce_value", 0.0)) - expected) > 1e-6:
        raise SystemExit(f"rank {rank} all_reduce mismatch: {payload}")
    if int(payload.get("broadcast_value", -1)) != 2048:
        raise SystemExit(f"rank {rank} broadcast mismatch: {payload}")
PY
