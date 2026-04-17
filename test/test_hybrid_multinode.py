#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

WORKER_CODE = textwrap.dedent(
    r"""
    import base64
    import ctypes
    import json
    import os
    import sys
    import traceback
    from pathlib import Path

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report_dir = Path(os.environ["FAKEGPU_REPORT_DIR"])
    report_path = report_dir / f"rank_{rank}.json"

    report = {
        "rank": rank,
        "world_size": world_size,
        "device_index": 0,
        "status": "starting",
    }
    comm = None

    def write_report() -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    try:
        import torch

        report["torch_version"] = torch.__version__

        torch.cuda.set_device(0)
        report["cuda_device"] = torch.cuda.current_device()

        lhs_cpu = (torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) + float(rank))
        rhs_cpu = (torch.arange(1, 17, dtype=torch.float32).reshape(4, 4).t() * 0.25)
        lhs = lhs_cpu.to("cuda")
        rhs = rhs_cpu.to("cuda")
        gpu_result = torch.matmul(lhs, rhs)
        torch.cuda.synchronize()
        gpu_result_cpu = gpu_result.cpu()
        cpu_reference = torch.matmul(lhs_cpu, rhs_cpu)
        max_abs_diff = float((gpu_result_cpu - cpu_reference).abs().max().item())
        report["matmul_checksum"] = float(gpu_result_cpu.sum().item())
        report["matmul_reference_checksum"] = float(cpu_reference.sum().item())
        report["matmul_max_abs_diff"] = max_abs_diff
        if max_abs_diff > 1e-5:
            raise AssertionError(f"matmul max_abs_diff={max_abs_diff} exceeded tolerance")

        lib = ctypes.CDLL(os.environ["FAKEGPU_NCCL_LIB"], mode=ctypes.RTLD_GLOBAL)

        class ncclUniqueId(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_char * 128)]

        ncclComm_t = ctypes.c_void_p

        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ncclComm_t),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
        ]
        lib.ncclCommInitRank.restype = ctypes.c_int
        lib.ncclCommDestroy.argtypes = [ncclComm_t]
        lib.ncclCommDestroy.restype = ctypes.c_int
        lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ncclComm_t,
            ctypes.c_void_p,
        ]
        lib.ncclAllReduce.restype = ctypes.c_int
        lib.ncclBroadcast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ncclComm_t,
            ctypes.c_void_p,
        ]
        lib.ncclBroadcast.restype = ctypes.c_int

        unique_id = ncclUniqueId()
        raw_unique_id = base64.b64decode(os.environ["FAKEGPU_UID_B64"])
        ctypes.memmove(
            ctypes.addressof(unique_id),
            raw_unique_id,
            min(len(raw_unique_id), ctypes.sizeof(unique_id)),
        )

        comm = ncclComm_t()
        init_result = lib.ncclCommInitRank(ctypes.byref(comm), world_size, unique_id, rank)
        report["comm_init_result"] = int(init_result)
        if init_result != 0:
            raise AssertionError(f"ncclCommInitRank failed with code {init_result}")

        all_reduce_value = torch.tensor([float(rank + 1)], device="cuda", dtype=torch.float32)
        all_reduce_result = lib.ncclAllReduce(
            ctypes.c_void_p(all_reduce_value.data_ptr()),
            ctypes.c_void_p(all_reduce_value.data_ptr()),
            1,
            7,
            0,
            comm,
            None,
        )
        report["all_reduce_result"] = int(all_reduce_result)
        if all_reduce_result != 0:
            raise AssertionError(f"ncclAllReduce failed with code {all_reduce_result}")

        broadcast_value = torch.tensor(
            [2048 if rank == 0 else 0],
            device="cuda",
            dtype=torch.int32,
        )
        broadcast_result = lib.ncclBroadcast(
            ctypes.c_void_p(broadcast_value.data_ptr()),
            ctypes.c_void_p(broadcast_value.data_ptr()),
            1,
            2,
            0,
            comm,
            None,
        )
        report["broadcast_result"] = int(broadcast_result)
        if broadcast_result != 0:
            raise AssertionError(f"ncclBroadcast failed with code {broadcast_result}")

        torch.cuda.synchronize()

        report["all_reduce_value"] = float(all_reduce_value.cpu().item())
        report["all_reduce_expected"] = float(world_size * (world_size + 1) // 2)
        if abs(report["all_reduce_value"] - report["all_reduce_expected"]) > 1e-6:
            raise AssertionError(
                f"all_reduce produced {report['all_reduce_value']}, "
                f"expected {report['all_reduce_expected']}"
            )

        report["broadcast_value"] = int(broadcast_value.cpu().item())
        if report["broadcast_value"] != 2048:
            raise AssertionError(
                f"broadcast produced {report['broadcast_value']}, expected 2048"
            )

        destroy_result = lib.ncclCommDestroy(comm)
        report["destroy_result"] = int(destroy_result)
        if destroy_result != 0:
            raise AssertionError(f"ncclCommDestroy failed with code {destroy_result}")
        comm = None

        report["status"] = "success"
        write_report()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "gap"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        if comm is not None and "lib" in locals():
            try:
                lib.ncclCommDestroy(comm)
            except Exception:
                pass
        write_report()
        print(
            f"[Rank {rank}] hybrid gap: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    """
)


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def load_unique_id(lib_path: Path) -> str:
    lib = ctypes.CDLL(str(lib_path))
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
    lib.ncclGetUniqueId.restype = ctypes.c_int

    unique_id = ncclUniqueId()
    result = lib.ncclGetUniqueId(ctypes.byref(unique_id))
    if result != 0:
        raise RuntimeError(f"ncclGetUniqueId failed with code {result}")
    payload = ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id))
    return base64.b64encode(payload).decode("ascii")


def load_report(report_dir: Path, rank: int) -> dict[str, Any]:
    report_path = report_dir / f"rank_{rank}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing rank report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid + simulate multinode smoke for FakeGPU")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--nccl-lib", type=Path, default=REPO_ROOT / "build" / "libnccl.so.2")
    args = parser.parse_args()

    if args.world_size != 2:
        raise SystemExit("only world_size=2 is supported by Step 19 validation")
    if not args.nccl_lib.exists():
        raise SystemExit(f"missing fake NCCL library: {args.nccl_lib}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    unique_id_b64 = load_unique_id(args.nccl_lib)

    processes: list[subprocess.Popen[str]] = []
    try:
        for rank in range(args.world_size):
            env = dict(os.environ)
            env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
            env.update(
                {
                    "FAKEGPU_MODE": "hybrid",
                    "FAKEGPU_DEVICE_COUNT": "1",
                    "FAKEGPU_UID_B64": unique_id_b64,
                    "FAKEGPU_NCCL_LIB": str(args.nccl_lib),
                    "FAKEGPU_REPORT_DIR": str(args.report_dir),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(args.world_size),
                    "LOCAL_RANK": "0",
                }
            )
            proc = subprocess.Popen(
                [str(args.python_bin), "-c", WORKER_CODE],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(proc)

        failed = False
        for rank, proc in enumerate(processes):
            stdout, stderr = proc.communicate(timeout=30.0)
            if stdout:
                print(stdout, end="")
            if stderr:
                sys.stderr.write(stderr)
            if proc.returncode != 0:
                failed = True
                print(f"rank {rank} exited with code {proc.returncode}", file=sys.stderr)

        reports = [load_report(args.report_dir, rank) for rank in range(args.world_size)]
        for report in reports:
            if report.get("status") != "success":
                failed = True
            if float(report.get("matmul_max_abs_diff", 1.0)) > 1e-5:
                failed = True
            if abs(
                float(report.get("all_reduce_value", 0.0))
                - float(report.get("all_reduce_expected", -1.0))
            ) > 1e-6:
                failed = True
            if int(report.get("broadcast_value", -1)) != 2048:
                failed = True

        return 1 if failed else 0
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
