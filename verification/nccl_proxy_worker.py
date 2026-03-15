#!/usr/bin/env python3

from __future__ import annotations

import base64
import ctypes
import json
import os
import time
import traceback
from pathlib import Path


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    report_path = Path(os.environ["REPORT_PATH"])
    uid_path = Path(os.environ["NCCL_UID_PATH"])

    report = {
        "rank": rank,
        "world_size": world_size,
        "mode": os.environ["RUN_MODE"],
        "status": "starting",
        "stage": "bootstrap",
    }
    comm = None

    def flush() -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    try:
        import torch

        report["stage"] = "torch_ready"
        flush()
        torch.cuda.set_device(0)
        report["stage"] = "cuda_ready"
        flush()

        lib = ctypes.CDLL(os.environ["NCCL_LIB_PATH"], mode=ctypes.RTLD_GLOBAL)
        report["stage"] = "nccl_loaded"
        flush()

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
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        lib.ncclGetUniqueId.restype = ctypes.c_int
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
        if rank == 0:
            result = lib.ncclGetUniqueId(ctypes.byref(unique_id))
            report["unique_id_result"] = int(result)
            if result != 0:
                raise AssertionError(f"ncclGetUniqueId failed with code {result}")
            payload = ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id))
            uid_path.write_text(base64.b64encode(payload).decode("ascii"), encoding="utf-8")
        else:
            deadline = time.monotonic() + 30.0
            while not uid_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for NCCL uid file: {uid_path}")
                time.sleep(0.05)
            raw_unique_id = base64.b64decode(uid_path.read_text(encoding="utf-8").strip())
            ctypes.memmove(
                ctypes.addressof(unique_id),
                raw_unique_id,
                min(len(raw_unique_id), ctypes.sizeof(unique_id)),
            )
        report["stage"] = "unique_id_ready"
        flush()

        comm = ncclComm_t()
        result = lib.ncclCommInitRank(ctypes.byref(comm), world_size, unique_id, rank)
        report["comm_init_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclCommInitRank failed with code {result}")
        report["stage"] = "comm_ready"
        flush()

        all_reduce = torch.tensor(
            [float(rank + 1), float(10 - rank)],
            device="cuda",
            dtype=torch.float32,
        )
        result = lib.ncclAllReduce(
            ctypes.c_void_p(all_reduce.data_ptr()),
            ctypes.c_void_p(all_reduce.data_ptr()),
            2,
            7,
            0,
            comm,
            None,
        )
        report["all_reduce_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclAllReduce failed with code {result}")
        report["stage"] = "all_reduce_done"
        flush()

        broadcast = torch.tensor(
            [111, 222] if rank == 0 else [0, 0],
            device="cuda",
            dtype=torch.int32,
        )
        result = lib.ncclBroadcast(
            ctypes.c_void_p(broadcast.data_ptr()),
            ctypes.c_void_p(broadcast.data_ptr()),
            2,
            2,
            0,
            comm,
            None,
        )
        report["broadcast_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclBroadcast failed with code {result}")
        report["stage"] = "broadcast_done"
        flush()

        torch.cuda.synchronize()

        report["all_reduce_value"] = [float(v) for v in all_reduce.cpu().tolist()]
        report["broadcast_value"] = [int(v) for v in broadcast.cpu().tolist()]
        report["status"] = "success"
        report["stage"] = "before_destroy"

        result = lib.ncclCommDestroy(comm)
        report["destroy_result"] = int(result)
        if result != 0:
            raise AssertionError(f"ncclCommDestroy failed with code {result}")
        comm = None
        report["stage"] = "done"
        flush()
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        if comm is not None and "lib" in locals():
            try:
                lib.ncclCommDestroy(comm)
            except Exception:
                pass
        flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
