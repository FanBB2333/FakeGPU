#!/usr/bin/env python3
"""Verify that hybrid clamp exposes a normal PyTorch CUDA OOM."""

from __future__ import annotations

import json


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "SKIP_NO_CUDA"}, sort_keys=True))
        return 3

    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    elements = total_memory // 4 + 1
    requested_bytes = elements * 4
    try:
        allocation = torch.empty((elements,), device="cuda", dtype=torch.float32)
    except torch.cuda.OutOfMemoryError as exc:
        print(
            json.dumps(
                {
                    "status": "PASS_OOM",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "requested_bytes": requested_bytes,
                    "total_memory": total_memory,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_WRONG_ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "requested_bytes": requested_bytes,
                    "total_memory": total_memory,
                },
                sort_keys=True,
            )
        )
        return 2

    del allocation
    print(
        json.dumps(
            {
                "status": "FAIL_NO_OOM",
                "requested_bytes": requested_bytes,
                "total_memory": total_memory,
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
