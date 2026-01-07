#!/usr/bin/env python3
"""
Minimal end-to-end correctness check for FakeGPU CPU simulation.

Runs a small matrix multiplication on "cuda" (FakeGPU) and compares it against
the CPU reference result.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"ERROR: failed to import torch: {e}")
        return 2

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False (FakeGPU not active?)")
        return 3

    torch.manual_seed(0)

    m, k, n = 8, 16, 12
    a_cpu = torch.randn(m, k, dtype=torch.float32)
    b_cpu = torch.randn(k, n, dtype=torch.float32)

    # Match FakeGPU CPU-sim accumulation (double) more closely.
    ref = (a_cpu.double() @ b_cpu.double()).float()

    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    out_gpu = torch.mm(a_gpu, b_gpu)
    out = out_gpu.cpu()

    ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
    if not ok:
        diff = (out - ref).abs()
        max_abs = float(diff.max().item())
        max_idx = int(diff.view(-1).argmax().item())
        got = float(out.view(-1)[max_idx].item())
        exp = float(ref.view(-1)[max_idx].item())
        print("FAIL: FakeGPU CPU-sim matmul mismatch")
        print(f"  max_abs={max_abs} idx={max_idx} got={got} expected={exp}")
        return 1

    print("OK: FakeGPU CPU-sim matmul matches CPU reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

