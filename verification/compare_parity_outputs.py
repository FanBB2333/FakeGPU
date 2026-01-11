#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path produced without FakeGPU.")
    parser.add_argument("--passthrough", required=True, help="Path produced with FakeGPU --mode passthrough.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    ns = parser.parse_args(argv)

    import torch

    a = torch.load(ns.baseline, map_location="cpu")
    b = torch.load(ns.passthrough, map_location="cpu")

    a_tok = int(a["next_token"])
    b_tok = int(b["next_token"])
    if a_tok != b_tok:
        print(f"Token mismatch: baseline={a_tok} passthrough={b_tok}")
        return 1

    a_logits = a["last_logits"]
    b_logits = b["last_logits"]
    if tuple(a_logits.shape) != tuple(b_logits.shape):
        print(f"Logits shape mismatch: baseline={tuple(a_logits.shape)} passthrough={tuple(b_logits.shape)}")
        return 1

    ok = torch.allclose(a_logits, b_logits, rtol=ns.rtol, atol=ns.atol)
    if not ok:
        diff = (a_logits - b_logits).abs()
        print("Logits allclose failed.")
        print(f"max_abs_diff={diff.max().item():.6g} mean_abs_diff={diff.mean().item():.6g}")
        print(f"rtol={ns.rtol} atol={ns.atol}")
        return 1

    print("passthrough_parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

