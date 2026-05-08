#!/usr/bin/env python3
from __future__ import annotations

import argparse

import fakegpu
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Small fakecuda preflight memory probe")
    parser.add_argument("--mode", choices=["pass", "oom"], default="pass")
    parser.add_argument("--oom-elements", type=int, default=300_000_000)
    ns = parser.parse_args()

    with fakegpu.stage("forward"):
        if ns.mode == "pass":
            x = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
            print(f"peak={torch.cuda.max_memory_allocated()}")
            del x
            return 0

        torch.empty((ns.oom_elements,), device="cuda", dtype=torch.float32)
        raise RuntimeError("expected FakeGPU OOM was not raised")


if __name__ == "__main__":
    raise SystemExit(main())
