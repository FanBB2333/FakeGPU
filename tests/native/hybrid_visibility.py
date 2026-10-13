"""Real-GPU regression: masked hybrid discovery and three training steps.

Run explicitly on a CUDA host:
    python tests/native/hybrid_visibility.py --build-dir build --device 1

``--device`` accepts a physical ordinal or GPU UUID. A missing CUDA device is
a failure, so this probe is separate from the CPU-only smoke suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def workload() -> dict:
    import torch

    torch.manual_seed(123)
    assert torch.cuda.is_available()
    assert not getattr(torch.cuda, "_fakegpu_simulated", False)
    assert torch.cuda.device_count() == 1
    name = torch.cuda.get_device_name(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 16)
    ).cuda()
    inputs = torch.randn(8, 64, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    parameters = b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    return {
        "name": name,
        "losses": losses,
        "signature": hashlib.sha256(parameters).hexdigest(),
        "peak": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--device", default="0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print("GPU_PROBE=" + json.dumps(workload(), sort_keys=True))
        return

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fakegpu import env as native_env

    base = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, FAKEGPU_TERMINAL_REPORT="0")
    results = {}
    for mode in ("real", "passthrough", "hybrid"):
        environment = native_env(
            build_dir=args.build_dir,
            base_env=base,
            mode="passthrough" if mode == "real" else mode,
            oom_policy="clamp",
            device_count=1,
        )
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=root, env=environment, text=True, capture_output=True, timeout=120,
        )
        assert process.returncode == 0, f"{mode}: {process.stdout}\n{process.stderr}"
        results[mode] = json.loads(next(
            line.removeprefix("GPU_PROBE=")
            for line in process.stdout.splitlines() if line.startswith("GPU_PROBE=")
        ))
    assert len({value["signature"] for value in results.values()}) == 1, results
    assert len({value["peak"] for value in results.values()}) == 1, results
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
