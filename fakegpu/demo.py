from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from .profile_catalog import ProfileCatalogError, get_profile, validate_catalog


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _print_plain(payload: dict[str, Any]) -> None:
    print("FakeGPU demo completed")
    print(f"  Profile: {payload['profile_id']} ({payload['device_name']})")
    print(
        "  Hardware: "
        f"{payload['architecture'].title()}, "
        f"compute capability {payload['compute_capability']} "
        f"({payload['compiler_target']})"
    )
    print(
        f"  Memory: {_format_bytes(payload['peak_memory_bytes'])} peak / "
        f"{_format_bytes(payload['total_memory_bytes'])} simulated"
    )
    print(
        f"  Tensor: device={payload['tensor_device']}, "
        f"is_cuda={str(payload['tensor_is_cuda']).lower()}"
    )
    print(
        f"  Training: {payload['steps']} steps, "
        f"final loss={payload['final_loss']:.6f}, backend={payload['backend']}"
    )


def _error(message: str, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"ok": False, "error": message}, indent=2))
    else:
        print(f"fakegpu demo: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu demo",
        description="Run a tiny CPU-backed PyTorch training step through FakeGPU.",
    )
    parser.add_argument(
        "--profile",
        default="a100",
        help="GPU profile ID to simulate (default: a100).",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=1,
        help="Number of fake CUDA devices to expose (default: 1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of tiny optimizer steps to execute (default: 2).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    if args.device_count <= 0:
        return _error("--device-count must be positive", as_json=args.json)
    if args.steps <= 0:
        return _error("--steps must be positive", as_json=args.json)

    try:
        profile = get_profile(args.profile)
        validation = validate_catalog()
    except ProfileCatalogError as exc:
        return _error(str(exc), as_json=args.json)
    if validation.errors:
        return _error(
            "invalid GPU profile catalog: " + "; ".join(validation.errors),
            as_json=args.json,
        )

    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
    started = time.perf_counter()
    try:
        from ._runtime import init

        runtime = init(
            runtime="fakecuda",
            profile=profile.id,
            device_count=args.device_count,
        )

        import torch

        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        ).to("cuda:0")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        inputs = torch.randn(6, 8, device="cuda:0")
        targets = torch.randn(6, 4, device="cuda:0")

        final_loss = None
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        parameter = next(model.parameters())
        capability = torch.cuda.get_device_capability(0)
        properties = torch.cuda.get_device_properties(0)
        total_memory = int(getattr(properties, "total_memory", profile.memory_bytes))
        payload: dict[str, Any] = {
            "ok": True,
            "profile_id": profile.id,
            "device_name": torch.cuda.get_device_name(0),
            "architecture": profile.architecture,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "compiler_target": f"sm_{capability[0]}{capability[1]}",
            "total_memory_bytes": total_memory,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "device_count": int(torch.cuda.device_count()),
            "tensor_device": str(parameter.device),
            "tensor_is_cuda": bool(parameter.is_cuda),
            "steps": args.steps,
            "final_loss": final_loss,
            "runtime": runtime.runtime,
            "backend": runtime.backend,
            "torch_version": str(torch.__version__),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", as_json=args.json)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plain(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
