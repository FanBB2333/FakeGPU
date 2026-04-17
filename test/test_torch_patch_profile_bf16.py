"""Regression test for profile-aware BF16 support in fakegpu.torch_patch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_with_profile(profile: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["FAKEGPU_PROFILES"] = profile
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fakegpu.torch_patch import _COMPUTE_MAJOR, _stub_is_bf16_supported; "
                "print(_COMPUTE_MAJOR); "
                "print(_stub_is_bf16_supported())"
            ),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def main() -> None:
    t4 = run_with_profile("t4:1").stdout.strip().splitlines()
    assert t4 == ["7", "False"], t4

    a100 = run_with_profile("a100:1").stdout.strip().splitlines()
    assert a100 == ["8", "True"], a100

    print("torch patch profile BF16 smoke passed")


if __name__ == "__main__":
    main()
