"""Wrapper-level smoke tests for the FakeGPU PrivateUse1 backend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from fakegpu.privateuse1 import init_privateuse1


def main() -> None:
    init_privateuse1()

    cpu = torch.randn(2, 2)
    fg = cpu.to("fgpu")

    assert fg.device.type == "fgpu"
    assert fg.is_fgpu is True
    assert hasattr(fg, "raw_data")
    assert torch.allclose(fg.raw_data, cpu)

    out = fg + fg
    assert out.device.type == "fgpu"
    assert out.is_fgpu is True
    assert hasattr(out, "raw_data")
    assert torch.allclose(out.raw_data, cpu + cpu)

    model = torch.nn.Linear(2, 2).fgpu()
    assert next(model.parameters()).device.type == "fgpu"

    print("privateuse1 wrapper smoke passed")


if __name__ == "__main__":
    main()
