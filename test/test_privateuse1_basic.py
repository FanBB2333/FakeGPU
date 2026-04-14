"""Minimal smoke test for the FakeGPU PrivateUse1 bootstrap."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from fakegpu.privateuse1 import init_privateuse1


def main() -> None:
    init_privateuse1()

    assert hasattr(torch, "fgpu")
    assert torch.device("fgpu:0").type == "fgpu"
    assert hasattr(torch.Tensor, "is_fgpu")
    assert hasattr(torch.nn.Module, "fgpu")
    assert torch.fgpu.is_available() is True
    assert torch.fgpu.device_count() >= 1
    assert torch.fgpu.current_device() == 0

    x = torch.tensor([1.0, 2.0]).to("fgpu")
    assert x.device.type == "fgpu"
    assert x.is_fgpu is True

    print("privateuse1 bootstrap smoke passed")


if __name__ == "__main__":
    main()
