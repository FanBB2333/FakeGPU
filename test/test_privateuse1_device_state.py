"""Device index smoke tests for the FakeGPU PrivateUse1 backend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from fakegpu.privateuse1 import init_privateuse1


def main() -> None:
    init_privateuse1()

    assert torch.fgpu.device_count() >= 3
    assert torch.fgpu.current_device() == 0

    x0 = torch.tensor([1.0]).to("fgpu")
    assert x0.device == torch.device("fgpu:0")

    torch.fgpu.set_device(1)
    assert torch.fgpu.current_device() == 1

    x1 = torch.tensor([2.0]).to("fgpu")
    assert x1.device == torch.device("fgpu:1")

    explicit = torch.tensor([3.0]).to("fgpu:2")
    assert explicit.device == torch.device("fgpu:2")

    with torch.fgpu.device(2):
        assert torch.fgpu.current_device() == 2
        x2 = torch.tensor([4.0]).to("fgpu")
        assert x2.device == torch.device("fgpu:2")

    assert torch.fgpu.current_device() == 1

    print("privateuse1 device-state smoke passed")


if __name__ == "__main__":
    main()
