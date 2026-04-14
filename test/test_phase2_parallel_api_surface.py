"""Parallel API surface smoke for the custom Phase 2 torch build."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
from torch.nn.parallel import comm, scatter_gather


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"

    x = torch.randn(8, 4, device="cuda")

    scattered = comm.scatter(x, devices=[0, 1], dim=0)
    assert len(scattered) == 2
    assert scattered[0].device.index == 0
    assert scattered[1].device.index == 1
    assert scattered[0].shape == (4, 4)
    assert scattered[1].shape == (4, 4)

    gathered = comm.gather(scattered, dim=0, destination=0)
    assert gathered.device.type == "cuda"
    assert gathered.device.index == 0
    assert gathered.shape == x.shape

    scattered_via_sg = scatter_gather.scatter(x, [0, 1], dim=0)
    assert len(scattered_via_sg) == 2
    assert scattered_via_sg[0].device.index == 0
    assert scattered_via_sg[1].device.index == 1

    gathered_via_sg = scatter_gather.gather(scattered_via_sg, 0, dim=0)
    assert gathered_via_sg.device.type == "cuda"
    assert gathered_via_sg.shape == x.shape

    model = nn.Linear(4, 2).cuda()
    ddp = nn.parallel.DistributedDataParallel(model)
    y = ddp(x)
    assert y.device.type == "cuda"
    assert y.shape == (8, 2)

    print("phase2 parallel api surface passed")


if __name__ == "__main__":
    main()
