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

    broadcasted = comm.broadcast(torch.randn(2, 4, device="cuda"), devices=[0, 1])
    assert len(broadcasted) == 2
    assert broadcasted[0].device.index == 0
    assert broadcasted[1].device.index == 1
    assert broadcasted[0].shape == (2, 4)

    src = torch.arange(8, dtype=torch.float32, device="cuda").view(2, 4)
    out_b0 = torch.empty(2, 4, device="cuda:0")
    out_b1 = torch.empty(2, 4, device="cuda:1")
    broadcasted_out = comm.broadcast(src, out=[out_b0, out_b1])
    assert broadcasted_out[0].device.index == 0
    assert broadcasted_out[1].device.index == 1
    assert torch.equal(broadcasted_out[0].cpu(), src.cpu())
    assert torch.equal(broadcasted_out[1].cpu(), src.cpu())

    broadcasted_coalesced = comm.broadcast_coalesced(
        [torch.randn(2, 4, device="cuda"), torch.randn(2, 4, device="cuda")],
        devices=[0, 1],
    )
    assert len(broadcasted_coalesced) == 2
    assert len(broadcasted_coalesced[0]) == 2
    assert broadcasted_coalesced[0][0].device.index == 0
    assert broadcasted_coalesced[1][0].device.index == 1

    scattered = comm.scatter(x, devices=[0, 1], dim=0)
    assert len(scattered) == 2
    assert scattered[0].device.index == 0
    assert scattered[1].device.index == 1
    assert scattered[0].shape == (4, 4)
    assert scattered[1].shape == (4, 4)

    out_s0 = torch.empty(4, 4, device="cuda:0")
    out_s1 = torch.empty(4, 4, device="cuda:1")
    scattered_out = comm.scatter(x, out=[out_s0, out_s1], dim=0)
    assert scattered_out[0].device.index == 0
    assert scattered_out[1].device.index == 1
    assert scattered_out[0].shape == (4, 4)
    assert scattered_out[1].shape == (4, 4)

    gathered = comm.gather(scattered, dim=0, destination=0)
    assert gathered.device.type == "cuda"
    assert gathered.device.index == 0
    assert gathered.shape == x.shape

    out_g = torch.empty_like(x, device="cuda:0")
    gathered_out = comm.gather(scattered, dim=0, out=out_g)
    assert gathered_out.device.index == 0
    assert gathered_out.shape == x.shape

    reduced = comm.reduce_add(
        (
            torch.ones(2, 4, device="cuda:0"),
            torch.full((2, 4), 2.0, device="cuda:1"),
        ),
        destination=0,
    )
    assert reduced.device.index == 0
    assert torch.equal(reduced.cpu(), torch.full((2, 4), 3.0))

    reduced_coalesced = comm.reduce_add_coalesced(
        [
            [torch.ones(2, 4, device="cuda:0"), torch.full((2, 4), 2.0, device="cuda:0")],
            [torch.full((2, 4), 3.0, device="cuda:1"), torch.full((2, 4), 4.0, device="cuda:1")],
        ],
        destination=0,
    )
    assert len(reduced_coalesced) == 2
    assert torch.equal(reduced_coalesced[0].cpu(), torch.full((2, 4), 4.0))
    assert torch.equal(reduced_coalesced[1].cpu(), torch.full((2, 4), 6.0))

    scattered_via_sg = scatter_gather.scatter(x, [0, 1], dim=0)
    assert len(scattered_via_sg) == 2
    assert scattered_via_sg[0].device.index == 0
    assert scattered_via_sg[1].device.index == 1

    gathered_via_sg = scatter_gather.gather(scattered_via_sg, 0, dim=0)
    assert gathered_via_sg.device.type == "cuda"
    assert gathered_via_sg.shape == x.shape

    like_on_cuda1 = torch.empty_like(x, device="cuda:1")
    assert like_on_cuda1.device.type == "cuda"
    assert like_on_cuda1.device.index == 1

    model = nn.Linear(4, 2).cuda()
    ddp = nn.parallel.DistributedDataParallel(model)
    y = ddp(x)
    assert y.device.type == "cuda"
    assert y.shape == (8, 2)

    print("phase2 parallel api surface passed")


if __name__ == "__main__":
    main()
