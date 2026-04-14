"""Distributed API smoke for the custom Phase 2 torch build."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.distributed as dist


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"

    assert dist.is_available() is True
    assert dist.is_nccl_available() is True
    assert dist.is_initialized() is False

    dist.init_process_group(backend="nccl", world_size=1, rank=0)

    assert dist.is_initialized() is True
    assert dist.get_backend() == "nccl"
    assert dist.get_world_size() == 1
    assert dist.get_rank() == 0

    x = torch.ones(2, 3, device="cuda")
    dist.all_reduce(x)
    assert torch.equal(x.cpu(), torch.ones(2, 3))

    y = torch.full((2, 3), 5.0, device="cuda")
    dist.broadcast(y, src=0)
    assert torch.equal(y.cpu(), torch.full((2, 3), 5.0))

    gathered = [torch.empty_like(y)]
    dist.all_gather(gathered, y)
    assert len(gathered) == 1
    assert torch.equal(gathered[0].cpu(), y.cpu())

    gathered_objects = [None]
    dist.all_gather_object(gathered_objects, {"epoch": 1})
    assert gathered_objects == [{"epoch": 1}]

    dist.barrier()
    work = dist.barrier(async_op=True)
    assert hasattr(work, "wait")
    work.wait()

    dist.destroy_process_group()
    assert dist.is_initialized() is False

    print("phase2 distributed api surface passed")


if __name__ == "__main__":
    main()
