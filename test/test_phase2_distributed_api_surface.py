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

    dist.barrier()
    work = dist.barrier(async_op=True)
    assert hasattr(work, "wait")
    work.wait()

    dist.destroy_process_group()
    assert dist.is_initialized() is False

    print("phase2 distributed api surface passed")


if __name__ == "__main__":
    main()
