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

    gathered_into = torch.empty_like(y)
    dist.all_gather_into_tensor(gathered_into, y)
    assert torch.equal(gathered_into.cpu(), y.cpu())

    gathered_into_base = torch.empty_like(y)
    dist._all_gather_base(gathered_into_base, y)
    assert torch.equal(gathered_into_base.cpu(), y.cpu())

    gathered_objects = [None]
    dist.all_gather_object(gathered_objects, {"epoch": 1})
    assert gathered_objects == [{"epoch": 1}]

    reduced = torch.full((2, 3), 7.0, device="cuda")
    dist.reduce(reduced, dst=0)
    assert torch.equal(reduced.cpu(), torch.full((2, 3), 7.0))

    gathered_single = [torch.empty_like(y)]
    dist.gather(y, gather_list=gathered_single, dst=0)
    assert len(gathered_single) == 1
    assert torch.equal(gathered_single[0].cpu(), y.cpu())

    scattered_single = torch.empty_like(y)
    dist.scatter(scattered_single, scatter_list=[y], src=0)
    assert torch.equal(scattered_single.cpu(), y.cpu())

    reduced_scatter_out = torch.empty_like(y)
    dist.reduce_scatter(reduced_scatter_out, [y])
    assert torch.equal(reduced_scatter_out.cpu(), y.cpu())

    reduced_scatter_tensor_out = torch.empty_like(y)
    dist.reduce_scatter_tensor(reduced_scatter_tensor_out, y)
    assert torch.equal(reduced_scatter_tensor_out.cpu(), y.cpu())

    reduced_scatter_base_out = torch.empty_like(y)
    dist._reduce_scatter_base(reduced_scatter_base_out, y)
    assert torch.equal(reduced_scatter_base_out.cpu(), y.cpu())

    all_to_all_out = [torch.empty_like(y)]
    dist.all_to_all(all_to_all_out, [y])
    assert len(all_to_all_out) == 1
    assert torch.equal(all_to_all_out[0].cpu(), y.cpu())

    all_to_all_single_out = torch.empty_like(y)
    dist.all_to_all_single(all_to_all_single_out, y)
    assert torch.equal(all_to_all_single_out.cpu(), y.cpu())

    object_list = [{"epoch": 2}]
    dist.broadcast_object_list(object_list, src=0)
    assert object_list == [{"epoch": 2}]

    dist.barrier()
    work = dist.barrier(async_op=True)
    assert hasattr(work, "wait")
    work.wait()

    gather_into_work = dist.all_gather_into_tensor(torch.empty_like(y), y, async_op=True)
    assert hasattr(gather_into_work, "wait")
    gather_into_work.wait()

    reduce_scatter_work = dist.reduce_scatter_tensor(
        torch.empty_like(y), y, async_op=True
    )
    assert hasattr(reduce_scatter_work, "wait")
    reduce_scatter_work.wait()

    dist.destroy_process_group()
    assert dist.is_initialized() is False

    print("phase2 distributed api surface passed")


if __name__ == "__main__":
    main()
