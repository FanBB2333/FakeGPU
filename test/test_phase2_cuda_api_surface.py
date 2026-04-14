"""CUDA management API surface smoke for the custom Phase 2 torch build."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"

    torch.cuda.manual_seed(1234)
    assert torch.cuda.initial_seed() == 1234

    torch.cuda.seed()
    seed_after_seed = torch.cuda.initial_seed()
    assert isinstance(seed_after_seed, int)

    torch.cuda.seed_all()
    seed_after_seed_all = torch.cuda.initial_seed()
    assert isinstance(seed_after_seed_all, int)

    assert torch.cuda.memory_stats() == {}
    assert torch.cuda.memory_summary() == "FakeGPU: no real CUDA memory to report.\n"
    assert torch.cuda.memory_snapshot() == []
    free_mem, total_mem = torch.cuda.mem_get_info()
    assert isinstance(free_mem, int)
    assert isinstance(total_mem, int)
    assert free_mem == total_mem
    assert total_mem > 0
    assert torch.cuda.max_memory_allocated() == 0
    assert torch.cuda.max_memory_reserved() == 0
    assert torch.cuda.memory_allocated() == 0
    assert torch.cuda.memory_reserved() == 0
    assert torch.cuda.memory_cached() == 0
    assert torch.cuda.max_memory_cached() == 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()

    assert "sm_80" in torch.cuda.get_arch_list()
    assert torch.cuda.cudart() is None
    torch.cuda.ipc_collect()
    assert torch.cuda.can_device_access_peer(0, 1) is True
    assert torch.cuda.is_current_stream_capturing() is False

    current_stream = torch.cuda.current_stream()
    default_stream = torch.cuda.default_stream(device=1)
    assert current_stream.device_index == 0
    assert default_stream.device_index == 1
    with torch.cuda.stream(default_stream):
        assert torch.cuda.current_stream(device=1).device_index == 1
        stream_tensor = torch.randn(2, device="cuda:1")
    assert stream_tensor.device.index == 1
    custom_stream = torch.cuda.Stream(device=2)
    torch.cuda.set_stream(custom_stream)
    assert torch.cuda.current_stream(device=2).device_index == 2
    device_tensor = torch.randn(2, device="cuda:2")
    with torch.cuda.device_of(device_tensor):
        assert torch.cuda.current_device() == 2

    import torch.cuda.memory as cuda_memory
    import torch.cuda.random as cuda_random

    assert cuda_memory.memory_allocated() == 0
    assert cuda_memory.memory_reserved() == 0
    assert cuda_memory.memory_cached() == 0
    assert cuda_memory.max_memory_cached() == 0
    assert cuda_memory.memory_stats() == {}
    assert cuda_memory.memory_summary() == "FakeGPU: no real CUDA memory to report.\n"
    assert cuda_memory.memory_snapshot() == []
    sub_free_mem, sub_total_mem = cuda_memory.mem_get_info()
    assert (sub_free_mem, sub_total_mem) == (free_mem, total_mem)
    cuda_memory.empty_cache()
    cuda_memory.reset_peak_memory_stats()
    cuda_memory.reset_max_memory_allocated()
    cuda_memory.reset_max_memory_cached()
    cuda_memory.reset_accumulated_memory_stats()

    cuda_random.manual_seed(4321)
    assert cuda_random.initial_seed() == 4321
    state = cuda_random.get_rng_state()
    assert state.dtype == torch.uint8
    assert state.numel() > 0
    cuda_random.set_rng_state(state)
    state_dev2 = cuda_random.get_rng_state(2)
    assert state_dev2.dtype == torch.uint8
    assert state_dev2.numel() == state.numel()
    cuda_random.set_rng_state(state_dev2, 2)
    all_states = cuda_random.get_rng_state_all()
    assert len(all_states) == torch.cuda.device_count()
    cuda_random.seed()
    assert isinstance(cuda_random.initial_seed(), int)
    cuda_random.seed_all()
    assert isinstance(cuda_random.initial_seed(), int)

    print("phase2 cuda api surface passed")


if __name__ == "__main__":
    main()
