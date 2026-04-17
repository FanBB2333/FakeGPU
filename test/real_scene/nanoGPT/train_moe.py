"""
MoE-GPT training script with Expert Parallelism support.

Usage:
  Single GPU:  python train_moe.py
  Multi-GPU:   torchrun --nproc_per_node=N train_moe.py
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from moe_model import MoEGPT, MoEGPTConfig


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def _load_config_file(args: argparse.Namespace) -> None:
    if not args.config:
        return
    config_path = Path(args.config)
    if not config_path.exists():
        return
    config_globals: dict[str, object] = {}
    exec(config_path.read_text(encoding="utf-8"), config_globals)
    for key, value in config_globals.items():
        if hasattr(args, key):
            setattr(args, key, value)


def _load_training_data() -> tuple[torch.Tensor, int]:
    data_dir = Path(__file__).parent / "data" / "shakespeare_char"
    train_data_path = data_dir / "train.bin"
    if train_data_path.exists():
        import numpy as np

        data = torch.from_numpy(np.memmap(str(train_data_path), dtype=np.uint16, mode="r").astype(int))
        vocab_size = int(data.max()) + 1
        vocab_size = ((vocab_size + 63) // 64) * 64
        return data, vocab_size

    print("[WARN] No training data found, generating random tokens")
    vocab_size = 256
    return torch.randint(0, vocab_size, (10_000,)), vocab_size


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE-GPT training for FakeGPU test")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--max-iters", type=int, default=50, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--num-experts-per-tok", type=int, default=2, help="Top-k experts per token")
    parser.add_argument("--expert-parallel", action="store_true", help="Enable Expert Parallelism")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()
    _load_config_file(args)

    ddp = int(os.environ.get("RANK", "-1")) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master_process = True

    device_type = "cuda" if "cuda" in device else "cpu"
    if args.expert_parallel and ddp:
        assert args.num_experts % world_size == 0

    if master_process:
        print("=" * 60)
        print("MoE-GPT Training")
        print(f"  World size: {world_size}")
        print(f"  Device: {device}")
        print(f"  Layers: {args.n_layer}, Heads: {args.n_head}, Embd: {args.n_embd}")
        print(f"  Experts: {args.num_experts}, Top-k: {args.num_experts_per_tok}")
        print(f"  Expert Parallel: {args.expert_parallel}")
        print(f"  Batch size: {args.batch_size}, Block size: {args.block_size}")
        print(f"  Max iters: {args.max_iters}")
        print(f"  Dtype: {args.dtype}")
        print("=" * 60)

    data, vocab_size = _load_training_data()
    config = MoEGPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        expert_parallel=args.expert_parallel and ddp,
    )
    model = MoEGPT(config).to(device)

    ptdtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if ddp:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])

    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate)

    if master_process and device_type == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"[Init] Model on {device}, memory allocated: {mem:.1f} MB")

    autocast_enabled = device_type == "cuda" and ptdtype != torch.float32
    start_time = time.time()
    for iteration in range(args.max_iters):
        x, y = get_batch(data, args.block_size, args.batch_size, device)

        with torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=autocast_enabled):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if master_process and (iteration % args.log_interval == 0 or iteration == args.max_iters - 1):
            elapsed = time.time() - start_time
            mem_msg = ""
            if device_type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                mem_msg = f" | mem {mem:.1f} MB"
            print(f"  iter {iteration:4d} | loss {loss.item():.4f}{mem_msg} | time {elapsed:.1f}s")

    if master_process:
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Training complete: {args.max_iters} iters in {total_time:.1f}s")
        if device_type == "cuda":
            for index in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(index) / 1024**2
                mem_peak = torch.cuda.max_memory_allocated(index) / 1024**2
                print(f"  Device {index}: allocated={mem_alloc:.1f} MB, peak={mem_peak:.1f} MB")
        print("=" * 60)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
