"""
Mixture-of-Experts GPT model for FakeGPU testing.

Based on the nanoGPT model structure with MoE replacing the dense MLP layer.
Supports Expert Parallelism (EP) via torch.distributed all_to_all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, GPTConfig, LayerNorm


@dataclass
class MoEGPTConfig(GPTConfig):
    """GPT config extended with MoE parameters."""

    num_experts: int = 4
    num_experts_per_tok: int = 2
    expert_parallel: bool = False
    aux_loss_weight: float = 0.01


class Router(nn.Module):
    """Top-k gating router for the MoE layer."""

    def __init__(self, n_embd: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        gate_values, expert_indices = torch.topk(probs, self.num_experts_per_tok, dim=-1)
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-9)

        num_tokens = x.shape[0]
        one_hot = F.one_hot(expert_indices.reshape(-1), self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=0)
        f = tokens_per_expert / (num_tokens * self.num_experts_per_tok + 1e-9)
        p = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * p).sum()

        return gate_values, expert_indices, aux_loss


class ExpertMLP(nn.Module):
    """Single expert MLP, mirroring the dense nanoGPT MLP."""

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with optional Expert Parallelism."""

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.router = Router(config.n_embd, config.num_experts, config.num_experts_per_tok)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_parallel = config.expert_parallel
        self.n_embd = config.n_embd
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_experts)])
        self.local_expert_ids = list(range(self.num_experts))

    def _local_dispatch(
        self,
        x: torch.Tensor,
        gate_values: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        for k in range(self.num_experts_per_tok):
            indices_k = expert_indices[:, k]
            gates_k = gate_values[:, k]
            for expert_idx, expert in zip(self.local_expert_ids, self.experts):
                mask = indices_k == expert_idx
                if not mask.any():
                    continue
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] += gates_k[mask].unsqueeze(-1) * expert_output

        return output

    def _ep_dispatch(
        self,
        x: torch.Tensor,
        gate_values: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        world_size = dist.get_world_size()
        top1_expert = expert_indices[:, 0]
        experts_per_rank = max(1, self.num_experts // world_size)
        dest_rank = torch.clamp(top1_expert // experts_per_rank, max=world_size - 1)

        send_counts = torch.zeros(world_size, dtype=torch.int64, device=x.device)
        for target_rank in range(world_size):
            send_counts[target_rank] = (dest_rank == target_rank).sum()

        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        return self._local_dispatch(x, gate_values, expert_indices)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.shape
        x_flat = x.reshape(batch_size * seq_len, channels)

        gate_values, expert_indices, aux_loss = self.router(x_flat)

        if self.expert_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            output = self._ep_dispatch(x_flat, gate_values, expert_indices)
        else:
            output = self._local_dispatch(x_flat, gate_values, expert_indices)

        return output.reshape(batch_size, seq_len, channels), aux_loss


class MoEBlock(nn.Module):
    """Transformer block with MoE replacing the dense MLP."""

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss


class MoEGPT(nn.Module):
    """GPT model with Mixture-of-Experts layers."""

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([MoEBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"MoEGPT model: {n_params / 1e6:.2f}M parameters, "
            f"{config.num_experts} experts, top-{config.num_experts_per_tok}"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        total_aux_loss = torch.zeros((), device=device)
        for block in self.transformer.h:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss + self.config.aux_loss_weight * total_aux_loss
            return logits, loss

        logits = self.lm_head(x[:, [-1], :])
        return logits, None
