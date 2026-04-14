"""Regression test for parameter aliasing in the FakeGPU PrivateUse1 path."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn

from fakegpu.privateuse1 import init_privateuse1


class SharedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 8, bias=False)
        self.proj.weight = self.emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.emb(token_ids).mean(dim=0, keepdim=True)
        return self.proj(hidden)


def main() -> None:
    init_privateuse1()

    model = SharedNet()
    assert model.proj.weight is model.emb.weight

    model = model.fgpu()

    assert model.proj.weight is model.emb.weight
    assert model.proj.weight.raw_data is model.emb.weight.raw_data

    token_ids = torch.tensor([0, 1, 2])
    loss = model(token_ids).sum()
    loss.backward()

    assert model.emb.weight.grad is model.proj.weight.grad
    assert model.emb.weight.grad.raw_data.shape == model.emb.weight.raw_data.shape

    print("privateuse1 aliasing smoke passed")


if __name__ == "__main__":
    main()
