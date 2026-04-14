"""Training smoke test for the FakeGPU PrivateUse1 backend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fakegpu.privateuse1 import init_privateuse1


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def main() -> None:
    init_privateuse1()

    device = torch.device("fgpu:0")
    model = TinyNet().fgpu()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = torch.randn(4, 3, 8, 8).to(device)
    labels = torch.randint(0, 4, (4,))

    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    assert logits.device.type == "fgpu"
    assert loss.device.type == "fgpu"

    print("privateuse1 training smoke passed")


if __name__ == "__main__":
    main()
