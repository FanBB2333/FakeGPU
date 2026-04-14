from __future__ import annotations

import torch


class FakeGpuDeviceGuard(torch._C._acc.DeviceGuard):
    def type_(self):
        return torch._C._autograd.DeviceType.PrivateUse1
