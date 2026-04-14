from __future__ import annotations

import torch


class FakeGpuPrivateUse1Hooks(torch._C._acc.PrivateUse1Hooks):
    def is_available(self) -> bool:
        return True

    def has_primary_context(self, dev_id) -> bool:
        return True

    def is_built(self) -> bool:
        return True
