#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fakegpu  # noqa: E402
from fakegpu.profile_catalog import load_profiles  # noqa: E402


def main() -> None:
    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
    catalog = load_profiles()
    profile_matrix = tuple(sorted(catalog))
    runtime = fakegpu.init(runtime="fakecuda", devices=profile_matrix)

    import torch

    assert torch.cuda.device_count() == len(profile_matrix)
    for index, profile_id in enumerate(profile_matrix):
        expected = catalog[profile_id]
        assert torch.cuda.get_device_name(index) == expected.torch_name
        assert torch.cuda.get_device_capability(index) == expected.compute_capability
        properties = torch.cuda.get_device_properties(index)
        assert int(properties.total_memory) == expected.memory_bytes
        assert int(properties.major) == expected.compute_major
        assert int(properties.minor) == expected.compute_minor

        tensor = torch.arange(4, device=f"cuda:{index}", dtype=torch.float32)
        result = tensor.square().add(1)
        assert result.device.type == "cuda"
        assert result.device.index == index
        assert result.is_cuda is True
        assert result.cpu().tolist() == [1.0, 2.0, 5.0, 10.0]

    capabilities = sorted(
        {catalog[profile_id].compute_capability_text for profile_id in profile_matrix},
        key=lambda value: tuple(int(part) for part in value.split(".")),
    )
    print(
        f"OK: fakecuda backend={runtime.backend} validated "
        f"{len(profile_matrix)} profiles across {len(capabilities)} compute capabilities"
    )


if __name__ == "__main__":
    main()
