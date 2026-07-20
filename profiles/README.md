# GPU profile catalog

`profiles/*.yaml` is the shared source of truth for FakeGPU's Python and native
runtimes. CMake embeds these files into native libraries at configure time;
Python loads the same files directly, or the packaged copies under
`fakegpu/_profiles` in an installed wheel.

## Coverage

| Architecture | Compute capability | Included profiles |
|---|---|---|
| Maxwell | 5.2 | `gtx980` |
| Pascal | 6.0, 6.1 | `p100`, `p4` |
| Volta | 7.0 | `v100` |
| Turing | 7.5 | `t4` |
| Ampere | 8.0, 8.6, 8.7 | `a100`, `a100-1g`, `a30`, `a10`, `a40`, `rtx3090ti`, `jetson-agx-orin-64gb`, `test-512m` |
| Ada | 8.9 | `l4`, `l40s` |
| Hopper | 9.0 | `h100`, `h200` |
| Blackwell | 10.0, 10.3, 11.0, 12.0, 12.1 | `b100`, `b200`, `b300`, `jetson-t5000`, `rtx-pro-5000-blackwell`, `rtx-pro-6000-blackwell`, `gb10` |

The architecture mapping follows NVIDIA's
[CUDA GPU compute-capability table](https://developer.nvidia.com/cuda/gpus),
[legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy), and
[CUDA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
Blackwell spans several product families and therefore several capability
numbers: 10.0, 10.3, 11.0, 12.0, and 12.1.

## Provenance fields

Each profile records:

- `name` and `torch_name`: native-interception and Python fakecuda device names.
- `official_model`: exact model text used by NVIDIA's capability table.
- `compute_major` and `compute_minor`: the reported CUDA capability.
- `compute_capability_source`: NVIDIA page supporting that mapping.
- `spec_source`: NVIDIA product page or data sheet for physical specifications.
- `memory_kind`: `dedicated`, `unified`, or `synthetic`.
- `profile_status`: `measured`, `reference`, or `synthetic`.
- `notes`: assumptions that affect interpretation.

`measured` profiles contain attributes observed on the project's real test
hardware. `reference` profiles combine NVIDIA product specifications with
architecture-level CUDA limits when a product page does not publish every
low-level device attribute. `synthetic` profiles intentionally change a real
model's memory size for OOM tests.

Jetson and GB10 use unified system memory. Their `memory_bytes` value describes
the full shared pool and should not be treated as GPU-exclusive usable memory.

## Refresh and validation

The checked-in `fakegpu/data/nvidia_compute_capabilities.json` file is generated
from NVIDIA's current and legacy CUDA GPU tables:

```bash
python3 scripts/update_nvidia_gpu_catalog.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

The checker compares model mappings while ignoring the snapshot date. Reviewing
a refresh is still required because NVIDIA can add, rename, or remove products.

Run catalog and runtime checks with:

```bash
fakegpu doctor --list-profiles
python3 -m pytest -q test/test_cli_commands.py
./ftest smoke
```

The Python validator and C++ loader both reject a declared architecture that
does not match its compute capability. The native smoke matrix selects one
profile for each of the 15 represented capabilities and reads device
attributes through the intercepted CUDA Driver API.
