# GPU profile catalog

`profiles/<architecture>/<segment>/*.yaml` is the shared source of truth for
FakeGPU's Python and native runtimes. CMake embeds these files into native
libraries at configure time; Python loads the same files directly, or the same
directory hierarchy under `fakegpu/_profiles` in an installed wheel.

## Directory layout

The first directory is the NVIDIA hardware architecture. The second is the
profile segment:

- `consumer`: GeForce products.
- `datacenter`: Tesla and NVIDIA data-center accelerators.
- `workstation`: RTX PRO and desktop AI systems.
- `embedded`: Jetson modules and systems.
- `test`: profiles created specifically for deterministic tests.

`profile_status` remains independent of the profile segment. For example,
`ampere/datacenter/a100-1g.yaml` is a synthetic profile of a data-center
product, while `ampere/datacenter/a100.yaml` is a reference profile.

```text
profiles/
├── maxwell/consumer/
├── pascal/
│   ├── consumer/
│   └── datacenter/
├── volta/datacenter/
├── turing/
│   ├── consumer/
│   ├── datacenter/
│   └── workstation/
├── ampere/
│   ├── consumer/
│   ├── datacenter/
│   ├── embedded/
│   ├── test/
│   └── workstation/
├── ada/
│   ├── consumer/
│   ├── datacenter/
│   └── workstation/
├── hopper/datacenter/
└── blackwell/
    ├── consumer/
    ├── datacenter/
    ├── embedded/
    └── workstation/
```

## Coverage

The catalog contains 82 profiles. Product families are grouped below so the
table stays readable; run `fakegpu doctor --list-profiles` for every profile
ID and its exact capacity.

| Architecture | Profiles | Segment(s) | Compute capability | Product families |
|---|---:|---|---|---|
| Maxwell | 1 | Consumer | 5.2 | GeForce GTX 900 series |
| Pascal | 9 | Consumer, data center | 6.0, 6.1 | GeForce GTX 10 series, Tesla P-series |
| Volta | 1 | Data center | 7.0 | Tesla V-series |
| Turing | 12 | Consumer, data center, workstation | 7.5 | GeForce RTX 20 series, Quadro RTX, T4 |
| Ampere | 22 | Consumer, data center, workstation, embedded, test | 8.0, 8.6, 8.7 | GeForce RTX 30 series, RTX A-series, A-series accelerators, Jetson |
| Ada | 17 | Consumer, data center, workstation | 8.9 | GeForce RTX 40 series, RTX Ada Generation, L-series accelerators |
| Hopper | 2 | Data center | 9.0 | H-series accelerators |
| Blackwell | 18 | Consumer, data center, workstation, embedded | 10.0, 10.3, 11.0, 12.0, 12.1 | GeForce RTX 50 series, RTX PRO Blackwell, B-series accelerators, Jetson and GB10 |

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

Some products ship with more than one memory capacity. A built-in ID selects
one documented reference configuration, and its `notes` field names that
choice. Create a custom YAML profile when the target board uses another
capacity, clock, or power limit.

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
does not match its compute capability. The Python loader also rejects an
architecture directory that does not match the YAML declaration. The native
smoke matrix automatically traverses every catalog entry and reads its device
attributes through the intercepted CUDA Driver API. The FakeCUDA matrix does
the same for reported identity, memory, and capability, then executes a tensor
operation on each logical device.
