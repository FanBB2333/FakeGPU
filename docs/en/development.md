# Development and testing

[Documentation index](README.md) · [简体中文](../zh-CN/development.md) ·
[繁體中文](../zh-TW/development.md)

## Work from a checkout

```bash
python3 -m pip install '.[torch,dev,monitoring]'
export PYTHONPATH="$PWD"
```

## macOS Conda environment

Install the Xcode Command Line Tools (`xcode-select --install`) if needed, then
run these commands from the checkout with a native macOS Conda installation:

```bash
conda env create -f environment.macos.yml
conda activate fakegpu
python -m fakegpu doctor
python -m fakegpu demo --profile l4
FAKEGPU_PYTHON="$CONDA_PREFIX/bin/python" scripts/test.sh all
```

If `fakegpu` already exists, use `conda env update -n fakegpu -f
environment.macos.yml` in place of `conda env create`. The environment uses
Python 3.12, CMake, native macOS PyTorch, and the development dependencies,
including Hugging Face, PEFT, datasets, and plotting integrations. Transformers
5.6.0 and Accelerate 1.11.0 match the native sampler planner's validated pair.
Installing the package compiles and bundles the macOS dynamic libraries and
coordinator; it does not need a separate Linux VM or CUDA installation.

`fakegpu.init()` selects CPU-backed FakeCUDA on macOS, so ordinary PyTorch
CUDA-device code can run for compatibility testing. Computation runs on the
CPU. The native runtime separately simulates CUDA/NVML/cuBLAS/NCCL APIs; it
does not add NVIDIA CUDA support to the macOS PyTorch build.

After changing Python or C++ sources, reinstall with
`python -m pip install '.[torch,dev,monitoring]'`. To check the installed package
independently of the checkout:

```bash
FAKEGPU_CHECKOUT="$PWD"
(
  unset PYTHONPATH FAKEGPU_BUILD_DIR FAKEGPU_LIB_DIR FAKEGPU_PROFILE_DIR
  cd "${TMPDIR:-/tmp}"
  python -I "$FAKEGPU_CHECKOUT/tests/support/installed_package_smoke.py"
)
```

Validated on 2026-10-16 with macOS arm64, Python 3.12.15, and PyTorch 2.14.1:
1,879 Python tests passed with no skips, all 306 profile checks passed, the native
smoke and CPU numerical suites passed, and both strict validation manifests
passed all 46 executions. The installed wheel also passed the isolated check
above. The optional native test requiring CUDA-enabled PyTorch is skipped on
macOS; CPU-backed FakeCUDA training is tested separately.


The installed NVML bridge was also verified outside the checkout with
`nvitop 1.7.1` and `nvidia-ml-py 13.595.45`: two publishers on two synthetic
devices, memory/process queries, allocation release and process exit cleanup,
`--once`, and the interactive TUI (including `q` to exit). See the
[monitoring guide](fakegpu-smi.md#use-pip-installed-nvitop-on-macos-or-linux).


The eight-device A100 reference monitor also passed an installed-wheel test:
real torch allocations grew from 32–60 MiB to 96–124 MiB per synthetic device,
then returned to their baseline and zero. A terminal emulator verified each
phase in the same nvitop TUI. Modeled sensor values use documented public idle
observations and CPU activity assumptions, not physical GPU measurements.

## Build

All native build behavior goes through one script:

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

Build directories and compiled artifacts are ignored by Git.

## Test

The maintained tests are grouped into four suites:

```bash
scripts/test.sh python   # Python regression tests
scripts/test.sh smoke    # native loading, reports, capabilities, SMI, coordinator
scripts/test.sh cpu      # CPU-backed cuBLAS simulation
scripts/test.sh all      # all of the above
```

Run a declarative validation manifest directly when needed:

```bash
python3 -m fakegpu validate \
  --manifest tests/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict

python3 -m fakegpu validate \
  --manifest tests/data/research_validation.yaml \
  --report-dir build/validation-research \
  --strict
```

The [release regression record](../../tests/data/research_release_evidence.json)
covers 1734 passing Python tests (one branch for CPU-only builds is skipped),
the native CPU suite, 104 packaged Python modules, 42 command help checks, and
68 offline CLI checks. All 152 historical predictions remain byte-identical.

Some tests check that the README and these documents stay in line with the
code, such as the command list and the published evidence numbers. If you
change a command or an evidence file, update the English, Simplified Chinese,
and Traditional Chinese documents together.

## GPU profiles

The catalog has 307 YAML profiles covering consumer, workstation, data-center,
and embedded NVIDIA GPUs from Maxwell through Blackwell. The Python and native
runtimes share them. The [full inventory](../../profiles/README.md) lists per-profile
specifications, sources, variant conventions, and unknown parameters.

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
python3 scripts/render_gpu_catalog.py --check
python3 scripts/derive_profile_throughput.py --check
```

Set `FAKEGPU_PROFILE` or pass `--profile` to select one profile. Use
`--devices` for a mixed list.

## Scripts

- `scripts/build.sh` configures and compiles the native targets.
- `scripts/test.sh` runs the maintained test suites.
- `scripts/update_nvidia_gpu_catalog.py` checks or updates the NVIDIA compute-capability snapshot.
- `scripts/render_gpu_catalog.py` regenerates the public inventory and website statistics from YAML profiles.
- `scripts/derive_profile_throughput.py` fills derived peak throughput from `fakegpu/data/throughput_rates.json`; run it after changing SM counts, clocks or rates.
- `scripts/validation/` holds shared report and artifact validators.
- `scripts/linux/` holds Linux GPU-management helpers.
- `scripts/macos/` holds macOS-to-Linux-VM helpers.

## Project structure

```text
FakeGPU/
├── docs/       Guides in English, Simplified Chinese, and Traditional Chinese
├── examples/   Fine-tuning, serving, and experiment configs
├── fakegpu/    Python package, CLI, runtimes, and estimators
├── profiles/   YAML GPU profile catalog
├── schemas/    JSON report and validation schemas
├── scripts/    Reusable build, test, platform, and validation tools
├── src/        Native C++ interception and coordinator implementation
└── tests/      Maintained regression tests and minimal native fixtures
```

Generated build directories, compiled libraries, test reports, caches, local
environments, binary assets, and design drafts are excluded through
`.gitignore`.
