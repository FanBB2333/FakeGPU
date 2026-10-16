# Development and testing

[Documentation index](README.md) · [简体中文](../zh-CN/development.md) ·
[繁體中文](../zh-TW/development.md)

## Work from a checkout

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

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

The catalog has 82 YAML profiles covering consumer, workstation, data-center,
and embedded NVIDIA GPUs from Maxwell through Blackwell. The Python and native
runtimes share them.

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
```

Set `FAKEGPU_PROFILE` or pass `--profile` to select one profile. Use
`--devices` for a mixed list.

## Scripts

- `scripts/build.sh` configures and compiles the native targets.
- `scripts/test.sh` runs the maintained test suites.
- `scripts/update_nvidia_gpu_catalog.py` checks or updates profile metadata.
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
