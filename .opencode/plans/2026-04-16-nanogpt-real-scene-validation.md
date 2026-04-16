# nanoGPT Real-Scene Validation Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate FakeGPU in real-world ML training scenarios (nanoGPT character-level Shakespeare) across macOS (no GPU) and Linux (real RTX 3090 Ti), covering baseline, partial-FakeGPU, full-FakeGPU, and OOM simulation.

**Architecture:** Create test runner scripts and a wrapper that initializes FakeGPU before running nanoGPT's `train.py`. Run 7 test scenarios across 2 machines, capture all output to log files, and summarize results in a single markdown report.

**Tech Stack:** Python 3, PyTorch, FakeGPU (C lib + Python), pytorch-fakegpu fork, nanoGPT, SSH (for Linux remote), conda environments.

---

## Background & Key Concepts

### Three FakeGPU Configurations

| Configuration | What is installed | How it works | Expected behavior |
|---|---|---|---|
| **Baseline (No FakeGPU)** | Standard PyTorch only | Nothing intercepted | macOS: fails (no CUDA); Linux: works (real GPU) |
| **Partial (fakegpu + native PyTorch)** | `fakegpu` Python package + standard PyTorch | `fakegpu.init()` → auto → "native" → loads C libs via ctypes | macOS: fails (CPU-only PyTorch ignores C libs); Linux: C libs intercept CUDA calls → works |
| **Full (fakegpu + pytorch-fakegpu)** | `fakegpu` package + modified PyTorch fork | `fakegpu.init()` → auto → "fakecuda" → Python-level monkeypatching | Both platforms: works (full Python-level CUDA simulation) |

### Runtime Selection Logic (`fakegpu/_runtime.py`)

```
fakegpu.init(runtime="auto")
  → Checks if `torch.fakegpu` module exists (from pytorch-fakegpu fork)
    → YES: selects "fakecuda" runtime → calls patch_torch() → full Python patching
    → NO:  selects "native" runtime  → loads C shared libs only (no Python patching)
```

### Environments

| Machine | Platform | Python Environment | Repos | GPU |
|---|---|---|---|---|
| Local | macOS (darwin) | To be determined (conda or system Python) | `/Users/l1ght/repos/FakeGPU`, `/Users/l1ght/repos/pytorch-fakegpu` | None |
| Remote | Linux (100.71.234.15) | Baseline/partial: `/home/l1ght/anaconda3/envs/fakegpu/bin/python`; full: `/home/l1ght/anaconda3/envs/fakegpu-macos/bin/python` | `~/repos/fakeGPU`, `~/repos/pytorch-fakegpu` | NVIDIA RTX 3090 Ti (24GB) |

### nanoGPT Training Configuration

All tests use character-level Shakespeare with reduced iterations for fast validation:
- Dataset: `shakespeare_char` (tiny Shakespeare, ~1MB)
- Model: Baby GPT (6 layers, 6 heads, 384 embedding dim) — ~10M parameters
- Training: **20 iterations** (reduced from 5000), `compile=False`
- Eval: every 250 iters → only step 0 eval runs in 20-iter test

### Test Scenario Summary

| # | Machine | Config | Expected Result |
|---|---|---|---|
| 1A | macOS | Baseline (no FakeGPU) | FAIL: CUDA not available |
| 1B | macOS | Partial (fakegpu + native PyTorch) | FAIL: C-level interception insufficient for CPU-only PyTorch |
| 1C | macOS | Full (fakegpu + pytorch-fakegpu) | PASS: Python-level patching enables training |
| 2 | macOS | Full + OOM (small virtual VRAM + large model) | FAIL: OOM error from FakeGPU memory tracking |
| 3A | Linux | Baseline (real GPU, no FakeGPU) | PASS: Real RTX 3090 Ti |
| 3B | Linux | Partial (fakegpu + native PyTorch) | PASS: C-level CUDA interception works |
| 3C | Linux | Full (fakegpu + pytorch-fakegpu) | PASS: Python-level patching works |

---

## File Structure

```
test/real_scene/nanoGPT/
├── (existing nanoGPT files: train.py, model.py, config/, data/, ...)
├── train_wrapper.py             # CREATE: Wrapper that inits FakeGPU before train.py
├── logs/                        # CREATE: Directory for all test log files
│   ├── macos_1a_baseline.log
│   ├── macos_1b_partial.log
│   ├── macos_1c_full.log
│   ├── macos_2_oom.log
│   ├── linux_3a_baseline.log
│   ├── linux_3b_partial.log
│   └── linux_3c_full.log
└── VALIDATION_REPORT.md         # CREATE: Final summarized results and analysis
```

---

## Task 0: Environment Discovery & Data Preparation

**Files:**
- Check: `test/real_scene/nanoGPT/data/shakespeare_char/` for existing `.bin` files
- Run: `test/real_scene/nanoGPT/data/shakespeare_char/prepare.py`

- [x] **Step 0.1: Verify macOS Python environment**

```bash
# Determine which Python/conda env has torch installed
conda env list
# Identify the correct environment and record it
# If no suitable env exists, create one:
# conda create -n fakegpu-test python=3.11 pytorch -c pytorch
```

Record the macOS Python path for later use (e.g., `/Users/l1ght/anaconda3/envs/<env>/bin/python`).

- [x] **Step 0.2: Prepare Shakespeare character-level dataset (macOS)**

```bash
cd /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT
python data/shakespeare_char/prepare.py
```

Expected output:
```
length of dataset in characters: 1,115,394
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens
```

Verify files created: `data/shakespeare_char/train.bin`, `data/shakespeare_char/val.bin`, `data/shakespeare_char/meta.pkl`.

- [x] **Step 0.3: Verify FakeGPU is built on macOS**

```bash
cd /Users/l1ght/repos/FakeGPU
ls build/*.dylib 2>/dev/null || echo "Need to build"
# If not built:
cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

Verify output libraries exist: `build/libnvidia-ml.dylib`, `build/libcudart.dylib`, `build/libcuda.dylib`, `build/libcublas.dylib`, `build/libnccl.dylib`.

- [x] **Step 0.4: Create logs directory**

```bash
mkdir -p /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT/logs
```

- [x] **Step 0.5: Verify Linux environment via SSH**

```bash
ssh 100.71.234.15 "
  echo '=== Python ==='
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python --version
  echo '=== PyTorch ==='
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python -c 'import torch; print(torch.__version__); print(\"CUDA:\", torch.cuda.is_available()); print(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")'
  echo '=== FakeGPU repo ==='
  ls ~/repos/fakeGPU/build/*.so 2>/dev/null || echo 'FakeGPU not built on Linux'
  echo '=== pytorch-fakegpu ==='
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python -c 'import torch.fakegpu; print(\"torch.fakegpu available\")' 2>/dev/null || echo 'torch.fakegpu not available'
  echo '=== fakegpu package ==='
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python -c 'import fakegpu; print(\"fakegpu version:\", fakegpu.__version__)' 2>/dev/null || echo 'fakegpu package not installed'
"
```

Record findings and adjust plan if needed (e.g., build FakeGPU on Linux if not done).

- [x] **Step 0.6: Sync code to Linux machine if needed**

```bash
# Push local changes
cd /Users/l1ght/repos/FakeGPU
git add -A && git commit -m "chore: add validation test scripts" && git push

# Pull on remote
ssh 100.71.234.15 "cd ~/repos/fakeGPU && git pull"
```

- [x] **Step 0.7: Prepare Shakespeare dataset on Linux**

```bash
ssh 100.71.234.15 "
  cd ~/repos/fakeGPU/test/real_scene/nanoGPT
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python data/shakespeare_char/prepare.py
  mkdir -p logs
"
```

---

## Task 1: Create the Training Wrapper Script

**Files:**
- Create: `test/real_scene/nanoGPT/train_wrapper.py`

This wrapper imports FakeGPU with different configurations and then runs nanoGPT's training code.

- [x] **Step 1.1: Create `train_wrapper.py`**

```python
#!/usr/bin/env python3
"""Wrapper to run nanoGPT train.py with different FakeGPU configurations.

Usage:
    python train_wrapper.py --mode baseline [train.py args...]
    python train_wrapper.py --mode partial [train.py args...]
    python train_wrapper.py --mode full [train.py args...]

Modes:
    baseline  - No FakeGPU, run nanoGPT as-is
    partial   - fakegpu.init() with native runtime (C-level interception only)
    full      - fakegpu.init() with auto detection (fakecuda if pytorch-fakegpu installed)
"""

import argparse
import os
import sys
import time
import traceback


def setup_fakegpu(mode: str) -> dict:
    """Initialize FakeGPU based on mode. Returns info dict."""
    info = {"mode": mode, "fakegpu_runtime": None, "torch_cuda_available": None}

    if mode == "baseline":
        print(f"[WRAPPER] Mode: baseline - No FakeGPU initialization")
        return info

    elif mode == "partial":
        print(f"[WRAPPER] Mode: partial - fakegpu + native PyTorch (C-level only)")
        try:
            import fakegpu
            result = fakegpu.init(runtime="native")
            info["fakegpu_runtime"] = result.runtime
            info["fakegpu_backend"] = result.backend
            print(f"[WRAPPER] fakegpu.init(runtime='native') -> runtime={result.runtime}, backend={result.backend}")
        except Exception as e:
            print(f"[WRAPPER] fakegpu.init() failed: {e}")
            info["fakegpu_init_error"] = str(e)

    elif mode == "full":
        print(f"[WRAPPER] Mode: full - fakegpu + pytorch-fakegpu (full patching)")
        try:
            import fakegpu
            result = fakegpu.init(runtime="auto")
            info["fakegpu_runtime"] = result.runtime
            info["fakegpu_backend"] = result.backend
            print(f"[WRAPPER] fakegpu.init(runtime='auto') -> runtime={result.runtime}, backend={result.backend}")
        except Exception as e:
            print(f"[WRAPPER] fakegpu.init() failed: {e}")
            info["fakegpu_init_error"] = str(e)

    return info


def main():
    # Parse our arguments (before --)
    parser = argparse.ArgumentParser(description="FakeGPU nanoGPT training wrapper")
    parser.add_argument("--mode", required=True, choices=["baseline", "partial", "full"],
                        help="FakeGPU configuration mode")
    parser.add_argument("--device-count", type=int, default=None,
                        help="Number of fake GPU devices")
    parser.add_argument("--total-memory-gb", type=float, default=None,
                        help="Total memory per fake device in GB (for OOM testing)")
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Arguments passed to train.py")

    args = parser.parse_args()
    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    # Set environment variables before import if needed
    if args.device_count is not None:
        os.environ["FAKEGPU_DEVICE_COUNT"] = str(args.device_count)
    if args.total_memory_gb is not None:
        total_bytes = int(args.total_memory_gb * 1024**3)
        os.environ["FAKEGPU_TOTAL_MEMORY"] = str(total_bytes)

    # Setup FakeGPU BEFORE importing torch
    print("=" * 70)
    print(f"[WRAPPER] nanoGPT Validation Test")
    print(f"[WRAPPER] Mode: {args.mode}")
    print(f"[WRAPPER] Device count: {args.device_count or 'default'}")
    print(f"[WRAPPER] Total memory/device: {args.total_memory_gb or 'default'} GB")
    print(f"[WRAPPER] Train args: {train_args}")
    print("=" * 70)

    info = setup_fakegpu(args.mode)

    # Now import torch and report status
    import torch
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"[WRAPPER] PyTorch version: {torch.__version__}")
    print(f"[WRAPPER] torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[WRAPPER] torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem
                print(f"[WRAPPER] Device {i}: {name}, Memory: {mem / 1024**3:.1f} GB")
            except Exception as e:
                print(f"[WRAPPER] Device {i}: error getting info: {e}")
    print("=" * 70)

    # Change to nanoGPT directory and run train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Inject train args into sys.argv so configurator.py can parse them
    sys.argv = ["train.py"] + train_args

    start_time = time.time()
    try:
        exec(open("train.py").read())
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"[WRAPPER] Training completed successfully in {elapsed:.1f}s")
        print(f"[WRAPPER] Result: PASS")
        print("=" * 70)
    except SystemExit as e:
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"[WRAPPER] Training exited with code {e.code} in {elapsed:.1f}s")
        print(f"[WRAPPER] Result: {'PASS' if e.code == 0 else 'FAIL'}")
        print("=" * 70)
    except Exception as e:
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"[WRAPPER] Training FAILED with error in {elapsed:.1f}s")
        print(f"[WRAPPER] Error type: {type(e).__name__}")
        print(f"[WRAPPER] Error message: {e}")
        print(f"[WRAPPER] Result: FAIL")
        traceback.print_exc()
        print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Task 2: macOS Test 1A — Baseline (No FakeGPU)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py`
- Output: `test/real_scene/nanoGPT/logs/macos_1a_baseline.log`

- [x] **Step 2.1: Run baseline test on macOS**

```bash
cd /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT
PYTHON=<macOS-python-path>

$PYTHON train_wrapper.py --mode baseline -- \
    config/train_shakespeare_char.py \
    --max_iters=20 \
    --log_interval=1 \
    --compile=False \
    --device=cuda \
    2>&1 | tee logs/macos_1a_baseline.log
```

**Expected:** FAIL — `torch.cuda.is_available()` returns `False`, and training with `device='cuda'` fails because macOS has no NVIDIA GPU and standard PyTorch is CPU-only. The script will fail at `model.to(device)` or `x.pin_memory().to(device)` when device='cuda' but CUDA is not available.

---

## Task 3: macOS Test 1B — Partial FakeGPU (fakegpu + native PyTorch)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py`
- Output: `test/real_scene/nanoGPT/logs/macos_1b_partial.log`

- [x] **Step 3.1: Run partial FakeGPU test on macOS**

```bash
cd /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT
PYTHON=<macOS-python-path>

$PYTHON train_wrapper.py --mode partial -- \
    config/train_shakespeare_char.py \
    --max_iters=20 \
    --log_interval=1 \
    --compile=False \
    --device=cuda \
    2>&1 | tee logs/macos_1b_partial.log
```

**Expected:** FAIL — The "native" runtime loads FakeGPU C shared libraries (.dylib) via ctypes, but macOS PyTorch is a CPU-only build. PyTorch never calls into CUDA libraries, so the C-level interception is invisible to it. `torch.cuda.is_available()` still returns `False`.

**Key observation to document:** On macOS, C-level library interception alone is insufficient. PyTorch's CPU-only build doesn't link against or call any CUDA libraries, so faking them at the C level has no effect. Full Python-level monkeypatching is required.

---

## Task 4: macOS Test 1C — Full FakeGPU (fakegpu + pytorch-fakegpu)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py`
- Output: `test/real_scene/nanoGPT/logs/macos_1c_full.log`

- [x] **Step 4.1: Ensure pytorch-fakegpu is installed/importable on macOS**

```bash
PYTHON=<macOS-python-path>

# Check if torch.fakegpu is importable
$PYTHON -c "import torch.fakegpu; print('torch.fakegpu is available')" 2>&1

# If not available, install pytorch-fakegpu in dev mode:
# cd /Users/l1ght/repos/pytorch-fakegpu
# pip install -e . -v --no-build-isolation
# (This is a full PyTorch build and takes a long time - likely already done)
```

- [x] **Step 4.2: Run full FakeGPU test on macOS**

```bash
cd /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT
PYTHON=<macOS-python-path>

$PYTHON train_wrapper.py --mode full -- \
    config/train_shakespeare_char.py \
    --max_iters=20 \
    --log_interval=1 \
    --compile=False \
    2>&1 | tee logs/macos_1c_full.log
```

Note: We do NOT pass `--device=cuda` explicitly because with full FakeGPU patching, nanoGPT's default `device = 'cuda'` should work automatically. The `torch.cuda.is_available()` check will return `True`, and all CUDA operations are redirected to CPU transparently.

**Expected:** PASS — `torch.cuda.is_available()` returns `True`, `torch.cuda.device_count()` returns 8 (default), model trains for 20 iterations with loss decreasing, output shows training progress logs.

---

## Task 5: macOS Test 2 — OOM Simulation

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py`
- Output: `test/real_scene/nanoGPT/logs/macos_2_oom.log`

- [x] **Step 5.1: Run OOM test with small virtual VRAM and large model parameters**

The strategy: Select a small-VRAM FakeGPU profile (1GB) and increase model parameters significantly (large n_layer, n_head, n_embd, batch_size) to exceed this limit.

```bash
cd /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT
PYTHON=<macOS-python-path>

# OOM test: 1GB virtual VRAM profile + large model
# GPT-2 Medium size: 24 layers, 16 heads, 1024 embd ~ 350M params ~ 1.4GB fp32
# Plus optimizer states, activations, gradients -> easily exceeds 1GB
$PYTHON train_wrapper.py --mode full --profile a100-1g -- \
    config/train_shakespeare_char.py \
    --max_iters=20 \
    --log_interval=1 \
    --compile=False \
    --n_layer=24 \
    --n_head=16 \
    --n_embd=1024 \
    --batch_size=64 \
    --block_size=512 \
    2>&1 | tee logs/macos_2_oom.log
```

**Expected:** FAIL — FakeGPU should expose a 1GB virtual device and detect that allocations exceed this limit, raising an OOM error. This demonstrates that FakeGPU can simulate small-memory GPU SKUs.

**Important note:** The OOM behavior depends on the patching approach:
1. **torch.fakegpu (from pytorch-fakegpu):** Check if it respects the selected GPU profile's memory size.
2. **torch_patch.py (from fakegpu package):** Verify patched device properties and `mem_get_info()` reflect the selected profile.
3. **Alternative approach if profile memory doesn't propagate fully:** Modify the wrapper to patch `torch.fakegpu` / `fakegpu.torch_patch` with the profile-derived memory before training.

**If OOM is not triggered with the above parameters:** Try even larger model params or smaller VRAM:
- Increase to `--n_layer=48 --n_head=24 --n_embd=2048 --batch_size=128`
- Or use an even smaller custom profile (e.g. 512MB)

---

## Task 6: Linux Test 3A — Baseline (Real GPU)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py` via SSH
- Output: `test/real_scene/nanoGPT/logs/linux_3a_baseline.log`

- [x] **Step 6.1: Sync code to Linux machine**

```bash
cd /Users/l1ght/repos/FakeGPU
git push  # assume already committed from Task 1
ssh 100.71.234.15 "cd ~/repos/fakeGPU && git pull"
```

- [x] **Step 6.2: Run baseline test on Linux (real GPU)**

```bash
ssh 100.71.234.15 "
  cd ~/repos/fakeGPU/test/real_scene/nanoGPT
  /home/l1ght/anaconda3/envs/fakegpu/bin/python train_wrapper.py --mode baseline -- \
      config/train_shakespeare_char.py \
      --max_iters=20 \
      --log_interval=1 \
      --compile=False \
      --device=cuda \
      2>&1 | tee logs/linux_3a_baseline.log
"

# Copy log back to local
scp 100.71.234.15:~/repos/fakeGPU/test/real_scene/nanoGPT/logs/linux_3a_baseline.log \
    /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT/logs/
```

**Expected:** PASS — Real RTX 3090 Ti (24GB VRAM) runs the baby Shakespeare model with no issues. `torch.cuda.is_available()` returns `True`, `torch.cuda.get_device_name(0)` shows "NVIDIA GeForce RTX 3090 Ti".

---

## Task 7: Linux Test 3B — Partial FakeGPU (fakegpu + native PyTorch)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py` via SSH
- Output: `test/real_scene/nanoGPT/logs/linux_3b_partial.log`

- [x] **Step 7.1: Run partial FakeGPU test on Linux**

```bash
ssh 100.71.234.15 "
  cd ~/repos/fakeGPU/test/real_scene/nanoGPT
  /home/l1ght/anaconda3/envs/fakegpu/bin/python train_wrapper.py --mode partial -- \
      config/train_shakespeare_char.py \
      --max_iters=20 \
      --log_interval=1 \
      --compile=False \
      --device=cuda \
      2>&1 | tee logs/linux_3b_partial.log
"

# Copy log back to local
scp 100.71.234.15:~/repos/fakeGPU/test/real_scene/nanoGPT/logs/linux_3b_partial.log \
    /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT/logs/
```

**Expected:** This is the most nuanced case. On Linux with CUDA-enabled PyTorch:
- `fakegpu.init(runtime="native")` loads the FakeGPU C shared libraries (.so) into the process via `ctypes.CDLL(RTLD_GLOBAL)`.
- If PyTorch's CUDA calls resolve to the fake libraries, CUDA will "work" with fake devices.
- However, the behavior depends on whether PyTorch has already loaded the real CUDA libraries or if the fake ones win resolution.
- Most likely: The `RTLD_GLOBAL` loading of fake libs before importing torch means the fake libraries intercept subsequent CUDA calls. `torch.cuda.is_available()` should return `True`, showing fake devices.

**Key observation to document:** On Linux, C-level interception works because the CUDA-enabled PyTorch build dynamically links against CUDA libraries. FakeGPU's C libraries, loaded first via ctypes, intercept these calls. This is fundamentally different from macOS where PyTorch is CPU-only.

---

## Task 8: Linux Test 3C — Full FakeGPU (fakegpu + pytorch-fakegpu)

**Files:**
- Run: `test/real_scene/nanoGPT/train_wrapper.py` via SSH
- Output: `test/real_scene/nanoGPT/logs/linux_3c_full.log`

- [x] **Step 8.1: Verify pytorch-fakegpu is installed on Linux**

```bash
ssh 100.71.234.15 "
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python -c 'import torch.fakegpu; print(\"OK\")'
"
```

- [x] **Step 8.2: Run full FakeGPU test on Linux**

```bash
ssh 100.71.234.15 "
  cd ~/repos/fakeGPU/test/real_scene/nanoGPT
  /home/l1ght/anaconda3/envs/fakegpu-macos/bin/python train_wrapper.py --mode full -- \
      config/train_shakespeare_char.py \
      --max_iters=20 \
      --log_interval=1 \
      --compile=False \
      2>&1 | tee logs/linux_3c_full.log
"

# Copy log back to local
scp 100.71.234.15:~/repos/fakeGPU/test/real_scene/nanoGPT/logs/linux_3c_full.log \
    /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT/logs/
```

**Expected:** PASS — Full Python-level patching intercepts all CUDA calls. `torch.cuda.is_available()` returns `True`, training completes successfully. Device names will show fake device names instead of RTX 3090 Ti.

---

## Task 9: Compile Results into VALIDATION_REPORT.md

**Files:**
- Create: `test/real_scene/nanoGPT/VALIDATION_REPORT.md`

- [x] **Step 9.1: Create the validation report**

After all tests complete, compile the results into a structured markdown report. The report should include:

```markdown
# FakeGPU Real-Scene Validation Report — nanoGPT

**Date:** YYYY-MM-DD
**Test model:** nanoGPT character-level Shakespeare (baby GPT, ~10M params)
**Training iterations:** 20

## Test Environment

### macOS
- Platform: ...
- Python: ...
- PyTorch version: ...
- GPU: None
- FakeGPU build: ...

### Linux
- Platform: ...
- Python: ...
- PyTorch version: ...
- GPU: NVIDIA RTX 3090 Ti (24GB)
- FakeGPU build: ...

## Test Results Summary

| # | Test | Platform | Config | Result | Key Output |
|---|------|----------|--------|--------|------------|
| 1A | Baseline | macOS | No FakeGPU | FAIL/PASS | ... |
| 1B | Partial | macOS | fakegpu + native PyTorch | FAIL/PASS | ... |
| 1C | Full | macOS | fakegpu + pytorch-fakegpu | FAIL/PASS | ... |
| 2 | OOM | macOS | Full + small VRAM | FAIL/PASS | ... |
| 3A | Baseline | Linux | Real GPU, no FakeGPU | FAIL/PASS | ... |
| 3B | Partial | Linux | fakegpu + native PyTorch | FAIL/PASS | ... |
| 3C | Full | Linux | fakegpu + pytorch-fakegpu | FAIL/PASS | ... |

## Detailed Results

### Test 1A: macOS Baseline (No FakeGPU)
**Command:** ...
**Result:** FAIL/PASS
**Key output:** (paste relevant log lines)
**Analysis:** (explain why it failed/passed)

(repeat for all 7 tests)

## Analysis & Conclusions

### macOS: Why C-level interception is insufficient
(Explain architectural difference between C-level and Python-level patching)

### Linux: C-level vs Python-level comparison
(Explain why both work on Linux, note device name differences)

### OOM Simulation Fidelity
(Explain how FakeGPU faithfully models memory limits)

### Partial vs Full FakeGPU Setup Comparison

| Aspect | Baseline | Partial (fakegpu only) | Full (fakegpu + pytorch-fakegpu) |
|--------|----------|------------------------|----------------------------------|
| macOS: torch.cuda.is_available() | False | False | True |
| macOS: Training status | FAIL | FAIL | PASS |
| Linux: torch.cuda.is_available() | True (real) | True (fake) | True (fake) |
| Linux: Training status | PASS | PASS | PASS |
| Linux: Device name | RTX 3090 Ti | FakeGPU device | FakeGPU device |
```

- [x] **Step 9.2: Verify all log files are present**

```bash
ls -la /Users/l1ght/repos/FakeGPU/test/real_scene/nanoGPT/logs/
# Should show all 7 log files
```

---

## Task 10: Commit Results

- [ ] **Step 10.1: Commit all test artifacts**

```bash
cd /Users/l1ght/repos/FakeGPU
git add test/real_scene/nanoGPT/train_wrapper.py
git add test/real_scene/nanoGPT/logs/
git add test/real_scene/nanoGPT/VALIDATION_REPORT.md
git commit -m "test: add nanoGPT real-scene validation tests with results"
```

---

## Troubleshooting Notes

### If `torch.fakegpu` is not available on a machine

The pytorch-fakegpu fork needs to be installed in the conda environment:
```bash
cd ~/repos/pytorch-fakegpu
pip install -e . -v --no-build-isolation
```
This is a full PyTorch build from source and can take 30+ minutes.

### If OOM test doesn't trigger OOM

1. Check if `torch.fakegpu` reads `FAKEGPU_TOTAL_MEMORY` env var. May need `TORCH_FAKEGPU_TOTAL_MEMORY`.
2. The `torch_patch.py` `_TOTAL_MEMORY` variable is set at module load time from env. Setting the env var before import should work.
3. Alternative: modify the wrapper to directly set `fakegpu.torch_patch._TOTAL_MEMORY` after import.
4. If no memory tracking exists at all, use progressively larger model sizes until system RAM OOM occurs (less ideal but still demonstrates resource limits).

### If nanoGPT train.py has import issues when exec'd

The `exec(open("train.py").read())` approach runs in the wrapper's namespace. If there are issues:
1. Switch to `subprocess.run` with the wrapper setting up env vars
2. Or use `importlib` to load train.py as a module
3. Or create a small shim: `import fakegpu; fakegpu.init(); exec(open("train.py").read())`

### If Linux SSH commands are too long

Break into a script and scp it over:
```bash
scp run_test.sh 100.71.234.15:~/repos/fakegpu/test/real_scene/nanoGPT/
ssh 100.71.234.15 "bash ~/repos/fakegpu/test/real_scene/nanoGPT/run_test.sh"
```

### macOS `compile=False` requirement

PyTorch `torch.compile()` does not work with FakeGPU's Python-level patching because the compiler tries to trace through the real CUDA backend. Always pass `--compile=False` for FakeGPU tests.

### `device='cuda'` vs default in nanoGPT

nanoGPT's `train.py` defaults to `device = 'cuda'` (line 72). When running with FakeGPU full mode, the default works. When running baseline/partial on macOS, explicitly passing `--device=cuda` makes the failure clear and expected. On Linux baseline, `--device=cuda` uses the real GPU.
