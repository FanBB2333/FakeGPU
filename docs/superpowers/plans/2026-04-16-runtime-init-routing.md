# Runtime Init Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a side-effect-free `fakegpu` import path and a unified `fakegpu.init(runtime="auto" | "native" | "fakecuda")` API.

**Architecture:** Keep preload initialization in `fakegpu._api`, add a lightweight runtime router in a new module, and convert heavy top-level exports into lazy wrappers. Extend the standalone fake-CUDA patch so checkpoint RNG save/restore works without `torch.fakegpu`.

**Tech Stack:** Python 3.11, PyTorch 2.x, FakeGPU preload runtime, lazy imports, TDD.

---

### Task 1: Lock Import And Runtime Routing Behavior

**Files:**
- Create: `test/test_runtime_init.py`
- Create: `fakegpu/_runtime.py`
- Modify: `fakegpu/__init__.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run the test to verify the current import side effect fails**
- [ ] **Step 3: Add the runtime router and lazy top-level wrappers**
- [ ] **Step 4: Run the test to verify routing/import behavior passes**

### Task 2: Add Standalone Fake-CUDA RNG Coverage

**Files:**
- Create: `test/test_torch_patch_standalone_runtime.py`
- Modify: `fakegpu/torch_patch.py`

- [ ] **Step 1: Write the failing standalone RNG/checkpoint test**
- [ ] **Step 2: Run the test to verify it fails on RNG state handling**
- [ ] **Step 3: Implement minimal standalone RNG/default-generator support**
- [ ] **Step 4: Run the test to verify it passes**

### Task 3: Update Callers And Docs To Use Explicit Runtime Choices

**Files:**
- Modify: `demo_usage.py`
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/getting-started.zh.md`
- Modify: `docs/quick-reference.md`
- Modify: `docs/quick-reference.zh.md`

- [ ] **Step 1: Update maintained examples that require preload semantics to call `runtime=\"native\"` explicitly**
- [ ] **Step 2: Document `runtime=\"auto\"`, `runtime=\"native\"`, and `runtime=\"fakecuda\"`**
- [ ] **Step 3: Re-run the targeted docs/examples verification commands**

### Task 4: Verify The Supported Paths

**Files:**
- Modify: `test/test_torch_training.py`

- [ ] **Step 1: Run the standalone fallback training smoke with `torch.fakegpu` blocked**
- [ ] **Step 2: Run the custom-torch bridge smoke**
- [ ] **Step 3: Run the native preload import/init smoke**
- [ ] **Step 4: Confirm the final behavior matches the design**
