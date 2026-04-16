# FakeGPU macOS 兼容性分析报告

> 本文档系统梳理了 FakeGPU 项目在 macOS 上运行所需的全部兼容性改动，覆盖构建系统、C++ 核心库、Python 封装层、脚本体系以及 macOS 平台特有限制。
>
> 更新（2026-04-14）：本文最初按“待改造项”整理。当前仓库已经完成并验证了以下关键项：macOS 构建链路、`fake_nccl` 链接修复、`fakegpu` Python 预加载/打包补齐 `libnccl`、SIP 告警、shared-memory 命名收敛、验证脚本改为 `./fgpu` 驱动、`pynvml`/示例脚本的 macOS 加载兼容。后文保留原始分析结构，作为设计与排查记录。

---

## 一、当前 macOS 适配现状总览

项目已经在以下层面做了 macOS 适配工作：

| 层面 | 适配情况 | 说明 |
|------|---------|------|
| CMake 编译器选择 | ✅ 已适配 | 在 `APPLE` 平台强制使用 AppleClang |
| 链接选项 | ✅ 已实现 | `fake_gpu`、`fake_cuda`、`fake_cudart`、`fake_cublas`、`fake_nccl` 已补齐 macOS linker flags |
| 库命名 (.dylib) | ✅ 已适配 | macOS 分支正确设置了 `OUTPUT_NAME` / `SOVERSION` |
| `librt` 链接 | ✅ 已适配 | macOS 正确跳过 `librt`（shm_open 在 libSystem 中） |
| Python `_api.py` | ✅ 已实现 | 平台检测、DYLD 变量、`libnccl` 预加载、SIP 告警均已处理 |
| 构建脚本 | ✅ 已适配 | `build_debug.sh`、`build_release.sh`、`fgpu`、`ftest` 均做了 macOS 分支判断 |
| C++ 核心 stubs | ✅ 平台无关 | `cuda_stubs.cpp`、`nvml_stubs.cpp` 等使用标准 C/C++ |
| IPC (shared memory) | ✅ POSIX 兼容 | `shm_open`/`mmap`/Unix socket 在 macOS 上均可用 |
| 验证脚本/示例 | ✅ 已实现 | 关键验证脚本、`demo_usage.py`、`test/test_cuda_direct.py` 已完成 macOS 适配并通过验证 |

**结论**：项目的核心 simulate 模式在 macOS 上**已完成基础适配并通过本机验证**。已验证项包括构建、`fgpu` 注入、NVML/Python 验证、CUDA runtime/driver 直连测试，以及主要验证脚本。后续仍可继续完善的是文档细节，以及未来 `hybrid`/`passthrough` 在 macOS 上的可选真实库探测路径。

---

## 二、需要修改的代码（按优先级排列）

### 2.1 【高优先级】CMakeLists.txt — 缺少 `-undefined dynamic_lookup`

**问题**：根目录 `CMakeLists.txt` 中，macOS 分支仅对 `fake_gpu` 和 `fake_cuda` 两个 target 设置了 `-undefined dynamic_lookup` 链接选项，但 `fake_cudart`、`fake_cublas`、`fake_nccl` 三个 target 缺少此选项。

**影响**：macOS 上这三个动态库在链接阶段可能会因未解析符号报错，导致构建失败。

**修改方案**：

```cmake
# CMakeLists.txt, macOS 分支 (if(APPLE) 块内)
if(APPLE)
    target_link_options(fake_gpu     PRIVATE "-undefined" "dynamic_lookup")
    target_link_options(fake_cuda    PRIVATE "-undefined" "dynamic_lookup")
    target_link_options(fake_cudart  PRIVATE "-undefined" "dynamic_lookup")  # 新增
    target_link_options(fake_cublas  PRIVATE "-undefined" "dynamic_lookup")  # 新增
    target_link_options(fake_nccl    PRIVATE "-undefined" "dynamic_lookup")  # 新增
    # ...
endif()
```

**涉及文件**：`CMakeLists.txt` (约第 195 行)

---

### 2.2 【高优先级】dl_intercept.cpp — `dlvsym()` 不存在于 macOS

**问题**：`src/core/dl_intercept.cpp` 使用了 `dlvsym()`，这是 glibc 特有的扩展函数，macOS 的 `dyld` 运行时不提供该符号。

**当前状态**：该文件已在 `src/core/CMakeLists.txt` 中被注释掉（`# dl_intercept.cpp`），因此**当前不影响构建**。

**修改方案（如未来需要启用）**：

```cpp
// 方案 A: 条件编译
#ifdef __APPLE__
    // macOS 不支持 dlvsym，使用 dlsym 替代
    // Apple dyld 不使用符号版本机制，dlsym 即可满足需求
    #define dlvsym(handle, symbol, version) dlsym(handle, symbol)
#endif

// 方案 B: 运行时回退
static void init_real_functions() {
    real_dlopen = (dlopen_fn)dlsym(RTLD_NEXT, "dlopen");
    real_dlsym = (dlsym_fn)dlsym(RTLD_NEXT, "dlsym");
#ifdef __linux__
    real_dlvsym = (dlvsym_fn)dlsym(RTLD_NEXT, "dlvsym");
#else
    real_dlvsym = nullptr;  // macOS 不支持
#endif
    real_dlclose = (dlclose_fn)dlsym(RTLD_NEXT, "dlclose");
}
```

同时 `should_intercept()` 函数需要添加 `.dylib` 匹配：

```cpp
static bool should_intercept(const char* filename) {
    if (!filename) return false;
    std::string name(filename);

    // Linux: .so, macOS: .dylib
    if (name.find("libnvidia-ml.so") != std::string::npos ||
        name.find("libnvidia-ml.dylib") != std::string::npos) {
        return true;
    }
    if (name.find("libcuda.so") != std::string::npos ||
        name.find("libcuda.dylib") != std::string::npos) {
        return true;
    }
    if (name.find("libcudart.so") != std::string::npos ||
        name.find("libcudart.dylib") != std::string::npos) {
        return true;
    }
    return false;
}
```

**涉及文件**：`src/core/dl_intercept.cpp`

---

### 2.3 【中优先级】backend_config.hpp — 库搜索路径仅含 Linux 路径

**问题**：`find_real_library_paths()` 中所有搜索路径和库名后缀均为 Linux 格式。

**当前影响**：在 simulate 模式下（macOS 唯一实际可用的模式），`use_real_cuda()` 返回 `false`，因此这些路径**不会被搜索**，不影响运行。

**修改方案（如未来需要支持 macOS 上的 hybrid 模式）**：

```cpp
void find_real_library_paths() {
#ifdef __APPLE__
    static const char* cuda_search_paths[] = {
        "/usr/local/cuda/lib",
        "/opt/homebrew/lib",
        "/usr/local/lib",
        nullptr
    };
    static const char* cuda_driver_candidates[] = {"libcuda.dylib", nullptr};
    static const char* cudart_candidates[] = {"libcudart.dylib", nullptr};
    static const char* cublas_candidates[] = {"libcublas.dylib", nullptr};
    static const char* nvml_candidates[] = {"libnvidia-ml.dylib", nullptr};
#else
    // 现有 Linux 路径和候选列表
    static const char* cuda_search_paths[] = {
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        // ...
    };
    // ...
#endif
    // ... 后续搜索逻辑不变
}
```

**涉及文件**：`src/core/backend_config.hpp` (约第 120–170 行)

---

### 2.4 【中优先级】real_cuda_loader.hpp — 回退库名使用 `.so` 后缀

**问题**：`load_library()` 中的回退库名（如 `"libcuda.so"`、`"libcudart.so"`）在 macOS 上找不到。

**当前影响**：与 2.3 相同，simulate 模式下不会执行此路径。

**修改方案**：

```cpp
void* load_library(const std::string& configured_path, const char* fallback_name) {
    // ... 尝试 configured_path ...

    // 平台相关的回退名
#ifdef __APPLE__
    std::string platform_fallback = std::string(fallback_name);
    // 将 .so 替换为 .dylib
    auto pos = platform_fallback.rfind(".so");
    if (pos != std::string::npos) {
        platform_fallback = platform_fallback.substr(0, pos) + ".dylib";
    }
    handle = dlopen(platform_fallback.c_str(), RTLD_NOW | RTLD_LOCAL);
#else
    handle = dlopen(fallback_name, RTLD_NOW | RTLD_LOCAL);
#endif
    return handle;
}
```

**涉及文件**：`src/core/real_cuda_loader.hpp` (约第 80–100 行)

---

### 2.5 【中优先级】nccl_passthrough.cpp — NCCL 搜索路径仅含 Linux 路径

**问题**：NCCL passthrough 模式的候选路径全部是 Linux 格式（`.so.2`）。

**当前影响**：macOS 上不存在真实 NCCL 库，passthrough 模式不可用。simulate 模式不受影响。

**修改方案（可选）**：

```cpp
#ifdef __APPLE__
static const char* nccl_candidates[] = {
    "libnccl.dylib",
    "libnccl.2.dylib",
    nullptr
};
#else
static const char* nccl_candidates[] = {
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
    "/usr/lib64/libnccl.so.2",
    "/usr/local/cuda/lib64/libnccl.so.2",
    // ...
    nullptr
};
#endif
```

**涉及文件**：`src/nccl/nccl_passthrough.cpp`

---

### 2.6 【低优先级】验证脚本硬编码 `LD_PRELOAD`

**问题**：以下验证脚本直接使用 `LD_PRELOAD` 和 `.so` 扩展名，无法在 macOS 上运行：

| 脚本 | 问题 |
|------|------|
| `verification/run_python_test.sh` | 硬编码 `LD_PRELOAD` |
| `verification/run_mode1_test.sh` | 硬编码 `LD_PRELOAD` |
| `verification/run_nvidia_smi_test.sh` | 硬编码 `libfake_gpu.so`、`libnvidia-ml.so.1`、`LD_PRELOAD` |
| `verification/run_pytorch_analysis.sh` | 硬编码 `LD_PRELOAD` |
| `verification/run_passthrough_test.sh` | 硬编码 `.so.1`/`.so.12`、`LD_PRELOAD` |

**修改方案**：每个脚本添加平台检测：

```bash
if [[ "$(uname -s)" == "Darwin" ]]; then
    LIB_EXT="dylib"
    PRELOAD_VAR="DYLD_INSERT_LIBRARIES"
    LIB_PATH_VAR="DYLD_LIBRARY_PATH"
else
    LIB_EXT="so"
    PRELOAD_VAR="LD_PRELOAD"
    LIB_PATH_VAR="LD_LIBRARY_PATH"
fi
```

或者统一改为使用 `./fgpu` wrapper（已跨平台适配）。

**涉及文件**：`verification/` 目录下的 5 个 shell 脚本

---

## 三、macOS 平台特有限制与注意事项

### 3.1 SIP（System Integrity Protection）对 `DYLD_*` 的限制

**这是 macOS 上最关键的平台限制。**

macOS 的 SIP 机制会对以下二进制文件**自动移除** `DYLD_INSERT_LIBRARIES` 和 `DYLD_LIBRARY_PATH` 环境变量：

- `/usr/bin/python3`（系统 Python）
- `/usr/bin/` 下所有系统二进制
- 任何带有 hardened runtime 或 entitlements 的签名二进制
- Apple 框架路径下的二进制文件

**解决方案**：

| 方案 | 推荐程度 | 说明 |
|------|---------|------|
| 使用 Homebrew Python | ✅ 推荐 | `brew install python` |
| 使用 conda/miniconda | ✅ 推荐 | `conda create -n fakegpu python=3.11` |
| 使用 pyenv | ✅ 推荐 | `pyenv install 3.11` |
| 禁用 SIP | ❌ 不推荐 | 会降低系统安全性 |

**建议**：在文档和错误信息中明确提醒用户必须使用非系统 Python。可在 `_api.py` 中添加检测：

```python
import sys, os

def _warn_system_python():
    if _is_macos() and sys.executable.startswith("/usr/bin/"):
        import warnings
        warnings.warn(
            "FakeGPU: System Python (/usr/bin/python3) is incompatible with "
            "DYLD_INSERT_LIBRARIES due to macOS SIP. "
            "Please use Homebrew, conda, or pyenv Python instead.",
            RuntimeWarning,
            stacklevel=2,
        )
```

### 3.2 macOS `shm_open` 名称长度限制

**问题**：macOS 对 `shm_open` 的名称限制为约 31 个字符（包括前导 `/`）。

当前命名格式：`/fakegpu-staging-r{rank}-s{staging_id}`

- `/fakegpu-staging-r0-s1` = 22 字符 ✅
- `/fakegpu-staging-r99-s99999` = 27 字符 ✅
- `/fakegpu-staging-r999-s999999` = 29 字符 ⚠️ 接近临界

**修改建议**：

```cpp
#ifdef __APPLE__
// macOS: 使用缩短的前缀
static constexpr const char* SHM_PREFIX = "/fgpu-s";  // 7 字符前缀
#else
static constexpr const char* SHM_PREFIX = "/fakegpu-staging-r";
#endif
```

**涉及文件**：`src/distributed/staging_buffer.cpp`

### 3.3 macOS 动态库加载行为差异

| 特性 | Linux | macOS |
|------|-------|-------|
| 注入环境变量 | `LD_PRELOAD` | `DYLD_INSERT_LIBRARIES` |
| 库路径环境变量 | `LD_LIBRARY_PATH` | `DYLD_LIBRARY_PATH` |
| 共享库后缀 | `.so` / `.so.N` | `.dylib` |
| 符号版本 (`dlvsym`) | ✅ 支持 | ❌ 不支持 |
| `RTLD_NEXT` | ✅ 支持 | ✅ 支持 |
| SIP 对 env vars 的过滤 | 无 | 对系统二进制生效 |
| 平坦命名空间 | 默认 | 需显式指定 (`-flat_namespace`) |
| 两级命名空间 | 不适用 | 默认行为 |

**关于两级命名空间（Two-Level Namespace）**：

macOS 默认使用两级命名空间，即每个符号绑定到特定的库。这意味着 `DYLD_INSERT_LIBRARIES` 注入的库只能拦截**未显式绑定到特定库**的符号查找。

如果目标程序（如 PyTorch）在编译时直接链接了 `libcudart.12.dylib`，那么 `DYLD_INSERT_LIBRARIES` 可能无法拦截这些调用。

**解决方案**：

1. **使用 `DYLD_LIBRARY_PATH`**（当前已实现）：将 FakeGPU 的库目录放入 `DYLD_LIBRARY_PATH`，使 dyld 优先加载我们的库。
2. **使用 `-flat_namespace` 编译 FakeGPU 库**（可选进一步保障）：

```cmake
if(APPLE)
    foreach(tgt fake_gpu fake_cuda fake_cudart fake_cublas fake_nccl)
        target_link_options(${tgt} PRIVATE "-flat_namespace")
    endforeach()
endif()
```

> **注意**：`-flat_namespace` 可能与某些第三方库冲突，需要测试。

### 3.4 Apple Silicon (ARM64) 特殊考虑

| 方面 | 状态 |
|------|------|
| `__int128` 类型 | ✅ AppleClang arm64 支持 |
| 内存对齐 | ✅ 代码使用标准 `malloc`，对齐由系统保证 |
| 指针大小 | ✅ 均为 64-bit |
| `__thread` TLS | ✅ AppleClang 支持 |
| Rosetta 2 兼容性 | ⚠️ x86_64 二进制通过 Rosetta 运行时，DYLD 行为一致 |
| 通用二进制 (Universal) | 需额外 CMake 配置 |

如果需要同时支持 Intel Mac 和 Apple Silicon，可添加通用二进制支持：

```cmake
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Build universal binary")
endif()
```

### 3.5 macOS 上的 PyTorch 兼容性

PyTorch 在 macOS 上的 CUDA 支持情况：

- PyTorch 官方**不提供** macOS CUDA wheel（Apple 不支持 CUDA）。
- `torch.cuda.is_available()` 在 macOS 上默认返回 `False`。
- PyTorch 可从源码编译 CUDA 支持，但极为少见。

**FakeGPU 的价值**：正是因为 macOS 无法运行真实 CUDA，FakeGPU 的 simulate 模式可以让开发者在 macOS 上：
1. 验证 CUDA 依赖代码的逻辑流程
2. 测试 GPU 内存管理逻辑
3. 调试分布式训练框架的通信层
4. 运行 `nvitop` 等 GPU 监控工具的 UI 开发

**拦截效果**：FakeGPU 通过 `DYLD_INSERT_LIBRARIES` + `DYLD_LIBRARY_PATH` 同时使用，使得 PyTorch（通过 `ctypes.cdll` 或 `dlopen`）加载 CUDA 库时优先找到 FakeGPU 的实现。由于 macOS PyTorch wheel 本身不含 CUDA 库，不存在冲突问题。

---

## 四、完整改动清单

### 4.1 必须改动（构建/运行可能失败）

| # | 文件 | 改动内容 | 工作量 |
|---|------|---------|--------|
| 1 | `CMakeLists.txt` | 为 `fake_cudart`、`fake_cublas`、`fake_nccl` 添加 `-undefined dynamic_lookup` | 小 |
| 2 | `CMakeLists.txt` | 为所有 target 添加 macOS backward-compat 符号链接（类似 Linux 的 `libfake_gpu.so`） | 小 |

### 4.2 建议改动（提升可靠性）

| # | 文件 | 改动内容 | 工作量 |
|---|------|---------|--------|
| 3 | `fakegpu/_api.py` | 添加系统 Python 检测警告（SIP 限制提示） | 小 |
| 4 | `src/distributed/staging_buffer.cpp` | macOS 上使用缩短的 shm 名称前缀 | 小 |
| 5 | `CMakeLists.txt` | 考虑对 macOS 添加 `-flat_namespace` 确保符号拦截完整性 | 小 |

### 4.3 可选改动（为未来 hybrid/passthrough 模式准备）

| # | 文件 | 改动内容 | 工作量 |
|---|------|---------|--------|
| 6 | `src/core/backend_config.hpp` | 添加 macOS 库搜索路径和 `.dylib` 候选名 | 中 |
| 7 | `src/core/real_cuda_loader.hpp` | 回退库名添加 `.dylib` 后缀 | 小 |
| 8 | `src/core/dl_intercept.cpp` | 用 `#ifdef` 排除 `dlvsym`，添加 `.dylib` 匹配 | 中 |
| 9 | `src/nccl/nccl_passthrough.cpp` | 添加 macOS NCCL 搜索路径 | 小 |

### 4.4 脚本与文档改动

| # | 文件 | 改动内容 | 工作量 |
|---|------|---------|--------|
| 10 | `verification/*.sh` (5 个脚本) | 添加平台检测或统一使用 `./fgpu` wrapper | 中 |
| 11 | `README.md` | 添加 macOS 安装说明、SIP 注意事项 | 小 |
| 12 | `docs/getting-started.md` | 添加 macOS 快速上手章节 | 小 |

---

## 五、推荐实施路径

```
Phase 1 — 基础构建修复（30 分钟）
├── 修复 CMakeLists.txt 缺失的 linker flags (#1, #2)
└── 验证 macOS 构建通过

Phase 2 — 运行时可靠性（1-2 小时）
├── 添加 SIP 检测警告 (#3)
├── 缩短 macOS shm 名称 (#4)
├── 评估 -flat_namespace 影响 (#5)
└── 端到端测试 simulate 模式

Phase 3 — 脚本与文档（1-2 小时）
├── 更新验证脚本 (#10)
├── 更新 README 和文档 (#11, #12)
└── 添加 macOS CI 流程

Phase 4 — 可选增强（按需）
├── 添加 macOS 库搜索路径 (#6, #7)
├── 修复 dl_intercept.cpp (#8)
└── 添加 NCCL macOS 路径 (#9)
```

---

## 六、测试验证方案

### 6.1 构建验证

```bash
# macOS 上构建
cmake -S . -B build && cmake --build build

# 检查生成的库文件
ls -la build/*.dylib
# 期望输出:
#   libnvidia-ml.1.dylib
#   libcuda.1.dylib
#   libcudart.12.dylib
#   libcublas.12.dylib
#   libnccl.2.dylib
```

### 6.2 基础功能验证

```bash
# 使用 fgpu wrapper（已跨平台适配）
./fgpu python -c "
import ctypes
print('FakeGPU libraries loaded successfully')
"

# 通过 Python API
python -c "
import fakegpu
fakegpu.init(runtime="native")
print('init OK, lib_dir:', fakegpu.library_dir())
"
```

### 6.3 PyTorch 集成验证

```bash
# 在 conda 环境中（避免 SIP）
conda activate fakegpu-test
./fgpu python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
"
```

### 6.4 SIP 限制验证

```bash
# 确认系统 Python 被正确拦截（应该失败或有警告）
/usr/bin/python3 -c "
import os
print('DYLD_INSERT_LIBRARIES:', os.environ.get('DYLD_INSERT_LIBRARIES', 'NOT SET'))
"
# 系统 Python 下 DYLD_INSERT_LIBRARIES 会被 SIP 移除
```

---

## 七、已确认无需改动的组件

以下组件经分析确认在 macOS 上**无需改动**：

- `src/core/global_state.cpp` — 纯标准 C++
- `src/core/device.cpp` — 纯标准 C
- `src/core/gpu_profile.cpp` — 纯标准 C++
- `src/cuda/cuda_stubs.cpp` — 使用 `malloc`/`free`/`memcpy`
- `src/cuda/cudart_stubs.cpp` — 标准 C++ 原子操作和互斥锁
- `src/cuda/cuda_driver_stubs.cpp` — simulate 模式路径无平台依赖
- `src/nvml/nvml_stubs.cpp` — 纯 C stub
- `src/cublas/cublas_stubs.cpp` — `__int128` 在 AppleClang 上受支持
- `src/monitor/monitor.cpp` — 标准文件 I/O
- `src/distributed/transport_local.cpp` — POSIX 兼容（Unix domain socket）
- `src/distributed/transport_tcp.cpp` — 标准 BSD socket
- `src/distributed/communicator.cpp` — 纯标准 C++
- `fakegpu/__init__.py` / `__main__.py` — 已适配 macOS
- `setup.py` / `pyproject.toml` — 已支持 darwin 平台
- `build_debug.sh` / `build_release.sh` / `fgpu` / `ftest` — 已做平台判断
