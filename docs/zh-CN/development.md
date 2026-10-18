# 开发与测试

[文档索引](README.md) · [English](../en/development.md) ·
[繁體中文](../zh-TW/development.md)

## 直接在源码目录中开发

```bash
python3 -m pip install '.[torch,dev,monitoring]'
export PYTHONPATH="$PWD"
```

## 原生 PTX 后端

Linux 构建的 PTX 执行核心要求支持 constexpr `std::variant` 的 C++20 编译器和
标准库（例如 GCC 12+）。CMake 下载固定提交的 PantheonSim，
并校验归档 SHA-256；构建和运行解释器不需要 CUDA 或 LLVM。该 Apache-2.0 依赖
链接进现有 CUDA 库，发行包包含其许可证。离线构建可指定已解压的固定版本源码：

```bash
scripts/build.sh -- -DFETCHCONTENT_SOURCE_DIR_PANTHEONSIM=/path/to/pantheonsim
```

`-DENABLE_FAKEGPU_PTX_EXECUTION=OFF` 可禁用该依赖，PTX 加载会返回不支持。
macOS 当前使用此配置。内核执行测试要求 Linux；编译 `.cu` 测试额外需要 `nvcc`，
直接加载 PTX 的测试不需要 CUDA 工具包。

## macOS Conda 环境

如尚未安装 Xcode Command Line Tools，先运行 `xcode-select --install`。使用原生
macOS Conda，在项目根目录执行：

```bash
conda env create -f environment.macos.yml
conda activate fakegpu
python -m fakegpu doctor
python -m fakegpu demo --profile l4
FAKEGPU_PYTHON="$CONDA_PREFIX/bin/python" scripts/test.sh all
```

如果 `fakegpu` 环境已存在，将创建命令替换为
`conda env update -n fakegpu -f environment.macos.yml`。该环境包含 Python 3.12、
CMake、原生 macOS PyTorch 和开发依赖，包括 Hugging Face、PEFT、datasets 和绘图
集成。Transformers 5.6.0 与 Accelerate 1.11.0 对应原生采样规划器已验证的版本组合。
安装 Python 包时会编译并打包 macOS 动态库及 coordinator，无需 Linux VM 或 CUDA 安装。

macOS 上的 `fakegpu.init()` 默认使用 CPU 支撑的 FakeCUDA，让普通 PyTorch
CUDA 设备代码可以执行兼容性测试，实际计算发生在 CPU。原生 runtime 单独模拟
CUDA/NVML/cuBLAS/NCCL API，不会为 macOS PyTorch 增加 NVIDIA CUDA 支持。

修改 Python 或 C++ 源码后，用 `python -m pip install '.[torch,dev,monitoring]'` 重新安装。
可以运行以下检查，确认安装后的包能独立于源码目录工作：

```bash
FAKEGPU_CHECKOUT="$PWD"
(
  unset PYTHONPATH FAKEGPU_BUILD_DIR FAKEGPU_LIB_DIR FAKEGPU_PROFILE_DIR
  cd "${TMPDIR:-/tmp}"
  python -I "$FAKEGPU_CHECKOUT/tests/support/installed_package_smoke.py"
)
```

2026-10-16 已在 macOS arm64、Python 3.12.15、PyTorch 2.14.1 环境验证：
1879 项 Python 测试全部通过，无跳过；306 个 profile 检查、原生 smoke 和 CPU 数值
测试全部通过；两个严格验证清单的 46 次执行全部通过。安装后的 wheel 也通过了上述
隔离检查。依赖 CUDA 版 PyTorch 的可选原生测试在 macOS 上跳过，CPU 支撑的
FakeCUDA 训练另有实际验证。


另已在源码目录外验证安装后的 NVML 桥接，使用 `nvitop 1.7.1` 与
`nvidia-ml-py 13.595.45`：两个发布进程、两个模拟设备、显存和进程查询、释放显存
及进程退出后的清理均通过；`--once` 和交互界面也通过，按 `q` 可正常退出。
用法见 [监控文档](fakegpu-smi.md#在-macos-或-linux-上使用-pip-安装的-nvitop)。


8 卡 A100 参考监控也通过安装后验证：真实 torch 分配使每张模拟卡从 32–60 MiB
增至 96–124 MiB，再回到初始值和零。终端解析器确认同一个 nvitop 交互界面显示
了各阶段。遥测数值使用有来源的公开空闲读数和 CPU 活跃度模型，不是物理 GPU 实测。

## 构建

所有原生构建行为都由一个脚本提供：

```bash
scripts/build.sh
scripts/build.sh --release
scripts/build.sh --debug
scripts/build.sh --build-dir build-custom -- -DSOME_CMAKE_OPTION=value
```

构建目录和编译产物不会被 Git 管理。

## 测试

项目维护的测试分为四组：

```bash
scripts/test.sh python   # Python 回归测试
scripts/test.sh smoke    # 原生库加载、报告、能力检查、SMI 和 coordinator
scripts/test.sh cpu      # CPU cuBLAS 模拟
scripts/test.sh all      # 以上全部
```

需要时可以直接执行声明式验证 manifest：

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

[发布回归记录](../../tests/data/research_release_evidence.json)包含 1734 项通过的
Python 测试（跳过一个 CPU-only build 分支）、原生 CPU 套件、104 个打包的 Python
模块、42 个命令帮助检查和 68 次离线 CLI 检查。152 份历史预测全部逐字节保持一致。

有些测试会检查 README 和这些文档是否与代码保持一致，比如命令列表和已公开的证据
数字。修改命令或证据文件时，请同时更新英文、简体中文和繁体中文文档。

## GPU profiles

目录包含 307 个 YAML profile，覆盖从 Maxwell 到 Blackwell 的消费级、工作站、
数据中心和嵌入式 NVIDIA GPU。Python 与原生 runtime 共用这些 profiles。
型号、参数来源、变体口径和未知字段见[完整目录](../../profiles/README.md)。

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
python3 scripts/render_gpu_catalog.py --check
python3 scripts/derive_profile_throughput.py --check
```

设置 `FAKEGPU_PROFILE` 或传入 `--profile` 可以选择一个 profile。用 `--devices`
配置混合设备列表。

## 脚本

- `scripts/build.sh` 配置并编译原生目标。
- `scripts/test.sh` 执行维护中的测试组。
- `scripts/update_nvidia_gpu_catalog.py` 检查或更新 NVIDIA compute capability 快照。
- `scripts/render_gpu_catalog.py` 从 YAML profile 重新生成公开目录与官网统计。
- `scripts/derive_profile_throughput.py` 按 `fakegpu/data/throughput_rates.json` 填写推导的峰值算力；修改 SM 数、频率或速率表后需要重新运行。
- `scripts/validation/` 存放共用的报告和产物验证工具。
- `scripts/linux/` 存放 Linux GPU 管理工具。
- `scripts/macos/` 存放 macOS 到 Linux VM 的辅助工具。

## 项目结构

```text
FakeGPU/
├── docs/       英文、简体中文和繁体中文指南
├── examples/   微调、serving 和实验配置
├── fakegpu/    Python 包、CLI、runtime 和估算器
├── profiles/   YAML GPU profile 目录
├── schemas/    JSON 报告和验证 schema
├── scripts/    可复用的构建、测试、平台和验证工具
├── src/        原生 C++ 拦截和 coordinator 实现
└── tests/      维护中的回归测试和最小原生测试 fixture
```

生成的构建目录、编译库、测试报告、缓存、本地环境、二进制资源和设计草稿均通过
`.gitignore` 排除。
