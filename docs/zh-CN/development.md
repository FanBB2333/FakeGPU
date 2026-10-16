# 开发与测试

[文档索引](README.md) · [English](../en/development.md) ·
[繁體中文](../zh-TW/development.md)

## 直接在源码目录中开发

```bash
python3 -m pip install pytest PyYAML jsonschema ruff
export PYTHONPATH="$PWD"
```

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

目录包含 306 个 YAML profile，覆盖从 Maxwell 到 Blackwell 的消费级、工作站、
数据中心和嵌入式 NVIDIA GPU。Python 与原生 runtime 共用这些 profiles。
型号、参数来源、变体口径和未知字段见[完整目录](../../profiles/README.md)。

```bash
python3 -m fakegpu doctor --list-profiles
python3 -m fakegpu demo --profile rtx4090
python3 -m fakegpu --build-dir build --devices "t4,a100:2,h100" \
  python3 your_script.py
python3 scripts/update_nvidia_gpu_catalog.py --check
python3 scripts/render_gpu_catalog.py --check
```

设置 `FAKEGPU_PROFILE` 或传入 `--profile` 可以选择一个 profile。用 `--devices`
配置混合设备列表。

## 脚本

- `scripts/build.sh` 配置并编译原生目标。
- `scripts/test.sh` 执行维护中的测试组。
- `scripts/update_nvidia_gpu_catalog.py` 检查或更新 NVIDIA compute capability 快照。
- `scripts/render_gpu_catalog.py` 从 YAML profile 重新生成公开目录与官网统计。
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
