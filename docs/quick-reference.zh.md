# 快速参考

## 编译命令

```bash
cmake -S . -B build
cmake --build build
```

开启 FakeGPU 日志：

```bash
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build
```

关闭 CPU-backed cuBLAS / cuBLASLt：

```bash
cmake -S . -B build -DENABLE_FAKEGPU_CPU_SIMULATION=OFF
cmake --build build
```

## 常用运行命令

```bash
fakegpu doctor --list-profiles
fakegpu demo --profile l4
fakegpu demo --profile b300 --json
fakegpu workspace-profiles --json
fakegpu capabilities --source-root . --strict
fakegpu analyze-repo . --json
fakegpu estimate-roofline --profile a100 --flops 1000000000000 --memory-bytes 4000000000
fakegpu validate --manifest verification/data/validation_smoke.yaml --strict
./fgpu nvidia-smi
./fgpu python3 your_script.py
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --devices "a100:4,h100:4" python3 your_script.py
./fgpu --unsupported-api error python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

`FAKEGPU_UNSUPPORTED_API` 可设为 `allow`、`warn` 或 `error`。默认值
`warn` 会保留兼容返回值，每种已识别的 no-op API 只提示一次，并把事件写入
native 报告。`error` 会让具有错误返回值的 API 返回 CUDA 801
（`NotSupported`）。

在 Python 进程内动态启用：

```bash
python3 -c "import fakegpu; fakegpu.init(runtime='native'); import torch; print(torch.cuda.device_count())"
```

Python 级 fake-CUDA 路由：

```bash
python3 -c "import fakegpu; print(fakegpu.init(runtime='auto').runtime)"
```

最小的 fake-CUDA 训练示例：

```bash
fakegpu demo --profile a100
```

较完整的 Transformer 示例仍可使用：

```bash
python3 demo_usage.py --test transformer
python3 demo_usage.py --test transformer --quiet
```

两种方式都使用 CPU-backed fake-CUDA runtime，不需要物理 GPU。

检查指定 profile，并输出结构化诊断信息：

```bash
fakegpu doctor --profile jetson-t5000
fakegpu doctor --profile rtx-pro-5000-blackwell --json
```

## LLM 推理估算与虚拟 SMI

无需加载权重即可检查 dense 或 MoE decoder safetensors checkpoint：

```bash
fakegpu estimate-llm \
  --model-dir /models/Qwen/Qwen3-8B \
  --prompt-tokens 9 \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --target-profile a100 \
  --json build/qwen-estimate.json
```

PEFT adapter 可以重复传入 `--adapter-dir`，MoE 通信估算使用
`--expert-parallel-size`。量化 base weight 使用 safetensors payload 的精确
字节数。profile 结果是 roofline 区间，不是 kernel 实测延迟。

让 FakeCUDA 进程发布显存状态，并在另一个终端查看：

```bash
FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi python3 inference.py
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi
fakegpu nvidia-smi --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

查看器会显示主机、逻辑 GPU、profile、进程、stage、tracking confidence，
并分别列出模拟进程显存、tensor requested bytes，以及 caching allocator
reserved memory 的当前值与峰值。模拟进程显存使用 reserved bytes，再加上
可选的实测 runtime overhead。`--loop` 每次刷新
都会重新扫描状态目录；省略 `--count` 时会持续刷新，配合 `--json` 时输出
NDJSON。

CPU 执行、真实 CUDA 校准、Qwen3-8B 实测数据和适用范围参见
[LLM 推理显存与计算量估算](llm-inference-estimation.md)。

## Preflight / OOM 检查

提交 Python 训练命令前，可以先跑 fakecuda preflight：

```bash
fakegpu preflight \
  --runtime fakecuda \
  --devices a100-1g:1 \
  --stage forward \
  --report-dir preflight-a100-1g \
  --allocation-stacks \
  --strict \
  -- python3 train.py --small-config
```

runner 会写出：

- `preflight_report.json`
- `preflight_report.md`
- `preflight_stdout.log`
- `preflight_stderr.log`

建议先用 `a100-1g` 这类小显存 profile 确认 OOM 能被检测到，再换成目标 profile。轻量回归测试也可以使用 `test-512m`，它是 512 MB 的 fakecuda/native 测试 profile。runner 会为 Python 命令自动初始化 fakecuda，并给出 `C3_torch_dispatch_lifetime` 可信度。除了分阶段峰值与 allocator 统计，dispatch 跟踪还会记录算子产生的新 storage、alias 和无法检查的输出。CUDA 后端内部 workspace 仍可能被低估；如果真实 GPU 校准显示 gap 大致固定，优先用 `--memory-safety-margin <bytes>`；只有当 gap 随 workload 规模增长时，再用 `--memory-safety-factor <factor>`。

`./ftest preflight_oom` 现在包含 profile 矩阵检查：同一个 560 MB allocation 在 `test-512m` 下必须失败，在 `a100` 下必须通过。

启用 `--strict` 后，child test 出现 skip 会被记为 `FAIL_RUNTIME`，不会作为通过的 preflight。

基于计算图的训练显存估算可以使用：

```bash
./ftest static_memory_validation
```

估算器通过 `make_fx` 和 `torch.func.grad_and_value` 捕获 fake-tensor ATen 前向/反向图，合并共享 storage 的 alias，并在最后一次图使用后释放 storage。PyTorch 含 CUDA 支持时，`target_device="auto"` 会使用 fake CUDA tensor，使 Attention 等设备相关算子选择 CUDA ATen 路径，但不会分配真实 GPU 显存。默认训练步骤会保留 module output，直到 backward 和 `optimizer.step()` 结束。graph 和 optimizer 两个阶段分别计算，不会叠加并不同时存在的峰值；Adam/AdamW 还会计算常驻 moment state。eager single-tensor optimizer 会按参数迭代顺序计算临时张量：当前参数的两个中间结果可能与上一个参数的 denominator 同时存在。CUDA Flash Attention auxiliary storage 按 query shape、dtype 和 64-token sequence tile 计算。FP32 Efficient Attention backward workspace 按 batch、sequence length 和 query storage 计算，并且只与对应 ATen 节点的 live storage 相加。CUDA 主机会分别记录 forward、backward 和 optimizer 峰值，测量一次 workload 释放后的 backend 常驻分配，再用 13 个 MLP/Transformer FP32/BF16 workload 对比 `torch.cuda.max_memory_allocated()`。维护中的套件会同时检查 allocator 和 requested-byte 低估是否超过 5%。

如果要求所有 workspace 候选调用都有非外推模型，可以运行：

```bash
python3 verification/static_memory_validation.py \
  --static-only \
  --min-workspace-coverage 1 \
  --reject-extrapolated-workspaces
```

Python API 也可以调用
`require_workspace_coverage(report, minimum_fraction=1.0,
allow_extrapolated=False, require_upper_bound=True)`。workspace profile
可以声明 lower/expected/upper；未匹配调用只有在
`estimate_module_memory(..., unknown_workspace_upper_bound_bytes=N)` 显式
传入每次调用的上界后，才能得到完整峰值区间。

查看内置及自定义 backend workspace catalog：

```bash
fakegpu workspace-profiles --json
fakegpu workspace-profiles --path my-workspaces.yaml --json
```

内置 catalog 包含 RTX 3090 Ti 与 RTX PRO 5000 实测的卷积和矩阵 profile，
只匹配对应的 operator、GPU profile、Compute Capability、PyTorch/CUDA
版本、dtype 和 shape。可以设置 `FAKEGPU_WORKSPACE_PROFILE_PATHS`，也可以
通过 Python estimator 的 `workspace_profile_paths` 参数加入 JSON/YAML
catalog。

## 声明式验证矩阵

```bash
fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

矩阵参数、依赖条件、结果断言、JSON 检查和跨主机报告格式见
[声明式验证清单](validation-manifests.md)。

不同 GPU 的报告可以这样比较：

```bash
python3 verification/aggregate_static_memory_validations.py \
  reports/3090ti/static_memory_validation.json \
  reports/pro5000/static_memory_validation.json \
  --output build/static_memory_validation_bundle.json
```

storage 大小计算本身与设备无关，但捕获的 ATen 图取决于目标设备。backend 常驻显存仍取决于 GPU、PyTorch、CUDA 和算子路径。实测报告会保存 allocator allocated 与 requested 对比、分阶段峰值、profile 覆盖率、profile 总字节数和实际影响峰值的增量；聚合报告还会保存缺失/不完整覆盖观测数，以及最低 modeled 和 non-extrapolated 覆盖率。这些字段用于区分 size-class 对齐误差、缺失的逻辑 storage 和算子 workspace。即使估算字节数相同，只要 graph fingerprint 变化，仍需要检查图结构差异。

真实 GPU 校准需要先运行缩小版 workload，再按环境能力对比 passthrough 或 hybrid：

```bash
./ftest real_gpu_calibration
python3 train.py --small-config
./fgpu --mode passthrough python3 train.py --small-config
./fgpu --mode hybrid --oom-policy clamp python3 train.py --small-config
```

校准套件会写出 `build/real_gpu_calibration/calibration_real_gpu.json` 和 `.md`。当前服务器会自动选择 `rtx-pro-5000-blackwell`；缺少 CUDA、PyTorch 或匹配 profile 时，报告会记录明确的 skip 原因。套件包含 tensor、MLP、Tiny Transformer、梯度累积、梯度 checkpointing、Hugging Face tiny GPT-2 和 PEFT LoRA tiny GPT-2 workload。默认先 warmup 1 次，再测量 3 次，使用最大观测峰值作为上界，同时保留 PyTorch allocated/reserved/requested 和 NVML 显存分布。NVML 能识别当前 PID 时会记录进程峰值；WSL 无法提供 PID 映射时，会明确标记进程采样不可用并保留设备显存增量。每个 workload 都会运行 real CUDA、passthrough、Hybrid clamp 和 fakecuda；报告会校验原生模式的结果签名，并执行受控的 Hybrid clamp OOM probe。校准结果只适用于完全匹配的 GPU profile、workload 签名和相近的软件环境。

多台机器的报告可以合并为实测数据集，再交给 preflight 按 profile 选择：

```bash
python3 verification/aggregate_real_gpu_calibrations.py \
  reports/3090ti/calibration_real_gpu.json \
  reports/pro5000/calibration_real_gpu.json \
  --output build/calibration_bundle.json \
  --markdown build/calibration_bundle.md

python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile rtx3090ti \
  --memory-calibration build/calibration_bundle.json \
  --calibration-workload tiny_transformer_step \
  --report-dir preflight-report \
  -- python3 train.py
```

这种模式直接采用重复实测中的物理显存上界，不拟合通用倍率。能够取得 NVML 进程峰值时，会把 CUDA context 和后端分配一起计入；WSL 无法提供进程数据时，则取 PyTorch allocator 峰值与 NVML 设备增量中的较大值，并在报告里注明数据来源。workload 名称对应多个签名时，命令会要求改用完整签名；不同 batch、序列长度或模型配置不能直接套用。

当前设计和限制见 [AI Researcher 提交前预检查](ai-researcher-preflight.md)。

## 测试命令

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
./ftest static_memory_validation
./ftest real_gpu_calibration
./ftest all
```

```bash
./test/run_comparison.sh
python3 verification/test_coordinator_smoke.py
python3 test/test_allreduce_correctness.py
python3 verification/test_allgather_correctness.py
python3 verification/test_group_semantics.py
./test/run_multinode_sim.sh 2
./test/run_multinode_sim.sh 4
./test/run_ddp_multinode.sh 4
./test/run_hybrid_multinode.sh 2
```

这些命令覆盖当前维护中的 simulate-mode DDP 路径；它们属于 smoke / 控制流验证，不代表完整的 PyTorch/NCCL 等价。

### 错误模拟测试

```bash
python test/run_error_simulation_suite.py   # all 23 tests + HTML report
python test/test_error_cross_device.py      # E1: cross-device
python test/test_error_oom.py               # E2: OOM
python test/test_error_device_index.py      # E3: invalid device
python test/test_error_dtype_autocast.py    # E4: dtype / autocast
python test/test_error_checkpoint_load.py   # E5: checkpoint load
python test/test_error_gradient.py          # E7: gradient
```

## 手动 preload

更推荐用 `./fgpu`。如果你需要手动控制：

### Linux

```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcublas.so.12:./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1 \
python3 your_script.py
```

### macOS

```bash
DYLD_LIBRARY_PATH=./build:$DYLD_LIBRARY_PATH \
DYLD_INSERT_LIBRARIES=./build/libcublas.dylib:./build/libcudart.dylib:./build/libcuda.dylib:./build/libnvidia-ml.dylib \
python3 your_script.py
```

Python API 在不同模式下会预加载不同的库：

| 计算模式 | `fakegpu.init(runtime=\"native\")` / `fakegpu.env()` 会加载的 fake 库 |
|---|---|
| `simulate` | cuBLAS + CUDA Runtime + CUDA Driver + NVML |
| `hybrid` | CUDA Runtime + CUDA Driver + NVML |
| `passthrough` | CUDA Runtime + CUDA Driver |

## 环境变量

### 计算与 profile

| 变量 | 含义 |
|---|---|
| `FAKEGPU_MODE` | `simulate`、`hybrid`、`passthrough` |
| `FAKEGPU_OOM_POLICY` | hybrid 模式下的超配策略 |
| `FAKEGPU_PROFILE` | 所有 fake device 使用同一 preset |
| `FAKEGPU_DEVICE_COUNT` | 暴露多少个 fake device |
| `FAKEGPU_PROFILES` | 每个设备分别指定 preset，例如 `a100:4,h100:4` |
| `FAKEGPU_REAL_CUDA_LIB_DIR` | 指定真实 CUDA 库目录 |

### 分布式

| 变量 | 含义 |
|---|---|
| `FAKEGPU_DIST_MODE` | `disabled`、`simulate`、`proxy`、`passthrough` |
| `FAKEGPU_CLUSTER_CONFIG` | cluster YAML 路径 |
| `FAKEGPU_COORDINATOR_TRANSPORT` | `unix` 或 `tcp` |
| `FAKEGPU_COORDINATOR_ADDR` | socket 路径或 `host:port` |
| `FAKEGPU_CLUSTER_REPORT_PATH` | cluster 级 JSON 报告输出路径 |
| `FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH` | 可选 Markdown 报告路径；默认与 cluster JSON 位于同一目录 |
| `FAKEGPU_STAGING_CHUNK_BYTES` | staging chunk 大小 |
| `FAKEGPU_STAGING_FORCE_SOCKET` | 设为 `1` 时强制走 socket fallback |

### 报告与调试

| 变量 | 含义 |
|---|---|
| `FAKEGPU_REPORT_PATH` | `fake_gpu_report.json` 输出路径 |
| `FAKEGPU_UNSUPPORTED_API` | 已识别 native no-op 的处理方式：`allow`、`warn`（默认）或 `error` |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | 调试分配路径时常用 |
| `TORCH_SDPA_KERNEL=math` | 避开 Flash Attention 特定路径时常用 |
| `CUDA_LAUNCH_BLOCKING=1` | 让错误更早、同步地暴露出来 |

### 错误模拟

| 变量 | 含义 |
|---|---|
| `FAKEGPU_CROSS_DEVICE_CHECK` | 跨设备操作守卫；设为 `0` 可关闭 |
| `FAKEGPU_MEMORY_TRACKING` | 每设备内存跟踪与 OOM 模拟；设为 `0` 可关闭 |
| `FAKEGPU_STRICT_COMPAT` | 严格 dtype 与架构兼容性检查；设为 `0` 可关闭 |

## 故障排查

终端状态异常时：

```bash
reset
```

查看导出的 NVML 符号：

Linux:

```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

macOS:

```bash
nm -gU ./build/libnvidia-ml.dylib | rg '\\bnvml'
```

查看动态库依赖：

Linux:

```bash
ldd ./build/libcuda.so.1
ldd ./build/libcudart.so.12
ldd ./build/libcublas.so.12
ldd ./build/libnvidia-ml.so.1
```

macOS:

```bash
otool -L ./build/libcuda.dylib
otool -L ./build/libcudart.dylib
otool -L ./build/libcublas.dylib
otool -L ./build/libnvidia-ml.dylib
```

## 相关页面

- [快速开始](getting-started.md)
- [项目结构与架构](project-structure.md)
- [报告与验证](reports-and-validation.md)
- [分布式模拟使用说明](distributed-sim-usage.md)
- [错误模拟](error-simulation.md)
