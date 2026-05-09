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
./fgpu nvidia-smi
./fgpu python3 your_script.py
./fgpu --profile t4 --device-count 2 python3 your_script.py
./fgpu --devices "a100:4,h100:4" python3 your_script.py
./fgpu --mode hybrid --oom-policy clamp python3 your_script.py
```

在 Python 进程内动态启用：

```bash
python3 -c "import fakegpu; fakegpu.init(runtime='native'); import torch; print(torch.cuda.device_count())"
```

Python 级 fake-CUDA 路由：

```bash
python3 -c "import fakegpu; print(fakegpu.init(runtime='auto').runtime)"
```

使用 `pytorch-fakegpu` 的 tiny Transformer 训练 demo：

```bash
python3 demo_usage.py --test transformer
python3 demo_usage.py --test transformer --quiet
```

这条路径会在 demo 内部调用 `fakegpu.torch_patch.patch()`，适合在 CPU-only
主机上做 fake-CUDA 训练 smoke 验证。

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

建议先用 `a100-1g` 这类小显存 profile 确认 OOM 能被检测到，再换成目标 profile。轻量回归测试也可以使用 `test-512m`，它是 512 MB 的 fakecuda/native 测试 profile。runner 会为 Python 命令自动初始化 fakecuda，并给出 `C2_torch_tensor_lifetime` 可信度，报告中包含分阶段峰值、top allocations、可选 allocation stack trace、粗粒度内存类别、共享 storage alias 处理和基础 logical-device 归属。autograd 保存的 activation 仍需要继续验证。

`./ftest preflight_oom` 现在包含 profile 矩阵检查：同一个 560 MB allocation 在 `test-512m` 下必须失败，在 `a100` 下必须通过。

启用 `--strict` 后，child test 出现 skip 会被记为 `FAIL_RUNTIME`，不会作为通过的 preflight。

如果要用 RTX 3090 Ti 做校准，先在真实 GPU 上跑缩小版 workload，再按环境能力对比 passthrough 或 hybrid：

```bash
./ftest rtx3090ti_calibration
python3 train.py --small-config
./fgpu --mode passthrough python3 train.py --small-config
./fgpu --mode hybrid --oom-policy clamp python3 train.py --small-config
```

校准套件会写出 `build/rtx3090ti_calibration/calibration_rtx3090ti.json` 和 `.md`。如果当前机器没有 CUDA 可见的 RTX 3090 Ti，它会写出明确 skip 原因，不会静默通过。

当前设计和限制见 [AI Researcher 提交前预检查](ai-researcher-preflight.md)。

## 测试命令

```bash
./ftest smoke
./ftest cpu_sim
./ftest python
./ftest preflight_oom
./ftest rtx3090ti_calibration
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
| `FAKEGPU_STAGING_CHUNK_BYTES` | staging chunk 大小 |
| `FAKEGPU_STAGING_FORCE_SOCKET` | 设为 `1` 时强制走 socket fallback |

### 报告与调试

| 变量 | 含义 |
|---|---|
| `FAKEGPU_REPORT_PATH` | `fake_gpu_report.json` 输出路径 |
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
