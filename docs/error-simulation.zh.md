# 错误模拟

FakeGPU 能够复现常见的真实 GPU 运行时错误，使开发者可以在没有物理 GPU 的机器上验证错误处理代码路径。E1–E5 和 E7 位于 Python `torch_patch` 层；E6 由 native NCCL shim 与分布式 coordinator 实现，需要显式启用。

## 错误类别

| 编码 | 类别 | 捕获内容 |
|------|------|----------|
| E1 | 跨设备 | 混合不同 CUDA 设备上张量的操作 |
| E2 | 显存不足 | 分配超过单设备显存限制 |
| E3 | 无效设备索引 | 引用超出已配置数量的设备 ID |
| E4 | dtype / autocast | 在不支持 bfloat16 的设备上使用 bfloat16 autocast（计算能力 < 8.0） |
| E5 | 检查点加载 | 加载为不同 GPU 架构保存的检查点 |
| E6 | 分布式 | 确定性的 collective rank 故障、持久 async error、communicator shrink 与存活 rank 恢复 |
| E7 | 梯度 | 对非叶子张量或已 detach 的张量进行梯度计算 |

## E1：跨设备操作

对张量运算（算术、matmul、`torch.cat`、`F.linear`）和模块前向传播进行检查。当操作数位于不同的 fake CUDA 设备时，抛出 `RuntimeError`，提示两个设备索引。

错误信息示例：

```
RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
a = torch.randn(3, device="cuda:0")
b = torch.randn(3, device="cuda:1")
a + b  # RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

## E2：显存不足

基于可配置 profile 的单设备显存跟踪。当分配超过设备的 `total_memory` 时，抛出 `torch.cuda.OutOfMemoryError`。

错误信息示例：

```
torch.cuda.OutOfMemoryError: CUDA out of memory on device cuda:0. Tried to allocate 37.25 GiB.
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# 默认 A100 profile：每设备 80 GB
torch.randn(100000, 100000, device="cuda:0")  # 可能超出显存限制
```

## E3：无效设备索引

在 `cudaSetDevice`、`torch.device` 及张量创建时验证设备索引。引用超出已配置设备数量的序号时抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: invalid device ordinal 99
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
torch.cuda.set_device(99)  # RuntimeError: invalid device ordinal 99
```

## E4：dtype / autocast 兼容性

在启用 bfloat16 autocast 前检查计算能力。计算能力低于 8.0 的设备（如 T4，计算能力 7.5）不支持 bfloat16，抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: bfloat16 autocast requires compute capability >= 8.0 (current device: T4, compute 7.5)
```

```python
# FAKEGPU_PROFILE=t4 python script.py
import fakegpu; fakegpu.patch_torch()
import torch
with torch.autocast("cuda", dtype=torch.bfloat16):  # 在 T4 上抛出 RuntimeError
    ...
```

## E5：检查点加载兼容性

验证检查点元数据是否与当前 fake GPU profile 匹配。检测架构不一致的情况。

错误信息示例：

```
RuntimeError: checkpoint was saved on A100 (compute 8.0) but current device is T4 (compute 7.5)
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# 在 "A100" profile 下保存，在 "T4" profile 下加载时抛出 RuntimeError
```

## E6：分布式 rank 故障与恢复

在 `FAKEGPU_DIST_MODE=simulate` 下，可以让一个全局 rank 在指定 direct
collective 提交前失败。所有参与 rank 都会收到 `ncclRemoteError`，父
communicator 上的 `ncclCommGetAsyncError` 会持续返回该错误。存活 rank
可以提供明确的排除列表并使用 `NCCL_SHRINK_ABORT` 调用
`ncclCommShrink`；子 communicator 的本地 rank 重新连续编号，报告仍保留
原始全局 rank。

维护中的四 rank 实验会让全局 rank 2 在第一次 All-Reduce 时失败，把
`[0, 1, 2, 3]` 缩减为全局 ranks `[0, 1, 3]`，随后在恢复后的
communicator 上验证第二次 All-Reduce：

```bash
FAKEGPU_MODE=simulate \
FAKEGPU_DIST_MODE=simulate \
FAKEGPU_NCCL_FAULT_RANK=2 \
FAKEGPU_NCCL_FAULT_SEQNO=1 \
FAKEGPU_NCCL_FAULT_OPERATION=all_reduce \
./build/fakegpu_nccl_direct_test --scenario fault-shrink
```

`python3 verification/test_fault_injection_recovery.py` 会校验报告 schema；
`./ftest distributed_resilience` 会执行完整的维护中异常套件。Cluster JSON
与 Markdown 报告会记录故障 rank、操作、观测到错误的 ranks、尝试传输量、
排除及存活 ranks 和恢复耗时。

## E7：梯度错误

捕获对非叶子张量访问 `grad` 以及对已 detach 张量调用 `backward()` 的情况，抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: cannot access grad of non-leaf tensor
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
x = torch.randn(3, device="cuda", requires_grad=True)
y = x * 2
y.grad  # 非叶子张量；误用时抛出 RuntimeError
```

## 环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `FAKEGPU_CROSS_DEVICE_CHECK` | `1`（启用） | 启用跨设备操作检查 (E1) |
| `FAKEGPU_MEMORY_TRACKING` | `1`（启用） | 启用单设备显存跟踪和 OOM 模拟 (E2) |
| `FAKEGPU_STRICT_COMPAT` | `1`（启用） | 启用严格的 dtype / 架构兼容性检查 (E4, E5) |
| `FAKEGPU_NCCL_FAULT_RANK` | 未设置 | direct 模拟 collective 中要失败的全局 rank（E6） |
| `FAKEGPU_NCCL_FAULT_SEQNO` | 未设置 | 触发故障的正整数 communicator 序号（E6） |
| `FAKEGPU_NCCL_FAULT_OPERATION` | `all_reduce` | collective 选择器：`all_reduce`、`reduce`、`broadcast`、`all_gather`、`reduce_scatter` 或 `all_to_all`（E6） |

前三个守卫变量可设为 `0` 来禁用对应检查。三个 E6 变量用于选择一个确定性
注入点，rank 与 seqno 必须同时设置；operation 可省略，此时使用
`all_reduce`。

## 运行错误模拟测试套件

```bash
python test/run_error_simulation_suite.py
```

这条命令执行 Python 层错误测试，并在 `test/report.html` 生成统一的 HTML 报告。报告包含分页导航，覆盖 Phase 1（设备发现）、Phase 2（训练流程）、Phase 3（MoE）和 Phase 4（错误模拟）。E6 依赖 native build，使用单独的维护入口：

```bash
python3 verification/test_fault_injection_recovery.py
./ftest distributed_resilience
```

各测试文件：

```bash
python test/test_error_cross_device.py      # E1：5 个测试
python test/test_error_oom.py               # E2：5 个测试
python test/test_error_device_index.py      # E3：4 个测试
python test/test_error_dtype_autocast.py    # E4：3 个测试
python test/test_error_checkpoint_load.py   # E5：3 个测试
python test/test_error_gradient.py          # E7：3 个测试
```

## 限制

- E1–E5 与 E7 是 Python 层检查；E6 经过 native NCCL shim 与 coordinator。
- E6 当前只覆盖 `simulate` 模式的 direct collective；尚未注入 grouped/P2P 故障，不会杀死操作系统进程、检测 heartbeat 丢失或重启训练框架。
- 当前维护的 shrink 路径要求显式提供排除列表；被排除的 rank 不能调用 `ncclCommShrink`。
- `tensor.device` 仍然显示 `cpu`——fake 设备索引在内部跟踪。
- 不支持 stream 和 event 的错误模拟。

## 相关页面

- [快速入门](getting-started.zh.md)
- [快速参考](quick-reference.zh.md)
- [报告与验证](reports-and-validation.zh.md)
