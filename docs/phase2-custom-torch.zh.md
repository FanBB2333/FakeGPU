# Torch Patch 系统

这份文档说明 FakeGPU 的 Python 层 torch patch 架构、工作原理、支持的 API 面和已知限制。

## 架构概览

FakeGPU 的 torch patch 采用两层架构：

1. **基础层** -- 内置的上游 `FakeCudaTensor` 后端（`fakegpu/_upstream.py`，源自 FanBB2333 的 `pytorch-fakegpu`）。通过 `torch.Tensor._make_subclass` + `__torch_function__` 协议实现核心 CUDA 重定向。
2. **增强层** -- FakeGPU 自有的增强功能（`fakegpu/torch_patch.py`），在上游基础上覆写和扩展：GPU profiles、内存跟踪与 OOM 模拟、autocast dtype 校验、GradScaler 透传、跨设备操作校验、终端摘要报告。

上游代码已经 vendor 到 FakeGPU 包内部，不再需要安装独立的 `pytorch-fakegpu` 自定义 PyTorch 构建。

## 工作原理（数据流）

1. `fakegpu.patch_torch()` 或 `fakegpu.init(runtime="fakecuda")` 调用 `torch_patch.py` 中的 `patch()`。
2. `patch()` 调用 `_activate_upstream(num_devices, device_name)`：
   - 优先尝试 `import torch.fakegpu`（已安装的自定义 PyTorch，如有）
   - 回退到 `fakegpu._upstream`（内置 vendor 副本，始终可用）
   - 调用 `upstream.enable()` 安装基础 FakeCudaTensor patch
3. `patch()` 再调用 `_apply_enhancements_over_upstream(upstream, torch)` 叠加 FakeGPU 增强。

## 关键机制 -- FakeCudaTensor

`FakeCudaTensor` 是基础层的核心实现：

- 使用 `torch.Tensor._make_subclass(cls, raw_data, requires_grad)` 创建子类，底层数据是 CPU tensor
- 覆写 `device` 属性 -- 返回 `torch.device(f"cuda:{device_index}")`
- 覆写 `is_cuda` 属性 -- 返回 `True`
- 使用 `__torch_function__` 协议：解包参数到 CPU -> 执行 CPU 运算 -> 重新包装结果为 `FakeCudaTensor`

这种方式解决了 `tensor.device` 和 `tensor.is_cuda` 是 C 级描述符、无法在普通 tensor 上 monkeypatch 的问题。

## FakeGPU 增强层

`_apply_enhancements_over_upstream` 在上游基础上叠加以下增强：

| 部分 | 功能 |
|---|---|
| 第 0 部分 | 设备索引越界校验 -- 替换上游宽松的 `_normalize_device_index`，使用匹配真实 CUDA 行为的 "CUDA error: invalid device ordinal" 错误 |
| 第 1 部分 | 内存跟踪器初始化，使用 GPU profile 中的每设备内存限制 |
| 第 2 部分 | Hook `upstream.wrap_tensor`，实现 tensor 创建时自动内存跟踪 |
| 第 3 部分 | 每设备 GPU profile（11 种 profile）-- 覆写 `get_device_name`, `get_device_capability`, `get_device_properties` |
| 第 4 部分 | 使用跟踪器的内存查询函数，替换上游返回零值的 stub |
| 第 5 部分 | Autocast dtype 校验（bf16 要求 compute capability >= 8.0）+ GradScaler 透传 |
| 第 6 部分 | 跨设备操作校验（tensor ops、loss functions、functional ops、binary dunders） |
| 第 7 部分 | RNG state 函数（上游未提供） |
| 第 8 部分 | 退出时的终端摘要报告 |

### 支持的 GPU profiles

共 11 种预设 profile：

| Profile | 说明 |
|---|---|
| `gtx980` | GeForce GTX 980 |
| `p100` | Tesla P100 |
| `v100` | Tesla V100 |
| `t4` | Tesla T4 |
| `a40` | NVIDIA A40 |
| `a100` | NVIDIA A100 |
| `a100-1g` | NVIDIA A100 (1g MIG) |
| `h100` | NVIDIA H100 |
| `l40s` | NVIDIA L40S |
| `b100` | NVIDIA B100 |
| `b200` | NVIDIA B200 |

## 已支持的 API 面

两层架构组合后支持的能力：

| 能力 | 状态 |
|---|---|
| `tensor.device == cuda:N` | 支持 |
| `tensor.is_cuda == True` | 支持 |
| `nn.DataParallel` | 支持 |
| `nn.DistributedDataParallel` | 支持 |
| `torch.distributed.*`（单进程 shim，覆盖所有 collective ops） | 支持 |
| Autocast / GradScaler with dtype validation | 支持 |
| GPU profiles（11 种预设） | 支持 |
| Memory tracking with OOM simulation | 支持 |
| Cross-device validation | 支持 |
| `torch.load` with `map_location` normalization | 支持 |
| Factory functions (`torch.randn`, `torch.zeros`, etc.) with `device="cuda"` | 支持 |
| Legacy tensor factories (`torch.cuda.FloatTensor`, etc.) | 支持 |
| Stream/Event API 兼容 stub | 支持 |

## 已验证 PyTorch 版本

目前唯一测试过的版本：**torch 2.9.1**。

## 配置

通过环境变量控制行为：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `FAKEGPU_DEVICE_COUNT` | Fake device 数量 | `8` |
| `FAKEGPU_PROFILE` | GPU profile 预设 | -- |
| `FAKEGPU_PROFILES` | 混合 profile（例: `a100:4,h100:4`） | -- |
| `FAKEGPU_DEVICE_NAME` | 自定义设备名 | -- |
| `FAKEGPU_STRICT_COMPAT` | 启用/禁用严格兼容校验 | `1` |

## 使用方式

### 基本用法

```python
import fakegpu
fakegpu.patch_torch()
import torch

x = torch.randn(3, 3, device="cuda")
assert x.device.type == "cuda"
assert x.is_cuda is True

model = torch.nn.Linear(3, 3).cuda()
y = model(x)
```

### 通过 init 接口

```python
import fakegpu
fakegpu.init(runtime="fakecuda")
```

## 已知限制

- 所有计算由 CPU 执行 -- 没有实际 GPU 执行。
- `__torch_function__` 开销：比直接 CPU tensor 操作慢约 2-3 倍（benchmark 测量值）。
- Stream/Event 仅为 API 兼容 stub（无真实异步）。
- Distributed 仅提供单进程语义兼容。
- CUDA 扩展、自定义 kernel、storage 级 CUDA allocator 不可用。
- 部分 PyTorch 内部路径可能绕过 `__torch_function__`（极少见）。

## 测试套件

- 12 个测试文件，共 58 个测试。
- 单独运行全部通过；同进程跨文件有隔离问题（pre-existing，由模块级 `_NUM_DEVICES` 全局状态引起）。
- 关键测试文件：`test_benchmark_overhead.py`, `test_dataloader_pin_memory.py`, `test_error_*.py`, `test_patch_advanced.py`, `test_hf_trainer.py`。

## Vendored 上游维护

- `fakegpu/_upstream.py` 是上游代码的原样副本。
- 不要直接修改 -- 在 `torch_patch.py` 中做增强。
- 更新方式是直接替换文件。
- 文件顶部保留 attribution header。
