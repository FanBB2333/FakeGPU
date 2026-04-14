# Phase 2 自定义 Torch 路线

这份文档说明当前 Phase 2 路线的状态：基于自定义 PyTorch 构建，在没有真实 CUDA 的 macOS / Linux 主机上暴露 CUDA 语义，并与 FakeGPU 配合使用。

## 目标

Phase 2 面向的是那些不能停留在 `fgpu` 这类 `PrivateUse1` 设备名上的场景。

目标是做本地调试和 smoke 验证层面的兼容，而不是做原生 CUDA 后端：

- `tensor.device.type == "cuda"`
- `tensor.is_cuda is True`
- `module.cuda()` / `module.to("cuda")`
- 常见 `torch.cuda.*` 控制流
- 足够支撑训练脚本的 `torch.distributed` / `DataParallel` / checkpoint 恢复

底层执行仍然是 CPU。

## 当前架构

- 源码基线：上游 `pytorch/pytorch` `v2.11.0`
- 分支仓库：本地 `pytorch-fakegpu`
- 集成入口：`torch.fakegpu.enable()`
- 桥接入口：`fakegpu.torch_patch.patch()` 在检测到自定义 torch 时会优先走这条路径

实现刻意放在 Python 层：

- 包装张量，暴露 fake CUDA 可见属性
- monkeypatch 张量、模块、device factory 入口
- `torch.load(..., map_location=...)` 先按 CPU 反序列化，再递归改写成 fake CUDA tensor
- 对部分 `torch.cuda`、`torch.nn.parallel`、`torch.distributed` 接口做单进程 shim

## 当前支持面

### CUDA 语义张量与模块

- `torch.device("cuda")`、`torch.device("cuda:N")`
- 使用 `device="cuda"` / `device="cuda:N"` 创建张量
- `.cuda()`、`.to("cuda")`、`.cpu()`
- `tensor.device`、`tensor.is_cuda`
- `module.cuda()`、`module.to("cuda")`
- `torch.cuda.FloatTensor` 这一类 legacy tensor factory

### CUDA 管理接口

- `torch.cuda.is_available()`、`device_count()`、`current_device()`、`set_device()`
- `get_device_name()`、`get_device_capability()`、`get_device_properties()`
- `Stream`、`Event`、`stream(...)`、`current_stream()`、`default_stream()`、`set_stream()`、`device_of(...)`
- `manual_seed()`、`manual_seed_all()`、`seed()`、`seed_all()`、`initial_seed()`
- `get_rng_state()`、`get_rng_state_all()`、`set_rng_state()`、`set_rng_state_all()`
- `memory_allocated()`、`memory_reserved()`、`mem_get_info()`
- `memory_stats()`、`memory_summary()`、`memory_snapshot()`
- `torch.cuda.memory` / `torch.cuda.random` 子模块中的对应别名

### Parallel / Distributed shim

- `torch.nn.DataParallel`
- `torch.nn.parallel.DistributedDataParallel`
- `torch.nn.parallel.comm.broadcast`、`scatter`、`gather`、`reduce_add`、`reduce_add_coalesced`
- 单进程语义兼容的 `torch.distributed`：
  - `init_process_group`、`destroy_process_group`
  - `barrier`
  - `all_reduce`、`broadcast`
  - `all_gather`、`all_gather_into_tensor`、`all_gather_object`
  - `reduce`、`gather`、`scatter`
  - `reduce_scatter`、`reduce_scatter_tensor`
  - `all_to_all`、`all_to_all_single`
  - `broadcast_object_list`
  - 私有别名 `_all_gather_base`、`_reduce_scatter_base`

### Checkpoint 与训练兼容

- `torch.save(...)`
- `torch.load(..., map_location="cuda:N")`
- `torch.load(..., map_location=torch.device("cuda:N"))`
- `torch.load(..., map_location={"cpu": "cuda:N"})`
- `torch.load(..., map_location={torch.device("cpu"): torch.device("cuda:N")})`
- 递归保持 `OrderedDict`、list、tuple 和共享 tensor alias
- model / optimizer / scheduler / AMP scaler / CUDA RNG state 的 checkpoint 恢复
- `torch.amp.autocast(device_type="cuda")`
- `torch.amp.GradScaler("cuda")`

## 已知限制

- 所有计算仍然由 CPU 执行。这条路线追求兼容，不追求性能。
- CUDA 显存统计是 stub，内存统计值固定为 0 或固定假总量。
- stream / event 只有 API 语义，没有真实异步执行。
- distributed 只提供单进程语义兼容，不提供真实多 rank 通信。
- `torch.load(..., map_location=<callable>)` 仍然保留上游 storage callback 语义；目前不支持把 callable 返回值翻译成 fake-CUDA 目标设备。
- 这条路线不能让 CUDA 扩展、自定义 kernel、真实 storage allocator 在 CPU-only 构建上工作。

## 当前维护的验证基线

FakeGPU 仓库里当前维护的 Phase 2 smoke 测试包括：

- `test/test_phase2_custom_torch_smoke.py`
- `test/test_phase2_cuda_api_surface.py`
- `test/test_phase2_parallel_api_surface.py`
- `test/test_phase2_distributed_api_surface.py`
- `test/test_phase2_checkpoint_state_surface.py`
- `test/test_phase2_torch_load_map_location_surface.py`
- `test/test_phase2_patch_bridge.py`
- `test/test_torch_patch.py`
- `test/test_torch_training.py`

## 推荐使用方式

### 1. 安装自定义 torch wheel

先在本地 `pytorch-fakegpu` 仓库构建 wheel，再装到 FakeGPU 使用的同一套 Python 环境里。

### 2. 选择一种激活方式

直接跑自定义 torch 测试：

```bash
TORCH_FAKEGPU_ENABLE=1 python test/test_phase2_custom_torch_smoke.py
```

已有脚本如果已经用了 FakeGPU patch：

```python
from fakegpu.torch_patch import patch
patch()
```

桥接层会尽量保持旧脚本不改，同时优先启用已安装的自定义 torch fake-CUDA 后端。

## 什么情况下不该继续扩 Phase 2

如果你的目标是下面这些场景，Phase 2 目前已经够用：

- 本地脚本 bring-up
- 在 CPU-only 环境里调试 CUDA 风格训练代码
- checkpoint / optimizer 恢复链路的 smoke 验证
- 需要 `torch.cuda` 和基础 `torch.distributed` 存在感的单进程兼容

如果下一个需求已经变成真实 CUDA 执行、真实 allocator 行为、或者真实分布式通信，那就不应该继续在这层往前堆实现。
