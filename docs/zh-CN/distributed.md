# 分布式规划与实验批量运行

[文档索引](README.md) · [English](../en/distributed.md) ·
[繁體中文](../zh-TW/distributed.md)

## 分布式训练规划

`plan-training` 读取 DeepSpeed、Accelerate、FSDP 和 FSDP2 配置，估算每个 rank 的
显存。

对于 FSDP2 `full_shard`，`plan-training --parameter-shapes layout.json` 接受
`{"shapes": [[12, 8], [12], [3, 12], [3]]}`，同时需要一致的 `--parameter-bytes`
总量和统一的 `--parameter-element-bytes`。它按原生方式对每个参数沿第 0 维切 chunk，
并以最大的 rank 做预算。没有提供 shape 时退回均衡切分近似。自定义 placement 不在
建模范围内。

DeepSpeed ZeRO 的 bucket 大小按元素个数计，默认值与 DeepSpeed 一致：
`reduce_bucket_size` 与 `allgather_bucket_size` 为 5e8，`stage3_prefetch_bucket_size`
为 5e7，`stage3_max_live_parameters` 与 `stage3_max_reuse_distance` 为 1e9。

- reduce bucket 占用元素个数乘以模型 dtype 的字节数，模型再小也不缩减。开启
  `contiguous_gradients`（默认）时，stage 1 和 2 在 backward 期间持有它，并在
  optimizer step 之前释放；`overlap_comm` 使其翻倍。stage 3 在整个运行期间持有
  一个 bucket。
- allgather bucket 原地写入参数，不占用显存。
- ZeRO-3 的 gathered window 约为 `stage3_max_reuse_distance` 的一半加上 prefetch
  bucket。单个 rank 不做 gather。
- 设为 `auto` 的大小在传入 `--hidden-size` 时取 HF Trainer 与 Accelerate 填入的值：
  reduce bucket 为 `hidden_size²`，prefetch bucket 为 `0.9·hidden_size²`。未传入时
  使用 DeepSpeed 默认值，并在 notes 中注明。

以上依据 DeepSpeed 0.15.3 与 0.19.8。FSDP 没有 bucket 设置，FSDP 规划仍沿用
FakeGPU 的启发式：每次 collective 取 `min(参数字节数, 500 MB)`。

FSDP `SHARD_GRAD_OP` 与 FSDP2 `reshard_after_forward=False`（`shard_grad_op`）
平时只保留每个 rank 的参数分片。规划从 forward 到 backward 结束额外计入完整的
未分片参数，并在 optimizer step 前重新分片，与 FSDP 在梯度同步步 backward 之后
的行为一致。`_HYBRID_SHARD_ZERO2`（`hybrid_shard_zero2`）在 shard group 内做同样
的事，在 group 之间复制。与 `HYBRID_SHARD` 一样，group 默认取 Accelerate 每台
机器的进程数。

各阶段描述的是稳定状态下的一次迭代：

- forward 与 backward 已经持有之前各步留下的 optimizer state 和 master 权重。
  ZeRO-3 还持有其梯度分区，梯度累积时也保留梯度。
- optimizer step 时，foreach Adam（torch 在 CUDA 上的默认实现）额外分配一个
  moment 大小的临时张量。
- DeepSpeed ZeRO 还会构建梯度分区的 FP32 副本。stage 1 和 2 会先释放 16 位梯度。

在一块 Ampere GPU（sm_80）上用 DeepSpeed 0.15.3 单 rank 训练 0.6B BF16 模型，
stage 1 到 3 的 step 峰值与预测相差不到 0.2%。这不能说明其他模型规模或 rank 数
下有同样的精度。

[版本化原生验证矩阵](../../tests/data/research_native_evidence.json)记录了六组
FP32/FP16/BF16 LoRA 与 attention 的 CPU_SIM 用例、tiny Qwen3 MoE 在 CPU 和 GPU 上
的路由与梯度，以及两个 CPU rank 上运行的原生 DDP 和 FSDP2。物理 TP/PP/CP/EP、
多 GPU ZeRO/FSDP 和多机 collective 不在当前范围内。复现入口为 `scripts/test.sh cpu`
和 `tests/native/research_distributed.py --help`。

<a id="multi-process-training-on-fakecuda"></a>

## FakeCUDA 上的多进程训练

以分布式作业方式启动的真实进程，会被 FakeCUDA 加入同一个组：`torchrun` 或
`accelerate launch` 的 worker，`mp.spawn` 或 `multiprocessing` 启动的进程，以及把
`init_method` 或 `store` 传给 `init_process_group` 的脚本。此时 `init_process_group` 会在
模拟 GPU 背后的 CPU tensor 上加入一个真实的 gloo 组，每个 rank 都能看到 NCCL
会算出的值：

- `get_rank()` 和 `get_world_size()` 返回真实的 rank 和 world size。
  `get_backend()` 仍报告请求的后端，例如 `nccl`；`get_backend(group)` 报告
  `new_group` 请求的后端。未指定后端时，`get_backend()` 与 torch 一样报告
  `undefined`。
- `all_reduce`（包括 `ReduceOp.AVG`）、`broadcast`、`all_gather`、
  `all_gather_into_tensor`、`reduce_scatter`、`reduce_scatter_tensor`、
  `reduce`、`gather`、`scatter`、`all_to_all`、`all_to_all_single`、
  `barrier`、`all_gather_object` 和 `broadcast_object_list` 交换真实的值。
  对整数 tensor 做 `ReduceOp.AVG` 时向零取整。与 NCCL 一致，对 CUDA tensor 使用
  `BAND`、`BOR`、`BXOR` 会抛出 `ValueError`，对 bool tensor 使用 `AVG` 会抛出
  `TypeError`，除非该组请求的是 gloo。由 `_make_nccl_premul_sum` 构造的
  `ReduceOp.PREMUL_SUM`（FSDP 就这样用）会先把每个 rank 的 tensor 乘以系数再求和，
  与 NCCL 一致；gloo 组仍会拒绝它。
- `DistributedDataParallel` 在构建时广播 rank 0 的参数和 buffer（设置
  `broadcast_buffers=False` 时不广播 buffer，设置 `init_sync=False` 时都不广播）。
  除非设置 `broadcast_buffers=False`，它会在第一次 forward 前、以及每次紧跟在
  `no_sync()` 之外的训练 forward 之后的 forward 前，再次广播 rank 0 的 buffer，
  与 DDP 一致。
  backward 结束时对梯度做跨 rank 平均；设置 `find_unused_parameters=True` 时，
  某个 rank 没用到的参数在该 rank 上按零计。未设置它或 `static_graph=True` 时，
  没有得到梯度的可训练参数会让下一次 forward 抛出 DDP 的 “Expected to have
  finished reduction in the prior iteration” 错误，模拟进程中也是如此。
  在 `no_sync()` 内梯度只在本地累积，与真实 DDP 一致。
- 通过 `DistributedDataParallel.register_comm_hook` 注册的 hook（例如 fp16、bf16
  或 PowerSGD hook）会拿到一个具备 `GradBucket` 方法（`buffer()`、`gradients()`、
  `parameters()`、`index()`、`is_last()` 和 `set_buffer()`）的 bucket，hook 的结果
  就是梯度。bucket 的划分与 DDP 的 reducer 一致：第一次同步的 backward（设置
  `static_graph=True` 时为第二次）之前只有一个 bucket，之后按梯度就绪顺序以
  `bucket_cap_mb` 重建；设置 `find_unused_parameters=True` 时保持最初的划分。
- `model.join()` 和 `Join([model])` 处理不均匀的输入：数据先用完的 rank 以零参与
  仍在训练的 rank 的梯度归约，`divide_by_initial_world_size`、
  `throw_on_early_termination`、`join_device` 和 `join_process_group` 的行为与 DDP
  一致。
- Accelerate 能看到每个进程、它的 rank 和 world size。

其他进程仍使用进程内模拟，此时 collective 只用本地 tensor：例如只指定 world size
的单个进程，或在集群 pod 环境里看到 `WORLD_SIZE` 和 `MASTER_ADDR` 的进程，这样它
不会去等永远不会启动的对端。对于 FakeGPU 识别不了的启动器（如 DeepSpeed 或
Lightning 的启动器），设置 `FAKEGPU_DIST_COLLECTIVES=real`；设置
`FAKEGPU_DIST_COLLECTIVES=simulate` 则让所有进程都使用模拟。模拟进程中，`gather`
用本地 tensor 填满 `gather_list` 的每个位置，`scatter` 把 `scatter_list` 中属于该模拟
rank 的那一项交给它。communication hook 在本地 bucket 上运行；由于 hook 里的
collective 只能看到本地 tensor，自带的 `allreduce_hook` 会让梯度除以 world size。

两种模式下，FakeCUDA 都像 torch 在 GPU 上那样检查设备和后端：

- 以 `nccl` 或不指定后端创建的组没有 CPU 后端，对 CPU tensor 做 collective 会抛出
  `RuntimeError: No backend type associated with device type cpu`。CPU tensor 请用
  gloo 组或 `cpu:gloo,cuda:nccl` 后端。
- 除非组使用 gloo（torch 2.14 起：除非组带有 CPU 后端），`monitored_barrier` 会抛出
  torch 的 `ValueError`。
- 在 `init_process_group` 之前和 `destroy_process_group` 之后，`get_rank()`、
  `get_world_size()` 和 `get_backend()` 会抛出 torch 的 “Default process group has
  not been initialized” `ValueError`，构建 `DistributedDataParallel` 也是如此。
- 不指定 mesh 的 `fully_shard` 以及 `FullyShardedDataParallel` 使用模拟的 CUDA
  设备，在 torch 本会选择 MPS 的 macOS 上也是如此。
  `FullyShardedDataParallel(module, device_id=torch.device("cuda", 0))` 会把 CPU 上的
  module 移到该设备。使用 `fully_shard` 时梯度计入显存，但在其 DTensor 参数上创建的
  优化器状态暂未计入，这些参数报告的设备也是 `cpu`。

跨进程的 FSDP 尚未有测试覆盖。峰值显存不包括真实 GPU 上 DDP 梯度 bucket 和 NCCL 缓冲区占用的
部分。如果想不改脚本就在模拟 GPU 上运行启动器，请使用
[`fakegpu run`](commands.md#run-a-command-on-simulated-gpus)；如果 `torchrun` 卡在
rendezvous，见[torchrun 卡在 rendezvous 时](commands.md#torchrun-hangs-at-rendezvous)。

## 拓扑、trace 与带宽

- `simulate-topology` 模拟 collective 路由和链路竞争。
- `replay-trace` 从 PyTorch Profiler、NCCL 风格或 FakeGPU trace 中汇总计算、通信、
  等待和显存时间线。
- `bandwidth` 验证模拟 TCP payload 并报告吞吐量。使用 `--listen` 且不提供
  `--cluster-config` 时，生成的拓扑按每个逻辑节点的平台（`--platform`，默认
  `auto`，见[平台与插槽](platforms.md)）给出 GPU 之间的带宽：取经过节点内所有 GPU
  的最佳环上最慢一跳的带宽，因为环形 collective 的速度取决于最慢的一跳。例如 8-GPU
  HGX A100 基板上为每方向 300 GB/s 的 NVLink（2400 Gbit/s），DGX-1 上为 50 GB/s
  的双 NVLink。
- `coordinator` 管理分布式模拟 coordinator。

分布式耗时包含 coordinator、内存复制、socket 和进程调度，不能当作 NCCL、NVLink
或 RDMA benchmark。

## 实验批量运行

有限的批量运行使用显式的 argv 数组、参数网格和重复次数。各次运行按顺序执行，并绑定
稳定的运行 ID。每次运行都会保留 stdout、退出状态、错误、产物哈希和数值指标。

```bash
python -m fakegpu plan-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-plan.json
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --json /tmp/research-results.json.gz
python -m fakegpu run-experiments --config examples/experiments/cpu-sweep.json \
  --workspace /tmp/research-sweep --resume --json /tmp/research-resumed.json
python -m fakegpu compare-experiments --workspace /tmp/research-sweep \
  --plot /tmp/research-sweep.svg --x size --metric seconds \
  --json /tmp/research-comparison.json
```

- `--resume` 跳过已核验的成功项。`--resume --retry-failed` 还会重试失败或中断的项。
- 输入、源码或环境改变后需要新的工作区。
- 命令必须在前台运行。指标路径支持嵌套 JSON。
- 绘图需要 Matplotlib。误差棒表示观测到的最小值和最大值，不是置信区间。
- 示例测量的是 CPU 工作量，不代表 GPU 吞吐。
