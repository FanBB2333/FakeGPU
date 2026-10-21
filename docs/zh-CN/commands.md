# 命令参考

[文档索引](README.md) · [English](../en/commands.md) ·
[繁體中文](../zh-TW/commands.md)

所有命令都以 `python3 -m fakegpu <command>` 的形式运行。用
`python3 -m fakegpu --help` 查看完整列表，用
`python3 -m fakegpu <command> --help` 查看某个命令的选项。凡是接受 JSON 路径的
地方，大型报告和计划都可以用 `.json.gz`。压缩输出采用原子写入，字节可复现。

## 环境与运行

| 命令 | 用途 |
|---|---|
| `fakegpu run` | 在模拟 GPU 上原样运行一条命令，并报告显存是否放得下（[说明](#run-a-command-on-simulated-gpus)） |
| `fakegpu doctor` | 检查安装、原生库、PyTorch 和 profiles |
| `fakegpu demo` | 在 CPU 上执行一个小型训练步骤，程序能看到 CUDA 设备 |
| `fakegpu preflight` | 将工作负载执行到指定阶段，判断能放下还是会 OOM |
| `fakegpu capabilities` | 列出或严格检查原生 API 分类 |
| `fakegpu validate` | 执行 JSON、TOML 或 YAML 声明式验证矩阵 |

## 分析与估算

| 命令 | 用途 |
|---|---|
| `fakegpu analyze-repo` | 统计仓库入口和仅支持 GPU 的风险 |
| `fakegpu analyze-kernel` | 检查 CUDA、PTX 和 SASS 资源与运算 |
| `fakegpu estimate-llm` | 估算 decoder 显存、通信量和 FLOP |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 和 VAE decode 阶段显存 |
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu workspace-profiles` | 验证并查看 workspace 估算 profiles |

## Serving

| 命令 | 用途 |
|---|---|
| `fakegpu plan-serving` | 规划同构或混合请求准入、chunked prefill 和分组共享 prefix KV 存储 |
| `fakegpu benchmark-serving` | 按计划发送混合流式请求，保留用量、延迟、SSE 事件和失败记录 |

## 微调

| 命令 | 用途 |
|---|---|
| `fakegpu estimate-finetune` | 根据框架配置估算单卡全量/LoRA/QLoRA 的步数、时间与显存 |
| `fakegpu probe-finetune` | 在真实 CUDA 显卡上采集固定形状 Hugging Face Trainer 的时间与显存 |
| `fakegpu profile-finetune-data` | 扫描本地 JSONL 的 token 长度、截断、padding 与处理后序列数 |
| `fakegpu run-finetune` | 执行 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl 原生训练，记录数据、CUDA 时间、显存及最终保存 |
| `fakegpu validate-finetune-windows` | 用独立的校准与 holdout 运行验证动态窗口耗时 |
| `fakegpu validate-finetune-events` | 用独立的原生事件记录验证评估与 checkpoint 成本 |
| `fakegpu plan-finetune-pipeline` | 从已校准的顺序数据推导未来原生训练轮次或更新数 |
| `fakegpu estimate-finetune-pipeline` | 估算完整原生执行时间和已覆盖的显存包络 |
| `fakegpu validate-finetune-pipeline` | 对照独立原生运行验证完整时间和显存预测 |
| `fakegpu plan-finetune-transfer` | 规划经过核验的随机或顺序原生输入 |
| `fakegpu estimate-finetune-transfer` | 估算显式的经验形状迁移预算 |
| `fakegpu validate-finetune-transfer` | 对照独立原生运行验证迁移预算 |
| `fakegpu calibrate-finetune` | 在可恢复的工作区中运行有限额的原生校准 |
| `fakegpu budget-finetune-workspace` | 只用 CPU 对保存的原生证据重新预算 |

## Diffusion 训练

| 命令 | 用途 |
|---|---|
| `fakegpu plan-diffusion-training` | 查看缓存特征 UNet 训练计划和持久存储 |
| `fakegpu run-diffusion-training` | 在一张真实 GPU 上执行缓存特征 UNet full/LoRA 训练 |
| `fakegpu estimate-diffusion-training` | 用兼容的原生缓存特征训练观测做预算 |
| `fakegpu validate-diffusion-training` | 对照独立的原生 diffusion 训练 holdout |

## 分布式与实验

| 命令 | 用途 |
|---|---|
| `fakegpu plan-training` | 统一分布式训练配置并估算单 rank 显存 |
| `fakegpu simulate-topology` | 模拟 collective 路由和链路竞争 |
| `fakegpu replay-trace` | 汇总计算、通信、等待和显存时间线 |
| `fakegpu coordinator` | 管理分布式模拟 coordinator |
| `fakegpu bandwidth` | 验证模拟 TCP payload 并报告吞吐量 |
| `fakegpu plan-experiments` | 展开有限额的参数网格并检查实际命令 |
| `fakegpu run-experiments` | 运行或显式恢复实验，保留每次尝试 |
| `fakegpu compare-experiments` | 核验原始产物，比较重复观测，可选导出图表 |

## 校准与监控

| 命令 | 用途 |
|---|---|
| `fakegpu calibrate` | 整理 serving 观测、对比显存报告并执行可靠性门槛检查 |
| `fakegpu nvidia-smi` | 查看设备、进程、建模拓扑/MIG、健康状态和可靠性事件 |
| `fakegpu nvitop` | 通过原生 NVML 桥接，用已安装的 nvitop 查看本机 FakeGPU 状态（macOS/Linux） |
| `fakegpu console` | 打开后台机器的终端控制台：GPU、进程、事件和设置（[说明](console.md)） |
| `fakegpu daemon` | 在后台运行一台模拟 GPU 机器，FakeCUDA 进程会自动接入；支持启动、停止、状态、事件和配置 |
| `fakegpu platform` | 列出主机平台，并显示一个设备布局得到的插槽、PCIe 链路和 NVLink（[说明](platforms.md)） |
| `fakegpu metrics` | 导出有数量上限的 Prometheus/JSON 指标，并提供短期内存历史 |

<a id="run-a-command-on-simulated-gpus"></a>

## 在模拟 GPU 上运行命令

`fakegpu run` 在模拟 GPU 上原样启动一条命令，结束时向 stderr 打印显存摘要：

```bash
python3 -m fakegpu run --gpu a100x8 -- python3 train.py
python3 -m fakegpu run --gpu h100:4,a100:4 -- \
  torchrun --nproc_per_node 8 train.py
python3 -m fakegpu run --gpu a100x4 --report-dir build/run -- \
  accelerate launch --multi_gpu --num_processes 4 train.py
```

这次运行会把一个 `sitecustomize` 模块放在 `PYTHONPATH` 最前面，所以命令的每个
Python 进程在首次导入 torch 时都会启动 FakeCUDA，包括 `torchrun`、
`accelerate launch` 和 `mp.spawn` 启动的 worker。已有的 `sitecustomize` 仍会执行。
这些 Python 必须能导入 FakeGPU；不能导入的会在导入 torch 时打印警告，并在没有模拟
GPU 的情况下运行。以 `-I`、`-E` 或 `-S` 启动的 Python 会忽略 `PYTHONPATH`，同样
没有模拟 GPU。`CUDA_VISIBLE_DEVICES` 会像在真实机器上一样，把进程限制在这次运行的
部分 GPU 上，摘要也把它的显存计在这些 GPU 上。命令 `PATH` 上的 `nvidia-smi`，以及
已安装时的 `nvitop`，会列出这次运行的 GPU，即使还没有进程使用它们。进程之间的 collective 传递真实的值，见
[FakeCUDA 上的多进程训练](distributed.md#multi-process-training-on-fakecuda)。

| 选项 | 含义 |
|---|---|
| `--gpu SPEC` | 要模拟的 GPU：`a100x8`、`8xa100`、`a100:8`，或 `h100:4,a100:4` 这样的列表。ID 来自 `fakegpu doctor --list-profiles`；未知 ID 会以状态 2 退出，并给出相近的候选 |
| `--report-dir DIR` | 保留 `run_report.json`，并把每个进程的记录保存在 `DIR/records/` 中，替换上一次运行的记录。不指定时，运行文件写入临时目录，结束后删除 |
| `--no-summary` | 不打印摘要 |

不带 `--gpu` 时，按以下顺序取第一个可用来源作为 GPU：

1. 正在运行的后台机器（`fakegpu daemon start` 或 `fakegpu console`）。这次运行会
   接入该机器，因此 `fakegpu console` 会显示它的进程。`FAKEGPU_MACHINE=off` 跳过
   这一步；
2. `FAKEGPU_PROFILES`；
3. `FAKEGPU_PROFILE` 和 `FAKEGPU_DEVICE_COUNT`，默认分别为 `a100` 和 8，所以什么都
   不设置时会得到八张 A100 GPU。

自带 GPU 的运行不会接入后台机器。脚本如果向 `fakegpu.init()` 传入自己的 profile，
会保留该 profile；此时摘要列出脚本实际使用的 GPU，并加一行 `Note` 说明。

### 退出状态与 Ctrl-C

`fakegpu run` 以命令的退出状态退出。命令被信号终止时返回 128 加信号编号，例如
Ctrl-C 后为 130。命令找不到时，`fakegpu run` 自身以 127 退出；命令不可执行时为
126；`--gpu` 有误或缺少命令时为 2。

Ctrl-C 由终端直接发给命令。`fakegpu run` 会等命令停止后仍打印摘要，摘要覆盖到
中断为止的运行。发给 `fakegpu run` 的 SIGTERM 和 SIGHUP 会转发给命令；运行不在
终端前台时，SIGINT 也会转发。

### 阅读摘要

对于一个在 `cuda:0` 上分配一个 1 GiB tensor 的脚本：

```text
── FakeGPU run summary ────────────────────────────────────────────────
 Command   python3 alloc.py
 Result    exit 0 after 1.6 s; 1 process used simulated GPUs
 GPUs      2 × a100 (NVIDIA A100-SXM4-80GB, 80.00 GiB each)  [catalog]

  GPU  Peak memory          Capacity     Headroom     Processes
    0     1.00 GiB  [C3]     80.00 GiB    79.00 GiB           1
    1            —           80.00 GiB    80.00 GiB           0

 Verdict   Fits. Largest peak: 1.00 GiB on GPU 0, 1.2% of its capacity.
 [C3]      Peak reserved memory: operator outputs tracked at PyTorch
           dispatch on the CPU; not calibrated against a real GPU (C4).
 [catalog] Capacity from the FakeGPU profile catalog.
 Excludes  the CUDA context, library workspaces and NCCL buffers a real
           GPU adds.
───────────────────────────────────────────────────────────────────────
```

- **Peak memory** 是每张 GPU 上的峰值 reserved 显存。同一张 GPU 上各进程的峰值会
  相加，所以当这些进程在不同时间达到峰值时，该数字可能偏高。
- 峰值后面的标签是显存跟踪级别，从 `[C0]`（没有可用的跟踪）、`[C2]`（tensor 生命
  周期）、`[C3]`（算子输出或原生分配）到 `[C4]`（已对照真实 GPU 校准）。多个进程
  共用一张 GPU 时显示其中最弱的级别。出现的每个标签都会在表格下方解释。
- `[catalog]` 标记取自 FakeGPU profile 目录的容量。
- `Excludes` 列出这次运行没有计入的显存：真实 GPU 在峰值之上还会占用的 CUDA
  context、库 workspace 和 NCCL 缓冲区。
- 结论（verdict）为 `Fits`、`Does not fit`（模拟的显存不足错误终止了命令，或某个
  峰值超过容量）、`Unresolved`（命令先失败或被中断；峰值覆盖到那时为止的运行），
  或未使用 GPU（没有任何东西放到 `cuda` 上，或 FakeCUDA 没有启动）。
  `run_report.json` 中分别记为 `fits`、`does_not_fit`、`unresolved` 和
  `no_gpu_use`。命令捕获模拟的显存不足错误后继续运行时仍算放得下，结论会说明捕获了
  几次。
- GPU 超过 16 张时，只列出被使用的 GPU。

<a id="limits-of-fakegpu-run"></a>

### `fakegpu run` 的限制

- 只有 Python 进程能获得模拟 GPU。原生 CUDA 程序仍需使用启动器
  `python3 -m fakegpu --build-dir build <program>`，见
  [原生库拦截](native-interception.md)。
- 运行中的 `nvidia-smi` 读取的是已发布的状态，可能比进程滞后约 250 ms。
- 每个进程每秒以及收到 SIGTERM 或 SIGHUP 时都会重写自己的记录，所以某个 rank
  失败后被 `torchrun` 停掉的其他 rank 仍会计入摘要。通过 `os._exit()` 或 SIGKILL
  结束的进程可能缺少最后一秒的数据。
- DDP 梯度 bucket 不计入。

<a id="torchrun-hangs-at-rendezvous"></a>

### torchrun 卡在 rendezvous 时

如果主机名通过代理的 fake-IP DNS 解析，`torchrun` 可能会在 rendezvous 阶段一直
等待，与是否使用 FakeGPU 无关。请传入回环地址和一个空闲端口：

```bash
python3 -m fakegpu run --gpu a100x2 -- torchrun --nproc_per_node 2 \
  --master_addr 127.0.0.1 --master_port 29500 train.py
```
