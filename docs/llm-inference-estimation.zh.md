# LLM 推理显存与计算量估算

FakeGPU 为 decoder-only 推理提供三种互补的验证方式：

| 方式 | 是否需要 GPU | 观测内容 |
|---|---:|---|
| 仅检查 checkpoint | 否 | 根据 safetensors/config 元数据计算参数、KV cache、临时张量和矩阵 FLOPs |
| FakeCUDA 执行 | 否 | 在 CPU 上执行 CUDA 形式的 PyTorch 代码，记录张量生命周期显存、生成 token 和实际矩阵 FLOPs |
| 真实 CUDA 校准 | 是 | 针对确定的 GPU、软件版本和算子路径记录 PyTorch allocator 与 NVML 进程显存 |

前两种方式可以在纯 CPU 主机上使用。真实 CUDA 校准可以补充 context
和后端开销，但只适用于相同的 GPU、PyTorch/CUDA、模型形状、dtype 与
attention 实现。

## 不加载权重检查 checkpoint

```bash
python3 -m fakegpu estimate-llm \
  --model-dir /models/Qwen/Qwen3-8B \
  --batch-size 1 \
  --prompt-tokens 9 \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --json build/qwen-estimate.json
```

该命令只读取 `config.json` 与 safetensors header，不读取 tensor payload，
也不会实例化模型或创建 CUDA context。报告包含：

- checkpoint 的精确参数量与存储信息
- 指定推理 dtype 下的参数字节数
- 根据层数、KV heads、head dimension、batch 和序列长度计算的 KV cache
- eager 或 SDPA 路径的临时张量估算
- prefill 和每个 decode step 的矩阵 FLOPs；一次乘加记作两个 FLOPs

当前形状模型适用于 dense decoder-only 架构。遇到 MoE 配置时会明确拒绝，
不会错误地按 dense 模型计算。

## 在不可见 GPU 的环境中执行模型

仓库中的验证 worker 会把完整 Transformers 模型加载到主机内存，并通过
CPU-backed FakeCUDA tensor 执行相同的 forward/decode 流程：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
python3 verification/qwen_inference_memory_worker.py \
  --mode fakecuda \
  --model-dir /models/Qwen/Qwen3-8B \
  --profile rtx-pro-5000-blackwell \
  --prompt Hello \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --smi-state /tmp/fakegpu-qwen.json \
  --output build/qwen-fakecuda.json
```

这里特意设置 `CUDA_VISIBLE_DEVICES=''`，用于确认推理没有使用物理 GPU。
对于已维护的 PyTorch 算子，FakeCUDA 会执行真实的 CPU 计算；它不预测
CUDA kernel 的耗时。

普通应用应在调用 `fakegpu.init(runtime="fakecuda")` 前配置状态文件。
进程运行期间，可以在另一个终端查看：

```bash
python3 -m fakegpu nvidia-smi --state /tmp/fakegpu-qwen.json
```

多进程场景可以设置 `FAKEGPU_SMI_STATE_DIR=/tmp/fakegpu-smi`，每个进程会
写入独立状态文件，查看器会按逻辑 GPU 汇总，并显示主机、profile、stage、
tracking confidence，以及模拟值和原始 tracked memory 的当前值与峰值。
下面的命令每秒刷新一次，共采样十次：

```bash
python3 -m fakegpu nvidia-smi \
  --state-dir /tmp/fakegpu-smi --loop 1 --count 10
```

查看器会在每次刷新时重新扫描状态目录，因此启动后出现的新进程也能被发现。
循环模式配合 `--json` 时，每行输出一个 JSON 对象。

## 与真实 CUDA 对比

真实和模拟执行必须使用相同的 prompt、生成长度、dtype 和 attention 实现：

```bash
python3 verification/qwen_inference_memory_worker.py \
  --mode real \
  --model-dir /models/Qwen/Qwen3-8B \
  --prompt Hello \
  --generated-tokens 2 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --output build/qwen-real.json

python3 verification/compare_qwen_memory.py \
  --real build/qwen-real.json \
  --fakecuda build/qwen-fakecuda.json \
  --output build/qwen-comparison.json \
  --markdown build/qwen-comparison.md
```

比较器要求参数量和生成 token ID 完全一致，加载与推理显存误差低于 1%，
静态 FLOPs 误差低于 0.01%，FakeCUDA 与真实 CUDA 观测到的矩阵 FLOPs
完全一致。

## Qwen3-8B 实测结果

以下数据来自 NVIDIA RTX PRO 5000 72GB Blackwell、PyTorch 2.9.1/CUDA
12.8、BF16 SDPA、一个 9-token prompt 和两个生成 token ID：

| 比较项 | 预测值 | 真实值 | 绝对误差 |
|---|---:|---:|---:|
| FakeCUDA 模型加载 vs CUDA allocator | 16,381,470,976 B | 16,383,586,816 B | 0.012914% |
| FakeCUDA 推理峰值 vs CUDA allocator | 16,385,992,936 B | 16,396,630,528 B | 0.064877% |
| 静态推理峰值 vs CUDA allocator | 16,385,606,472 B | 16,396,630,528 B | 0.067234% |
| 虚拟 SMI 进程显存 vs NVML 进程显存 | 16,825,298,920 B | 16,835,936,256 B | 0.063182% |
| FakeCUDA 实际矩阵 FLOPs vs CUDA | 151,415,620,864 | 151,415,620,864 | 0% |
| 静态矩阵 FLOPs vs CUDA | 151,415,619,584 | 151,415,620,864 | 0.000001% |

虚拟 SMI 使用了本次真实执行测得的 `442,049,024` 字节开销，即 NVML
进程显存减去 CUDA allocator 当前显存。这个值不能当作通用 CUDA context
常量。

## 精度适用范围

目前可以确认：FakeGPU 对上述 dense Qwen 推理范围足够准确；它还不是可对
任意仓库给出同等保证的完整 GPU 模拟器。

以下场景需要单独建模或重新校准：

- MoE 路由、量化 checkpoint、adapter 或 tensor 间混合 dtype
- 自定义 CUDA extension、Triton kernel 或 FakeCUDA 尚未覆盖的算子
- 模型特有的常驻 buffer 或动态变化的控制流
- 不同 attention backend、PyTorch/CUDA 版本、allocator 策略或 GPU
- 吞吐量和延迟预测；FakeGPU 当前不提供这类估算

对于任意训练仓库，还应结合
[`fakegpu preflight`](ai-researcher-preflight.md) 或 ATen 计算图估算器，而不是
只使用 checkpoint 推理估算。
