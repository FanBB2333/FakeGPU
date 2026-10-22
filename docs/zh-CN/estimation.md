# 显存分析与估算

[文档索引](README.md) · [English](../en/estimation.md) ·
[繁體中文](../zh-TW/estimation.md)

FakeGPU 有两种检查显存的方式。`preflight` 在 FakeCUDA 上把代码执行到指定阶段。
静态估算器只读取模型元数据，从不加载权重。在线 serving 请看
[在线 LLM serving](serving.md)。

## 运行前检查显存

将命令执行到指定阶段，并把报告写入 Git 忽略的构建目录：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight 跟踪实际执行路径上用到的显存，并报告选定的 profile 能否容纳该工作负载。

## 分析仓库

```bash
# 查找 GPU 入口、依赖、原生源码和兼容性风险。
python3 -m fakegpu analyze-repo .

# 检查 PTX、SASS 或 CUDA 源文件中的指令、FLOP、寄存器、
# shared memory 和 occupancy。
python3 -m fakegpu analyze-kernel kernel.ptx --profile a100 --threads-per-block 256
```

## 估算 LLM 显存

```bash
# 估算 checkpoint、KV cache、临时 tensor、adapter 和 MoE 显存。
python3 -m fakegpu estimate-llm \
  --model-dir /models/example \
  --batch-size 1 \
  --prompt-tokens 128 \
  --generated-tokens 32 \
  --dtype bfloat16 \
  --kv-cache-strategy paged \
  --kv-cache-block-tokens 16 \
  --target-profile a100 \
  --json build/llm-estimate.json

# 对齐只把最后一个 token 投影到 logits 的生成 runtime。
python3 -m fakegpu estimate-llm \
  --model-dir /models/example --prompt-tokens 128 --generated-tokens 32 \
  --logits-to-keep 1 --json build/last-token-estimate.json
```

默认值 `--logits-to-keep 0` 会投影全部 token。请按你的 runtime 实际设置填写。它
影响 LM head 的 FLOP 和显存，不改变其他层的 token 数。

LLM 估算器的工作方式：

- 只读取 safetensors header。
- `--kv-cache-strategy` 可选 `dynamic`、`static`、`quantized` 或 `paged`。报告
  分别列出逻辑存储量、量化节省量、static 预留、paged 额外占用和 sliding-window
  上限。
- sliding-window 层按 Transformers 解析配置的方式确定：`layer_types`、Qwen 风格的
  `use_sliding_window` 加 `max_window_layers`，或 `sliding_window_pattern`
  （Gemma 3 默认每 6 层中的第 6 层为 full attention）。dynamic cache 在 prefill
  后这些层仍保存整个 prompt，decode 一步后只保留最近 `sliding_window` 个 token；
  paged cache 保留窗口加上本步调度的 token。full attention 层保留完整上下文。
  `--kv-cache-window-tokens` 则对所有层套用同一个窗口。
- 量化 cache（Transformers `QuantizedCache`）在 prefill 时量化整个 prompt。生成的
  token 先以计算 dtype 保存，累积到 `--kv-cache-residual-tokens`（默认 128）个时
  整个 cache 重新量化，因此最多有该值减 1 个 token 未量化。每
  `--kv-cache-group-size`（默认 64）个量化元素另存一个计算 dtype 的 scale 和
  zero point。每一层都保留完整历史。
- MHA、GQA 和 MQA 使用独立的 key/value 存储。配置同时包含 `kv_lora_rank`、
  `qk_nope_head_dim`、`qk_rope_head_dim` 和 `v_head_dim` 时识别为 MLA。
  `--mla-kv-cache-layout` 选择 runtime 缓存的内容：`decompressed` 为逐 head 的
  key（`qk_nope_head_dim + qk_rope_head_dim`）和 value（`v_head_dim`），与
  Transformers 一致；`latent` 为压缩 latent 加 RoPE key，与 vLLM 一致。默认
  `auto` 在 paged 策略下用 `latent`，其余用 `decompressed`。MLA backend
  workspace 仍需校准。
- 对带视觉编码器的模型，只计入 checkpoint 权重，不含 vision/projector 激活与
  FLOP。报告中会标注这一点。decoder token 数需要已经包含视觉 token。

<a id="estimate-diffusion-generation-memory"></a>

## 估算 diffusion 生成显存

```bash
# 比较 diffusion 生成阶段和显存优化选项。
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# 检查本地 Diffusers pipeline，不加载 tensor payload。
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json
```

Diffusion 估算器的工作方式：

- 分别计算 text encoding、重复 denoising 和 VAE decode。`--model-dir` 只读取
  `model_index.json`、组件的 `config.json` 和选定的 safetensors header。不执行
  远端代码，也不加载 tensor payload。
- checkpoint 存储字节与运行时权重字节分开报告。本地 Euler/PNDM scheduler 配置
  决定 denoiser 调用次数。其他 scheduler 报告 `denoiser_evaluations: null`。
- UNet、交叉注意力 Transformer（DiT/PixArt）和 SD3/Flux 类联合注意力
  Transformer 使用不同的 activation 公式。CFG 只在正负分支都存在时让 denoiser
  batch 翻倍。
- `--weight-variant fp16|bf16|fp32` 指定 checkpoint 文件族。内置 profile 包含
  `pixart-sigma-xl-2-1024-ms` 和两个 Stable Diffusion UNet。
- `--offload model`、`--attention-slicing`、`--vae-slicing` 和 `--vae-tiling`
  对应 Diffusers 常见的显存选项。没有同配置的真实 GPU 对比时，报告保持
  `Modeled` 和 `uncalibrated`，不给出误差百分比或预测区间。
- 与 Diffusers 一致，`--attention-slicing` 只影响 UNet 去噪器。Transformer
  去噪器（PixArt、SD3、Flux）会忽略 pipeline 的 attention slicing。
- `--vae-tile-size` 默认取 VAE config 的 `sample_size`，即
  `enable_vae_tiling()` 实际使用的 tile（SDXL 为 1024，所以 1024x1024 解码不会
  分块）。config 没有该字段时回退为 512。

训练 diffusion 模型请看 [Diffusion 训练](diffusion-training.md)。

## 其他估算器

- `estimate-roofline` 生成与 profile 相关的分析延迟区间，不是实测 kernel 延迟。
  `--compute-precision` 选择 profile 中某条计算路径的 dense 峰值，例如 `tensor_bf16`、
  `tensor_fp8`（默认 `fp32`）；`estimate-llm` 和 `estimate-finetune` 也支持这个选项。
  峰值来自 NVIDIA 公布值，或由架构的每 SM 速率推导，[profile 目录](../../profiles/README.md)中有标注。
- `plan-training` 估算分布式训练的单 rank 显存。见
  [分布式规划与实验批量运行](distributed.md)。
- `estimate-finetune` 估算单卡微调的时间和显存。见
  [微调时间与显存](finetuning.md)。

凡是接受 JSON 路径的地方，大型报告和计划都可以用 `.json.gz`。压缩输出采用原子
写入，字节可复现。
