# Diffusion 训练

[文档索引](README.md) · [English](../en/diffusion-training.md) ·
[繁體中文](../zh-TW/diffusion-training.md)

FakeGPU 可以规划、运行缓存特征 UNet 训练，并为其做预算。支持全量微调和 LoRA，
可选 `--ema-decay`。生成阶段的显存请看
[显存分析与估算](estimation.md#estimate-diffusion-generation-memory)。

## 工作方式

- `plan-diffusion-training` 对本地权重和 FP32 缓存张量计算哈希。
- `run-diffusion-training` 在一张 CUDA GPU 上训练，使用 BF16 autocast、FP32 可训练
  参数和 AdamW。
- `estimate-diffusion-training` 根据兼容的原生运行生成预算。
- `validate-diffusion-training` 将预算与一次独立运行对比。

不包括 VAE 和 text encoder 训练，也不包括原始图像预处理。

## 示例

下面的 fixture 会生成随机特征，用于集成测试：

```bash
python tests/native/prepare_diffusion_fixture.py \
  --model-dir /path/to/stable-diffusion-v1-4 --output-dir /tmp/diff-fixture \
  --num-samples 5 --epochs 3 --method lora --gradient-checkpointing
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-1 \
  --json /tmp/diff-cal-1.json
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-diffusion-training \
  --plan /tmp/diff-fixture/plan.json --output-dir /tmp/diff-cal-2 \
  --json /tmp/diff-cal-2.json
python -m fakegpu estimate-diffusion-training --plan /tmp/diff-fixture/plan.json \
  --epochs 5 --calibration-run /tmp/diff-cal-1.json \
  --calibration-run /tmp/diff-cal-2.json --json /tmp/diff-budget.json.gz
```

## 要求与限制

- 没有原生观测之前，activation 峰值和 reserved 峰值都是未知的。
- 预算需要两个独立进程，未来的每类窗口都要有重复样本。某个 shape 第一次出现时的
  开销单独计算。
- `--cache-regime` 的行为与[微调流程](finetuning.md#kernel-caches)相同，默认是
  `uncontrolled`。Allocator、GPU UUID、软件、输入和执行身份都必须一致。
- `validate-diffusion-training` 对照一次独立的 `--holdout-run`。超过
  `--max-total-time-error-percent` 或 `--max-memory-underestimate-percent` 时返回 1。

## 证据

[真实 diffusion 记录](../../tests/data/diffusion_native_evidence.json)包含 SD/SDXL
生成观测和十二个训练进程。六次训练 holdout 的总时间误差为 0.044%–3.185%，
allocator 包络没有低估。生成阶段的 VAE allocated 峰值低估了 12.0%–13.8%，
reserved 峰值最多低估 31.6%。
