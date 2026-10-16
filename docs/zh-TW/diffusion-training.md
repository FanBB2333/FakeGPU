# Diffusion 訓練

[文件索引](README.md) · [English](../en/diffusion-training.md) ·
[简体中文](../zh-CN/diffusion-training.md)

FakeGPU 可以規劃、執行快取特徵 UNet 訓練並編列預算。支援 full 與 LoRA，可選用
`--ema-decay`。生成所需的記憶體請參閱
[記憶體分析與估算](estimation.md#estimate-diffusion-generation-memory)。

## 運作方式

- `plan-diffusion-training` 計算本機權重與 FP32 快取張量的雜湊。
- `run-diffusion-training` 在一張 CUDA GPU 上訓練，使用 BF16 autocast、FP32 可訓練
  參數與 AdamW。
- `estimate-diffusion-training` 根據相容的原生執行結果建立預算。
- `validate-diffusion-training` 將預算與一次獨立執行比較。

VAE、text encoder 訓練與原始影像前處理不在範圍內。

## 範例

以下 fixture 產生隨機特徵，僅供整合測試使用：

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

## 需求與限制

- 沒有原生觀測時，activation 與 reserved 峰值保持未知。
- 預算需要兩個獨立程序，且未來每類視窗都要有重複樣本。某個形狀第一次出現的開銷
  另外計算。
- `--cache-regime` 的行為與[微調流程](finetuning.md#kernel-caches)相同，預設為
  `uncontrolled`。Allocator、GPU UUID、軟體、輸入與執行身分都必須一致。
- `validate-diffusion-training` 對照獨立的 `--holdout-run`。超過
  `--max-total-time-error-percent` 或 `--max-memory-underestimate-percent` 時
  回傳 1。

## 證據

[原生 diffusion 記錄](../../tests/data/diffusion_native_evidence.json)包含 SD/SDXL
生成觀測與 12 個訓練程序。六次訓練 holdout 的總時間誤差為 0.044%–3.185%，
allocator 包絡沒有低估。生成方面，VAE allocated 峰值低估 12.0%–13.8%，reserved
峰值最多低估 31.6%。
