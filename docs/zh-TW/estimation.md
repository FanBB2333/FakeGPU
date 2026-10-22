# 記憶體分析與估算

[文件索引](README.md) · [English](../en/estimation.md) ·
[简体中文](../zh-CN/estimation.md)

FakeGPU 有兩種方式檢查記憶體。`preflight` 在 FakeCUDA 上把程式執行到指定階段。
靜態估算器只讀取模型中繼資料，不會載入權重。線上 serving 請參閱
[線上 LLM serving](serving.md)。

## 執行前檢查 GPU 記憶體

將指令執行到指定階段，並把報告寫入 Git 忽略的建置目錄：

```bash
python3 -m fakegpu preflight \
  --runtime fakecuda \
  --profile a100 \
  --stage forward \
  --report-dir build/preflight \
  --strict \
  -- python3 train.py
```

Preflight 追蹤實際執行路徑上用到的 GPU 記憶體，並判斷選定的 profile 能否容納
該工作負載。

## 分析儲存庫

```bash
# 尋找 GPU 入口、相依套件、原生原始碼與相容性風險。
python3 -m fakegpu analyze-repo .

# 檢查 PTX、SASS 或 CUDA 原始檔中的指令、FLOP、暫存器、
# shared memory 與 occupancy。
python3 -m fakegpu analyze-kernel kernel.ptx --profile a100 --threads-per-block 256
```

## 估算 LLM 記憶體

```bash
# 估算 checkpoint、KV cache、暫存 tensor、adapter 與 MoE GPU 記憶體。
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

# 對齊只把最後一個 token 投影到 logits 的生成 runtime。
python3 -m fakegpu estimate-llm \
  --model-dir /models/example --prompt-tokens 128 --generated-tokens 32 \
  --logits-to-keep 1 --json build/last-token-estimate.json
```

預設 `--logits-to-keep 0` 會投影所有 token。請使用 runtime 實際採用的設定。它只
影響 LM head 的 FLOP 與記憶體，其他層的 token 數不變。

LLM 估算器的運作方式：

- 只讀取 safetensors header。
- `--kv-cache-strategy` 可選 `dynamic`、`static`、`quantized` 或 `paged`。報告
  分別列出邏輯儲存量、量化節省量、static 預留、paged 額外占用與 sliding-window
  上限。
- 量化 cache 預設將最近 128 個 token 保留為計算 dtype，可用
  `--kv-cache-residual-tokens` 調整。
- MHA、GQA 與 MQA 使用分離的 key/value 儲存。設定同時包含 `kv_lora_rank`、
  `qk_nope_head_dim`、`qk_rope_head_dim` 與 `v_head_dim` 時辨識為 MLA（壓縮
  latent + 獨立 RoPE）。MLA backend workspace 仍需校準。
- 對帶有 vision encoder 的模型，會計入 checkpoint 權重，但不含 vision/projector
  activation 與 FLOP，報告會標示這一點。decoder token 數需已包含視覺 token。

<a id="estimate-diffusion-generation-memory"></a>

## 估算 diffusion 生成記憶體

```bash
# 比較 diffusion 生成階段與 GPU 記憶體最佳化選項。
python3 -m fakegpu estimate-diffusion --list-profiles
python3 -m fakegpu estimate-diffusion \
  --model-profile stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 --batch-size 4 \
  --attention-backend sdpa --vae-slicing \
  --offload model --target-profile a100 \
  --json build/diffusion-estimate.json

# 檢查本機 Diffusers pipeline，不載入 tensor payload。
python3 -m fakegpu estimate-diffusion \
  --model-dir /models/pixart-or-flux \
  --height 1024 --width 1024 --text-tokens 300 \
  --dtype bfloat16 --attention-backend sdpa \
  --offload model --target-profile a100 \
  --json build/local-diffusion-estimate.json
```

Diffusion 估算器的運作方式：

- 分別計算 text encoding、重複 denoising 與 VAE decode。`--model-dir` 只讀取
  `model_index.json`、元件 `config.json` 與選定的 safetensors header，不執行遠端
  程式碼，也不讀取 tensor payload。
- checkpoint 儲存位元組與執行期權重位元組分別列出。本機 Euler/PNDM scheduler
  設定決定 denoiser 呼叫數，其他 scheduler 回報 `denoiser_evaluations: null`。
- UNet、交叉注意力 Transformer（DiT/PixArt）與 SD3/Flux 類聯合注意力
  Transformer 使用不同的 activation 公式。CFG 只在存在正負分支的架構上讓 denoiser
  batch 加倍。`encode_prompt` 分別編碼 prompt 與 negative prompt，因此 text
  encoder 依 prompt batch 計算。
- 對 SD3，`--text-tokens` 是 T5 的 `max_sequence_length`（預設 256）。聯合注意力
  序列另含 77 個 token 的 CLIP 前綴，回報為 `inputs.denoiser_text_tokens`。
  沒有 `text_encoder_3` 時，T5 部分固定為 77 個 token 的佔位張量。
- 沒有記錄呼叫預設值的 pipeline 類別沿用 `StableDiffusionPipeline` 的預設值
  （50 步、guidance 7.5、77 個 token）。`generation_defaults.assumed_inputs` 與
  `assumptions` 會列出每個以此方式填入的值。請以該 pipeline 自己的預設值傳入
  `--steps`、`--guidance-scale` 與 `--text-tokens`。
- `--weight-variant fp16|bf16|fp32` 指定權重檔案族。內建 profile 包含
  `pixart-sigma-xl-2-1024-ms` 與兩個 Stable Diffusion UNet。
- `--offload model`、`--attention-slicing`、`--vae-slicing` 與 `--vae-tiling`
  對應 Diffusers 常見的 GPU 記憶體取捨。沒有相同設定的實體 GPU 比較時，報告保持
  `Modeled` 與 `uncalibrated`，不給出誤差百分比或預測區間。
- 與 Diffusers 一致，`--attention-slicing` 只影響 UNet 去噪器。Transformer
  去噪器（PixArt、SD3、Flux）會忽略 pipeline 的 attention slicing。
- `--vae-tile-size` 預設取 VAE config 的 `sample_size`，即
  `enable_vae_tiling()` 實際使用的 tile（SDXL 為 1024，因此 1024x1024 解碼不會
  分塊）。config 沒有該欄位時退回 512。

訓練 diffusion 模型請參閱 [Diffusion 訓練](diffusion-training.md)。

## 其他估算器

- `estimate-roofline` 產生與 profile 相關的分析延遲區間，不代表實測的 kernel 延遲。
  `--compute-precision` 選擇 profile 中某條運算路徑的 dense 峰值，例如 `tensor_bf16`、
  `tensor_fp8`（預設 `fp32`）；`estimate-llm` 與 `estimate-finetune` 也支援此選項。
  峰值來自 NVIDIA 公布值，或由架構的每 SM 速率推導，[profile 目錄](../../profiles/README.md)中有標註。
- `plan-training` 估算分散式訓練的單 rank GPU 記憶體。請參閱
  [分散式規劃與實驗批次執行](distributed.md)。
- `estimate-finetune` 估算單卡微調的時間與 GPU 記憶體。請參閱
  [微調時間與 GPU 記憶體](finetuning.md)。

大型報告與計畫在所有接受 JSON 路徑的位置都可以使用 `.json.gz`。壓縮輸出採用原子
寫入，位元組可重現。
