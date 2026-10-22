# 微調時間與 GPU 記憶體

[文件索引](README.md) · [English](../en/finetuning.md) ·
[简体中文](../zh-CN/finetuning.md)

FakeGPU 可以根據設定檔估算單卡微調，也可以執行真實的訓練框架並記錄過程，再用獨立
執行檢查之後的預測。支援的框架為 Hugging Face Trainer/PEFT、LLaMA-Factory 與
Axolotl。

目錄：

1. [依設定估算](#estimate-from-a-config)
2. [掃描資料集](#profile-the-dataset)
3. [記錄原生執行](#record-a-native-run)
4. [驗證耗時視窗與事件](#validate-timing-windows-and-events)
5. [完整流程預算](#budget-a-complete-pipeline)
6. [隨機採樣與更換資料集](#random-sampling-and-changed-datasets)
7. [長度分組與 packing](#length-grouping-and-packing)
8. [Kernel 快取](#kernel-caches)
9. [原生執行準備](#native-run-setup)
10. [以 Hugging Face Trainer 取樣](#probe-with-hugging-face-trainer)
11. [驗證紀錄](#validation-records)

<a id="estimate-from-a-config"></a>

## 依設定估算

`estimate-finetune` 讀取本機模型中繼資料與框架設定，估算單卡全量、LoRA 與 4-bit
QLoRA 的 GPU 記憶體與**訓練迴圈**時間。它不會匯入訓練框架，也不會載入權重。

- 樣本數指過濾、切分與 packing 後的訓練序列。序列長度包含模板 token 與 padding。
- `max_steps` 優先於 epochs。`--max-steps -1` 可清除設定中的步數限制。
- 未打包的 Axolotl 設定依 Axolotl 自己的規則換算 epochs：
  `ceil(samples * epochs / (micro_batch_size * gradient_accumulation_steps))`
  次更新，Axolotl 會把它作為 `max_steps` 傳給 Trainer。
- Axolotl 只在設定了 `max_grad_norm` 時才把它傳給 Trainer，未設定即採用 Trainer
  的預設值 1.0。Axolotl 的 `lora_r` 沒有預設值（PEFT 會拒絕）；未設定時估算以 8
  計算，並在 `assumptions` 中註明。
- 沒有 `parameter_dtype` 或 `torch_dtype` 時，權重儲存精度依框架規則決定。
  LLaMA-Factory 的 LoRA/QLoRA 凍結權重保持本機 checkpoint `config.json` 宣告的
  dtype（`torch_dtype="auto"`），全參數微調為 FP32。Axolotl 依精度開關的 dtype
  載入權重（先 `bf16`，再 `fp16` 或 `load_in_8bit`）。HF Trainer 設定與未宣告
  dtype 的 checkpoint 使用 FP32。`assumptions` 會寫明採用的規則。

```bash
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --sequence-length 128 --json estimate.json
```

- 三類框架的設定範例位於 [`examples/finetuning/`](../../examples/finetuning/)。設定
  轉換已通過 CPU 測試，但沒有針對每種框架與執行環境組合做過基準測試。
- 分散式或 offload 執行、不支援的 optimizer，以及無法解析的 adapter 目標會明確報錯。
  分散式規劃請使用 `plan-training`。
- Qwen3.5 的線性 attention 與完整 attention 層分開計算。DeltaNet backward
  workspace、快速核心與 MoE dispatch 只有部分建模。`--text-only` 選擇多模態
  checkpoint 的文字骨幹。

<a id="profile-the-dataset"></a>

## 掃描資料集

`profile-finetune-data` 完整掃描本機 JSONL 的每筆資料，記錄內容指紋、token 長度、
截斷、padding 與序列數。原始 `text` 或 `messages` 資料使用本機 tokenizer。
`--packing concatenate` 只是 token 預算情境，不會重現任何框架的 packing 演算法。

要取得實際訓練集的計數，請匯出最終的 `input_ids`（可附帶 `attention_mask` 與
`labels`），並加上 `--processed`：

```bash
python -m fakegpu profile-finetune-data \
  --data /path/to/processed-train.jsonl --format input_ids --processed \
  --sequence-length 128 --json dataset.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --dataset-report dataset.json \
  --epochs 3 --max-steps -1 --json estimate.json
```

`--dataset-report` 會固定樣本數與序列上限。它取代設定中的樣本數、拒絕衝突的覆寫，
並要求長度上限一致。`--source-framework` 只是來源標籤。Packed 訓練需要最終處理後的
序列計數。

<a id="record-a-native-run"></a>

## 記錄原生執行

`run-finetune` 在一張實體 CUDA GPU 上執行已安裝的 HF Trainer/PEFT、LLaMA-Factory
或 Axolotl 流程。支援全量、LoRA 與 bitsandbytes 4-bit QLoRA，使用本機未量化的文字
checkpoint。除下文另有說明外，資料不做 packing。訓練採用完整的原生設定，相對路徑
以設定檔所在目錄解析。執行報告記錄：

- 原生前處理、有效訓練資料、實際 batch、optimizer 更新、allocator 峰值、評估與
  checkpoint 事件，以及最終儲存
- `observation.updates[].batches`：每個 microbatch 的形狀、epoch 位置、token 槽位、
  非 padding token 與有效 labels
- `observation.batch_profile`：依完整且有序的梯度累積視窗分組的耗時。每組有兩個
  穩態 update 後才標記為 `sufficient`。這只表示樣本足夠，與準確度無關。
- `observation.phase_events`：實際的評估與儲存呼叫樹。階段總量只累計根事件，儲存後
  的檔案清單也不代表磁碟寫入量。

Warmup 與統計開銷不計入穩態樣本，但包含在總牆鐘時間中。

<a id="validate-timing-windows-and-events"></a>

## 驗證耗時視窗與事件

`validate-finetune-windows` 使用另外記錄的 holdout 執行驗證耗時預測。

- 至少需要兩份模型、資料、軟體及原生設定相同的獨立校準報告。
- 排除全域 warmup 與視窗首次出現後，每個視窗在每份校準報告中仍須有兩個樣本。
  缺少涵蓋的視窗直接拒絕。
- 它只驗證歷史證據，不預測 shuffle 順序、warmup、總牆鐘時間或動態 GPU 記憶體。
  原生 packing 與 MoE 視窗尚未支援。

```bash
python -m fakegpu validate-finetune-windows \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json window-validation.json
```

`validate-finetune-events` 以同樣方式驗證評估與儲存的子總時間。開始、首次、重複與
最終事件分開計算。目前支援無 packing 的 dense 文字模型、只計算 loss 的評估，以及
本機模型/adapter 或可恢復 checkpoint 的儲存。它不預測事件次數或完整流程時間。

```bash
python -m fakegpu validate-finetune-events \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-validation.json
```

`--event-context` 也會匹配每個視窗前最近的評估與 checkpoint 邊界，包括有序呼叫、
工作量、階段，以及各邊界之後的 update 數。每個「視窗＋上下文」組合在每份校準報告
中需要兩個樣本。它驗證週期事件後重複出現的視窗，不涵蓋首次儲存後的恢復耗時或完整
訓練時間。

```bash
python -m fakegpu validate-finetune-windows --event-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json event-window-validation.json
```

`--epoch-context` 也會匹配每個 microbatch 在 epoch 內的位置，用來區分包含新一輪
取數或 worker 啟動的視窗。可以單獨使用，也可以與 `--event-context` 組合：

```bash
python -m fakegpu validate-finetune-windows --event-context --epoch-context \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json epoch-window-validation.json
```

<a id="budget-a-complete-pipeline"></a>

## 完整流程預算

`plan-finetune-pipeline` 從校準執行中完整觀測到的 epoch 推導未來的輪次或更新次數。
`estimate-finetune-pipeline` 組合至少兩次校準，得到完整的原生預算。預算涵蓋
setup、warmup、有序訓練視窗、評估、checkpoint、Trainer 間隙與收尾。

- 計時止於原生呼叫及最後一次 CUDA 同步，不含 Python 啟動、報告建構與 CLI 輸出。
- 時間區間取各程序平均值的極值，不能當作信賴區間。任一部件缺少涵蓋都會明確報錯。
- 支援無 packing、保留尾批次的循序文字資料：HF Trainer（Transformers 5.6.0）與
  LLaMA-Factory 0.9.5 需設定 `disable_shuffling: true`，Axolotl 0.18.0
  （Transformers 5.14.1）需設定 `curriculum_sampling: true`。
- `--epochs` 依框架自己的 `num_epochs` 含義換算。HF 與 LLaMA-Factory 會跑完每個
  epoch 末尾不足一個累積視窗的 update。Axolotl 執行
  `ceil(rows * epochs / (micro_batch_size * gradient_accumulation_steps))` 次
  update，即它傳給 Trainer 的 `max_steps`。`plan-finetune-transfer` 與
  `run-finetune --planning-epochs` 採用同一規則。打包的 Axolotl 執行須以步數指定
  範圍（`--max-steps`，`run-finetune` 用 `--planning-max-steps`），因為 Axolotl
  依 multipack 長度估計換算打包 epoch。
- 資料、模型、硬體與執行設定須與校準一致，資料筆數改變時需要重新校準。日誌須關閉
  或每次 update 都記錄。不支援 checkpoint 輪替刪除與最佳模型重新載入。

```bash
python -m fakegpu plan-finetune-pipeline \
  --run-report calibration-1.json --epochs 30 --json future-plan.json
python -m fakegpu estimate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --plan future-plan.json --memory-budget-gib 24 --reserve-gib 1 --json budget.json
python -m fakegpu validate-finetune-pipeline \
  --calibration-run calibration-1.json --calibration-run calibration-2.json \
  --holdout-run holdout.json --json validation.json
```

記憶體結果是同一張 GPU 與 allocator 上完整執行期間觀測到的 allocated 與 reserved
峰值，不能推算到更小的顯示卡。可選的保留量只是你自己設定的餘量，不能證明不會 OOM。
`validate-finetune-pipeline` 對照獨立 holdout 回報時間誤差與記憶體低估。

<a id="random-sampling-and-changed-datasets"></a>

## 隨機採樣與更換資料集

使用預設隨機採樣或更換訓練資料時，請改用經驗遷移流程。它讀取原生 prepared
dataloader，在 CPU 上計畫 sampler 與 collator 的輸出，並在校準時核對實際消費的列號
與輸入張量雜湊。支援 HF 與 LLaMA-Factory 搭配 Accelerate 1.11.0，以及 Axolotl 搭配
Accelerate 1.13.0。隨機採樣使用原生 data seed 加上 epoch。（Axolotl 0.18.0 不會把
YAML 的 `data_seed` 傳給 Trainer。）

```bash
python -m fakegpu calibrate-finetune \
  --config target-training.json --framework huggingface \
  --profile "$TARGET_GPU_PROFILE" --python /path/to/framework/python \
  --cuda-device 0 --workspace ./calibration-target \
  --calibration-steps 24 --warmup-steps 2 --epochs 3 \
  --timeout-seconds 600 --json calibration.json
python -m fakegpu budget-finetune-workspace \
  --workspace ./calibration-target --epochs 2 --json revised-budget.json
```

- `calibrate-finetune` 固定啟動兩次全新的原生程序，限制步數與執行時間，預設各自
  使用新的空 kernel 快取目錄。它保留所有嘗試，以 SHA256 封存報告，並支援
  `--resume`。它不會在背後自行延長或重跑校準。
- 在 CPU 上重估預算不需要 GPU 或框架，但不能超過已儲存的採樣範圍。
- 工作區需保留原本的絕對路徑，並放在已記錄的輸入目錄之外。巢狀工作區會被拒絕。
  只支援有限的本機資料與 POSIX 鎖語義。Axolotl 與 LLaMA-Factory 的 prepared 資料
  隔離在工作區的 `prepared/` 目錄中，恢復時若原始輸入有變更會被拒絕。
- `--reuse-calibration-run PATH`（可重複指定）可重用另一份資料的相容訓練證據。
  啟動與收尾成本仍由目標資料上的兩次新校準提供，且不會依樣本數放大。不同 token、
  mask 與監督分布之間的訓練成本重用屬於**近似值**。缺少涵蓋時傳回 1 並列出缺口。
- `run-finetune --planning-epochs N`（或 `--planning-max-steps N`）直接記錄 CPU
  規劃範圍。`plan-finetune-transfer`、`estimate-finetune-transfer` 與
  `validate-finetune-transfer` 不經過工作區管理也能完成相同步驟。streaming、自訂
  採樣器與自動 batch size 調整尚未支援。

<a id="length-grouping-and-packing"></a>

## 長度分組與 packing

### 長度分組

HF 或 LLaMA-Factory 設定 `train_sampling_strategy: group_by_length`，Axolotl 設定
`group_by_length: true`，並向 `calibrate-finetune` 或 `run-finetune` 傳入
`--length-grouping-seed 123`。

- 它沿用原生分組演算法，搭配以 `123 + epoch` 為種子的獨立 CPU 生成器。依賴全域
  RNG 的原生分組無法重現。
- 必須指定規劃範圍。一個有限評估資料集使用同一個明確種子，不會推進訓練 sampler。
  自訂評估 sampler 與預先準備的 loader 會被拒絕。
- Sampler plan v2 把分組長度與 batch size 納入校準身分，因此長度組合不同時需要各自
  的證據。

### Packing

- [packing 語義審計](../../tests/native/finetune_packing_semantics.py)顯示，在極小的
  Qwen3.5 模型中，所測的一般 HF 與 LLaMA-Factory 混合路徑在分塊 mask 與重設位置後
  仍有跨段影響。所測的 Axolotl patch 也與 Transformers 5.14.1 的 decoder 介面不相容。
- 選用的[分段參考審計](../../tests/native/finetune_segment_reference.py)檢查專案自有
  的 Qwen3.5 介面。每段 linear attention 都從空的卷積與遞迴狀態開始執行，結果與獨立
  原生序列一致，包括 checkpoint 重算與重新載入。
- 向 `run-finetune` 或 `calibrate-finetune` 傳入 `--packing-seed 123` 即可啟用。
  需要 dense `qwen3_5_text` LoRA、指定規劃範圍、啟用原生 packing（LLaMA-Factory
  設定 `packing: true` 與 `neat_packing: true`，Axolotl 設定 `sample_packing: true`），
  並關閉評估。
- HF processed 資料另需 `position_ids`，`attention_mask` 使用連續的正整數 segment ID
  （右側可用零 padding）。後續每一段的首個 label 必須忽略。
- Sampler plan v3 記錄 epoch bins、邏輯序列數、張量列數、分段幾何、dtype 與輸入
  雜湊，不同資料需要各自校準。原生 packing 報告須使用 transfer 指令。
- 不對 varlen 核心吞吐量做任何承諾。全量微調、QLoRA 與 grouped packing 尚未涵蓋。
  另見 [PEFT direct-forward 審計](../../tests/native/finetune_segment_peft.py)與
  [packing 預算證據](../../tests/data/finetune_packing_budget_evidence.json)。

<a id="kernel-caches"></a>

## Kernel 快取

`run-finetune --cache-regime empty_kernel_cache` 要求明確設定五個互不相同、已存在且
為空的目錄：`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、
`TORCH_EXTENSIONS_DIR` 與 `PYTORCH_KERNEL_CACHE_PATH`。`populated_kernel_cache`
記錄其中每個檔案的相對路徑、大小與 SHA256。runner 不會建立、清理或預熱這些快取。

預設的 `uncontrolled` 模式不對快取狀態做任何承諾，也不能與受控校準混用。Triton
中繼資料可能引用原本的快取路徑，請保持這些檔案不變。

<a id="native-run-setup"></a>

## 原生執行準備

- Unpacked HF 執行（`--framework huggingface`）使用平鋪設定，以及每行恰好包含
  `input_ids`、`attention_mask` 與 `labels` 的 processed JSONL。三者須為等長的整數
  陣列，mask 為二值，長度介於 2 到 `sequence_length` 個 token。不會隱式斷詞，
  tokenizer 須有 pad token。參見[原生 LoRA 範例](../../examples/finetuning/huggingface-native-lora.yaml)。
- 梯度檢查點：LLaMA-Factory 使用 `disable_gradient_checkpointing: false`，並以
  `use_reentrant_gc: true` 或 `false` 選擇實作。Axolotl 使用
  `gradient_checkpointing: true` 與 `gradient_checkpointing_kwargs: {use_reentrant: true}`
  （或 `false`）。實際生效的檢查點與量化設定都納入校準身分。
- 原生訓練需要文字 checkpoint，`--text-only` 不會轉換權重。對於本機 Qwen3.5，先在
  CPU 上匯出文字 checkpoint，並產生用於整合檢查的合成 Alpaca 資料：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/fakegpu-finetune/qwen35-text \
  --dataset-dir /tmp/fakegpu-finetune/data
```

匯出需要支援 `Qwen3_5ForCausalLM` 的 Transformers。原生範例針對 LLaMA-Factory 0.9.5
（Transformers 5.6.0）與 Axolotl 0.18.0（Transformers 5.14.1）。請分別在獨立環境
安裝對應框架，以及相容的 CUDA PyTorch、PEFT 與資料相依套件。每次執行使用新的
`output_dir`。Axolotl 的前處理快取位於 `/tmp/fakegpu-finetune/axolotl-prepared`。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu run-finetune \
  --config examples/finetuning/axolotl-native-lora.yaml --framework axolotl \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --output run.json
python -m fakegpu estimate-finetune \
  --run-report run.json --epochs 3 --max-steps -1 --json estimate.json
```

- LLaMA-Factory 使用 `--framework llamafactory` 與
  [`llamafactory-native-lora.yaml`](../../examples/finetuning/llamafactory-native-lora.yaml)。
- 請在同一個 Python 環境重用 `--run-report`。CLI 會核對本機權重、模型設定與軟體
  身分，GPU profile 預設沿用報告中的值。
- 這個單報告的 `estimate-finetune` 介面要求固定形狀的完整 batch。改變
  `--num-samples` 只提供訓練迴圈情境，不含前處理與載入成本，也不含 Python 啟動或
  下載。

512-token NF4 QLoRA 搭配梯度檢查點的範例見
[`llamafactory-native-qlora.yaml`](../../examples/finetuning/llamafactory-native-qlora.yaml)
與 [`axolotl-native-qlora.yaml`](../../examples/finetuning/axolotl-native-qlora.yaml)。
兩者都使用 double quantization 與 BF16，需要 bitsandbytes。以已匯出的 tokenizer
產生較長的資料：

```bash
python tests/native/prepare_finetune_fixture.py \
  --model /tmp/fakegpu-finetune/qwen35-text --dataset-only \
  --dataset-dir /tmp/fakegpu-finetune/data-long \
  --sequence-length 1024 --num-samples 64
```

fixture 長度只是下限。各框架會依設定截斷至 512 或 1024 個 token，執行報告記錄實際
採用的序列。

使用 [`tests/native/finetune_framework_matrix.py`](../../tests/native/finetune_framework_matrix.py)
採集獨立的校準與 holdout 程序。序列長度、microbatch、梯度累積與 seed 都從原生設定
讀取：

```bash
python tests/native/finetune_framework_matrix.py \
  --framework axolotl --config examples/finetuning/axolotl-native-qlora.yaml \
  --python /path/to/axolotl/venv/bin/python \
  --output-dir /tmp/fakegpu-finetune/axolotl-qlora-matrix \
  --gpu-index 1 --calibration-steps 6 --warmup-steps 2 \
  --holdout-runs 2 --holdout-steps 16
```

步數包含 warmup。預設為 6 次校準更新、2 次 warmup，以及三次各 32 次更新的
holdout。每次執行使用新的輸出目錄，預測使用正式估算器與實際的 microstep 計數。

<a id="probe-with-hugging-face-trainer"></a>

## 以 Hugging Face Trainer 取樣

想取得以實測為基礎的估算，可以先在實體 GPU 上短跑取樣，再於**同一個 Python 環境**
重用觀測結果。模型、GPU、精度、adapter、attention 實作、batch、序列長度與梯度累積
也必須一致。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fakegpu probe-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --warmup-steps 2 --measure-steps 5 \
  --output observation.json
python -m fakegpu estimate-finetune \
  --config examples/finetuning/huggingface-lora.yaml \
  --model /path/to/Qwen3.5-0.8B --text-only \
  --profile "$TARGET_GPU_PROFILE" --num-samples 10000 --epochs 3 \
  --observation observation.json --json calibrated.json
```

probe 以 Hugging Face Trainer 和固定長度的合成文字執行，需要 Transformers、
Accelerate 與 CUDA PyTorch。adapter 另需 PEFT，QLoRA 另需 bitsandbytes。warmup
另行記錄，啟動峰值與穩態峰值都會保留。推估時間不含資料準備、模型載入、評估與
checkpoint I/O。checkpoint 檔案會計算雜湊，替換權重後原本的校準就會失效。

<a id="validation-records"></a>

## 驗證紀錄

精確的工作負載、版本與適用範圍見 [`refactor.md`](../../refactor.md)。

- [完整流程驗收](../../tests/data/finetune_pipeline_validation_evidence.json)：三種
  框架共 30 個原生程序，包含 allocator OOM 對照。
- [採樣與遷移驗收](../../tests/data/finetune_transfer_validation_evidence.json)：
  Blackwell GPU 上 18 個 Qwen3.5-0.8B 程序，包括訓練資料從 25 筆改為 73 筆，以及
  隨機採樣搭配週期性評估與儲存。每個條件一次 holdout。
- [Qwen3.5-2B 紀錄](../../tests/data/finetune_qwen2b_evidence.json)：512/1024-token
  LoRA 與 512-token NF4 QLoRA（啟用梯度檢查點），每個條件兩次校準、兩次 holdout。
  總時間誤差為 0.054%–1.578%。
- [原生驗證歸檔](../../tests/data/finetune_framework_validation_evidence.json)：
  先前的 128-token LoRA 執行。最大牆鐘誤差為 3.931%（LLaMA-Factory）與 1.125%
  （已快取的 Axolotl）。把 Axolotl 首次執行的校準用在之後已快取的執行上，誤差為
  364.885%（首次 update 花了 66.542 秒）。所以模型與軟體簽章相同，並不代表 cache
  與 JIT 開銷可以重用。
- [擴充驗證歸檔](../../tests/data/finetune_extended_validation_evidence.json)：六組
  設定，涵蓋 512/1024 tokens、batch 1/2、梯度檢查點與 NF4 QLoRA，共 18 個測量程序。
- [Qwen3.5-0.8B probe 紀錄](../../tests/data/finetune_validation_evidence.json)：
  全量、LoRA 與 QLoRA 的校準與 holdout 執行，可用
  [`tests/native/finetune_matrix.py`](../../tests/native/finetune_matrix.py) 重現。

這些短跑結果不能證明通用的準確度或 OOM 安全。
