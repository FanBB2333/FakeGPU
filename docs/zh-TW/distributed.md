# 分散式規劃與實驗批次執行

[文件索引](README.md) · [English](../en/distributed.md) ·
[简体中文](../zh-CN/distributed.md)

## 分散式訓練規劃

`plan-training` 讀取 DeepSpeed、Accelerate、FSDP 與 FSDP2 設定，並估算每個 rank
的 GPU 記憶體。

FSDP2 `full_shard` 可向 `plan-training --parameter-shapes layout.json` 提供
`{"shapes": [[12, 8], [12], [3, 12], [3]]}`，並搭配一致的總 `--parameter-bytes` 與
統一的 `--parameter-element-bytes`。它依原生每參數第 0 維 chunk 計算，並以最大的
rank 編列預算。未提供形狀時使用均衡切分近似。自訂 placement 不在建模範圍內。

DeepSpeed ZeRO 的 bucket 大小以元素個數計，預設值與 DeepSpeed 相同：
`reduce_bucket_size` 與 `allgather_bucket_size` 為 5e8，`stage3_prefetch_bucket_size`
為 5e7，`stage3_max_live_parameters` 與 `stage3_max_reuse_distance` 為 1e9。

- reduce bucket 佔用元素個數乘以模型 dtype 的位元組數，模型再小也不縮減。開啟
  `contiguous_gradients`（預設）時，stage 1 與 2 在 backward 期間持有它，並在
  optimizer step 之前釋放；`overlap_comm` 使其加倍。stage 3 在整個執行期間持有
  一個 bucket。
- allgather bucket 原地寫入參數，不佔用記憶體。
- ZeRO-3 的 gathered window 約為 `stage3_max_reuse_distance` 的一半加上 prefetch
  bucket。單一 rank 不做 gather。
- 設為 `auto` 的大小在傳入 `--hidden-size` 時取 HF Trainer 與 Accelerate 填入的值：
  reduce bucket 為 `hidden_size²`，prefetch bucket 為 `0.9·hidden_size²`。未傳入時
  使用 DeepSpeed 預設值，並在 notes 中註明。

以上依據 DeepSpeed 0.15.3 與 0.19.8。FSDP 沒有 bucket 設定，FSDP 規劃仍沿用
FakeGPU 的啟發式：每次 collective 取 `min(參數位元組數, 500 MB)`。

FSDP `SHARD_GRAD_OP` 與 FSDP2 `reshard_after_forward=False`（`shard_grad_op`）
平時只保留每個 rank 的參數分片。規劃從 forward 到 backward 結束額外計入完整的
未分片參數，並在 optimizer step 前重新分片，與 FSDP 在梯度同步步 backward 之後
的行為一致。`_HYBRID_SHARD_ZERO2`（`hybrid_shard_zero2`）在 shard group 內做同樣
的事，在 group 之間複製。與 `HYBRID_SHARD` 一樣，group 預設取 Accelerate 每台
機器的程序數。

各階段描述的是穩定狀態下的一次迭代：

- forward 與 backward 已持有先前各步留下的 optimizer state 與 master 權重。
  ZeRO-3 還持有其梯度分區，梯度累積時也保留梯度。
- optimizer step 時，foreach Adam（torch 在 CUDA 上的預設實作）額外配置一個
  moment 大小的暫存張量。
- DeepSpeed ZeRO 還會建立梯度分區的 FP32 副本。stage 1 與 2 會先釋放 16 位元梯度。

在一張 Ampere GPU（sm_80）上以 DeepSpeed 0.15.3 單 rank 訓練 0.6B BF16 模型，
stage 1 到 3 的 step 峰值與預測相差不到 0.2%。這不能說明其他模型規模或 rank 數
下有相同的精度。

[版本化原生驗收矩陣](../../tests/data/research_native_evidence.json)記錄六組
FP32/FP16/BF16 LoRA 與 attention CPU_SIM 案例、CPU 與 GPU 上 tiny Qwen3 MoE 的
路由與梯度，以及兩個 CPU rank 執行原生 DDP 與 FSDP2。實體 TP/PP/CP/EP、多 GPU
ZeRO/FSDP 與多機 collective 不在目前範圍內。重現入口為 `scripts/test.sh cpu` 與
`tests/native/research_distributed.py --help`。

<a id="multi-process-training-on-fakecuda"></a>

## 在 FakeCUDA 上進行多程序訓練

以分散式工作啟動時，FakeCUDA 會把多個真實程序加入同一個 group：`torchrun` 或
`accelerate launch` 的 worker，`mp.spawn` 或 `multiprocessing` 啟動的程序，以及把
`init_method` 或 `store` 傳給 `init_process_group` 的腳本。此時 `init_process_group` 會在
模擬 GPU 背後的 CPU tensor 上加入真實的 gloo group，因此每個 rank 都看到 NCCL
會產生的數值：

- `get_rank()` 與 `get_world_size()` 回傳真實的 rank 與 world size。
  `get_backend()` 仍回報要求的 backend，例如 `nccl`；`get_backend(group)` 回報
  `new_group` 要求的 backend。
- `all_reduce`（包括 `ReduceOp.AVG`）、`broadcast`、`all_gather`、
  `all_gather_into_tensor`、`reduce_scatter`、`reduce_scatter_tensor`、`reduce`、
  `gather`、`scatter`、`all_to_all`、`all_to_all_single`、`barrier`、
  `all_gather_object` 與 `broadcast_object_list` 交換真實數值。對整數 tensor 做
  `ReduceOp.AVG` 時向零取整。與 NCCL 相同，對 CUDA tensor 使用 `BAND`、`BOR`、
  `BXOR` 會拋出 `ValueError`，對 bool tensor 使用 `AVG` 會拋出 `TypeError`，除非
  該 group 要求的是 gloo。
- `DistributedDataParallel` 建立時會廣播 rank 0 的參數與 buffer（設定
  `broadcast_buffers=False` 時不廣播 buffer，設定 `init_sync=False` 時都不廣播）。
  除非設定 `broadcast_buffers=False`，它會在第一次 forward 前、以及每次緊接在
  `no_sync()` 之外的訓練 forward 之後的 forward 前，再廣播一次 rank 0 的 buffer，
  與 DDP 相同。
  backward 結束時於各 rank 之間對梯度取平均；設定 `find_unused_parameters=True`
  時，某個 rank 沒用到的參數在該 rank 上以零計。未設定它或 `static_graph=True` 時，
  沒有得到梯度的可訓練參數會讓下一次 forward 拋出 DDP 的 “Expected to have
  finished reduction in the prior iteration” 錯誤，模擬程序中也是如此。
  在 `no_sync()` 內梯度只在本地累積，與真實 DDP 相同。
- Accelerate 能看到每個程序、它的 rank 與 world size。

其他程序仍使用程序內模擬，collective 只用到本地 tensor：例如只指定 world size 的
單一程序，或在叢集 pod 環境中看到 `WORLD_SIZE` 與 `MASTER_ADDR` 的程序，如此它不會
去等永遠不會啟動的對端。對於 FakeGPU 無法辨識的啟動器（如 DeepSpeed 或 Lightning
的啟動器），設定 `FAKEGPU_DIST_COLLECTIVES=real`；設定
`FAKEGPU_DIST_COLLECTIVES=simulate` 則讓每個程序都使用模擬。跨程序的 FSDP
尚未有測試涵蓋。峰值記憶體不包括真實 GPU 上的 DDP 梯度 bucket 與 NCCL 緩衝區。
若要不修改腳本就在模擬 GPU 上執行啟動器，請使用
[`fakegpu run`](commands.md#run-a-command-on-simulated-gpus)；若 `torchrun` 卡在
rendezvous，見[torchrun 卡在 rendezvous 時](commands.md#torchrun-hangs-at-rendezvous)。

## 拓撲、trace 與頻寬

- `simulate-topology` 模擬 collective 路由與鏈路競爭。
- `replay-trace` 從 PyTorch Profiler、NCCL 風格或 FakeGPU trace 彙整運算、通訊、
  等待與 GPU 記憶體時間軸。
- `bandwidth` 檢查模擬 TCP payload 並回報吞吐量。使用 `--listen` 且不提供
  `--cluster-config` 時，產生的拓撲依每個邏輯節點的平台（`--platform`，預設
  `auto`，見[平台與插槽](platforms.md)）給出 GPU 之間的頻寬：取經過節點內所有 GPU
  的最佳環上最慢一跳的頻寬，因為環形 collective 的速度取決於最慢的一跳。例如 8-GPU
  HGX A100 基板上為每方向 300 GB/s 的 NVLink（2400 Gbit/s），DGX-1 上為 50 GB/s
  的雙 NVLink。
- `coordinator` 管理分散式模擬 coordinator。

分散式耗時包含 coordinator、記憶體複製、socket 與程序排程，不能當作 NCCL、
NVLink 或 RDMA benchmark。

## 實驗批次執行

有限的實驗批次使用明確的 argv 陣列、參數網格與重複次數。各次執行依序進行，並綁定
穩定的執行 ID。每次執行都保留 stdout、退出狀態、錯誤、產物雜湊與數值指標。

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

- `--resume` 跳過已核驗成功的項目。`--resume --retry-failed` 也會重試失敗或中斷
  的項目。
- 輸入、原始碼或環境改變時需要新的工作區。
- 指令須在前景執行。指標路徑支援巢狀 JSON。
- 繪圖需要 Matplotlib。誤差棒是觀測到的最小值與最大值，不代表信賴區間。
- 範例測量的是 CPU 工作，不代表 GPU 吞吐量。
