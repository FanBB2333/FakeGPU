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

[版本化原生驗收矩陣](../../tests/data/research_native_evidence.json)記錄六組
FP32/FP16/BF16 LoRA 與 attention CPU_SIM 案例、CPU 與 GPU 上 tiny Qwen3 MoE 的
路由與梯度，以及兩個 CPU rank 執行原生 DDP 與 FSDP2。實體 TP/PP/CP/EP、多 GPU
ZeRO/FSDP 與多機 collective 不在目前範圍內。重現入口為 `scripts/test.sh cpu` 與
`tests/native/research_distributed.py --help`。

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
