# FakeGPU 文件

[返回專案 README](../../README.zh-TW.md) · [English](../en/README.md) ·
[简体中文](../zh-CN/README.md)

安裝方式與快速導覽請先看[專案 README](../../README.zh-TW.md)。以下各頁提供更詳細的說明。

## 使用 FakeGPU

- [指令參考](commands.md)：依任務分組的全部指令
- [記憶體分析與估算](estimation.md)：`preflight`、`analyze-repo`、
  `analyze-kernel`、`estimate-llm` 與 `estimate-diffusion`
- [線上 LLM serving](serving.md)：serving 容量、vLLM 預算、speculative decoding、
  混合請求長度與 `benchmark-serving`
- [微調時間與 GPU 記憶體](finetuning.md)：依設定估算，以及 HF Trainer、
  LLaMA-Factory 與 Axolotl 的原生執行、校準和驗證
- [Diffusion 訓練](diffusion-training.md)：快取特徵 UNet 訓練計畫與預算
- [分散式規劃與實驗批次執行](distributed.md)
- [背景機器與控制台](console.md)：`fakegpu console`、背景機器、事件與 `fakegpu daemon`
- [平台與插槽](platforms.md)：HGX/DGX 基板、Grace 超級晶片、PCIe 插槽與 NVLink 橋，
  以及它們如何決定 PCIe 鏈路、NVLink 與 peer access
- [FakeGPU-SMI 與監控](fakegpu-smi.md)：`nvidia-smi` 檢視、NVLink、MIG、故障建模
  與 Prometheus 指標
- [原生函式庫攔截](native-interception.md)：執行未修改的程式，以及處理不支援的 API

## 準確度

- [對照實體 GPU 校準](calibration.md)：採集峰值並以門檻檢查估算
- [驗證與可靠性](validation.md)：已公開的證據、可靠性標籤與建模比較

## 專案

- [開發與測試](development.md)：建置、測試、GPU profiles 與專案結構
- [限制](limitations.md)
