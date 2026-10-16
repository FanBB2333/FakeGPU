# FakeGPU 文档

[返回项目 README](../../README.zh-CN.md) · [English](../en/README.md) ·
[繁體中文](../zh-TW/README.md)

安装方法和快速上手请先看[项目 README](../../README.zh-CN.md)。下面各页讲得更细。

## 使用 FakeGPU

- [命令参考](commands.md)：按用途分组列出所有命令
- [显存分析与估算](estimation.md)：`preflight`、`analyze-repo`、
  `analyze-kernel`、`estimate-llm` 和 `estimate-diffusion`
- [在线 LLM serving](serving.md)：serving 容量、vLLM 预算、speculative
  decoding、混合请求长度和 `benchmark-serving`
- [微调时间与显存](finetuning.md)：面向 HF Trainer、LLaMA-Factory 和 Axolotl
  的配置估算、原生运行、校准和验证
- [Diffusion 训练](diffusion-training.md)：缓存特征 UNet 训练计划和预算
- [分布式规划与实验批量运行](distributed.md)
- [FakeGPU-SMI 与监控](fakegpu-smi.md)：`nvidia-smi` 视图，NVLink、MIG 和故障
  建模，以及 Prometheus 指标
- [原生库拦截](native-interception.md)：运行未修改的程序，处理不支持的 API

## 准确性

- [对照真实 GPU 校准](calibration.md)：采集峰值，给估算设门槛
- [验证与可靠性](validation.md)：已公开的证据、可靠性标签和建模对比

## 项目

- [开发与测试](development.md)：构建、测试、GPU profiles 和项目结构
- [限制](limitations.md)
