# 命令参考

[文档索引](README.md) · [English](../en/commands.md) ·
[繁體中文](../zh-TW/commands.md)

所有命令都以 `python3 -m fakegpu <command>` 的形式运行。用
`python3 -m fakegpu --help` 查看完整列表，用
`python3 -m fakegpu <command> --help` 查看某个命令的选项。凡是接受 JSON 路径的
地方，大型报告和计划都可以用 `.json.gz`。压缩输出采用原子写入，字节可复现。

## 环境与运行

| 命令 | 用途 |
|---|---|
| `fakegpu doctor` | 检查安装、原生库、PyTorch 和 profiles |
| `fakegpu demo` | 在 CPU 上执行一个小型训练步骤，程序能看到 CUDA 设备 |
| `fakegpu preflight` | 将工作负载执行到指定阶段，判断能放下还是会 OOM |
| `fakegpu capabilities` | 列出或严格检查原生 API 分类 |
| `fakegpu validate` | 执行 JSON、TOML 或 YAML 声明式验证矩阵 |

## 分析与估算

| 命令 | 用途 |
|---|---|
| `fakegpu analyze-repo` | 统计仓库入口和仅支持 GPU 的风险 |
| `fakegpu analyze-kernel` | 检查 CUDA、PTX 和 SASS 资源与运算 |
| `fakegpu estimate-llm` | 估算 decoder 显存、通信量和 FLOP |
| `fakegpu estimate-diffusion` | 估算 diffusion 的 text encode、denoise 和 VAE decode 阶段显存 |
| `fakegpu estimate-roofline` | 生成与 profile 相关的分析延迟区间 |
| `fakegpu workspace-profiles` | 验证并查看 workspace 估算 profiles |

## Serving

| 命令 | 用途 |
|---|---|
| `fakegpu plan-serving` | 规划同构或混合请求准入、chunked prefill 和分组共享 prefix KV 存储 |
| `fakegpu benchmark-serving` | 按计划发送混合流式请求，保留用量、延迟、SSE 事件和失败记录 |

## 微调

| 命令 | 用途 |
|---|---|
| `fakegpu estimate-finetune` | 根据框架配置估算单卡全量/LoRA/QLoRA 的步数、时间与显存 |
| `fakegpu probe-finetune` | 在真实 CUDA 显卡上采集固定形状 Hugging Face Trainer 的时间与显存 |
| `fakegpu profile-finetune-data` | 扫描本地 JSONL 的 token 长度、截断、padding 与处理后序列数 |
| `fakegpu run-finetune` | 执行 HF Trainer/PEFT、LLaMA-Factory 或 Axolotl 原生训练，记录数据、CUDA 时间、显存及最终保存 |
| `fakegpu validate-finetune-windows` | 用独立的校准与 holdout 运行验证动态窗口耗时 |
| `fakegpu validate-finetune-events` | 用独立的原生事件记录验证评估与 checkpoint 成本 |
| `fakegpu plan-finetune-pipeline` | 从已校准的顺序数据推导未来原生训练轮次或更新数 |
| `fakegpu estimate-finetune-pipeline` | 估算完整原生执行时间和已覆盖的显存包络 |
| `fakegpu validate-finetune-pipeline` | 对照独立原生运行验证完整时间和显存预测 |
| `fakegpu plan-finetune-transfer` | 规划经过核验的随机或顺序原生输入 |
| `fakegpu estimate-finetune-transfer` | 估算显式的经验形状迁移预算 |
| `fakegpu validate-finetune-transfer` | 对照独立原生运行验证迁移预算 |
| `fakegpu calibrate-finetune` | 在可恢复的工作区中运行有限额的原生校准 |
| `fakegpu budget-finetune-workspace` | 只用 CPU 对保存的原生证据重新预算 |

## Diffusion 训练

| 命令 | 用途 |
|---|---|
| `fakegpu plan-diffusion-training` | 查看缓存特征 UNet 训练计划和持久存储 |
| `fakegpu run-diffusion-training` | 在一张真实 GPU 上执行缓存特征 UNet full/LoRA 训练 |
| `fakegpu estimate-diffusion-training` | 用兼容的原生缓存特征训练观测做预算 |
| `fakegpu validate-diffusion-training` | 对照独立的原生 diffusion 训练 holdout |

## 分布式与实验

| 命令 | 用途 |
|---|---|
| `fakegpu plan-training` | 统一分布式训练配置并估算单 rank 显存 |
| `fakegpu simulate-topology` | 模拟 collective 路由和链路竞争 |
| `fakegpu replay-trace` | 汇总计算、通信、等待和显存时间线 |
| `fakegpu coordinator` | 管理分布式模拟 coordinator |
| `fakegpu bandwidth` | 验证模拟 TCP payload 并报告吞吐量 |
| `fakegpu plan-experiments` | 展开有限额的参数网格并检查实际命令 |
| `fakegpu run-experiments` | 运行或显式恢复实验，保留每次尝试 |
| `fakegpu compare-experiments` | 核验原始产物，比较重复观测，可选导出图表 |

## 校准与监控

| 命令 | 用途 |
|---|---|
| `fakegpu calibrate` | 整理 serving 观测、对比显存报告并执行可靠性门槛检查 |
| `fakegpu nvidia-smi` | 查看设备、进程、建模拓扑/MIG、健康状态和可靠性事件 |
| `fakegpu nvitop` | 通过原生 NVML 桥接，用已安装的 nvitop 查看本机 FakeGPU 状态（macOS/Linux） |
| `fakegpu console` | 打开后台机器的终端控制台：GPU、进程、事件和设置（[说明](console.md)） |
| `fakegpu daemon` | 在后台运行一台模拟 GPU 机器，FakeCUDA 进程会自动接入；支持启动、停止、状态、事件及配置导入导出（[说明](console.md)） |
| `fakegpu platform` | 列出主机平台、显示设备链路，并解释拷贝路径、瓶颈和建模耗时（[说明](platforms.md)） |
| `fakegpu metrics` | 导出有数量上限的 Prometheus/JSON 指标，并提供短期内存历史 |
