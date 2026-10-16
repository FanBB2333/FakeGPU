# 对照真实 GPU 校准

[文档索引](README.md) · [English](../en/calibration.md) ·
[繁體中文](../zh-TW/calibration.md)

某个工作负载的估算，只有在同一软件栈上和真实测量对比过之后才值得信任。本页介绍
如何采集这些测量值，以及如何用它们给估算设门槛。已公开的结果汇总在
[验证与可靠性](validation.md)。

## 静态显存验证

在 CUDA 主机上重新执行项目维护的受控 ATen 对比：

```bash
python3 scripts/validation/static_memory_validation.py \
  --output build/static-memory-validation.json \
  --markdown build/static-memory-validation.md \
  --max-underestimate-percent 5
```

在只有 CPU 的主机上加 `--static-only`，只检查估算流程，不产生准确性结果。

<a id="serving-observations"></a>

## Serving 观测

### 手动记录阶段峰值

将 `plan-serving` 报告与多次真实 GPU 阶段峰值对比：

```bash
python3 -m fakegpu calibrate observe-serving \
  build/speculative-serving-plan.json \
  --prefill-peak-bytes 24800000000,24900000000,25000000000 \
  --prefill-peak-bytes 24950000000,25050000000 \
  --decode-peak-bytes 25200000000,25300000000,25250000000 \
  --decode-peak-bytes 25350000000,25400000000 \
  --source torch.cuda.max_memory_reserved \
  --strict \
  --json build/serving-observation.json
```

命令保留全部样本，每个阶段取最大值。默认每阶段至少要五次观测，不足时 `--strict`
返回 1。`ready_for_comparison` 只表示样本数够了，不代表 `GPU-validated`。Plan
必须设置 `--target-profile`。

### 自动采集多次运行

使用框架对应的 runner：

```bash
python3 -m fakegpu calibrate collect-serving \
  build/speculative-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/serving-observation.json \
  -- python3 benchmark_serving.py
```

同构、非 speculative 的 Transformers 工作负载可以用内置 runner。Plan 必须使用
dynamic KV cache：

```bash
python3 -m fakegpu plan-serving \
  --model-dir /models/Qwen/Qwen3-0.6B \
  --active-sequences 4 \
  --max-batch-size 8 \
  --prompt-tokens 2048 \
  --generated-tokens 128 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --kv-cache-strategy dynamic \
  --target-profile a100 \
  --json build/transformers-serving-plan.json

python3 -m fakegpu calibrate collect-serving \
  build/transformers-serving-plan.json \
  --repetitions 5 \
  --timeout-seconds 900 \
  --strict \
  --json build/transformers-serving-observation.json \
  -- python3 -m fakegpu calibrate sample-transformers
```

`sample-transformers` 做了这些事：

- 重新核对本地配置、Safetensors header 与 plan，再测量完整 prefill 和逐 token
  decode。
- 默认报告 `torch.cuda.max_memory_reserved`。只看张量分配峰值时用
  `--metric allocated`。
- 不接受混合请求、speculative decoding、paged 或 quantized cache、共享前缀或
  chunked prefill plan、量化 checkpoint 和 adapter。
- 需要在 CUDA 环境中预先装好 PyTorch 与 Transformers。

### 用自己的测量结果构建样本

vLLM 或自定义测量程序已经有可信的阶段峰值时，可以直接构建样本：

```bash
python3 -m fakegpu calibrate emit-serving-sample \
  build/vllm-serving-plan.json \
  --prefill-peak-bytes 24900000000 \
  --decode-peak-bytes 25300000000 \
  --metric nvml.process_family_peak_bytes \
  --framework vllm \
  --framework-version 0.10.2
```

Python 中对应 `fakegpu.build_cuda_serving_sample(...)`。vLLM 的 cache usage 指标
是 block 使用比例，不是字节峰值，所以 benchmark 需要提供真实的阶段字节数。

### 样本协议

每次重复都在新进程中运行，并且必须输出一个 `fakegpu.serving_peak_sample.v1`
对象。可以放在最后一行，也可以单独一行并加上 `FAKEGPU_SERVING_SAMPLE=` 前缀：

```json
{
  "schema_version": "fakegpu.serving_peak_sample.v1",
  "workload_signature": "<FAKEGPU_SERVING_WORKLOAD_SIGNATURE>",
  "run_index": 1,
  "metric": "torch.cuda.max_memory_reserved",
  "phases": {
    "prefill": {"peak_bytes": 24900000000},
    "decode": {"peak_bytes": 25300000000}
  },
  "environment": {
    "backend": "cuda",
    "simulated": false,
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "gpu_uuid": "GPU-...",
    "compute_capability": [8, 0],
    "total_memory_bytes": 85899345920,
    "software": {
      "framework": "transformers",
      "framework_version": "...",
      "cuda_version": "...",
      "torch_version": "..."
    }
  }
}
```

- FakeGPU 通过 `FAKEGPU_SERVING_*` 环境变量传入 plan 路径、工作负载签名、目标
  profile、compute capability、当前次数和总次数。内置 runner 与协议构建器会自动
  生成这个对象，这些框架也不会因此成为 FakeGPU 的依赖。
- 以下情况会让采集失败：模拟 CUDA、容量超出 profile 2% 容差、compute capability
  不同、软件元数据不一致、超时或非零退出。
- 报告不保存原始命令参数，只记录可执行文件名和命令指纹。
- `simulated: false` 由 runner 提供，不是硬件证明。声明 `GPU-validated` 仍需保留
  日志并通过校准门槛。

## 对比与门槛检查

对任意兼容的预测报告和观测报告执行对比，再做门槛检查：

```bash
python3 -m fakegpu calibrate compare \
  build/prediction.json \
  build/observation.json \
  --json build/calibration-comparison.json

python3 -m fakegpu calibrate verify \
  build/calibration-comparison.json \
  --max-underestimate-percent 5 \
  --max-absolute-percentage-error-percent 5 \
  --min-interval-coverage-percent 90 \
  --capacity-bytes 25769803776 \
  --json build/calibration-verification.json
```

`compare` 报告各阶段的有符号误差和绝对误差、区间覆盖率和建议的安全余量。`verify`
在任一门槛未通过时返回 1。门槛包括最大低估、误差中位数/95 分位数/最大值、区间
覆盖率、指定容量下的 false-safe 判断，以及工作负载一致性。结果只适用于相同的
工作负载签名、shape、dtype、软件栈和 GPU profile。

## 阶段显存校准

Allocated 张量存储与 allocator reserved 显存分开统计。`calibrate phase-sample`
在一个全新 CUDA 进程中同时采集两者。`phase-estimate` 用至少五个进程封存各阶段
最大值，`phase-validate` 对照之后的一次独立运行。每份样本和预算都用新的输出路径。

```bash
# plan.json: a compatible local plan-serving or estimate-diffusion report
for i in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
    --json "/tmp/phase-cal-$i.json"
done
python -m fakegpu calibrate phase-estimate \
  --calibration-sample /tmp/phase-cal-1.json \
  --calibration-sample /tmp/phase-cal-2.json \
  --calibration-sample /tmp/phase-cal-3.json \
  --calibration-sample /tmp/phase-cal-4.json \
  --calibration-sample /tmp/phase-cal-5.json --json /tmp/phase-budget.json.gz
CUDA_VISIBLE_DEVICES=1 python -m fakegpu calibrate phase-sample plan.json \
  --json /tmp/phase-holdout.json
python -m fakegpu calibrate phase-validate --budget /tmp/phase-budget.json.gz \
  --holdout-sample /tmp/phase-holdout.json --max-absolute-error-percent 20 \
  --max-underestimate-percent 10 --json /tmp/phase-validation.json
```

- 采集支持同构 Transformers dynamic-cache serving，以及本地 SD/SDXL 的 eager 或
  SDPA attention、attention slicing、VAE slicing、VAE tiling 及其组合。不支持
  offload。
- Tiling 时，`--vae-tile-size` 同时设置原生 sample 与 latent tile 上限。实际
  processor、VAE 开关和 tile 上限都纳入原生身份。
- 计划、所选权重和配置、GPU UUID、Torch/CUDA/cuDNN 版本、allocator 与实现必须
  一致。不做 shape 迁移，也不加隐式安全系数。kernel cache 不受控。
- 默认门槛为绝对误差 20%、低估 10%，未通过时返回 1。离线估算和验证只依赖标准库。

目前的证据都是 experimental，只适用于已测试的 GPU、软件和固定输入：

- [阶段显存验证](../../tests/data/phase_memory_evidence.json)：补上 VAE 上采样卷积
  后，四种分辨率的 VAE allocated 误差为 0.045%–1.172%。六个条件各五次校准、五次
  holdout（共 60 个进程）都通过了两项门槛。
- [扩展显存验证](../../tests/data/memory_generalization_evidence.json)：25 个单卡
  条件、74 个进程，覆盖 SD 1.4/SDXL、Qwen3-0.6B/1.7B/Qwen2.5-0.5B、
  FP16/BF16/FP32、不同 batch、矩形图像、guidance、eager attention、slicing 与
  tiling。allocated 最大绝对误差 10.93%，最大低估 3.58%（门槛 20%/10%）。精确
  匹配条件的预算与全部十二次 holdout 吻合，68 次跨条件复用被拒绝。这些不算
  `GPU-validated` 数据，因为 21 个条件不足五次观测，优化模式用的是归档的原生
  探针，没有走公开 CLI。
- [公开优化模式验证](../../tests/data/optimized_phase_memory_evidence.json)：七个
  SD/SDXL 条件、70 个进程。210 项 holdout 比较都与校准最大值一致，解析式
  allocated 最大绝对误差 6.69%，最大低估 3.38%。42 次跨模式和 21 次实际模式
  身份不匹配被拒绝。

这些结果都不能说明可以迁移到新 shape、具备预测区间或物理上不会 OOM。
