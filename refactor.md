# FakeGPU 重构分析报告

> 基于对全仓库的静态扫描（2026-08-16，main @ 20a95f7）。本文档只列问题与方向，不包含具体改动；所有条目在动手前应先在当前代码上复核 file:line 是否仍然成立。
> **约束：重构后功能保持不变。** 文末单独列出了扫描中顺带发现的疑似 bug——它们的修复会改变行为，应与重构分开提交。

## 实施进度（滚动更新）

**2026-08-16 第一批落地**：§8 建议顺序中的第 2 步（纯删除）全部完成，第 3 步（公共设施）完成两项，第 6 步（性能）完成两项。净删约 **2,500 行**（+250/−2,700 量级），功能不变。

| Commit | 阶段 | 内容 |
|---|---|---|
| `e451bd2` | §1.1 | 删除 torch_patch standalone 不可达路径（约 50 个定义、~1,200 行）；`_activate_upstream` 改为显式 raise；`_runtime` 恒真探测简化；顺带删除与 upstream 逐字相同的 no-op 补丁 |
| `bc77d9f` | §1.2 | 删除零消费者的 `fsdp_memory.py`（970 行；可从 git 历史恢复） |
| `6b00725` | §1.3–1.7 | 恒 None 遥测链整链删除（含 `native_smi.cpp`）；死状态字段（`memory_utilization_percent`、`reserved_stage_peaks`）；死 env 导出与 `efficiency` 死参数；各处局部死分支；`_preflight_bootstrap` 冗余兜底 |
| `432c21c` | §2.2 | 同体重复 except、不可能失败的 try、恒真 hasattr、setup.py 死 wheel 兜底、17 处"子类⊂基类"冗余异常项 |
| `91243b8` | §5.2 | workspace catalog 每 FX 节点重载 → 每图一次；torch 版本串 lru_cache |
| `f6891c8` | §3.4+§5.4 | `structured_io.emit_json` 收敛 8 处 print-or-write `--json` 块并消除双重序列化；dtype 表与 `_iter_tensor_leaves` 各去重一份 |

**验证矩阵**：每阶段本地全量 `pytest`（222 passed）+ ruff；`scripts/test.sh smoke` 完整通过（含 native C++ 重建，覆盖 `native_smi.cpp` 改动）；CI 同款 manifest 校验（7+39 cases）通过；远端真机单测——**406**（Python 3.12 + torch 2.9.1+cu128，`~/refactor-test/FakeGPU`）与 **gem12**（torch 2.12.1+cu130）均 221 passed + 1 skipped（skip 为 CPU-only 构建专属测试，预期）。

**2026-08-17 第二批落地**：§8 第 3 步（公共设施）剩余项之一完成。

| Commit | 阶段 | 内容 |
|---|---|---|
| `a9da7f8` | §3.3 | 新增 `_api._FakeGpuRuntimeConfig`（frozen dataclass，10 个 FAKEGPU_* 配置字段）收敛 `_api.py`/`_runtime.py` 内部转发；`init()`/`env()`/`run()` 三个公开入口的关键字签名保持逐字不变（外部调用方零感知），仅内部把 10 个散装 kwarg 换成一个 `config` 对象；新增 `_build_env()` 消除 `run()`→`env()` 的整段 10-kwarg 转发；`_apply_env_inplace`/`_apply_config_env_inplace`/`_apply_config_env`/`_runtime._init_native_runtime` 四处签名同步收敛。净减约 **77 行**（220 处/− 变化集中在 `_api.py`），行为不变 |

验证：本地全量 `pytest`（222 passed，与改动前一致）+ ruff clean；CI 同款 manifest 校验（7+39 cases）通过；`scripts/test.sh smoke`（含 native 重建）通过；额外手工冒烟——`fakegpu.env()`/`fakegpu.init(runtime="native")`/`fakegpu.run()` 三个公开入口直接调用验证派生的 `FAKEGPU_*` 环境变量、`device_count` 校验报错、子进程环境注入均与改动前一致。远端 406/gem12 真机复测见第一批的验证矩阵说明；`f6891c8`（emit_json，改动了 CLI `--json` 输出代码路径）与 `a9da7f8` 已于 2026-08-17 在两台机器上复跑通过（各 221 passed + 1 skipped）。

**2026-08-17 §7 bug 批量修复**：12 项中 10 项确认为真实缺陷并修复（各一个 `fix:` commit），1 项（#6）在修复过程中发现是**未完成且有 bug 的半成品**（假设了 catalog schema 里不存在的 `"group"` 字段，`native_capability_report()` 一调用就 `KeyError`）——已改正为正确实现；1 项（#12）复核后判定**不是 bug，未改**。

| Commit | Bug # | 内容 |
|---|---|---|
| `1a7a337` | #1 | rank 正则去掉多余转义，ShardMetadata placement 重写恢复生效 |
| `3ab2fb8` | #2 | 释放回调改为绑定分配时的 tracker 实例（storage finalizer + saved-tensor 两处），二次 `patch()` 后不再向新 tracker 误释放 |
| `14a9f5f` | #3 | `successful_writes` 改为写入前预增、失败回滚，去掉序列化里的 `+1` 特例 |
| `d519e97` | #4/#5 | `_evaluate_json_check` 捕获元组补 `IndexError`；`finished_at_ns`/`duration_seconds` 改为同一时钟读数 |
| `2f483f5` | #6 | 过滤下沉到 `native_capability_report()`，在算 summary **之前**过滤 apis（并修正了 WIP 里假设不存在的 `"group"` 关联字段导致的 KeyError）；`group_count` 保持目录级、不随 API 过滤变化（匹配既有测试 `== 5`） |
| `6a87c50` | #7 | 超时 rank 仍读取已写出的诊断 JSON；顺序 join 改为共享 deadline，避免 N 个慢 rank 把超时叠加放大 |
| `a270e59` | #8 | 文本编码器均未暴露宽度且 denoiser 也未声明时改为 raise（而非静默取 1）；真正无文本编码器角色的管线不受影响 |
| `19e1999` | #9 | `_bool_value` 只保留已通过测试验证的 `"auto"`（DeepSpeed 真实哨兵值，**不是 bug，保留**）；仅对词表外的未知字符串收紧为 raise，不再默认 True |
| `97dc3f6` | #10 | `assert` 改为显式 `if...raise`，避免 `python -O` 下失效 |
| `4a3ae3b` | #11 | `_process_name` 只取入口脚本 basename，不再拼接可能含路径/密钥的后续 argv |

**#12 复核结论（未改）**：`_validate_collector_environment_contract` 只在校验的 `FAKEGPU_SERVING_*` 环境变量已设置时才比对，起初怀疑"未设置应报错"。但 `build_cuda_serving_sample` 是文档化的公开 API（README "The same adapter is available as `fakegpu.build_cuda_serving_sample(...)` for use inside a Python benchmark"），供 vLLM/自定义 runner **在 FakeGPU 自身采集子进程之外直接调用**；`tests/test_analysis_extensions.py:757` 就是在这些环境变量全部未设置的情况下调用并断言成功。若改成"未设置即报错"会直接打破这个文档化用例和对应测试。当前"仅在存在时比对"是该函数**双重用途设计下的正确行为**，不是缺陷，保留不动。

验证：本轮改动后本地全量 `pytest`（222 passed）+ ruff clean + CI 同款 manifest 校验（7+39 cases）通过；`_process_name`/`native_capability_report` 过滤逻辑另有手工冒烟验证。**本轮未做远端 406/gem12 复测**——多数改动是纯逻辑修正（正则、计数器时序、assert→raise、字符串校验）或范围明确的行为收紧，本地+CI 已提供等价确认；如需更保险可再补一次远端跑。

**复核后有意不改的条目**（与文档原建议不同处）：
- `_aggregate_report` 的 error 分支（§1.5/2.3）：rc=0 但报告为 error 的边缘情况下可达，保留。
- `_stage.py` 重复 env 赋值（§2.2）：嵌套 stage 退出时有恢复语义，保留。
- `_format_bytes` 11 份（§3.4）：实测 6 份实现 5 种输出格式（负数/None/整除格式化各不同），全量合并会改 CLI 输出，仅 demo/doctor 两份本来相同，不值得为此建共享模块。**放弃**。
- `_percentile` 2 份（§3.4）：calibration 是线性插值、bandwidth_worker 是就近排名，**不同算法**，不可合并。
- `_ceil_div` 3 份（§3.4）：单行惯用式，零漂移风险，合并收益不抵跨模块依赖，保留。
- `--include-exited`（smi/metrics）、`distributed_cli` 的 `--markdown-report` 等：有真实工作路径，属缺测试而非死代码，保留。
- smi 发布的 state JSON 不再含恒 null 的 `telemetry.*`、`topology.numa_node/pcie_generation` 键：死数据移除，`typical_power_usage_mw` 等真实值字段保留。

**待决策项**（需要项目层面拍板，未动）：`privateuse1/`（866 行）、`capabilities.audit_native_exports`（约 150 行）、`flop_counter.MatmulFlopCounterMode`（88 行，公开导出）、`schemas/*.schema.json`（三个孤立 schema）、legacy schema 支持是否还需要（`test_virtual_smi.py` 是唯一消费者）。

---

## 0. 规模总览

| 部分 | 规模 |
|---|---|
| `fakegpu/`（主包） | 约 37,100 行 Python（含 `privateuse1/` 866 行） |
| `src/`（C/C++ stubs） | 约 34,600 行 |
| `tests/` | 约 9,200 行 |
| 最大的文件 | `smi.py` 4353 行、`torch_patch.py` 4346 行、`serving_plan.py` 3722 行、`calibration.py` 2873 行、`diffusion_estimator.py` 2327 行 |

ruff（F401/F811/F841）检查是干净的——没有未使用的 import / 变量这类低级冗余。真正的体积问题集中在四类：**成块的不可达/无消费者代码**、**大面积复制粘贴**、**多层重复校验与兜底**、**超大文件/函数**。粗略估计，前两类合计可以删掉或合并 **4,000–5,000 行** 而不改变任何可观察行为。

---

## 1. 角度一：删除死代码与不可达路径（收益最大，风险最低）

### 1.1 `torch_patch.py` 的 standalone fallback 路径（约 900 行，不可达）

`_activate_upstream`（`torch_patch.py:3333-3384`）只有在 `import torch.fakegpu` **和** `from . import _upstream` 同时失败时才返回 `None`。而 `_upstream.py` 是随包内置（vendored）的模块，正常安装下必然可导入，因此：

- `patch()` 的 standalone 分支（`torch_patch.py:3857-4341`）在正常安装下永远走不到；
- 只被该分支引用的约 43 个 `_stub_*` 函数（`:2354-2691`）、standalone 版 `_FakeStream`/`_FakeEvent`/`_FakeStreamCtx`/`_FakeDeviceProperties`（`:2207-2346, 2699-2736`）、`_patched_tensor_to`/`_patched_module_cuda` 等（`:2056-2199`）随之全部变成死代码；
- 例外：`_stub_is_bf16_supported`、`_stub_get/set_rng_state*`、`_normalize_device_index` 也被 upstream 路径使用，需保留。

同类问题：`_runtime.py:108-136` 的 `_detect_custom_torch_fakegpu_available()` 末尾 `find_spec("fakegpu._upstream") is not None` 恒为 True，导致 `init(runtime="auto")` 永远选不到 `"native"`，前面的 `sys.path` 扫描与 `torch.fakegpu` 探测（`:109-127`）全部不可达。`tests/test_runtime_init.py:71-82` 只在 mock 掉该函数的情况下测试，所以测试抓不到。

**处置建议**：要么整体删除 standalone 分支并同步修改模块 docstring（`torch_patch.py:16-19`），要么用显式环境变量把它变成可达、可测的路径——现在它既不可达也无覆盖，是最差的状态。

### 1.2 `fsdp_memory.py` 整个模块无消费者（970 行）

4 个公开函数（`build_full_shard_plan:7`、`build_fully_shard_plan:74`、`estimate_full_shard_sft_memory:461`、`estimate_fully_shard_sft_memory:598`）在 `fakegpu/`、`tests/`、`scripts/`、README、`__init__.py` 中**零引用**。docstring（`:467`）声明的唯一消费者 `qwen_sft_memory_worker.py` 已不在仓库里。git 历史显示它曾被积极维护（CHANGELOG 记录过 real-GPU 验证），属于"消费者被删掉后遗留"而非从未用过。

**处置建议**：先确认是否有仓库外的消费者；没有则删除，有则补 `__init__.py` 导出 + 测试。二者取一，不要维持现状。

### 1.3 永远为 `None` 的遥测子系统（贯穿 5 层）

`gpu_utilization_percent` / `temperature_c` / `fan_speed_percent` / `power_usage_mw` 从未被写入非 None 值（Python 侧 `smi.py:1391-1397`，C++ 侧 `src/core/native_smi.cpp:874-879` 均硬编码 `None`），却被 `smi.py:3308-3320` setdefault、`:153-173` 暴露为查询字段、`:2881-2888` 渲染，`_format_percent`/`_format_temperature`（`:4264-4279`）几乎专为它们存在。同类："永远是 None"的 `topology.numa_node`/`pcie_generation`（`smi.py:578-579, 3781-3782` 及查询/渲染点）。

**处置建议**：如果近期没有实现真实遥测的计划，整链删除；否则至少收敛到一处 `TELEMETRY_FIELDS` 表，避免 5 处手写 None。

### 1.4 写了没人读的状态字段（每 250ms 白写一次）

- `reserved_stage_peaks`：`smi.py:3087` 初始化、`:3162-3165` 累加，无任何读取方。
- `memory_utilization_percent`（`smi.py:1353-1357`）：读取方自己重算（`:3367-3369`）；同类还有 `free_memory`、`headroom_bytes/percent`、`identity_source`（恒为 `"synthetic"`，`:1331`）。
- `trace_replay.py:452-453` 的 `event["memory_bytes"]` 写后无读；`:488-489` 的 `excluded_from_aggregate`/`exclusion_reason` 打上标记后立刻在 `:490-491` 被过滤丢弃，整个复制遍历可以简化为一个 filter。

### 1.5 无消费者的 CLI 参数与环境变量

- `preflight --steps`（`preflight.py:69`）导出的 `FAKEGPU_PREFLIGHT_STEPS` 与 `FAKEGPU_PREFLIGHT_TARGET_STAGE`（`:639-643`）全仓库无读取方，是完整的 no-op。
- `FAKEGPU_PREFLIGHT_MEMORY_SAFETY_FACTOR`/`_MARGIN`（`preflight.py:839, 853`）只有读取处，无任何写入/文档/测试。
- `distributed_cli.py` 的 `--markdown-report`、`--cluster-markdown-report`、`--interconnect-*`、`--ranks-per-node` 无文档无测试，且部分组合下被静默忽略（`:706-718`）；`_aggregate_report` 的 error 分支（`:499`）因上游先 raise 而不可达，`_print_bandwidth_summary` 恒打印 PASS（`:555`）。
- `smi.py --include-exited`、`metrics.py --include-exited` 无任何测试或脚本使用。
- `performance_model.py` 的 `efficiency` 参数（`:98`）无任何调用方传入，其约 36 行 override 校验（`:298-333`）全部是死代码。

### 1.6 只导出不使用、或只在模块内自用的公开 API

- `MatmulFlopCounterMode`（`flop_counter.py`，88 行整个模块）：`__init__.py` 导出，但无内部调用、无测试。
- `capabilities.audit_native_exports`（`capabilities.py:222-328` + 6 个 helper，约 150 行）：仅 `--build-dir` 一条路径可达，无测试无 CI 调用。
- `privateuse1/` 子包（866 行）：仅经 `init_privateuse1` 暴露，测试只断言 callable（`tests/test_runtime_init.py:54`），从未真正执行——属于实验性平行后端，需要决定去留。
- `validation.expand_validation_cases`/`render_validation_markdown`、`workspace_profiles.default_workspace_profile_path`、`profile_catalog.CatalogValidation`/`profile_directory` 等公开命名但仅模块内使用；`calibration.py`/`repository_analyzer.py` 的 `__all__` 与 `__init__.py` 实际导出不一致——建议统一"什么算公开 API"的口径，缩小公开面。

### 1.7 局部死分支（小而明确）

- `diffusion_estimator.py:1416-1429`：`StableDiffusion` 分支与 fallback 返回完全相同的 dict。
- `serving_plan.py:3150-3164`、`:1796-1806`、`llm_estimator.py:862-864`：结构上不可能触发的防御性 raise。
- `kernel_analysis.py:150-152`：左侧 opcode 集合被右侧正则完全覆盖；`:287` 的 `limiting = []` 立即被重新赋值。
- `torch_patch.py:78-81` `_PROFILE_SUPPORTED_TYPES` 导入时构建、从未读取；`:2690` `_stub_cudart` 从未绑定；`_orig_tensor_cuda`/`_orig_tensor_pin_memory`/`_orig_torch_compile`（`:2051-2053` 等）被保存"以便恢复"但不存在任何 unpatch 路径。
- `_preflight_bootstrap.py:127-158`：32 行 fallback 伸手进 `torch_patch` 私有属性，重新推导公开函数 `memory_snapshot()` 已经返回的内容。
- `schemas/*.schema.json` 三个 JSON Schema 完全孤立——无代码、无 MANIFEST.in、无 CI 引用，而 `validation.py:278-318` 手写了一份更弱且已漂移的校验。要么让代码真正使用 schema，要么删掉。

---

## 2. 角度二：收敛兜底/防御逻辑（对可读性影响最大）

全包 61 处宽 `except`，其中 `torch_patch.py` 独占 60 个 `try`、39 个 `except Exception`、27 个 `hasattr`。原则建议：**版本兼容性 guard 保留并注释清楚；"不可能失败"的 guard 删除；会吞掉真实故障的宽捕获收窄并留诊断。**

### 2.1 应当保留的（真实版本/平台差异）

`torch.distributed.fsdp` 私有模块 import guard（`torch_patch.py:3205-3258`）、`torch._C._cuda_*` 的 CUDA/CPU 构建差异探测（`:2812, 4104-4128`）、`allow_tf32` 的 property 分叉（`:2840-2853`）、`hasattr(os, "RTLD_NOW")`（`_api.py:168`）等。这类保留，但建议统一加一行"为哪个 torch 版本/构建差异服务"的注释，防止未来被误删或被继续模仿。

### 2.2 应当删除的（守护不可能发生的条件）

- 完全相同的重复 handler：`torch_patch.py:1372-1375` 与 `:1603-1608`，窄捕获后紧跟同体宽捕获。
- 不可能抛异常的操作外套 try：`getattr(x, k, default)`（`:1684-1690`）、`os.path.dirname`（`:1461-1469`）、`torch.nn.Module` 属性访问（`:1808-1812, 1909-1912`）、已在 `patch()` 里无条件 import 过的子模块再次 try-import（`:3642-3662, 3743-3751, 1536-1539`）。
- 在声明支持窗口（torch 2.6–2.11）内恒为 True 的 `hasattr`：`torch.compile`（`:2130`）、`torch.amp.autocast`（`:3299, 3329`）、对自带 vendored 模块 `_upstream` 的 4 次属性探测（`:3358-3374, 3808-3810`）。
- `_stage.py:26, 30` 两次无条件重复赋同一个环境变量。
- `setup.py:12-15, 114-130` 对 `wheel` 的 try/except——`pyproject.toml` 已在 build-system.requires 里声明它。
- 各 CLI `except` 元组里"子类 ⊂ 基类"的冗余项：`FileNotFoundError ⊂ OSError`、`json.JSONDecodeError ⊂ ValueError`、自定义 `*Error ⊂ ValueError`（`serving_plan.py:2202-2208`、`training_plan.py:431-437`、`topology.py:483-489`、`capabilities.py:365-370`、`smi.py:1862` 等 6+ 处）。副作用：所有命令实际都在裸捕 `ValueError`，内部 bug 会被当作用户输入错误报出。

### 2.3 应当收窄的（宽捕获掩盖真实故障）

- `torch_patch.py:3350-3356`：vendored 模块 import 失败被静默降级到（不可达的）standalone 路径——打包坏了却无任何告警；`:3344-3347` 应只捕 `ImportError`，否则损坏的 custom-torch 构建与"未安装"不可区分。
- `torch_patch.py:1759-1776, 1794-1804`：saved-tensor 追踪安装/恢复失败被吞，激活内存追踪静默关闭且 hooks 永久丢失。
- `workspace_profiles.py:371-379`：`import torch` 失败退化成空字符串**作为 profile 匹配键**参与 fnmatch——一次 import 失败静默改变匹配结果；`:363-368` 把 `get_profile` 的一切失败折叠成"unknown target GPU profile"。
- `repository_analyzer.py:558-564, 600-605`：损坏的 `pyproject.toml` 静默视为"无依赖/无入口"，最终却报告 `static_analysis_complete: True`；`:326-358` git 失败静默退化为 `os.walk`（ignore 策略不同，文件清单会变）。
- `smi.py:1495-1500`：publisher 主循环裸捕后 `continue`，状态文件持续超限会永远失败而无信号；`:1226-1229` `stop()` 吞掉最终写入失败，进程可能永远显示 running。
- `smi.py:4019-4053`：坏掉的 profile catalog（`ProfileCatalogError` 是 `ValueError`）被 `_catalog_metadata` 折叠为 `{}`，所有输出静默降级为 N/A。
- `calibration.py:2548-2561`：定位 sample 的第四层 fallback 会捡起任意一行日志 JSON，随后在下游报出令人困惑的 schema 错误，而不是"未找到 sample"。
- `demo.py:101-155`：54 行主体套一个 `except Exception` 折叠为 exit 2，torch import 失败与算术错误不可区分。
- `metrics.py:1207-1218` `_number` 把负数/NaN 一律钳到 0（时钟回拨产生的负 age 变成 0，无告警）。

### 2.4 重复出现的兜底"模式"应工厂化

- "patch 某属性、防重复打补丁"六连块（`torch_patch.py:1813-1941, 3260-3294`）→ 一个 `once_patched(target, name)` 装饰器，约省 90 行。
- "invalid device ordinal" raise 有 11 份拷贝、3 种消息格式（`:1991-4028` 各处）→ 一个 `_require_valid_device_index()`，顺带统一报错文案。
- "`_memory_tracker` 非 None 则委托否则默认值"10 连块（`:2443-2506`）→ 表驱动。
- 多层 Optional 传播链（`calibration.py` `_first_integer→_memory_points→_canonical_point→_match_points` 四层 None/空集合接力才 raise 一个错误；`workspace_profiles` 同型）→ 在边界一次性 raise，中间层不再传 Optional。

### 2.5 同一输入被 3–4 层重复校验

一次 `estimate_serving_plan` 调用中，`prompt_tokens` 等参数在 `serving_plan.py:570-573`、`:62-65`、`llm_estimator.py:66-82` 各验一遍；`kv_cache_strategy` 验 4 遍（argparse choices + 三层函数），且每层抛不同的异常类型，还都在二分搜索内部重复执行。`training_plan.py:48-373` 对 `normalize_training_config` 已经归一化过的输出再套一层 `x or default` 链（`:84-358` 多处）。`_normalize_serving_requests`（108 行）对同一份数据在 `serving_plan.py:1027, 2648, 2670, 2693` 反复全量归一化。`build_cuda_serving_sample` 甚至校验自己刚构造的输出（`calibration.py:399-414`）。写侧/读侧独立重算同一派生值的还有 `smi.py` 的 MIG mode、health status、NVLink 汇总（写 `:1091-1097` 等 vs 读 `:3631-3715` 等）。

**方向**：确立"入口边界校验一次，内部信任"的约定；内部函数用注释或类型标注声明前置条件，删除重复层。

---

## 3. 角度三：消除重复（第二大删减来源）

### 3.1 `torch_patch.py` 对 `_upstream.py` 的成块复刻

- `_patched_copy_collective_tensor`（`torch_patch.py:3041-3063`）与 `_upstream.py:483-499` 语义逐字相同——整个补丁是 no-op，直接删。
- `_patched_dist_init/destroy_process_group`（`:3135-3181`）复刻 `_upstream.py:381-438`，仅状态读取方式不同。
- 12 个 `_tracked_*` 内存闭包在 upstream 路径（`:3581-3662`）与 standalone 路径（`:3909-3981`）写了两遍（约 150 行）；跨设备 wrapping 块两遍（`:3682-3736` vs `:4191-4249`，其中 `_BINARY_DUNDERS` 列表实际有三份）；`_FakeGradScaler` 两遍（`:3669-3677` vs `:4293-4301`）。若按 1.1 删除 standalone 路径，这些自动消失。
- `torch_patch` 影子实现了 `_upstream` 已有的 `_FakeStream`/`_FakeEvent`/`_FakeDeviceProperties`/`_normalize_device_index`/`_install_legacy_cuda_types`/torch_load 处理等 8 组类与函数——统一改为复用。

### 3.2 `serving_plan.py`：两条主路径 82% 逐行相同

- `estimate_serving_plan`（`:537-993`，457 行）与 `estimate_serving_request_set`（`:996-1494`，499 行）经机械 diff 有 **375 行逐字相同**（capacity 解析、vLLM 分支、limiting_factor 九连 if、报告尾部全部一致）。
- `_serving_batch_memory`（`:2276-2626`）与 `_serving_request_set_memory`（`:2629-3083`）64% 相同。
- 三份 token 级相同的二分搜索（`:3086-3140`）→ 一个 `_largest_fitting(limit, predicate)`。

**方向**：抽出共享的 `_build_serving_report(...)`，两个入口收敛为薄封装。这是全仓库杠杆最大的单项重构（涉及约 1760 行）。

### 3.3 `_api.py`/`_runtime.py`：10 参数配置签名写了 8 遍

`init`/`env`/`run`/`_apply_env*` 加 `_runtime` 两处，共 8 处完整签名 + 6 处逐参转发调用，约 200 行纯转发代码。引入一个 frozen 的 `FakeGpuConfig` dataclass 作为单参数即可全部收敛；`_apply_config_env_inplace` 是 `_apply_env_inplace` 的严格子集，合并为带 flag 的一个函数。

### 3.4 全包级公共 helper 缺失（建议建立 `_common`/扩展 `structured_io`）

| 重复项 | 份数 | 位置举例 |
|---|---|---|
| `_format_bytes` 字节格式化 | **11** | `doctor.py:21` 与 `demo.py:13` 逐字节相同；`serving_plan.py:3717`、`llm_cli.py:141`、`smi.py:4244`、`torch_patch.py:1203`、`preflight.py:1346`、`distributed_cli.py:543`、`diffusion_estimator.py:2280`、2 个 scripts |
| JSON/TOML/YAML 结构化读取 | 3 | `structured_io.load_mapping`、`validation.py:617`、`workspace_profiles.py:188`（各带一份"install fakegpu[validation]"提示） |
| `--json` 输出块（含双重序列化 bug，见 §7） | 8+ | `calibration.py` 一个函数内 4 份、`trace_replay.py:170`、`kernel_analysis.py:344`、`repository_analyzer.py:287` 等 |
| `write_json` 内联复刻 | 7 | `llm_cli.py:105`、`distributed_cli.py:860`、`preflight.py:160`、`serving_plan.py:2210`、`performance_model.py:254` 等（`structured_io.write_json` 已存在） |
| dtype→bytes 表 | 2（逐字节相同） | `llm_estimator.py:16-32` == `diffusion_estimator.py:31-47`（后者已 import 前者的其他函数） |
| `_positive_integer`/`_nonnegative_integer` 校验 | 3+ | `serving_plan.py:3695`、`diffusion_estimator.py:2164`、`training_plan.py:694`（仅异常类不同）；`calibration.py` 内部同型判断约 15 处 |
| `_ceil_div` | 3 | `training_plan.py:733`、`diffusion_estimator.py:2234`、`topology.py:924` |
| `_percentile` | 2 | `calibration.py:2022`、`_bandwidth_worker.py:32` |
| `git rev-parse` 子进程封装 | 3 | `preflight.py:1329`、`validation.py:700`、`repository_analyzer.py:1000` |
| `_iter_tensor_leaves`/`_tensor_bytes` | 2（逐字节相同） | `memory_estimator.py:1771` == `workspace_profiles.py:633` |
| `_positive_float`/非负 int 钳制 | 3 | `smi.py:4332`、`metrics.py:1261`、`topology.py:894` |
| 字节数量*解析*（单位表还不一致） | 2 | `distributed_cli.py:34-49` vs `preflight.py:859-886` |
| 区间合并算法（同文件内两份） | 2 | `trace_replay.py:736-742` == `:768-774` |
| severity 等级表 | 3 | `smi.py:44`、`metrics.py:1239`（每次调用重建 dict）、`native_smi.cpp:502` |
| 手写 YAML 迷你解析器 | 3 | `profile_catalog.py:423`（并强迫整数字段必须是 str）、`structured_io`（真 YAML）、`tests/native/...:88`；另有 `scripts/lib/fakegpu_uv_deps.py:18-99` 的 82 行手写 TOML 解析器和 `scripts/validation/check_preflight_report.py:84-135` 的手写迷你 JSON-Schema 校验器（CI 已装 jsonschema） |

### 3.5 Python 与 C++ 的双实现（改动必须双写）

`smi.py:433-1116` 的三个 `_modeled_*` 环境解析器（约 680 行）与 `src/core/device.cpp:271-702` 逐条重复，错误字符串逐字节一致；JSON 发布再与 `src/core/native_smi.cpp:374-880` 重复。短期至少要加一个**共享 fixture 断言两侧输出一致**；长期考虑单侧生成或以数据文件驱动。`smi.py` 内部这三个解析器本身也是同构骨架，"可选 `+` 前缀整数解析"复制了 6 遍（`:483-984` 各处），可先在 Python 侧内聚。

### 3.6 CLI 层样板重复

21 个独立 `main(argv)` 各建 `ArgumentParser`，`prog="fakegpu <name>"` 字符串须与 `__main__.py:12-33` 的注册表手工同步；`--build-dir`/`--profile`/`--devices` 在 5+ 处重复声明；`--json` 有 4 种互不兼容的拼法（store_true / dest 字符串 / type=Path）；`llm_cli.py` 与 `serving_plan.py` 重复声明约 12 个同名 flag 且**默认值不同**（`dynamic/eager` vs `paged/sdpa`）；退出码约定 5 种。`preflight`/`doctor`/`demo` 三者独立校验 profile catalog，`preflight.py:716-745` 甚至用两个独立循环解析同一个 `--devices` 字符串。**方向**：一个小型 `register(name, build_parser, handler)` 注册器 + 共享 flag 工厂，统一 `--json`/退出码约定。`CMakeLists.txt:198-282` 的 APPLE/else 双分支中 5 个相同的 `set_target_properties` 块（约 85 行→30 行）同理。

---

## 4. 角度四：拆分超大文件与函数

### 4.1 超大函数一览（重构时优先处理）

| 行数 | 位置 | 说明 |
|---:|---|---|
| 666 | `torch_patch.py:466` `_DeviceMemoryTracker` | 混杂字节记账、allocator segment 模型、stage 记账、报告格式化 |
| 573 | `torch_patch.py:3769` `patch()` | 含 485 行不可达 standalone 安装 |
| 499/457 | `serving_plan.py:996 / :537` | 两条 82% 相同的主路径 |
| 498 | `diffusion_estimator.py:311` | 内含 156 行字面量返回 dict |
| 474 | `memory_estimator.py:107` `estimate_module_memory` | |
| 421 | `metrics.py:123` `render_prometheus` | 9 个 accumulator 机械重复 |
| 412 | `calibration.py:1104` `main` | argparse 构建 + 六路 dispatch |
| 376 | `serving_plan.py:1898` `main` | 同型 |
| 375 | `torch_patch.py:3387` `_apply_enhancements_over_upstream` | 已有 0–8 节编号，逐节提取即可 |
| 365 | `smi.py:2623` `render_detail` | runtime 段与 device 段互相独立 |
| 322 | `smi.py:1503` `main` | |
| 269 | `calibration.py:748` `verify_calibration_reports` | 8 个 gate 手写 if，天然表驱动 |

各 CLI 的 `main` 都可按同一模板拆为 `_build_parser()` + 每个子命令一个 `_run_<action>(args)`，顺带消掉 dispatch 长梯子和重复 JSON 输出块。

### 4.2 模块拆分方案（均为机械移动，不改行为）

- **`torch_patch.py`（4346 → ~6 个模块）**：`_profiles.py`（profile 解析，无 torch 依赖，`:63-226`）；`_cross_device.py`（`:228-368`）；`_allocator.py`（`_DeviceMemoryTracker` 及其自由函数，约 810 行，`:371-1184`，是最干净的一刀）；`_reporting.py`（`:1187-1332`，注意 `_dump_terminal_summary` 直接摸 tracker 私有字段，先给 tracker 一个公开 `report_rows()`）；`_ecosystem_compat.py`（HF/accelerate/FSDP/NCCL 第三方 shim，`:2777-3330`）；standalone 部分按 §1.1 处置。
- **`smi.py`（4353 → ~6 个模块）**：查询 schema（`:53-403` + `:3845-3961`）；modeled 环境解析（`:406-1146`，纯函数）；publisher（`:1149-1500`，`torch_patch` 唯一依赖的部分）；inventory/归一化（`:1841-2046` + `:3042-3833`，`metrics.py` 真正需要的部分——现在 `metrics.py:18-23` 在 import `smi` 的私有函数，是缺共享模块的最明确信号）；renderers（约 900 行）；CLI。
- **`calibration.py`（2873 → 4 个模块）**：比较/验证/bundle；serving observation 协议；transformers real-CUDA 适配器（唯一 import torch/transformers 的部分，现在为躲循环 import 在 `:2131` 懒加载 `llm_estimator`——强烈暗示它本就该在别处）；CLI。
- **`serving_plan.py`（3722 → ~5 个模块）**：KV-pool 数学、vLLM 预算、speculative decoding、request manifest、CLI（详见 §3.2 合并后再拆）。
- **`diffusion_estimator.py`（2327 → 4 个模块）**：pipeline inspection、profile 加载/校验（注意 `_local_profile` 产物绕过 `_validate_profile`，两套 schema 编码会漂移）、activation 模型、CLI。
- **`preflight.py`（1355）**：报告组装 / 状态分类 / 经验校准（约 260 行）/ Markdown 渲染四个关注点分离；`render_markdown_report`（`:305-503`）约 200 行。

---

## 5. 角度五：效率

### 5.1 热路径（每次张量操作/每次分配都执行）

- `torch_patch.py` 内 20 处函数内 `import torch`，其中 `_wrap_tensor_binary_op.wrapper:343` 在**每次** `+`/`*`/`@` 上执行，`_DeviceMemoryTracker.allocate:527` 在每次分配上执行（只为拿 `torch.cuda.OutOfMemoryError`）。模块级绑定一次即可。同类：`memory_estimator.py:1771` / `workspace_profiles.py:633` 的 `_iter_tensor_leaves` 在**每层递归**都 try-import torch。
- `_allocate_allocator_block`（`torch_patch.py:929-946`）每次分配对全部 segment×block 做嵌套扫描（持锁），随分配数近似二次增长；`_free_allocator_block:1011` 线性扫段（已有 `segment_id` 却不建索引）。
- `snapshot()`（`:790-855`）O(设备×存活分配)，被后台线程每 250ms 持锁驱动，直接与分配热路径抢锁——应改为在 allocate/release 时维护增量汇总。
- 每个被追踪张量 5 次 `os.environ` 读取（`:565, 1414, 1424, 1428, 1507`），其中仅 stage 一个真正会变。

### 5.2 每个 FX 节点重新加载全目录

`memory_estimator.py:1034-1044` 在节点循环内调 `match_workspace_profile`，后者每次 `load_workspace_profiles`：每个 catalog 文件一次 `stat()`、每个 profile 一次 `dict()` 拷贝、外加 `_software_stack` 的 try-import torch（`workspace_profiles.py:51-74, 91-92`）。数千节点的图 = 数千次 syscall 与目录重建。把 catalog + software stack 提升到循环外是这批文件里**单项收益最大**的优化。

### 5.3 二分搜索内的重复计算（serving_plan）

- 每个 request 的 transient 只依赖自身（`batch_size=1`），却在每次 prefix 二分探测中全量重算（`:2718-2804` × `:3105`），6N·logN → 6N。
- request-set 路径有 `memory_by_request_count` 缓存（`:1120-1147`），homogeneous 路径的 `batch_memory`（`:665-687`）却没有，且每次探测都带着 §2.5 的多层校验重跑完整 KV 估算栈。
- `prefill_chunk_tokens is None`（默认）时 `unchunked_prefill_transient` 与 target 逐位相同却算两遍（`:2381-2388, 2758-2765`）。
- `_worst_concurrent_transient:3438-3469` 对 6 个分量各做一次全排序取前 k（k 通常 1–2）→ `heapq.nlargest`。

### 5.4 反复深拷贝与多遍扫描

- `metrics.py`：history 双端队列最多 1440 个快照，`snapshots()` 每次 `GET /api/v1/history` 深拷贝**整个队列**（`:565`）；一次 `/metrics` 抓取额外 2 次深拷贝（`:673-682`）；`health_payload:712` 为读 6 个标量深拷贝整个快照。
- `trace_replay.py`：每个 event dict 拷贝两遍（`:429, :475`）；`_rank_summary` O(ranks×events) 外加每 rank 3 次重扫；`replay_trace` 后续约 10 遍全量扫描可合并为一遍累加。
- `repository_analyzer.py`：每个 `.py` 读两遍、`pyproject.toml` 读三遍解析两遍；每文件约 7 次 `ast.walk`（一个 NodeVisitor 可合一）；files 列表 8 遍扫描。
- `compare_memory_reports`（`calibration.py:101-193`）对 comparisons 约 10 遍生成器；`verify_calibration_reports` 又从原始数据重推一遍 summary 里已有的统计量（两套公式需人工保持同步，见 §3 scripts 里还有第三、四份）。
- 8+ 处 `--json` 块的**双重序列化**：`payload = json.dumps(...)` 算完即弃，`write_json` 再序列化一遍——大报告（trace_replay 内嵌全部 event）CPU 和峰值内存×2。
- `llm_estimator.py:485-506`：`decode_steps` 每生成 token 一个 dict 全量嵌入报告（4096 token = 4096 个 dict），且逐步重算本可闭式求和的 FLOPs——建议聚合 + `include_decode_steps` 开关（报告 shape 变化需过一次消费者确认）。
- `topology.py`：`_simulate_flows` 每轮对每条链路重扫全部 flow 路径（`:556-567, 658-664`）；ring collective 对相同 (src,dst,payload) 重复跑 Dijkstra `2*(N-1)` 轮无 memo（`trace_replay.py:57` 反而有缓存，可对齐）。
- 小项：`profile_catalog.official_compute_capabilities` 无缓存每次读文件（`:220-230`）；`preflight._contains_oom_marker` 每行 stderr 编译 4 个正则（`:1223-1232`）；`preflight.build_report:178` 从磁盘重读自己刚写的日志；`capabilities.py` lru_cache 因调用形参不同缓存同一文件两份（`:29-32`）。

### 5.5 启动开销

`__main__.py:8` 的 `from ._api import env` 触发 `fakegpu/__init__.py` 全量 eager import（实测 `import fakegpu` 165ms，其中 `calibration` 82ms），使 `:42` 的 `importlib` 懒加载完全失效——每个子命令（包括只需 `_api` 的 passthrough）都付这笔钱。`__init__.py` 改为 `__getattr__` 懒导出或让 `__main__` 绕开包 `__init__` 即可。

---

## 6. 角度六：测试与验证侧的配套

- **先补保护网再动刀**：`distributed_cli.py`（876 行）零测试覆盖，是 CLI 层最大的裸奔面；`torch_patch` standalone 路径、`preflight --steps`、`--include-exited`、`llm_cli` 多个 flag 均无覆盖。重构前对将保留的路径补冒烟测试。
- `scripts/validation/` 三个 bespoke assert 脚本与 `fakegpu validate` 的声明式 manifest 机制重叠，多数可转成 manifest case；静态内存统计在 `calibration.py`、两个 scripts 里共有 4 份独立实现（§3.4），收敛后 CI 校验逻辑只剩一份。
- `tests/test_virtual_smi.py` 是唯一让 `LEGACY_SCHEMA_VERSION` 和 `render_table` 重建路径活着的消费者——确认 legacy 支持是否还需要，不需要则连测试一起删。

---

## 7. 扫描中顺带发现的疑似 bug（与重构分开处理）

这些修复会**改变行为**，不应混入"功能不变"的重构 commit；已逐一确认、单独提交（状态见下，详见"实施进度"2026-08-17 §7 批量修复一节）：

1. ✅ 已修复（`1a7a337`）`torch_patch.py:3088`：`re.sub(r"rank:\\d+/", ...)` 在 raw string 里 `\\d` 是字面反斜杠+d，替换恒为 no-op——ShardMetadata placement 重写从未生效。
2. ✅ 已修复（`3ab2fb8`）`torch_patch.py:3817-3820`：重复调用 `patch()` 时新建 `_DeviceMemoryTracker`，但旧 tracker 注册的 `weakref.finalize` 仍会对新 tracker 调 `release()`，二次 `init()` 后内存记账漂移。
3. ✅ 已修复（`14a9f5f`）`smi.py:1453-1456`：`successful_writes` 在写入**之前**+1 序列化进状态，写失败时计数超前。
4. ✅ 已修复（`d519e97`）`validation.py:539` 捕获元组缺 `IndexError`，而 `_json_pointer:583` 对 list 索引越界正是抛它——错一个 pointer 会让整次验证崩溃而非记为 case 失败。
5. ✅ 已修复（`d519e97`）`validation.py:163-164`：`finished_at_ns` 与 `duration_seconds` 来自两次独立取时钟，报告内部自相矛盾。
6. ✅ 已修复（`2f483f5`）`capabilities.py:373-388`：`--library`/`--classification` 过滤发生在 summary 计算之后，打印的汇总与表格互相矛盾。**注**：中途一版未完成的修复曾假设 `apis`/`groups` 间存在实际不存在的 `"group"` 关联字段，会导致 `KeyError`——已改正为正确实现，过滤下沉到 `native_capability_report()` 内部、summary 计算之前。
7. ✅ 已修复（`6a87c50`）`distributed_cli.py:448`：rank 超时后 `continue` 发生在读取 report 之前，超时 rank 的诊断 JSON 被丢弃；`:438` 顺序 join 使 N 个慢 rank 串行放大超时。
8. ✅ 已修复（`a270e59`）`diffusion_estimator.py:1106-1107`：text encoder 均未暴露 hidden_size 时 conditioning width 静默取 1，产出接近 0 的错误估算而非报错（扫描发现的**最高风险静默兜底**）。
9. ✅ 已修复（`19e1999`）`training_plan.py:695-696, 730`：`_positive_int("auto")` 确认是 DeepSpeed 真实哨兵值且有测试覆盖，**保留不改**；`_bool_value` 对词表外未知字符串默认 True 才是真问题，已改为 raise。
10. ✅ 已修复（`97dc3f6`）`calibration.py:713-714`：用 `assert` 做控制流，`python -O` 下消失后 `None` 会被嵌进 observation。
11. ✅ 已修复（`4a3ae3b`）`smi.py:4322-4325`：`_process_name` 拼接 `sys.argv[:3]` 写入状态文件与 Prometheus label，可能泄露命令行中的路径/敏感参数，且是无界 label 基数来源。
12. ⛔ **复核后判定不是 bug，未改**。`calibration.py:2250-2272`：`_validate_collector_environment_contract` 只在 env var 已设置时比对。起初怀疑应改成"未设置即报错"，但 `build_cuda_serving_sample` 是文档化公开 API，供 vLLM/自定义 runner 在 FakeGPU 采集子进程之外直接调用（README 明确写了这个用法），`tests/test_analysis_extensions.py:757` 就是在这些变量全部未设置时调用并断言成功——"未设置即报错"会打破这个文档化用例。当前行为是该函数双重用途设计下的正确实现。

---

## 8. 实施顺序与后续计划

原建议顺序中，第 1 步的决策项已部分拍板（`fsdp_memory.py`、standalone 路径已删），第 2 步已全部完成，第 3、6 步各完成一部分。剩余工作按以下优先级推进：

1. ~~决策项~~ → 剩余待拍板：`privateuse1/`、`audit_native_exports`、`MatmulFlopCounterMode`、`schemas/*.json`、legacy schema（见"实施进度"一节的待决策清单）。
2. ~~纯删除（§1 + §2.2）~~ ✅ 已完成（`e451bd2`/`bc77d9f`/`6b00725`/`432c21c`）。
3. **公共设施（剩余，中风险）**：~~`FakeGpuConfig` dataclass 收敛 `_api.py`/`_runtime.py` 的 8 处 10 参数签名~~ ✅ 已完成（`_FakeGpuRuntimeConfig`，2026-08-17，−77 行）。剩余：CLI 注册器 + 共享 flag 工厂 + 统一 `--json`/退出码约定（注意 `llm_cli` 与 `serving_plan` 同名 flag 默认值不同，统一时需显式保留差异）。`--json` 输出块已由 `emit_json` 收敛（`f6891c8`）。
4. **大合并**：`serving_plan` 双路径合一（§3.2，全仓杠杆最大单项，约 1,760 行）——**先决条件**：给 `estimate_serving_plan` 与 `estimate_serving_request_set` 补 golden-output 测试，锁住两条路径的输出差异后再抽 `_build_serving_report`；`_patched_dist_init/destroy` 对 `_upstream` 的复刻清理（仅状态读取方式不同，合并时逐字段比对）。
5. **机械拆分（§4.2）**：`torch_patch`（−1,200 后约 3,100 行，`_allocator.py` 一刀约 810 行最干净）、`smi.py`（拆出 metrics 真正需要的 inventory/归一化层，消除 `metrics.py` import `smi` 私有函数）、`calibration.py`、`serving_plan.py`、`diffusion_estimator.py`。只做移动与 import 调整，旧模块 re-export 保持兼容。
6. **兜底收敛与性能（剩余）**：§2.3 逐个评估（`workspace_profiles` 的 import 失败退化、`smi.py` publisher 主循环裸捕、`calibration` 第四层 sample fallback 等——行为敏感，建议转 warning 而非静默）；§5.1 allocator `snapshot()` 增量汇总与 `_allocate_allocator_block` 索引化（需 benchmark）；§5.4 metrics 深拷贝三处、`decode_steps` 聚合 + 开关（报告 shape 变化需过消费者）。
7. ~~**bug 修复（§7，与重构完全分开走 `fix:` commit）**~~ ✅ 已完成（2026-08-17）：12 项中 10 项修复、1 项（#6）修复过程中改正了一处 WIP 引入的 KeyError、1 项（#12）复核后判定不是 bug，保留不动。详见"实施进度"与 §7 逐项状态。

原第 2–5 步预计的 4,000–5,000 行削减目标，目前已完成约 2,500 行；剩余大头集中在第 4 步（serving_plan 合并，约 −400 行净减 + 消除最大维护面）与第 5 步拆分（不减行数但改善结构）。§7 的 bug 修复批次已收尾，后续如无新发现不再单独安排。
