# 声明式验证清单

`fakegpu validate` 可以把 JSON、TOML 或 YAML 清单展开成可重复执行的测试
矩阵。GPU profile、dtype、框架或主机不同，但需要共用一份测试定义和报告
格式时，可以使用这个入口。

## 运行仓库中的 smoke 矩阵

```bash
fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

runner 会生成 `validation_report.json`、`validation_report.md`，并为每个展开
后的 case 保存一组 `stdout.log` 和 `stderr.log`。报告包含 Git commit、主机、
命令、矩阵参数、耗时、结果，以及 skip 或失败原因。

`--strict` 会把缺少依赖的 case 记为失败；不使用该选项时，这类 case 记为
skipped。`--case NAME` 可以只选择指定 case，`--dry-run` 只检查矩阵展开，
`--fail-fast` 会在首次失败后停止。

## 清单示例

```yaml
schema_version: fakegpu.validation_manifest.v1

defaults:
  cwd: ..
  timeout_seconds: 60
  env:
    FAKEGPU_TERMINAL_REPORT: "0"
  expect:
    exit_code: 0

cases:
  - name: allocator-api
    matrix:
      profile: [rtx3090ti, rtx-pro-5000-blackwell]
    requires:
      python_modules: [torch]
    command:
      - "{python}"
      - -c
      - >-
        import fakegpu;
        fakegpu.init(runtime="fakecuda", profile="{profile}", device_count=1);
        import torch;
        x=torch.empty(1024, device="cuda");
        assert torch.cuda.memory_reserved() >= torch.cuda.memory_allocated()
    expect:
      exit_code: 0
```

所有 matrix axis 会按笛卡尔积展开。命令、路径、环境变量、依赖条件和结果
断言都可以使用这些占位符：

- `{python}`、`{case}`、`{execution_id}`
- `{manifest_dir}`、`{report_dir}`、`{root_report_dir}`
- 各个 matrix axis，例如 `{profile}` 或 `{dtype}`

命令会以参数列表执行，不经过 shell。字符串命令支持类似 shell 的引号拆分，
但不会执行管道、重定向等 shell 运算符。

## 依赖条件与结果断言

`requires` 可以检查 `platforms`、可执行 `commands`、`python_modules`、环境
变量和文件。`expect` 可以检查：

- 退出码与最长耗时
- stdout/stderr 必须或不得包含的文本
- 命令应当生成的文件
- 通过 RFC 6901 JSON Pointer 指定的 JSON 值

JSON 检查支持 `eq`、`ne`、`lt`、`le`、`gt`、`ge`、`contains` 和
`approx`。完整契约见仓库根目录的
[`validation_manifest.schema.json`](https://github.com/FanBB2333/FakeGPU/blob/main/validation_manifest.schema.json)。

## 跨主机使用

先提交并推送清单，再让每台主机执行 `git pull --ff-only`。随后在各主机运行
同一条命令并保留各自的报告目录。报告中的 `git_commit` 一致，说明各主机
测试的是同一份源码；主机、Python 和平台字段则记录环境差异。

清单 runner 负责组织实验，不会让 CPU-only 测试等同于真实 CUDA，也不会
改变底层验证器本身的精度范围。
