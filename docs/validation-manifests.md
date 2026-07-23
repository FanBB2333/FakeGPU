# Declarative Validation Manifests

`fakegpu validate` turns a JSON, TOML, or YAML manifest into a repeatable test
matrix. It is intended for checks that differ by GPU profile, dtype, framework,
or host but should use one definition and one report format.

## Run the maintained smoke matrix

```bash
fakegpu validate \
  --manifest verification/data/validation_smoke.yaml \
  --report-dir build/validation-smoke \
  --strict
```

The runner writes `validation_report.json`, `validation_report.md`, and one
`stdout.log`/`stderr.log` pair per expanded case. The report records the Git
commit, host, command, matrix values, duration, result, and skip/failure
details.

`--strict` makes a missing prerequisite fail the run. Without it, such a case
is reported as skipped. Use `--case NAME` to select cases, `--dry-run` to
inspect expansion without executing commands, and `--fail-fast` to stop after
the first failed case.

## Manifest example

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

Every matrix axis is expanded as a Cartesian product. Placeholders are
available in commands, paths, environment values, prerequisites, and
expectations:

- `{python}`, `{case}`, and `{execution_id}`
- `{manifest_dir}`, `{report_dir}`, and `{root_report_dir}`
- every matrix axis, such as `{profile}` or `{dtype}`

Commands are executed as argument lists without a shell. A string command is
split with shell-like quoting, but shell operators are not evaluated.

## Prerequisites and expectations

`requires` can check `platforms`, executable `commands`, `python_modules`,
environment variables, and files. `expect` can check:

- exit code and maximum duration
- required or forbidden text in stdout/stderr
- files created by the command
- JSON values addressed by RFC 6901 JSON Pointer

JSON checks support `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `contains`, and
`approx`. The repository-root
[`validation_manifest.schema.json`](https://github.com/FanBB2333/FakeGPU/blob/main/validation_manifest.schema.json)
defines the complete contract.

## Cross-host use

Commit and push the manifest first, then use `git pull --ff-only` on every
host. Run the same command and retain each host's report directory. Matching
`git_commit` values establish that the machines tested the same source; host,
Python, and platform fields preserve the environment differences.

The manifest runner organizes experiments. It does not make a CPU-only test
equivalent to real CUDA, and it does not relax the accuracy boundary of the
underlying validator.
