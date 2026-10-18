# Limitations

[Documentation index](README.md) · [简体中文](../zh-CN/limitations.md) ·
[繁體中文](../zh-TW/limitations.md)

- Linux native simulation executes supported PTX instructions on CPU. SASS-only
  cubins, unsupported instructions and launch features fail explicitly. macOS
  does not yet execute PTX. See [framework compatibility](frameworks.md).
- FakeCUDA covers maintained Python and PyTorch behavior. Binary CUDA
  extensions are not covered.
- Static analysis cannot resolve every dynamic import, generated kernel,
  runtime shape, or data-dependent branch.
- Memory estimates can miss backend-private allocations, custom operators,
  allocator policies, and workspaces that were never matched.
- Native cuBLAS algorithms without numeric implementations, unsupported
  GEMM/Lt dtypes, layouts, or pointer modes, and skipped CPU computation all
  follow `FAKEGPU_UNSUPPORTED_API`. With `warn` a warning is recorded. With
  `error` the call returns `CUBLAS_STATUS_NOT_SUPPORTED` before writing output.
- For models with a vision encoder, `estimate-llm` and `plan-serving` count
  checkpoint weights but not vision or projector activations and FLOPs. The
  reports flag this. Decoder token counts must already include visual tokens.
- Diffusion profiles model fixed reference pipelines. Custom ControlNet,
  IP-Adapter, LoRA, safety-checker, refiner, video, and DiT components need
  extra metadata and real-GPU validation.
- Online-serving plans do not model request arrival distributions, cache
  eviction, preemption or reordering, self-speculative, Medusa, or EAGLE
  variants, tokenizer remapping, tensor-parallel execution, throughput, or
  latency.
- Roofline output is an analytical interval, not a measured kernel latency.
- Distributed timing includes coordinator work, memory copies, sockets, and
  process scheduling. It is not an NCCL, NVLink, or RDMA benchmark.
- Modeled fault events only appear in reports. They do not change CUDA
  execution and do not represent real ECC or Xid data.
- Hybrid and passthrough modes require a compatible physical CUDA stack.
- macOS System Integrity Protection can remove `DYLD_*` variables from system
  binaries. Use a Homebrew, conda, or pyenv Python for native interception.

## Still planned

- Expand executable native CUDA operations and cuBLAS coverage.
- Validate distributed and MoE estimates across more GPU and software stacks.

See the [open issues](https://github.com/FanBB2333/FakeGPU/issues) for other
proposals and known problems.
