# Limitations

[Documentation index](README.md) · [简体中文](../zh-CN/limitations.md) ·
[繁體中文](../zh-TW/limitations.md)

- Linux native simulation executes supported PTX and compatible cubin/SASS
  instructions on CPU. Unsupported instructions and launch features fail explicitly.
  macOS does not yet execute these kernels. See [framework compatibility](frameworks.md).
- FakeCUDA covers maintained Python and PyTorch behavior. Binary CUDA
  extensions are not covered.
- FakeCUDA random numbers repeat with the CUDA seed but differ from a real
  GPU's. Each random operation advances the CUDA generator's offset by 4
  (modeled); a GPU advances large launches further.
- Static analysis cannot resolve every dynamic import, generated kernel,
  runtime shape, or data-dependent branch.
- Memory estimates can miss backend-private allocations, custom operators,
  allocator policies, and workspaces that were never matched.
- Native cuBLAS algorithms without numeric implementations, unsupported
  GEMM/Lt dtypes, layouts, or pointer modes, and skipped CPU computation all
  follow `FAKEGPU_UNSUPPORTED_API`. With `warn` a warning is recorded. With
  `error` the call returns `CUBLAS_STATUS_NOT_SUPPORTED` before writing output.
- Simulated device memory cannot be shared with another process: CUDA IPC
  calls and VMM shareable-handle export and import return not supported.
  Freeing a pointer FakeGPU never allocated returns success unless
  `FAKEGPU_UNSUPPORTED_API=error`, where it fails like the driver.
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
- Multi-process FakeCUDA collectives run over gloo on the CPU. FSDP across
  processes is not covered by tests yet, and launchers other than `torchrun`,
  `accelerate launch` and `multiprocessing` need
  `FAKEGPU_DIST_COLLECTIVES=real`.
- `fakegpu run` gives simulated GPUs to Python processes only; native CUDA
  programs need the native launcher. Its summary can miss the last second
  of a process that ends through `os._exit()` or SIGKILL, and does not count
  DDP gradient buckets, the CUDA context, library workspaces, or NCCL
  buffers. See
  [Limits of `fakegpu run`](commands.md#limits-of-fakegpu-run).
- Modeled fault events only appear in reports. They do not change CUDA
  execution and do not represent real ECC or Xid data.
- Hybrid and passthrough modes require a compatible physical CUDA stack.
- `CUDA_DEVICE_ORDER=FASTEST_FIRST`, the CUDA default, is not modeled:
  simulated GPUs enumerate in PCI bus order, as with `PCI_BUS_ID`. FakeCUDA
  does not expose MIG instances as CUDA devices, so a `MIG-` entry in
  `CUDA_VISIBLE_DEVICES` ends the list there.
- macOS System Integrity Protection can remove `DYLD_*` variables from system
  binaries. Use a Homebrew, conda, or pyenv Python for native interception.

## Still planned

- Expand executable native CUDA operations and cuBLAS coverage.
- Validate distributed and MoE estimates across more GPU and software stacks.

See the [open issues](https://github.com/FanBB2333/FakeGPU/issues) for other
proposals and known problems.
