# NVIDIA data-center catalog research

Research date: **2026-10-16**. The data-center directory contains **65 selectable profiles**, including the existing synthetic `a100-1g` and archival provisional `b100`. This expansion adds **49 profiles**. Specifications describe public product configurations; they do not identify a maintainer's hardware or establish measured performance or calibration compatibility.

| Architecture | Profiles | Product coverage |
| --- | ---: | --- |
| Maxwell | 6 | Tesla M4, M6, M10, M40 12/24 GB, M60 |
| Pascal | 6 | Tesla P4, P6, P40; P100 PCIe 12/16 GB and SXM2 16 GB |
| Volta | 8 | V100 PCIe/SXM2 16/32 GB, FHHL 16 GB, SXM3/DGXS 32 GB, V100S PCIe 32 GB |
| Turing | 1 | T4 |
| Ampere | 19 | A2, A10/A10G, A16, A30/A30X, A40, A100 PCIe/SXM4 40/80 GB, A100X, regional A800 variants and AX800; synthetic A100 validation profile |
| Ada | 6 | L2, L4, L20 air/liquid, L40, L40S |
| Hopper | 14 | H100 PCIe/SXM/NVL memory variants, H200 SXM/NVL, H20 96/141 GB, H800 PCIe/SXM/NVL, GH200 96/144 GB |
| Blackwell | 5 | B200, B300, GB200, GB300; archival B100 |

## Device and unit conventions

Each YAML describes **one CUDA-visible GPU**. M10 has four GPUs, each with 8 GiB; M60 has two, each with 8 GiB; A16 has four, each with 16 GiB. Their shared board TDPs are recorded in notes, while unverified per-device power limits remain `0`. The [M10 product brief](https://images.nvidia.com/content/pdf/tesla/Tesla-M10-Product-Brief.pdf), [M60 product brief](https://images.nvidia.com/content/pdf/tesla/tesla-m60-product-brief.pdf), and [A16 product brief](https://images.nvidia.com/content/Solutions/data-center/vgpu-a16-product-brief.pdf) distinguish these quantities.

H100 NVL uses 94 GiB per GPU, not a bridged pair's 188 GB. H200 NVL uses 141 GiB per GPU. GH200 profiles include GPU-local HBM and exclude Grace CPU LPDDR. GB200/GB300 profiles represent one GPU of a two-GPU Superchip; neither Superchip nor rack memory becomes one GPU's capacity. NVIDIA's [H100 NVL brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf), [Grace Hopper page](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/), and [GB200/GB300 compute-tray specification](https://docs.nvidia.com/dsx/ncp/software-reference-guide/data-center-architecture) supply the corresponding product context.

`memory_bytes` follows the existing catalog convention of nominal advertised capacity multiplied by 2^30. It is not an exact `cudaMemGetInfo` measurement: ECC, firmware reservations, and SKU-specific usable capacity can reduce what a driver reports. In particular, Blackwell nominal capacities must not be mistaken for measured free memory.

`memory_clock_mhz` uses the project's bandwidth convention:

```text
bandwidth_bytes_per_second = memory_bus_width_bits / 8 * memory_clock_mhz * 2e6
```

Published memory clocks are retained when they fit this convention. When only bandwidth and active bus width are available, the half-data-rate clock is rounded to whole MHz and identified as derived in the profile notes. Such a value is a bandwidth-model input, not a claim about the physical oscillator. GPU boost clock, memory data rate, and video-engine clock are different quantities.

SM counts are either published directly or derived by dividing published CUDA core count by the documented cores per SM for that GPU architecture. They are **enabled product SMs**, not the full silicon maximum. Shared memory, register, block, and thread limits come from the [CUDA compute-capability appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) and the older [Maxwell](https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html), [Pascal](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html), and [Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) tuning guides. GP100 retains a 48 KiB per-block limit despite its 64 KiB per-SM shared memory.

## Primary-source map

The YAML `spec_source` and additional source comments preserve more specific provenance. The following sources cover product groups and variant distinctions.

| Products | Primary evidence |
| --- | --- |
| M4/M40 | [Original technical specifications](https://developer.nvidia.com/blog/accelerating-hyperscale-datacenter-applications-tesla-gpus/) list SMs, clocks, cache, memory, and TDP; [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) confirms M40 up to 24 GB. |
| M6/M10/M60, P6/P40 | [GRID packaging guide](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/161207-GRID-Packaging-and-Licensing-Guide.pdf) distinguishes GPU counts and per-GPU memory; [M6 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/nvidia-m6-datasheet.pdf), [P6 product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/solutions/resources/documents1/Tesla-P6-Product-Brief.pdf), and [P40 datasheet](https://images.nvidia.com/content/pdf/grid/data-sheet/nvidia-p40-datasheet.pdf) supply individual specifications. |
| P100 | [PCIe product brief](https://images.nvidia.com/content/pdf/tesla/NV-tesla-p100-pcie-PB-08248-001-v01.pdf) and [Pascal technical comparison](https://developer.nvidia.com/blog/inside-pascal/) distinguish PCIe/SXM2, 12/16 GB, bus width, boost clocks, cache, and TDP. |
| V100/V100S | [V100 family datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf), [PCIe product brief](https://images.nvidia.com/content/tesla/pdf/Tesla-V100-PCIe-Product-Brief.pdf), and [driver product table](https://download.nvidia.com/solaris/580.95.05/README/supportedchips.html) distinguish memory and form-factor variants. |
| A2/A10/A16/A30/A40 | [A2 specifications](https://www.nvidia.com/en-us/data-center/products/a2/), [A10 brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/A10-Product-Brief.pdf), [A16 brief](https://images.nvidia.com/content/Solutions/data-center/vgpu-a16-product-brief.pdf), [A30 brief](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-product-brief.pdf), [A40 brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/NVIDIA%20A40%20Product%20Brief.pdf). |
| A100 | [Product specification table](https://www.nvidia.com/en-us/data-center/a100/), [40 GB PCIe brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf), and [80 GB PCIe brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) distinguish bandwidth, memory clock, power, and PCI IDs. |
| A10G/A30X/A100X/A800/AX800 | [AWS G5 specifications](https://aws.amazon.com/ec2/instance-types/g5/), [RAPIDS AWS inventory](https://docs.nvidia.com/datascience/deployment/nightly/cloud/aws/), [NVIDIA marketplace](https://marketplace.nvidia.com/es-es/enterprise/data-center/), [vGPU user guide](https://docs.nvidia.com/vgpu/18.0/grid-vgpu-user-guide/index.html), and [A800/AX800 tables](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/ampere-a800.html) establish identity, architecture, and memory. |
| T4/L4/L40/L40S | [Ada whitepaper appendices C/D](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) publish enabled SMs, cache, memory bandwidth, and TDP; [L40S product specifications](https://www.nvidia.com/en-us/data-center/l40s/) and [NIM hardware specifications](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance) provide L40S data. |
| L2/L20 | [Detailed vGPU tables](https://docs.nvidia.com/vgpu/18.0/grid-vgpu-user-guide/index.html) and [architecture-specific vGPU features](https://docs.nvidia.com/ai-enterprise/release-4/latest/infra-software/vgpu/features.html) establish 24/48 GB and Ada identity. The licensing overview's inconsistent L2 memory entry is not used. |
| H100/H200 | [Hopper technical description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/), [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/), [H100 NVL brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf), [H200 NVL brief linked by NVIDIA](https://dam-cdn.nvd.orangelogic.com/AssetLink/7n7vya4684sdccfyy6kv37ek5lw702h7.pdf), [CUDA optimization presentation](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf), [NVIDIA GTC H100 NVL SM count](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73168/), and [Supermicro H200 NVL core specifications](https://www.supermicro.com/white_paper/white_paper_Accelerating_Edge_Innovation.pdf). |
| H20/H800/GH200 | [MIG supported-GPU table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html), [H800 reference](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html), [vGPU feature tables](https://docs.nvidia.com/ai-enterprise/release-4/latest/infra-software/vgpu/features.html), and [GPU Operator MIG configurations](https://github.com/NVIDIA/gpu-operator/blob/main/assets/state-mig-manager/0400_configmap.yaml). |
| B200/B300/GB200/GB300 | [HGX component table](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html), [compute-tray specifications](https://docs.nvidia.com/dsx/ncp/software-reference-guide/data-center-architecture), [NVIDIA B200 148-SM example](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/), and [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html). |

A10G has 80 enabled SMs, derived from AWS's 80 RT cores and 320 third-generation Tensor Cores. It does not inherit A10's 72-SM count. H100 NVL has 132 SMs in the NVIDIA GTC session; H200 NVL has 132 SMs derived from Supermicro's 16,896 CUDA cores and Hopper's 128 FP32 cores per SM.

## Corrections and remaining gaps

The existing IDs remain available. Corrections include V100 SXM2's PCI ID and boost clock; A10's PCI ID; A40's boost clock and 300 W TDP; T4's 70 W TDP; P100 PCIe's 250 W TDP and per-block shared-memory limit; A100 SXM4 80 GB's memory clock and PCI ID; and H100 SXM's core/memory clocks. T4, P4, A40, and L40S memory clocks now agree with their published bandwidths instead of mixing oscillator frequencies with the project's half-data-rate convention.

The H100 NVL brief publishes a **6016-bit active memory bus**, 2619 MHz memory, and 3938 GB/s. The H200 NVL brief publishes **6016 bits**, 3201 MHz, and 4813 GB/s. A full-chip 6144-bit assumption is therefore not substituted for active product width. H200 SXM bus and memory clock remain unverified. B200 uses the documented 148-SM configuration; the former 160-SM value represented full silicon rather than this product configuration.

**Zero means unknown**, including a `0` enabled-SM count in 27 reference profiles. Regional variants, special V100 form factors, converged accelerators, and Blackwell SKUs often have verified identity and memory but incomplete internal specifications. These profiles support selecting CUDA compatibility and nominal memory; missing throughput inputs prevent performance modeling. P100 PCIe 12 GB L2 remains `0` because the generic NVIDIA comparison lists 4 MiB while a [published SKU-specific benchmark](https://pub.hpc-technologies.co.jp/wp-content/uploads/2022/08/p100_12G_PCIe_ro.pdf) reports 3 MiB. Cache sizes are not copied between related SKUs, and board TDP is not passed off as typical workload power. All public reference profiles use `typical_power_usage_mw: 0`.

`b100` is retained for compatibility with an explicit archival/provisional note. Its pre-existing nominal 192 GiB capacity has not been reconfirmed as a shipping configuration; its former unsourced SM, bus, clocks, cache, power, and PCI values are now `0`. No claim is made that this catalog is exhaustive: A10M, T4G, OEM-only or restricted configurations, and additional cooling/board revisions still need sufficiently specific primary evidence. Rubin is outside this Maxwell-through-Blackwell expansion. The synthetic `a100-1g` remains explicitly synthetic and is not counted as a physical NVIDIA product.
