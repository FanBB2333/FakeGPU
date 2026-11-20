# Throughput and platform specification research

Reviewed **2026-10-16**. All 307 profiles were checked for per-precision peak throughput, memory type and bandwidth, ECC, form factor, PCIe, NVLink, NVLink-C2C, MIG, base clock, power range, PCI device ID and driver device name. As in the other catalog research, values come from NVIDIA primary sources: product pages, datasheets and product briefs, architecture whitepapers, docs.nvidia.com, and the driver supported-chips lists ([595.99.02](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/supportedchips.html) for current GPUs, [580.95.05](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/supportedchips.html) for Maxwell, Pascal and Volta). Discontinued product pages are cited through Internet Archive copies of the original nvidia.com, geforce.com or nvidia.cn page. Third-party databases were used only to locate NVIDIA documents.

## Throughput fields

Thirteen dense peak rates per GPU are recorded in TFLOPS (TOPS for integers) at the boost clock: FP64, FP32, FP16 and BF16 on CUDA cores, and Tensor Core FP64, TF32, FP16, BF16, FP8, FP6, FP4, INT8 and INT4. Tensor FP16, BF16 and FP8 rates assume FP32 accumulation, which is what PyTorch matmuls use; GeForce parts run that path at half the FP16-accumulate rate. Each profile lists every value under `throughput_published` or `throughput_derived`.

| Basis | Values | Rule |
| --- | ---: | --- |
| Published | 523 | NVIDIA's figure for this configuration. Sparse-only tensor figures are halved only when NVIDIA's footnote says dense is one half; Blackwell Ultra FP4, where dense is not one half, uses the stated dense value. |
| Derived | 1,852 | Per-SM, per-clock rate × enabled SMs × boost clock, from [`throughput_rates.json`](../fakegpu/data/throughput_rates.json). |

The rate table has 20 compute-capability entries, separate GeForce and professional entries where FP32-accumulate tensor or FP64 rates differ, and 22 per-profile exceptions such as TITAN RTX (professional tensor rate), GTX 16-series and T-series Turing parts without Tensor Cores, and L40. Non-tensor rates follow the CUDA Programming Guide arithmetic-throughput table; tensor rates follow the Volta, Turing, Ampere, Ada, Hopper, Blackwell and RTX Blackwell whitepapers. The table records 139 checks of derived against published values; 133 agree within 3%. The six exceptions are H100 SXM and H100 NVL tensor rates, whose published figures correspond to lower clocks than the profiles' maximum boost (those profiles keep the published values), and Tesla P4 FP32 (5.5 published, 5.7 derived).

`scripts/derive_profile_throughput.py` writes derived values and removes them when an SM count, clock or rate is unknown; `--check` and the test suite fail when they are stale. 37 profiles have no throughput because neither a published figure nor both SM count and boost clock are known (for example regional H800, H20 and A800 variants, H100 SXM5 64/94 GB, V100 SXM3, and several OEM laptop parts). Estimates for those profiles still reject throughput-dependent requests.

The RTX PRO 6000 and RTX PRO 4500 Blackwell Server Edition pages publish tensor figures without saying whether they are sparse. RTX PRO 6000 Server TF32 (234) matches the dense derivation and is recorded; the remaining figures are about twice the dense derivation, so the derived dense values are used instead.

## Platform fields

| Field | Profiles with a value |
| --- | ---: |
| `form_factor` | 290 |
| `memory_type` | 266 |
| `pcie_generation` | 239 |
| `memory_bandwidth_gbps` | 232 |
| `mig_max_instances` | 165 |
| `base_clock_mhz` | 119 |
| `nvlink_generation` | 91 |
| `ecc` | 75 |

An absent key means unknown. `nvlink_generation: 0` and `mig_max_instances: 0` mean NVIDIA states the product has no NVLink or no MIG; MIG support follows the [MIG user guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html). NVLink bandwidth is NVIDIA's total bidirectional figure per GPU (for bridged boards, the bridge bandwidth). Memory bandwidth is NVIDIA's figure, or the published data rate × bus width where only those are given.

## Corrections to existing fields

- **Device names.** 89 `torch_name` values now follow the driver supported-chips list, which is what `torch.cuda.get_device_name` reports: older Quadro and Tesla parts have no `NVIDIA` prefix (`Quadro P4000`, `Tesla T4`), Pascal notebook GPUs have no `Laptop GPU` suffix, and memory variants of one device ID share a name.
- **Boost clocks.** 13 boost clocks changed:
  - Most move from rounded web figures to whitepaper values (for example RTX 4070 2475 MHz and RTX 5090 2407 MHz, from 2480 and 2410).
  - Tesla P4 moves from its base clock to the 1531 MHz boost.
  - The GTX 1050 3 GB profile now uses NVIDIA's reference clocks rather than an OEM board's.
  - The RTX PRO 6000 Workstation Edition uses the whitepaper's 2617 MHz.
- **Other existing fields.** RTX 5070 Ti L2 is 48 MiB (was 64 MiB). Three power limits use the top of NVIDIA's published range. 147 PCI device IDs were added where the driver name identifies the configuration unambiguously.
- **Not applied.** RTX 2070's 185 W is the Founders Edition rating, so the reference 175 W is kept.
