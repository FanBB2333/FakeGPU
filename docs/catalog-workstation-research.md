# Workstation catalog research

Reviewed **2026-10-16**. This expansion adds **91 reference configurations** for NVIDIA professional desktop and laptop GPUs, from Maxwell through Blackwell. Configurations with different published memory capacities or materially different mobile resources have separate IDs. Catalog entries are public product descriptions; they make no claim about maintainer hardware or calibration coverage.

| Architecture | Added configurations | Coverage |
| --- | ---: | --- |
| Maxwell | 19 | Maxwell-based Quadro K620/K1200/K2200; M2000/M4000/M5000/M6000 desktop; M6000 24GB; M-series mobile configurations |
| Pascal | 22 | Quadro P400 through P6000, P2200, GP100; P500/P520/P600/P620/P1000/P2000/P3000/P3200/P4000/P4200/P5000/P5200 mobile |
| Volta | 1 | Quadro GV100 |
| Turing | 17 | Quadro RTX 3000/4000/5000/6000 mobile; Quadro T1000/T2000 mobile; T400/T600/T1000 desktop and T500/T550/T600/T1200 laptop configurations |
| Ampere | 15 | RTX A400/A1000 desktop, A2000 6GB, RTX A500 through A5500 laptop configurations, A800 40GB Active |
| Ada | 7 | RTX 500/1000/2000/3000/3500/4000/5000 Ada Generation laptop GPUs |
| Blackwell | 10 | Six RTX PRO laptop GPUs, RTX PRO 5000 nominal 48GB/72GB, RTX PRO 6000 Max-Q, preliminary RTX PRO 5500 |

Existing desktop RTX A-series, Ada, Quadro RTX and Blackwell IDs remain available. The two existing Blackwell profiles that used 128 KiB as shared memory now use the CUDA 12.x shared-memory capacity of 100 KiB. The 128 KiB unified L1/shared-memory storage must not be substituted for allocatable shared memory. The RTX PRO 4500 official model spelling follows NVIDIA's current Workstation Edition name.

## Source map

Every YAML includes `spec_source`, `compute_capability_source` and qualification notes. The following NVIDIA primary sources provide the product-family overview; individual desktop profiles prefer their linked model datasheets.

| Source | Profiles / evidence |
| --- | --- |
| [NVIDIA previous-generation desktop archive](https://www.nvidia.com/en-us/products/workstations/previous-quadro-desktop-gpus/) | Discovery links to individual K/M/P, GV100 and Quadro RTX PDFs. The PDF contents take precedence over erroneous quick-spec text. |
| [NVIDIA desktop comparison](https://www.nvidia.com/en-in/drivers/quadro-desktop-gpu-specs/) | Maxwell desktop memory interfaces and M6000 24GB GDDR5 configuration; Pascal comparisons. |
| [February 2016 professional graphics line card](https://images.nvidia.com/content/quadro/product-literature/line-card/ProGraphicsLineCard-M6000-24GB-FEB16-US-fnl-HR.pdf) | M5000M/M4000M/M3000M/M2000M/M1000M/M600M CUDA cores and memory. |
| [Media and entertainment line card](https://images.nvidia.com/content/quadro/gpu-accelerated/pdf/208584-ME-ProGraphicsLineCard-US-FNL-HR.pdf) | M500M configuration. |
| [NVIDIA Quadro VR solutions](https://images.nvidia.com/content/virtual-reality/pdf/NVIDIA-Quadro-VR-fnl.pdf) | M5500 CUDA core count and memory. |
| [NVIDIA AutoCAD guide](https://images.nvidia.com/content/quadro/industry-solution/pdf/364261-AutodeskAutoCAD-SolutionOverview-US-NV-Fnl-WEB.pdf) | M620/M1200/M2200 memory capacities; unverified internal specifications remain zero. |
| [NVIDIA PTC Creo guide](https://images.nvidia.com/content/quadro/quadro-mobile/pdf/341544-PTC-CREO-SolutionOverview-US-NV-FNL-HR.pdf) | P3000/P4000/P5000 mobile memory capacities. Desktop core counts are not reused for these laptop GPUs. |
| [August 2019 mobile line card, revision 7](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/quadro-rtx-mobile-line-card-us-nvidia-r7-web.pdf) | Pascal mobile and original Quadro RTX/T mobile configurations, bus widths, bandwidth and power envelopes. |
| [Ampere-era mobile line card](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/quadro-mobile-line-card-n18-11x8.5-r4-hr.pdf) | Original RTX A2000/A3000/A4000/A5000 laptop GPUs; T500 options and later GDDR6 Quadro T1000. |
| [March 2022 mobile line card](https://www.nvidia.com/content/dam/en-zz/ja/Solutions/design-visualization/documents/ampere-mobile-linecard-master-mar22-jp.pdf) | RTX A5500/A4500/A3000 12GB/A2000 8GB/A1000/A500 and T1200/T600/T550 laptop resources and maximum TGP ranges. |
| [March 2023 professional line card](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/rtx-6000-l40-linecard-nvidia-us-2653097-r7-web.pdf) | RTX A1000 6GB laptop core count and memory; cross-checks of desktop and laptop product differences. |
| [Current laptop comparison](https://www.nvidia.com/en-us/products/workstations/professional-laptops/compare/) | All seven Ada and six Blackwell professional laptop GPUs, including CUDA cores, memory, bus width, bandwidth and configurable TGP ranges. |
| [Current desktop lineup](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/) | Desktop memory variants and links to current model PDFs. |
| [RTX A400](https://www.nvidia.com/en-us/products/workstations/rtx-a400/) / [RTX A1000](https://www.nvidia.com/en-us/products/workstations/rtx-a1000/) | Product pages link their NVIDIA PDF datasheets, which publish the core counts and memory interfaces missing from the short HTML tables. |
| [A800 40GB Active](https://www.nvidia.com/en-us/products/workstations/a800/) | Workstation GA100 resources, HBM2 interface, bandwidth, power and L2 capacity. |
| [RTX PRO 5000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf) | Nominal 48GB and 72GB configurations with the same CUDA-core count and interface. |
| [RTX PRO 6000 Max-Q datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-us-5349650-web.pdf) | Desktop Max-Q Workstation Edition; separate from laptop Max-Q operating points. |
| [RTX PRO 5500](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5500/) | Explicitly preliminary, announced 84GB product. Unpublished core count, interface and clocks remain zero. |

Model compute capabilities follow the [current CUDA GPU table](https://developer.nvidia.com/cuda/gpus) and [legacy table](https://developer.nvidia.com/cuda/gpus/legacy). Where a newer or mobile marketing name is absent from those tables, the profile explicitly identifies the capability as an architecture-family inference and cites the [CUDA technical appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html); such a profile does not imply an independently listed per-model CUDA-table entry. Shared-memory and occupancy limits also use the [Maxwell](https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html), [Pascal](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html) and [Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) guides.

## Derived and unavailable fields

- `sm_count` is calculated from the model's published CUDA-core count and the documented per-SM organization: 128 for Maxwell, Pascal 6.1 and RTX Ampere/Ada/Blackwell; 64 for GP100, Volta, Turing and GA100. A full-chip maximum is never substituted for an unverified SKU count.
- `memory_clock_mhz` uses the catalog's equivalent half-data-rate convention. Where both bandwidth and interface width are published, it is `round(bandwidth_GBps * 4000 / bus_bits)`. This reproduces the advertised bandwidth approximately; rounded marketing bandwidth can produce a slightly different value from a physical oscillator or `nvidia-smi` clock. Each derived value is labelled in the YAML notes. Seventy-five new configurations have this field populated.
- Advertised GB capacities use the catalog's binary-capacity convention (`GB * 1024**3`) and describe nominal capacity. Real driver-visible allocatable memory can be lower.
- Core clock, cache sizes, PCI ID and typical operating power remain zero when no verified model specification was found. Public maximum TGP is stored separately from typical power. Zero means unavailable, not a real zero-capacity resource.
- Laptop maximum power is the top of the published configurable operating envelope, including Dynamic Boost where the source says so. It is not a claim that every OEM laptop operates at that power or achieves the same clocks.
- Turing T-series profiles do not advertise Tensor Core data types or throughput. Format support such as `fp16` does not establish Tensor Core acceleration.
- The existing `rtx-pro-5000-blackwell` ID retains its original measured compatibility values. New `rtx-pro-5000-blackwell-48gb` and `rtx-pro-5000-blackwell-72gb` IDs use public nominal specifications. These are different evidence classes; a reference profile is not a calibration record.

## Source conflicts and remaining gaps

The NVIDIA archive HTML incorrectly labels several Maxwell boards as HBM2; their model PDFs and comparison table identify GDDR5. Its M6000 24GB datasheet link points to a P6000 PDF, so the 24GB profile uses the matching NVIDIA comparison entry. K2200/K620 power values also differ between archive quick specs and their individual PDFs; the model PDFs are used.

The early [Quadro mobile line card, revision 1](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/quadro-mobile-pro-graphics-line-card-us-r1-hr.pdf) lists 2,304 CUDA cores for RTX 3000, while revision 7 lists 1,920. Only the later revision is selectable: those conflicting documents alone do not establish a separately shipped 2,304-core GPU. Quadro T1000 has documented original GDDR5 and later GDDR6 configurations with different core counts; these have separate labelled profiles. The M5500 product spelling differs from the legacy CUDA table's M5500M; its profile records the alias. Some Maxwell mobile capabilities in that CUDA table differ from common secondary references; the table value is retained explicitly rather than silently changed.

This catalog is broad public-reference coverage, not proof that every historical OEM SKU has been enumerated. Remaining gaps include M520/K620M, special embedded/SE/medical boards, OEM-only capacity combinations, historical laptop Max-Q operating points, and Pascal mobile refreshes lacking complete primary specifications. P3000/P4000/P5000 mobile and M620/M1200/M2200 currently have memory-capacity support but unverified SM counts and clocks. The RTX A1000 6GB laptop interface and TGP remain unknown. Many workstation core clocks and L2 cache capacities are unpublished in the referenced product sheets. Quadro K-series products based on Kepler are outside the requested Maxwell-through-Blackwell architecture range.
