# Consumer GPU catalog research

Public-source review date: **2026-10-16**. The consumer catalog contains **124 selectable configurations**, including desktop, notebook, memory-capacity and memory-type variants. This update adds 84 configurations to the original 40 consumer profiles. These are public compatibility references, not a list of devices owned or tested by the maintainer. A configuration count is not a count of distinct silicon dies.

## Coverage

- Maxwell: GTX 950/960/970/980/980 Ti and TITAN X; GTX 950M/960M/965M/970M/980M, including independently documented notebook memory capacities.
- Pascal: the GTX 1050–1080 Ti desktop family, explicit 1050 2/3 GB and 1060 3/5/6 GB configurations, GT 1010 GDDR5/DDR4 (including 4 GB DDR4) and GT 1030 GDDR5/DDR4, TITAN X and TITAN Xp; GTX 1050/1050 Ti/1060/1070/1080 notebook configurations.
- Volta: the public 12 GB TITAN V.
- Turing: the desktop RTX 20 family and 2060 12 GB, all seven GTX 16 desktop configurations in NVIDIA's comparison table, GTX 16 notebook configurations, all five RTX 20 notebook model families, and TITAN RTX.
- Ampere: desktop RTX 30 models and documented 3050 6 GB/OEM, 3060 8 GB, 3060 Ti GDDR6X and 3080 12 GB variants; RTX 30 notebook models and capacity options; RTX 2050 notebook.
- Ada: desktop RTX 40/SUPER models, explicit 4060 Ti 8/16 GB and 4070 GDDR6, RTX 4090 D, the five RTX 40 notebook models, and RTX 3050 A notebook.
- Blackwell: desktop RTX 5050–5090, both 5060 Ti capacities, original RTX 5090 D and D v2, and the six RTX 50 notebook model families including the currently listed RTX 5070 notebook 8/12 GB options.

Existing IDs remain available. In particular, `rtx4060ti` retains its existing 16 GB default, with explicit `rtx4060ti-8gb` and `rtx4060ti-16gb` selections. `gtx1050` retains its previous compatibility capacity; `gtx1050-2gb` identifies the documented standard desktop configuration. `rtx3090ti-reference` supplies public nominal specifications independently of the pre-existing measured compatibility profile; it does not alter calibration records.

## Primary sources

Each YAML stores its exact `spec_source` and `compute_capability_source`; supplemental sources and variant restrictions appear in `notes`.

| Source | Evidence used |
| --- | --- |
| [NVIDIA GeForce desktop comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/) | Desktop RTX 20–50 and GTX 16 cores, boost clocks, capacities, buses, board power and series compute capabilities. |
| [NVIDIA GTX 900](https://www.nvidia.com/en-us/geforce/900-series/) and [Maxwell TITAN X specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/geforce-gtx-titan-x/specifications/) | Maxwell reference desktop specifications. |
| [NVIDIA laptop comparison](https://www.nvidia.com/en-us/geforce/laptops/compare/) | Ada/Blackwell notebook CUDA cores, capacity, bus, clock and subsystem-power ranges. |
| [NVIDIA RTX 30 notebook comparison](https://www.nvidia.com/en-in/geforce/laptops/compare/30-series/) | Ampere notebook buses and power ranges; footnote 3 explicitly establishes that RTX 3050 A is **Ada, CC 8.9**, with 1792 cores, 4 GB/64-bit memory and 35–50 W. |
| [NVIDIA RTX 30 and GTX 16 notebooks](https://www.nvidia.com/en-ph/geforce/laptops/30-series/) | Capacity options and GTX 16 notebook envelopes, including memory data rates. |
| [NVIDIA Pascal notebook launch table](https://www.nvidia.com/content/apac/event/en/pascal-notebook-campaign/notebooks.html) and [entry notebook table](https://www.nvidia.com/content/apac/event/au/overwatch-campaign/geforce-gtx-10-series-laptops.html) | Pascal notebook core counts, maximum capacities, boost clocks and memory rates. CC follows the CUDA table where a campaign architecture label is inconsistent. |
| [NVIDIA TITAN Xp](https://www.nvidia.com/nb-no/titan/titan-xp/), [TITAN V](https://www.nvidia.com/nb-no/titan/titan-v/), [TITAN RTX](https://www.nvidia.com/de-at/titan/titan-rtx/), [Pascal developer page](https://developer.nvidia.com/pascal) | Surviving official TITAN specifications; the Pascal TITAN X entry leaves fields absent from its surviving source unknown. |
| [NVIDIA RTX 4090 D](https://www.nvidia.cn/geforce/graphics-cards/40-series/rtx-4090-d/), [RTX 5090 D launch](https://www.nvidia.cn/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/), [RTX 5090 D v2](https://www.nvidia.cn/geforce/graphics-cards/50-series/rtx-5090-d-v2/) | Distinct regional configurations. The original 5090 D product URL now redirects to D v2, so the launch announcement preserves evidence of the original 32 GB model; ASUS shipping specs corroborate its 512-bit/28 Gbps memory while original reference boost/power remain unknown. |
| [Eurocom MXM catalog](https://eurocom.com/ec/data/whitepapers/34.pdf), [X170SM components](https://eurocom.com/ec-components-472-x170sm/), [Maxwell launch](https://eurocom.com/ec/release%28267%29ec), [M5 Pro components](https://eurocom.com/ec/components%28257%29M5Pro), [ASUS G751 table](https://rog.asus.com/articles/product-news/rog-g751jl-gaming-laptop-launches-with-nvidia-gtx-965m/) | Older notebook CUDA counts, capacities and selected OEM module ratings/base clocks. X170SM explicitly verifies **2560** RTX 2070 SUPER notebook CUDA cores; contradictory Sky-series web entries are not copied. |
| [Connect Tech GTX 950M module sheet](https://connecttech.com/pdf/PAR-M3N950M-FN_SpecSheet.pdf) | A documented 2 GB/128-bit GDDR5 GTX 950M module, 640 cores and 55 W. |
| [Colorful GTX 1060 5 GB](https://www.colorful.cn/home/product?id=1527&mid=102), [GT 1010 GDDR5](https://www.colorful.cn/home/product?id=1947&mid=102), [GT 1010 DDR4 2 GB](https://www.colorful.cn/home/product?id=1946&mid=102), [GT 1010 DDR4 4 GB](https://www.colorful.cn/home/product?id=1957&mid=102) | Manufacturer shipping configurations. Their Specs tabs retrieve public `/Home/GetAttrbuteValue?mid=102&id=...` JSON; exact endpoint URLs are in the profiles. GT 1010 CC follows the documented GP108 Pascal silicon, with cache and PCI ID unspecified. |
| MSI specification pages linked in individual profiles | GTX 1050 3 GB board, GTX 960 4 GB and additional memory-data-rate verification. OEM-specific clocks are labeled. |

The RTX 2050 is another branding exception: its name does not establish Turing architecture. Its profile uses a [public original CUDA deviceQuery report](https://forums.developer.nvidia.com/t/confusion-about-setting-kernel-block-and-grid-size-for-maximum-occupancy/287857) for CC 8.6, 16 SMs, 64-bit memory bus and 1 MiB L2. The report's outdated helper prints an incorrect CUDA-core total; this catalog derives 2048 cores from 16 Ampere SMs × 128. NVIDIA independently [announces the model](https://www.nvidia.com/en-in/geforce/news/geforce-rtx-laptop-new-options/). This is a reference assembled from public evidence, not a new measured/calibrated profile; clocks and power remain unknown.

## Field conventions and limits

CUDA resource limits come from NVIDIA's [compute-capability appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html). They describe architectural limits, not a measurement of every board. SM counts are derived from published CUDA-core counts using 128 FP32 lanes per SM for Maxwell/Pascal/Ampere/Ada/Blackwell and 64 for Volta/Turing. GTX 16 profiles omit Tensor-core-only INT4 support.

`memory_clock_mhz` is **half the effective memory data rate**, matching the catalog's DDR bandwidth formula; it is not necessarily the oscillator clock shown by a board utility. For example, 7 Gbps is stored as 3500 MHz. Where bandwidth and bus are given, the same rate can be derived from those values. Public nominal GB capacities use GiB byte counts consistently with the existing catalog.

**Zero means unknown or unspecified**, particularly L2 cache, PCI ID, notebook memory data rate, or older OEM power/clock fields. It does not assert the hardware lacks a cache or memory interface. A capacity/compatibility profile can therefore be usable while a throughput estimate remains unavailable. Laptop boost and power fields usually store upper published endpoints; these are not guaranteed simultaneous operating points, and Dynamic Boost is not added. OEM examples state when a base clock or specific module rating is used. Board/subsystem power ratings are not measured typical draw or unlocked firmware maxima.

## Remaining boundaries

This is broad researched coverage, not an exhaustive enumeration of every OEM part number or board revision. Separate profiles are still needed for verified GTX 950/960 OEM silicon variants, GTX 1060 GDDR5X regional refreshes, the MX notebook family, additional DDR3/GDDR5 notebook capacities, TITAN V CEO Edition, and independently documented Max-Q configurations. Factory overclock, cooler, branding and connector-only board variants are not expanded into duplicate entries. The GIGABYTE 5 GB specification URL was unavailable during review; Colorful supplied independent primary evidence for that capacity instead. GTX 950/960 OEM core-count variants were not added because surviving NVIDIA links lead to the generic product page and no sufficiently clear primary specification was recovered. The same marketing name can conceal multiple chips or power envelopes; unsupported values are not filled by copying an adjacent desktop card.
