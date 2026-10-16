# Platforms and slots

[Documentation index](README.md) · [简体中文](../zh-CN/platforms.md) ·
[繁體中文](../zh-TW/platforms.md)

A real GPU sits in a slot, and the slot decides how it reaches the CPU and the
other GPUs. SXM modules mount on a baseboard built for one GPU generation and
talk to each other over NVLink, usually through NVSwitch chips. PCIe cards go
into PCIe slots, run at the slower of the card's and the slot's link, and can
pair up with NVLink bridges. Grace superchips join the GPU to the CPU over
NVLink-C2C. FakeGPU models this with platform templates.

## Platform templates

`fakegpu platform list` prints the templates; `fakegpu platform show` places a
device layout into one.

| Platform | Slots | GPU links |
|---|---|---|
| `hgx-a100-8`, `hgx-h100-8`, `hgx-b200-8`, `hgx-b300-8` | 8 SXM | NVSwitch: every GPU reaches every other over all of its NVLinks (NV12 for A100, NV18 for Hopper and Blackwell) |
| `hgx-a100-4`, `hgx-h100-4` | 4 SXM | Direct links, 4 (A100) or 6 (H100) per pair |
| `dgx-1-p100`, `dgx-1-v100` | 8 SXM2 | Hybrid cube-mesh with NVIDIA's published pair wiring |
| `dgx-2` | 16 SXM3 | NVSwitch across two baseboards; two levels of PCIe switches |
| `gh200` | 1 superchip | NVLink-C2C to the Grace CPU, 450 GB/s per direction |
| `gb200-nvl72-tray`, `gb300-nvl72-tray` | 4 superchip GPUs | Two Grace CPUs with C2C; 18 NVLinks per GPU into the rack switch trays |
| `dgx-station-gb300` | 1 superchip + 1 PCIe Gen5 x16 | C2C for the GB300; one extra RTX PRO card |
| `dgx-spark` | 1 superchip | GB10 GPU sharing unified memory with the CPU |
| `pcie-server` | 8 PCIe Gen5 x16 | Two sockets, four PCIe switches with two slots each |
| `workstation` | 4 PCIe Gen5 x16 | One socket, each slot on its own root complex |
| `desktop` | 2 PCIe Gen5 | x16 for one card, x8/x8 for two |
| `laptop`, `jetson` | 1 | Laptop PCIe link; Jetson unified memory |
| `generic` | up to 64, any GPU | No constraints: each GPU at its own link behind one switch, no NVLink |

The baseboard and superchip layouts follow NVIDIA's DGX/HGX guides, the Fabric
Manager user guide and architecture whitepapers; each template in
[`platforms.json`](../../fakegpu/data/platforms.json) lists its sources. The
PCIe server, workstation, desktop and laptop templates are representative
layouts, not particular products.

## Choosing a platform

`auto` (the default) picks a template from the GPUs' form factor:

- SXM modules go on the baseboard that takes that module, the smallest one with
  enough sockets: `a100:2` lands on the 4-GPU HGX A100 board, `a100:8` on the
  8-GPU board, `v100:8` on DGX-1.
- Superchips go on their Grace platform, and a GB300 DGX Station may add one
  PCIe card.
- PCIe cards go into a desktop (up to two consumer cards), a workstation (up to
  four, or any workstation card) or the PCIe server (data-center cards or more
  than four cards). A GPU with an unknown form factor counts as a PCIe card.

Layouts that cannot exist are rejected with a hint: one baseboard holds one GPU
model, so `a100:4,h100:4` fails; SXM modules and PCIe cards never share a
platform; and a platform has a fixed number of slots. `generic` accepts any
layout.

## How links are modeled

- **PCIe.** A GPU trains at the lower generation and width of the card and the
  slot. Usable bandwidth per direction is 0.985 GB/s per lane for Gen3, 1.97
  for Gen4, 3.94 for Gen5 (128b/130b encoding) and 7.375 for Gen6 (236 data
  bytes per 256-byte FLIT). The PCIe slot generation of a platform can be
  lowered or raised for every PCIe slot.
- **CPU link.** Data moves between host memory and a GPU over its PCIe link,
  over NVLink-C2C on superchips, or within shared memory on DGX Spark and
  Jetson (half the memory bandwidth, since a copy reads and writes every
  byte). A narrower upstream link, such as the Gen5 x16 host link behind each
  B300's Gen6 ConnectX-8, caps the CPU link.
- **NVLink.** Each GPU has the NVLink ports its profile records (A100 12, H100
  18, B200 18, V100 6, RTX 3090 4), at 20 GB/s per link and direction for
  NVLink 1, 25 for NVLink 2 to 4 and 50 for NVLink 5, or the per-link rate its
  bridge bandwidth implies (14.0625 GB/s for GA102 cards). NVSwitch boards bond
  all ports to every peer (`NV12`, `NV18`); direct wiring gives each pair its
  own links (`NV4`, `NV6`, `NV1`/`NV2` on DGX-1).
- **Peer access.** `cudaDeviceCanAccessPeer` and
  `torch.cuda.can_device_access_peer` are true over NVLink, and over PCIe when
  both GPUs sit under the same CPU socket (`PIX`, `PXB`, `PHB` or `NODE`).
  They are false across sockets (`SYS`), for the GeForce RTX 4090 (NVIDIA
  disables its PCIe peer access), and on superchip platforms between GPUs
  without NVLink, where NVIDIA documents no peer access.

Native programs see the same model. NVML reports the PCIe generation and width
(current, maximum and the GPU's own maximum), each NVLink port with a GPU or
NVSwitch at the far end, `nvmlDeviceGetTopologyCommonAncestor` and
`nvmlDeviceGetP2PStatus`. `cudaDeviceCanAccessPeer`, `cuDeviceCanAccessPeer`
and `cuDeviceGetP2PAttribute` answer from the topology, and
`cudaDeviceEnablePeerAccess` returns `cudaErrorPeerAccessUnsupported` for a
pair without peer access.

`fakegpu nvidia-smi topo -m` shows the result with NVIDIA's legend:

```text
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NUMA Affinity
GPU0  X     NV1   NV1   NV2   NV2   SYS   SYS   SYS   0
```

## Configure the machine

The background machine (see the [console guide](console.md)) keeps the
platform with its device layout:

```bash
fakegpu daemon config --devices rtx3090:2 --nvlink-bridges 0-1
fakegpu daemon config --devices l40s:8 --platform pcie-server --pcie-gen 4
fakegpu daemon config --devices h100:8 --platform auto
```

`--platform` takes `auto` or a template ID, `--pcie-gen` overrides the
generation of PCIe slots (`0` keeps the template's), and `--nvlink-bridges`
joins PCIe cards with NVLink bridges (`0-1,2-3`; H200 NVL also takes four-way
bridges such as `0-1-2-3`). Only cards with an NVLink connector accept a
bridge, and both ends must be the same model. The console's Machine tab has
the same settings and shows the slot map and the link matrix. FakeCUDA
processes that use the machine layout get its platform.

## Configure a single process

Outside the machine, a process has no platform unless it asks for one:

```bash
fakegpu run --devices a100-pcie-80gb:4 --platform workstation ./app
FAKEGPU_PLATFORM=auto FAKEGPU_PROFILES=h100:8 python train.py
```

```python
import fakegpu
fakegpu.init(devices="v100:8", platform="auto")
```

`FAKEGPU_PCIE_GEN` and `FAKEGPU_NVLINK_BRIDGES` set the PCIe slot generation
and the bridges. Without a platform, every GPU sits behind one PCIe switch at
its own link generation and width, without NVLink, and every pair has peer
access, as before platforms existed.

The platform reaches FakeCUDA processes as three environment variables, which
`fakegpu platform env --devices ...` prints for programs started some other
way. A native program that sets only `FAKEGPU_PLATFORM` reports a
configuration error, since resolving a platform needs Python:

| Variable | Contents |
|---|---|
| `FAKEGPU_PLATFORM` | Template ID |
| `FAKEGPU_SLOTS` | One `;`-separated entry per GPU with `type`, `pcie` (`GENxWIDTH` or `none`), `gpu` (the GPU's own maximum), `host` (narrower upstream link), `link` (`pcie`, `c2c` or `unified`), `hostbw` (GB/s per direction), `numa`, `bridge`, `switch` (`a` or `a.b`) and `p2p` (`0` disables PCIe peer access) |
| `FAKEGPU_NVLINK_FABRIC` | `;`-separated groups `KIND:MEMBERS:LINKS:GBPS`: `mesh` gives every pair of members LINKS direct links, `switch` gives each member LINKS links into an NVSwitch fabric; GBPS is per link and direction |

Explicit `FAKEGPU_SLOTS`, `FAKEGPU_NVLINK_FABRIC` or the older
`FAKEGPU_NVLINK_GROUPS` win over a platform. `FAKEGPU_NVLINK_GROUPS="0,1;2,3"`
still gives each pair in a group one link at `FAKEGPU_NVLINK_BANDWIDTH_GBPS`
(default 900).

## Limits

- Some modules do not record their NVLink ports (for example A800 and H800
  SXM, H20, V100 SXM3); they get the baseboard's ports, and
  `fakegpu platform show` says so.
- NVIDIA publishes 900 GB/s of C2C bandwidth per GB200/GB300 superchip without
  saying how its two GPUs share it; each GPU is modeled with 450 GB/s per
  direction.
- The H200 4-GPU HGX wiring, the GB200 NVL4 bridge map and the DGX Spark C2C
  rate are not published, so no template uses them.
- PCIe peer access on GeForce cards other than the RTX 4090 follows the
  socket rule above; NVIDIA documents no per-model policy for them.
