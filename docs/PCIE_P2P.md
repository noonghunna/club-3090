# PCIe Topology & Enabling P2P (multi-GPU, no NVLink)

This is the home for **getting the most out of a PCIe-only multi-GPU rig** — understanding your topology, and (optionally) enabling GPU↔GPU peer-to-peer (P2P) over the PCIe bus when you don't have NVLink.

**You don't need any of this to run the stack.** The default dual/multi-card path is PCIe-only with P2P *off* (`NCCL_P2P_DISABLE=1`, custom all-reduce disabled) — it's robust, needs no tuning, and works out of the box on any consumer rig. This doc is for two audiences: anyone who wants to **read their topology correctly** (why does `topo -m` say `PHB`?), and enthusiasts who want to **squeeze a workload-dependent few-to-~20% more** out of the PCIe bus via P2P. If you have an NVLink bridge, see [HARDWARE.md → NVLink](HARDWARE.md#nvlink) instead — that path auto-detects.

> **Example rig used throughout:** ASRock Rack **ROMED8-2T** (single-socket EPYC SP3) + 2× RTX 3090. It's just a concrete illustration (one maintainer's box) — the principles are board-agnostic; substitute your own slot/BIOS specifics.

---

## 1. Reading your topology: why `PHB`, not `PIX`

`nvidia-smi topo -m` labels each GPU↔GPU link by the *closest common point* the two cards share:

| Code | Meaning | Relative speed |
|---|---|---|
| `NV#` | NVLink (# = number of links) | fastest |
| `PIX` | a single PCIe **switch** (one bridge hop) | fast |
| `PXB` | multiple PCIe switches | good |
| `PHB` | a PCIe **Host Bridge** (the CPU root complex) | PCIe-bound |
| `NODE` | across host bridges within one NUMA node | slower |
| `SYS` | across NUMA nodes / sockets | slowest |

`PIX` requires a physical **PCIe switch chip** (PLX/PEX) sitting between the two slots. Most server/workstation boards — the ROMED8-2T included — have **no PLX switch**: every slot routes straight to the CPU's IO die. So two GPUs in different slots meet at the **CPU host bridge**, and **`PHB` is the correct, expected result** — not a misconfiguration, and not something a "better slot" will turn into `PIX`. (You only see `PIX` on boards with an onboard PCIe switch, or via a PLX riser.)

`PHB` is **not** a dead end for P2P. It just means peer traffic crosses the CPU root complex rather than a dedicated switch. Whether P2P actually *engages* over `PHB` depends on three more things: NUMA placement (§2), BIOS/ACS (§4), and the driver (§5).

---

## 2. NUMA: keep both cards in one domain

EPYC (and multi-socket Xeon) can expose the socket as 1 or 4 NUMA nodes — `NPS1` / `NPS4` in BIOS. Under `NPS4`, two GPUs in different CPU quadrants can report `NODE` (or worse) instead of `PHB`, adding cross-die latency to every all-reduce.

**Set `NPS1`** (one NUMA node per socket) so both GPUs share a domain and report `PHB` — the cleanest single-socket layout for TP=2. On a true multi-socket box, keep both GPUs on the **same socket** (otherwise you get `SYS`, the worst case).

---

## 3. Physical slot choice (two triple-slot GPUs)

A 3-slot (triple-width) card like most 3090s covers its own slot **plus the two below it**, so you need slots spaced **≥3 positions apart**.

- Use the **first usable x16 slot + one three positions down** so the coolers don't collide and both cards train the full x16 width. On the ROMED8-2T (7× PCIe 4.0 x16 slots) that's typically the top slot paired with one ~3 slots lower — **check your board's manual block diagram for the exact pair**, since the spacing and which slots are full-x16 vary.
- **Mind lane-sharing with onboard M.2 / NVMe.** Many boards bifurcate or share lanes between a PCIe slot and an onboard M.2 (often jumper- or BIOS-gated). A populated M.2 can silently drop your second GPU slot to **x8**, or disable it. Consult the manual's lane-allocation / jumper table before committing a pair.
- **Always verify the *trained* width** after seating — a slot can negotiate lower than its physical size:
  ```bash
  nvidia-smi --query-gpu=index,pcie.link.width.current,pcie.link.gen.current --format=csv
  # or:  sudo lspci -vv | grep -E 'LnkCap|LnkSta'
  ```
  `report.sh` captures this automatically (it flags a slot that trained narrower than the GPU's capability).

---

## 4. BIOS settings that matter

| Setting | Set to | Why |
|---|---|---|
| **Above 4G Decoding** | **Enabled** | Required to map large GPU BARs above the 4 GB boundary; prerequisite for ReBAR and for P2P BAR access. |
| **Re-Size BAR / Smart Access Memory** | **Enabled** | Lets the CPU address the full VRAM aperture; helps both model load and P2P. |
| **IOMMU** | **Off / passthrough** for bare-metal P2P; **On** only for VFIO/VM passthrough | An *enforcing* IOMMU + ACS routes peer traffic up to the root and back, defeating direct P2P. |
| **ACS (Access Control Services)** | **Disabled** for bare-metal P2P | ACS-redirect on the upstream port forces P2P transactions through the root complex — the **#1 silent P2P killer**. Leave **On** only if you need VM isolation (a genuine tradeoff). |
| **NPS** (EPYC) | **NPS1** | Keeps both GPUs in one NUMA domain (§2). |

> **⚠️ Ampere consumer cards: the ReBAR toggle is gated by GPU firmware, not the motherboard** (#734, @alexpolo1). On RTX 3090-class cards the BAR1 size ceiling comes from the **VBIOS** — with a launch-era (pre-ReBAR) VBIOS the BIOS toggle silently does nothing. Check before assuming:
>
> ```
> sudo lspci -vv -s <bus> | grep -A4 "Physical Resizable BAR"
> ```
>
> If `BAR 1: supported:` tops out at 256MB, the card needs a ReBAR VBIOS from its **board vendor** before any BIOS setting matters. And a ReBAR-*era* VBIOS date is not sufficient evidence: the reference rig's two 3090s run `94.02.42.*` (the ReBAR-era family) and still advertise `supported: 256MB` only. Trust the `lspci` `supported:` list, never the VBIOS date.

---

## 5. Enabling P2P on consumer GPUs

Two hard truths set expectations before you start:

1. **The stock NVIDIA driver refuses P2P on GeForce cards over `PHB`.** Even with perfect topology and BIOS, the consumer driver disables peer access. Enabling it requires a **patched kernel module** — the community [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules) fork ([Sam McLeod's walkthrough](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/)). This is a custom DKMS module — weigh the maintenance cost. (Should the walkthrough link ever rot, the shape of it: clone the fork matching your driver branch → build + install via DKMS in place of the stock `nvidia` kernel module → reboot → `nvidia-smi topo -p2p r` should now report `OK` between your GPUs.)
2. **`PHB` P2P is PCIe-bounded** (~25 GB/s on PCIe 4.0 x16), well under NVLink. So the win is real but modest and workload-shaped (§6).
3. **Large BAR1 is a hard prerequisite for the patched-module path, not a nice-to-have** (#734). The patch maps the **full VRAM aperture through BAR1** (static BAR1 mapping) rather than using mailbox windows — a 256MB BAR1 cannot map 24GB, so on a card whose `lspci` `supported:` list caps at 256MB (§4 note) this path is unreachable **regardless of topology, IOMMU, or ACS**. Fix the firmware first or stop here.

> **Identifying your board for a VBIOS hunt** (#734): `nvidia-smi` reports `Board Part Number: N/A` on many AIB cards. A non-destructive ROM read surfaces the real board ID:
>
> ```
> echo 1 | sudo tee /sys/bus/pci/devices/<dbdf>/rom
> sudo cat /sys/bus/pci/devices/<dbdf>/rom > card.rom
> echo 0 | sudo tee /sys/bus/pci/devices/<dbdf>/rom
> strings card.rom | head -20
> ```
>
> Match the exact board string **and revision** against the vendor's VBIOS — same-vendor ROMs for a different cooler/board variant will flash but can brick or misbehave. Note the sysfs dump is often truncated (fine for ID, **not a backup**); use `nvflash --save` for a real pre-flash backup.

**On this stack**, once the patched module is installed you don't edit composes — set one env var:

```bash
# in your repo-root .env
NVLINK_MODE=pcie_p2p
```

`scripts/detect_nvlink.sh` then flips the dual/multi composes to `NCCL_P2P_LEVEL=PHB` + custom-all-reduce **ON** (and strips the `expandable_segments` alloc token that's incompatible with the custom-all-reduce IPC path — see [UPSTREAM.md → #42609](UPSTREAM.md)). The other `NVLINK_MODE` values: `auto` (default), `force_on` (NVLink present), `force_off` (PCIe, P2P off).

> If `nvidia-smi topo -p2p rw` already shows `OK` between your GPUs *without* the patched module (some server boards / layouts genuinely expose P2P), `detect_nvlink.sh` auto-enables the PCIe-P2P path on its own — no env var needed.

---

## 6. Realistic expectations

From cross-rig data on this stack (2× 3090, TP=2):

| Path | Measured gain | Source |
|---|---|---|
| `dual.yml` (fp8 KV) — patched P2P vs unpatched | **+2% narrative / +9% code** | [#91](https://github.com/noonghunna/club-3090/issues/91) |
| DFlash / spec-decode path — patched P2P | **+19–22%** | [#95](https://github.com/noonghunna/club-3090/issues/95) |
| NVLink hardware (reference, power-matched A/B) | **~+15%** | [#77](https://github.com/noonghunna/club-3090/issues/77) |

**Translation:** code / spec-decode workloads see a real lift (the K+1 cross-card verify is bandwidth-bound, so it benefits most); narrative decode barely moves. The gain also grows with GPU count (more all-reduce traffic at TP=4). For most users the stock no-P2P PCIe path is already perfectly fine — **P2P is an enthusiast tuning lever, not a requirement.**

---

## 7. Verifying P2P actually engaged

Capability (`topo -m` / `topo -p2p`) tells you it *can* — it doesn't tell you it *did*. After launching a serving container:

```bash
bash scripts/report.sh
```

Read the **"Interconnect verdict"** line under *Boot log highlights* — the report cross-references host capability against the running container's engagement automatically: `✓ engaged`, `⚠ WARN` (NVLink bridge present but idle), or `ℹ` (P2P-capable driver, container not using it), each naming the fix. The raw evidence sits directly above it: the `[nvlink]` boot line plus the resolved `NCCL_P2P_LEVEL` + custom-all-reduce env. On rigs with no P2P capability the verdict line is deliberately absent — silence means "nothing to gain here", not "check failed". (This is exactly the round-trip the field was added to avoid — [#446](https://github.com/noonghunna/club-3090/issues/446), [#488](https://github.com/noonghunna/club-3090/issues/488).)

---

## 8. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `topo -m` shows `NODE` / `SYS`, not `PHB` | Wrong NUMA placement → set `NPS1`; reseat both GPUs in same-NUMA (same-socket) slots. |
| Second GPU trains at **x8** or disappears | A populated M.2 / adjacent slot is stealing its lanes → move the card or clear the bifurcation jumper (board manual). |
| `topo -p2p rw` shows `CNS` ("chipset not supported") | Stock driver refusing P2P on consumer GPU → install the patched module (§5), then re-check. |
| `topo -p2p rw` shows `GNS` **and** `lspci` `BAR 1: supported:` caps at 256MB | **Pre-ReBAR / BAR1-capped VBIOS** — a firmware gate, not a driver or topology problem (#734). No BIOS setting or driver swap helps; the §5 patched path needs large BAR1. Vendor ReBAR VBIOS first (§4 note + the board-ID tip in §5), then re-check `supported:`. |
| Boot crash after enabling P2P: `custom_all_reduce.cuh … invalid argument` | Known `expandable_segments` ↔ custom-all-reduce IPC clash → `detect_nvlink.sh` strips the token on the P2P path automatically; ensure you're on a current pin ([UPSTREAM.md → #42609](UPSTREAM.md)). |
| Enabled it but TPS didn't move | Check it actually engaged (§7); then check your workload — narrative decode barely benefits, code/spec-decode does (§6). |

---

**See also:** [HARDWARE.md → NVLink](HARDWARE.md#nvlink) (the bridge path) · [DUAL_CARD.md → NVLink auto-detection](DUAL_CARD.md#nvlink-auto-detection) · [BENCHMARKS.md](../BENCHMARKS.md) (cross-rig interconnect rows) · [CONTAINER_RUNTIMES.md](CONTAINER_RUNTIMES.md) (P2P/NVLink under VM passthrough) · [UPSTREAM.md](UPSTREAM.md) (#42609 alloc-conf fix).
