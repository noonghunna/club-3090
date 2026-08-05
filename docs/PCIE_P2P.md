# PCIe Topology & Enabling P2P (multi-GPU, no NVLink)

This is the home for **getting the most out of a PCIe-only multi-GPU rig** — understanding your topology, and (optionally) enabling GPU↔GPU peer-to-peer (P2P) over the PCIe bus when you don't have NVLink.

**You don't need any of this to run the stack.** The default dual/multi-card path is PCIe-only with P2P *off* (`NCCL_P2P_DISABLE=1`, custom all-reduce disabled) — it's robust, needs no tuning, and works out of the box on any consumer rig. This doc is for two audiences: anyone who wants to **read their topology correctly** (why does `topo -m` say `PHB`?), and enthusiasts who want to **squeeze a workload-dependent few-to-~20% more** out of the PCIe bus via P2P. If you have an NVLink bridge, see [HARDWARE.md → NVLink](HARDWARE.md#nvlink) instead — that path auto-detects.

> **Example rig used throughout:** ASRock Rack **ROMED8-2T** (single-socket EPYC SP3) + 2× RTX 3090. It's just a concrete illustration (one maintainer's box) — the principles are board-agnostic; substitute your own slot/BIOS specifics.

---

## The three layers of "is P2P on?" — read this first

"P2P" is three independent questions stacked on top of each other, and every confused triage in our tracker came from conflating them ([disc #773](https://github.com/noonghunna/club-3090/discussions/773) has the worked example):

| Layer | Question | How to check |
|---|---|---|
| **1. Driver** | Is direct GPU↔GPU access *granted*? | `nvidia-smi topo -p2p rw` (= OK), module flavor (§5); strongest: a transfer-verified cache (§7) |
| **2. NCCL** | Is the granted path *used* for transfers? | `NCCL_P2P_LEVEL` set + layer 1 granted — this is where most of the P2P benefit flows, at any GPU count |
| **3. vLLM's custom all-reduce** | Is vLLM's *extra* kernel on top of NCCL active? | vLLM's own log: the `Custom allreduce is disabled…` line means no. At >2 GPUs it requires a **full NVLink mesh** and never consults P2P ([#786](https://github.com/noonghunna/club-3090/issues/786)) — so on 3+-card PCIe rigs layer 3 is always off, *by design, not misconfiguration*, and P2P still pays through layer 2 |

### Why layer 3's gate asks about NVLink and not peer access (2026-07-30)

Read from `vllm/distributed/device_communicators/custom_all_reduce.py` (verified identical in v0.24.0 and v0.25.1). **The NVLink test was never intended as the requirement — it is a cheap pre-filter in front of an expensive one**, and its own comments say so:

```python
# test nvlink first, this will filter out most of the cases
# where custom allreduce is not supported
fully_connected = current_platform.is_fully_connected(physical_device_ids)
if world_size > 2 and not fully_connected:
    logger.warning("Custom allreduce is disabled because it's not supported on"
                   " more than two PCIe-only GPUs. ...")
    return                      # <-- disqualifies, instead of falling through
# test P2P capability, this checks software/cudaruntime support
# this is expensive to compute at the first time
# then we cache the result
if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
```

`gpu_p2p_access_check()` spawns subprocesses to test real transfers — slow on first call, hence cached. `is_fully_connected` exists to avoid paying that when it would fail anyway. The narrow defect: at `world_size > 2` the pre-filter `return`s rather than deferring to the P2P check a patched PCIe rig would pass. An early-out optimisation became a hard gate.

Two details confirming it is a heuristic and not a kernel limit:

- The condition is literally `world_size > 2`. **At TP=2 the NVLink test is skipped entirely**, which is exactly why dual-card PCIe rigs get layer 3 and 4-card rigs do not — the peer-access result is never consulted at world>2.
- `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]` — the kernel itself supports 4 cards. Patching `is_fully_connected` works rather than crashing.

**Why `>2` plausibly exists** (inference from the algorithm shape, *not* stated upstream): custom AR has every rank write into every peer's buffer. At 2 GPUs that is one peer over one link; at 4 on a shared PCIe fabric the N−1 peer writes per rank contend for the same host-bridge bandwidth, while NVLink is point-to-point per pair and NCCL's ring/tree is topology-aware. It reads as an NVSwitch-era assumption never retested against P2P-patched consumer hardware.

**The 8 MiB cap is why forcing the gate is a tradeoff, not a free win.** `CustomAllreduce(max_size=8192 * 1024)` — tensors above 8 MiB fall back to NCCL regardless. So decode-sized tensors take the custom path and win, while prefill-sized ones exceed the cap, go through NCCL anyway, and still pay the registration overhead.

**We measured the bypass and deliberately do not ship it.** Forcing `is_fully_connected → True` at TP=4 on a patched 4×3090 gives **≈ +15% decode**, independently reproduced by [@superalesha](https://github.com/noonghunna/club-3090/discussions/773#discussioncomment-17834828) at **+14.8%** (80.9 → 92.9 tok/s single-stream, 144 → 165 at 16 users) — **with prefill paying for it** (TTFT +9% at c16, consistent with the cap above). Note the flag is read twice, once to build the communicator and once in `should_custom_ar`, so a half-patch does nothing. club-3090 keeps the gate: the tradeoff is workload-dependent, and monkeypatching an engine's topology gate is not something we want in a default path on other people's rigs. The clean upstream fix is to let `world_size > 2` fall through to `_can_p2p` and gate on measured benefit instead of link type.

**Where NVLink fits:** a bridge is the native version of layer 1 (no patched driver needed) and a faster layer 2 for the bridged pair. Layer 3 follows the same mesh rule: **2× 3090 + bridge → all three layers on**; **4× 3090 with two pairwise bridges → layer 3 still off** (consumer cards bridge exactly two GPUs; full meshes are NVSwitch/SXM territory). `report.sh`'s *Interconnect verdict* (§7) resolves all three layers for you.

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
| **IOMMU** | **Passthrough** (`iommu=pt`) for bare-metal P2P; **enforcing** only for VFIO/VM passthrough | An *enforcing* IOMMU + ACS routes peer traffic up to the root and back, defeating direct P2P. What matters is that the IOMMU isn't *translating* for these devices, which `iommu=pt` gives you while leaving it up. ⚠️ **The patched-module path documents `pt` specifically, not "off"** — aikitoria's setup step 1 is `amd_iommu=on iommu=pt` (Intel: `intel_iommu=on iommu=pt`) in `/etc/default/grub`. Switching the IOMMU off in BIOS also stops translation, but it is off their documented path; if you're on the patched module and P2P misbehaves, match their setup before debugging anything else ([#873](https://github.com/noonghunna/club-3090/issues/873)). |
| **ACS (Access Control Services)** | **Disabled** for bare-metal P2P | ACS-redirect on the upstream port forces P2P transactions through the root complex — the **#1 silent P2P killer**. Leave **On** only if you need VM isolation (a genuine tradeoff). |
| **NPS** (EPYC) | **NPS1** | Keeps both GPUs in one NUMA domain (§2). |

> **⚠️ Ampere consumer cards: the ReBAR toggle is gated by GPU firmware, not the motherboard** (#734, @alexpolo1). On RTX 3090-class cards the BAR1 size ceiling comes from the **VBIOS** — with a launch-era (pre-ReBAR) VBIOS the BIOS toggle silently does nothing. Check before assuming:
>
> ```
> sudo lspci -vv -s <bus> | grep -A4 "Physical Resizable BAR"
> ```
>
> If `BAR 1: supported:` tops out at 256MB, the card *may* need a ReBAR VBIOS from its **board vendor** — but **read §4a first if there is any chance you are inside a VM**, because a guest can report exactly this while the physical card supports 32GB.
>
> ⚠️ **CORRECTED 2026-08-05.** This note previously cited the reference rig's two 3090s as ReBAR-era VBIOS (`94.02.42.*`) that "still advertise `supported: 256MB` only", and concluded the VBIOS was the cap. **That was wrong, and it was wrong in an instructive way.** That reading was taken *inside a QEMU guest*, which exposes **no Resizable BAR capability at all**. Read on the *host*, the same two cards report `supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB`. Nothing was VBIOS-capped; the capability was simply invisible from where we looked. A `supported:` list is only evidence about the *hardware* when you read it on bare metal.

---

## 4a. Virtualised rigs (VFIO passthrough): diagnose on the HOST, not the guest

If your GPUs are passed through to a VM, **every BAR/ReBAR reading taken inside the guest is untrustworthy** and the fix lives on the host. Confirm where you are before believing anything else:

```
systemd-detect-virt                 # qemu / kvm / none
cat /sys/class/dmi/id/product_name  # "Standard PC (Q35 + ICH9, ...)" = emulated
```

**Worked example (reference rig, 2026-08-05).** Guest showed BAR1 `256M`, no Resizable BAR capability, `topo -p2p r` = `CNS`. The host told a completely different story — and the actual gate was a single BIOS toggle:

| where | before | after enabling **Re-Size BAR Support** in host BIOS |
|---|---|---|
| host `lspci` BAR1 | 256M | **32G** |
| host bridge prefetchable window | **288M** (= 256M BAR1 + 32M BAR3, i.e. sized exactly to the current BARs — the tell) | **32800M** |
| guest `nvidia-smi -q -d MEMORY` BAR1 | 256 MiB | **32768 MiB** |

**The bridge window is the real diagnostic, not the BAR.** A window sized to exactly `BAR1 + BAR3` means firmware allocated no room to grow, which is what ReBAR-disabled looks like. The `resource1_resize` sysfs knob will exist and still fail, because there is nowhere to expand into.

Note also that **Above 4G Decoding being on is not the same as ReBAR being on** — the reference rig had 64-bit prefetchable regions at ~4.9 TB (so Above 4G was clearly working) while BAR1 stayed at 256M.

> ### ⛔ A large BAR1 is necessary but NOT sufficient in a VM
>
> With BAR1 at 32 GB in the guest **and** the open kernel modules loaded (`modinfo -F license nvidia` → `Dual MIT/GPL`, driver 610.57.04), `topo -p2p r` **still reported `CNS`**.
>
> `CNS` is *Chipset Not Supported* — the driver consulting its **supported-chipset table**. A QEMU guest presents an emulated **Q35/ICH9**, which will never appear on that list at any aperture size. **The aperture was never the gate at that layer.**
>
> Two things follow, both measured on this rig rather than inferred:
> - **Stock `nvidia-open` alone does not grant P2P on consumer 3090s under virtualisation.** (It has been reported working on bare-metal server boards — see #688 — so this is a statement about VMs, not about the open modules.)
> - Defeating a chipset-table verdict is exactly what the §5 patched module removes from the source. Registry keys like `PeerMappingOverride` relax peer *mapping* restrictions; they are not documented to bypass the chipset check.
>
> ⚠️ **And even a patched module may not be enough under VFIO.** Peer DMA between two passed-through devices must be routed by the host IOMMU, and ACS on the root ports — the very thing giving you clean per-GPU IOMMU groups — pushes peer traffic upstream. **Do not reflexively disable ACS to chase this**: on the reference rig each GPU sits alone with its audio function in its own IOMMU group, and merging those groups can break passthrough outright. That is trading "no P2P" for "no GPUs". Treat ACS as a deliberate, reversible experiment, never a default.

**Order of operations on a virtualised rig:** confirm you're a guest → read BAR1 + bridge window **on the host** → host BIOS Re-Size BAR (Above 4G stays on; leave ACS and IOMMU alone) → verify the host window grew → verify the guest sees the large BAR → *then* driver/patch work → then a **transfer** check (§7), never the grant alone.

---

## 5. Enabling P2P on consumer GPUs

> **Not sure whether your rig is even a candidate?** `bash scripts/report.sh` answers that under *GPU hardware → NVLink*. On a 2+-card PCIe rig with peer access off it prints one line naming **which gate you're actually behind**: cards on separate root complexes (unreachable regardless of driver — §1–§2), a BAR1 too small to back the static full-VRAM mapping (fix §4 / the VBIOS **first**; this section can't help until then), or the stock driver's `CNS` refusal — which is the one case this section fixes. It stays silent on single-card rigs, NVLink rigs, and rigs where P2P is already on, so a line appearing means there's something real to decide.

Two hard truths set expectations before you start:

1. **The stock NVIDIA driver refuses P2P on GeForce cards over `PHB`.** Even with perfect topology and BIOS, the consumer driver disables peer access. Enabling it requires a **patched kernel module** — the community [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules) fork (**the 50-series flag set that makes this work on Blackwell is [@vladie](https://github.com/vladie)'s** — see [aikitoria/open-gpu-kernel-modules#25](https://github.com/aikitoria/open-gpu-kernel-modules/issues/25); the two-key recipe below is our bisection of their five flags, not our discovery) ([Sam McLeod's walkthrough](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/)). This is a custom DKMS module — weigh the maintenance cost. (Should the walkthrough link ever rot, the shape of it: clone the fork matching your driver branch → build + install via DKMS in place of the stock `nvidia` kernel module → reboot → `nvidia-smi topo -p2p r` should now report `OK` between your GPUs.)
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

> **⚠️ Large BAR1 is necessary but NOT sufficient — the driver also has to *use* it. Confirmed the hard way** ([#873](https://github.com/noonghunna/club-3090/issues/873)). The patch depends on the driver selecting the static full-VRAM BAR1 mapping, and nothing guarantees it picks that path on your card and driver branch. On a 2× RTX 5090 rig (@paulp83) the aperture was **already full-size — `nvidia-smi -q -d MEMORY` reported BAR1 Total 32768 MiB against 32607 MiB of VRAM** — the driver advertised peer access (`topo -p2p r` → `OK`), and every vLLM boot still hung inside NCCL init. **A healthy BAR1 number does not clear you.** What fixed it was forcing the mapping and the peer-mapping override through the kernel module:
>
> ```
> # /etc/modprobe.d/nvidia-p2p.conf
> options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;PeerMappingOverride=1"
> ```
>
> then `sudo update-initramfs -u`, reboot, and confirm with `cat /proc/driver/nvidia/params | grep -iE "Peer|P2P|Bar"`.
>
> ### ⚠️ SCOPE: this override is BLACKWELL evidence — do NOT apply it prophylactically
> Every datapoint for these keys comes from **one 2× RTX 5090 rig**. Before you paste them:
> - **3090 and 4090 P2P is confirmed working WITHOUT them** on this project's rigs — @TheFuzy's
>   4× 3090 ([#688](https://github.com/noonghunna/club-3090/issues/688)) and @MIkamal88's 2× 3090
>   ([#719](https://github.com/noonghunna/club-3090/discussions/719)) both ran the aikitoria BAR1
>   path months before these keys were discovered.
> - **They are not free.** Measured on the 5090 rig: with P2P *off*, the override costs
>   **−3.2% narrative / −1.8% code decode** — you pay that whenever peer access isn't engaged.
> - aikitoria's README documents **no** registry keys as required, for any card.
>
> **So: try the patched module alone first. Reach for these only if `topo -p2p` says `OK` and NCCL
> still hangs** (§8) — which so far has only been observed on 50-series.
>
> **Those two keys are the bisected minimum** — the reporter first got it working with five (`ForceP2P=17;RMForceP2PType=1;RMPcieP2PType=1;PeerMappingOverride=1;RMForceStaticBar1=1`) and then confirmed the pair above boots P2P healthily on its own, across two slugs. `RMForceStaticBar1=1` names the mechanism; `PeerMappingOverride=1` lifts the peer-mapping restriction. (aikitoria's README documents only `RMForceP2PType=1`, and there as a *testing* knob to make 3090s prefer PCIe over NVLink — it isn't needed here.) Undocumented RM registry keys are a support-free surface: if a boot goes bad, delete the file and rebuild the initramfs.

> **Which P2P path your card uses (aikitoria's README):**
>
> | card | path |
> |---|---|
> | **RTX 3090** | **Pairwise NVLink where available, PCIe BAR1 otherwise** — the only card with two |
> | RTX 4090 | PCIe BAR1 |
> | RTX 5090 | PCIe BAR1 (also works cross-device within a generation, e.g. 5090 ↔ RTX PRO 6000 Blackwell) |
>
> This matters for 3090 owners specifically: **with bridges fitted you are on the NVLink path, not
> BAR1**, so the BAR1 prerequisites (§4/§5) and the Blackwell override above are irrelevant to you —
> and `RMForceP2PType=1` is the documented knob to force PCIe *instead*, for testing. Reference
> throughput from the same README: 5090 pair ≈ **55–56 GB/s** unidirectional / **111 GB/s**
> bidirectional with P2P, against **~43 GB/s** without — useful as a sanity target for
> `p2pBandwidthLatencyTest` (§7).

**On this stack**, once the patched module is installed you don't edit composes — set one env var:

```bash
# in your repo-root .env
NVLINK_MODE=pcie_p2p
```

`scripts/detect_nvlink.sh` then flips the dual/multi composes to `NCCL_P2P_LEVEL=PHB` + custom-all-reduce **ON** (and strips the `expandable_segments` alloc token that's incompatible with the custom-all-reduce IPC path — see [UPSTREAM.md → #42609](UPSTREAM.md)). The other `NVLINK_MODE` values: `auto` (default), `force_on` (NVLink present), `force_off` (PCIe, P2P off).

> If `nvidia-smi topo -p2p rw` already shows `OK` between your GPUs *without* the patched module (some server boards / layouts genuinely expose P2P), `detect_nvlink.sh` auto-enables the PCIe-P2P path on its own — no env var needed.

> **Which composes auto-detect, and which don't.** All **24 multi-GPU vLLM composes** source `detect_nvlink.sh` from their own entrypoint, so auto-enable happens **in-container on every boot — `launch.sh`/`switch.sh` and a raw `docker compose up` alike**. The trigger is *only* `nvidia-smi topo -p2p r` reporting `OK` on every pair: it does not inspect your topology, ACS, IOMMU or BAR1, which is why a driver that grants peer access it can't deliver takes you straight to the §8 hang rather than to a slow path.
>
> The **llama.cpp-family** composes (`llama-cpp`, `ik-llama`, `beellama`) do no interconnect detection — they don't route peer traffic through NCCL, so there's nothing to toggle. The one **SGLang** dual compose (`eagle3-experimental`) also doesn't: it ships `--disable-custom-all-reduce` and `NCCL_P2P_DISABLE=1` deliberately, and P2P there is opt-in with `NCCL_P2P_DISABLE=0` (untested on a patched rig — report back if you try it).

⚠️ **The flip side of auto-enable: installing the patched module changes launcher behavior by itself.** The next launch after the module is in place, `detect_nvlink.sh` sees the new `OK` grant and switches every dual/multi compose to the P2P path (`NCCL_P2P_LEVEL=PHB` + custom-all-reduce) with no config change on your side. A driver *grant* is not the same as *working transfers* — if the grant doesn't actually carry bytes (**the driver not using the patch's static full-VRAM BAR1 mapping** — what forcing it resolved in [#873](https://github.com/noonghunna/club-3090/issues/873) — patch branch not matching your exact driver version, or ACS/IOMMU redirecting peer TLPs), NCCL blocks forever on its first peer operation and **every vLLM slug hangs silently at `pynccl` init with weights never loading** (§8). The escape hatch is always `NVLINK_MODE=force_off` in `.env`. Before trusting a fresh grant, run the transfer check (§7, `VLLM_SKIP_P2P_CHECK=0`) or cuda-samples `p2pBandwidthLatencyTest` — both move real bytes; the topo matrix does not.

> **Blackwell / 50-series — boots, but needs one extra step (updated 2026-08-04, [#873](https://github.com/noonghunna/club-3090/issues/873)).** The first field report (2026-08-01: 5090 pair, driver 610.43.03 + patch) got a granted `OK` matrix and the silent `pynccl` hang above, and this guide read that as *possibly hard-locked, treat as experimental*. **That reading was wrong** — the same rig now boots vLLM with P2P engaged after forcing the static BAR1 mapping via the two-key `NVreg_RegistryDwords` line above, and `report.sh` reports `✓ interconnect: PCIe P2P engaged (patched driver, custom all-reduce ON)`. So on 50-series, plan on the patched module alone being insufficient and the registry override being part of the recipe. What's validated so far is a **clean P2P boot**, not a throughput result: no P2P-on-vs-off A/B exists on Blackwell yet (§6). Still confirm a fresh grant with `p2pBandwidthLatencyTest` or §7's transfer check, and expect the patch branch to lag new driver releases.

---

## 6. Realistic expectations

From cross-rig data on this stack. ⚠️ **The gain is strongly card-dependent** — the Ampere rows are 2× 3090 / TP=2; the Blackwell row is an order of magnitude larger. Read the row that matches your hardware, not the first one:

| Path | Measured gain | Source |
|---|---|---|
| `dual.yml` (fp8 KV) — patched P2P vs unpatched | **+2% narrative / +9% code** | [#91](https://github.com/noonghunna/club-3090/issues/91) |
| DFlash / spec-decode path — patched P2P | **+19–22%** | [#95](https://github.com/noonghunna/club-3090/issues/95) |
| **2× RTX 5090, P2P on-vs-off at fixed TP=2** (`qwen-35b-a3b-dual-nvfp4`, same slug, same sitting) | **decode +32.5% · prefill@90K +33.9%** (net, vs a pristine no-dwords system) — the isolated interconnect delta is **+36.9%**, of which ~3% is given back by the `NVreg` override the 50-series path requires. ⚠️ **Ratio is clean, absolutes are not** — that sitting's P2P-off baseline runs 17.7% below the same rig's own earlier measurement of the same slug (BENCHMARKS `⤷ P2P on-vs-off A/B`), so a cross-session read gives only ~+9%. Treat +32.5% as the interconnect delta, **not** as a promised upgrade gain | [#873](https://github.com/noonghunna/club-3090/issues/873) (@paulp83) |
| NVLink hardware — workload-shaped (same-host A/B) | **decode +3–5% · prefill/long-ctx +35–49%** | [#698](https://github.com/noonghunna/club-3090/issues/698) — supersedes the flat ~+15% from [#77](https://github.com/noonghunna/club-3090/issues/77) (older v7.72.2 image) |

**Translation:** code / spec-decode workloads see a real lift (the K+1 cross-card verify is bandwidth-bound, so it benefits most); narrative decode barely moves. For most users the stock no-P2P PCIe path is already perfectly fine — **P2P is an enthusiast tuning lever, not a requirement.**

⚠️ **These are DUAL-card measurements — do not extrapolate them to 3+ GPUs.** At world_size > 2 without NVLink, **vLLM force-disables its custom all-reduce kernel** (its gate queries NVML for NVLink and never consults peer access — [#786](https://github.com/noonghunna/club-3090/issues/786)), so whatever P2P is worth at TP=4 arrives **through NCCL peer transfers only** — a lower ceiling than the dual-card custom-kernel path above. An earlier revision claimed the gain "grows with GPU count"; that was a projection, not a measurement, and stays withdrawn. **UPDATE 2026-07-30 — a measured multi-GPU A/B now exists, on ONE rig:** [disc #773](https://github.com/noonghunna/club-3090/discussions/773) (4× 3090, patched P2P, vLLM 0.25.1, MTP n=3, 220 W, one sitting) reports TP=2 → TP=4 as **prefill +55% @10K / +62% @90K, TTFT −38% @90K, decode +4.0% prose / −2.6% code**. Read it as *four cards read faster; they do not write faster* — the win is prefill and TTFT, and the decode column is inside run-to-run noise. It is one rig, one sitting, and it is a **TP-scaling** A/B (2 vs 4 cards, P2P on throughout), **not** a P2P-on-vs-off A/B at fixed TP. **That A/B now exists — see the Blackwell row below.** So: do not extrapolate "P2P scales with GPU count" from it, and do not cite the decode figures as a P2P result.

---

## 7. Verifying P2P actually engaged

Capability (`topo -m` / `topo -p2p`) tells you it *can* — it doesn't tell you it *did*. After launching a serving container:

```bash
bash scripts/report.sh
```

Read the **"Interconnect verdict"** line under *Boot log highlights* — the report cross-references host capability against the running container's engagement automatically: `✓ engaged`, `⚠ WARN` (NVLink bridge present but idle), or `ℹ` (P2P-capable driver, container not using it), each naming the fix. The raw evidence sits directly above it: the `[nvlink]` boot line plus the resolved `NCCL_P2P_LEVEL` + custom-all-reduce env. On rigs with no P2P capability the verdict line is deliberately absent — silence means "nothing to gain here", not "check failed". (This is exactly the round-trip the field was added to avoid — [#446](https://github.com/noonghunna/club-3090/issues/446), [#488](https://github.com/noonghunna/club-3090/issues/488).)

**On 3+ PCIe cards, expect "engaged via NCCL", not "custom all-reduce ON".** vLLM vetoes its custom kernel at world_size > 2 without NVLink and logs `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs` — that line is **expected on every 3+-card PCIe rig, patched or not**, and is not a misconfiguration. The same veto fires on **pairwise NVLink bridges** at 3+ cards (2 bridges on 4x 3090 is never a full 1-hop mesh — consumer cards bridge exactly two GPUs), so a quad-3090-with-bridges rig is also NCCL-only in vLLM; only NVSwitch/SXM-class full meshes keep the custom kernel at world>2. The report folds it into the verdict automatically ([#786](https://github.com/noonghunna/club-3090/issues/786)); P2P remains active on the NCCL path.

**Transfer-verified P2P — the strongest evidence tier.** Everything above is ultimately a driver *assertion* (the topo matrix, the module license, a clean boot — all the same query asked three ways). vLLM ships a functional check that actually moves bytes: boot once with `VLLM_SKIP_P2P_CHECK=0` and it performs an IPC write/read-back across every directed GPU pair, caching the result to `~/.cache/vllm/gpu_p2p_access_cache_for_<devices>.json`.

> **How to actually run it:** the variable has to reach the *container*, so every multi-GPU vLLM compose now carries the bare passthrough and you just prefix the boot:
>
> ```bash
> VLLM_SKIP_P2P_CHECK=0 bash scripts/launch.sh <slug>
> ```
>
> Then `bash scripts/report.sh` prints the **Transfer check** line. Until 2026-08-04 no compose declared the variable, so exporting it did nothing and the check silently never ran — [#873](https://github.com/noonghunna/club-3090/issues/873) hit exactly that. If you're on an older checkout, or on a single-card compose (where it's meaningless), add `- VLLM_SKIP_P2P_CHECK` to the `environment:` block yourself.

`report.sh` reads that cache automatically when present (host first, then the serving container) and adds a **"Transfer check"** line — `✓ N/N directed pairs OK` upgrades the verdict from driver-asserted to *measured*, and a partial result flags advertised-but-broken peer access that no driver query can see. The check costs seconds and the cache is a durable, paste-able artifact (idea from [disc #773](https://github.com/noonghunna/club-3090/discussions/773)).

⚠️ **Delete that cache after any hardware, BIOS, or driver change.** Its filename is keyed *only* on the device-id list (`gpu_p2p_access_cache_for_0,1.json`), and vLLM recomputes it only when the file is **absent** — no driver version, no BIOS state, nothing that moves when you re-seat a card or flip a toggle. A `true` written while P2P worked outlives whatever broke it, and the stale verdict is what everything downstream reads. `rm ~/.cache/vllm/gpu_p2p_access_cache_for_*.json` before re-running the check. (Related: `VLLM_SKIP_P2P_CHECK` defaults to **skip** — by default vLLM never verifies, it trusts the driver.)

---

## 8. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `topo -m` shows `NODE` / `SYS`, not `PHB` | Wrong NUMA placement → set `NPS1`; reseat both GPUs in same-NUMA (same-socket) slots. |
| Second GPU trains at **x8** or disappears | A populated M.2 / adjacent slot is stealing its lanes → move the card or clear the bifurcation jumper (board manual). |
| `topo -p2p rw` shows `CNS` ("chipset not supported") | Stock driver refusing P2P on consumer GPU → install the patched module (§5), then re-check. |
| `topo -p2p rw` shows `GNS` **and** `lspci` `BAR 1: supported:` caps at 256MB | **First rule out virtualisation (§4a)** — a QEMU guest exposes no Resizable BAR capability, so it reports this while the physical card supports 32GB. Run `systemd-detect-virt`; if it says `qemu`/`kvm`, re-read on the host, and the fix is host BIOS Re-Size BAR, not a VBIOS. **On bare metal**, this is a genuine **pre-ReBAR / BAR1-capped VBIOS** — a firmware gate, not a driver or topology problem (#734). No BIOS setting or driver swap helps; the §5 patched path needs large BAR1. Vendor ReBAR VBIOS first (§4 note + the board-ID tip in §5), then re-check `supported:`. |
| BAR1 is large, driver is open (`Dual MIT/GPL`), but `topo -p2p r` still says **`CNS`** | **Chipset-table verdict, not an aperture problem** (§4a). Expected inside a VM: the emulated Q35/ICH9 is not on NVIDIA's supported list, and no BAR size or registry key changes that. Needs the §5 patched module — and even then, verify with a **transfer** check, because VFIO peer DMA can still be blocked by host IOMMU/ACS routing. |
| `[nvlink] WARNING: … BAR1 is far smaller than VRAM …` | The launcher's BAR1 sanity check fired: P2P is being enabled, but the aperture is too small to back the patch's static full-VRAM mapping (§5) — the firmware-gated #734 class, caught *before* the hang instead of after. It **warns rather than gating**; the config is applied unchanged, so if this boot then hangs at `pynccl`, that's the row below. ⚠️ **Silence is not a clean bill of health** — [#873](https://github.com/noonghunna/club-3090/issues/873) had a full-size 32 GB BAR1 and still needed the §5 override, so the absence of this warning does not mean the mapping is in use. If P2P demonstrably works on your rig *despite* the warning, say so in an issue and bring `nvidia-smi -q -d MEMORY`. |
| Boot crash after enabling P2P: `custom_all_reduce.cuh … invalid argument` | Known `expandable_segments` ↔ custom-all-reduce IPC clash → `detect_nvlink.sh` strips the token on the P2P path automatically; ensure you're on a current pin ([UPSTREAM.md → #42609](UPSTREAM.md)). |
| **vLLM slugs HANG at `pynccl` init after installing a patched driver/module** (last line `vLLM is using nccl==…`, weights never load, no error) | The driver now *grants* P2P, so `detect_nvlink.sh` auto-enabled the P2P path (§5) — but the grant doesn't carry actual transfers, so NCCL blocks on its first peer op. **Unblock: `NVLINK_MODE=force_off` in `.env`, relaunch** (back to pre-patch behavior). Then validate the grant with raw transfers: cuda-samples `p2pBandwidthLatencyTest`, or §7's `VLLM_SKIP_P2P_CHECK=0` transfer check. If raw P2P hangs/reads garbage, in order: (1) **force the static BAR1 mapping** with the two-key registry override in §5 — this is what resolved the 5090 case ([#873](https://github.com/noonghunna/club-3090/issues/873)), and note it was needed *even though* `nvidia-smi -q -d MEMORY` already showed a full-size 32 GB BAR1, so **a healthy BAR1 number does not rule this out**; (2) match the patch branch to your **exact** driver version; (3) check ACS (`lspci -vvv \| grep ACSCtl` — ACS redirect stalls peer TLPs) and confirm `iommu=pt` per §4. Common right after a driver upgrade: the patch fork lags new driver branches. |
| Raw `p2pBandwidthLatencyTest` passes but vLLM still hangs | The grant works; the issue is in the NCCL/custom-AR layer. Rerun one slug with `NCCL_DEBUG=INFO` and read the last transport lines; try `NCCL_P2P_DISABLE=1` in the compose env to split NCCL peer transport from the custom-all-reduce path, and re-check the `expandable_segments` row above. |
| Enabled it but TPS didn't move | Check it actually engaged (§7); then check your workload — narrative decode barely benefits, code/spec-decode does (§6). |

---

**See also:** [HARDWARE.md → NVLink](HARDWARE.md#nvlink) (the bridge path) · [DUAL_CARD.md → NVLink auto-detection](DUAL_CARD.md#nvlink-auto-detection) · [BENCHMARKS.md](../BENCHMARKS.md) (cross-rig interconnect rows) · [CONTAINER_RUNTIMES.md](CONTAINER_RUNTIMES.md) (P2P/NVLink under VM passthrough) · [UPSTREAM.md](UPSTREAM.md) (#42609 alloc-conf fix).
