# Running club-3090 on Windows (WSL2) — from scratch

A start-to-finish path for getting club-3090 running on a Windows machine via **WSL2** (Windows Subsystem for Linux). If you're already on native Linux, ignore this — use the [Quick start](../README.md#quick-start).

**What works on Windows:**

| Engine | Native Windows | WSL2 |
|---|---|---|
| vLLM | ❌ (Linux + CUDA only) | ✅ |
| llama.cpp / ik_llama | ⚠️ *engine* only — no club-3090 tooling | ✅ (Docker, matches the recipes) |

> ⚠️ **club-3090 itself requires WSL2 (or native Linux).** Its scripts, composes, and `setup.sh` / `launch.sh` / `switch.sh` are bash + Docker + Linux-path based — **none of them run on *native* Windows.** There, you can drive the *upstream* llama.cpp binary by hand against the GGUF weights, but with none of this repo's helpers (no picker, no SHA-verify, no VRAM-budget composes, no bench/verify scripts). For the full stack, use WSL2 (this guide) or native Linux.

This guide uses **WSL2 + Docker** so the commands match the rest of the repo. The bulk of the work is one-time host setup (steps 1–6); after that it's the normal [Quick start](../README.md#quick-start).

> **Runtime tuning lives elsewhere — this guide links to it, doesn't repeat it.** Once you're booting, the WSL2-specific VRAM budget, TDR timeout, and boot-crash fixes are in [FAQ.md → Windows/WSL2](FAQ.md#does-this-work-on-windows--wsl2) and [HARDWARE.md → WSL2/Windows](HARDWARE.md#note-for-wsl2--windows-users). Steps 8–9 point you at them.

---

## 1. Install WSL2 + Ubuntu

From an **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Reboot when prompted. Then confirm the distro is on **version 2** (not 1):

```powershell
wsl -l -v        # VERSION column must read 2
```

Everything from here runs **inside the Ubuntu (WSL) shell** unless a step explicitly says "Windows / PowerShell".

## 2. NVIDIA driver + GPU passthrough

Install the **Windows** NVIDIA driver (580.x+ for vLLM's CUDA 13 runtime) from nvidia.com. **Do not install a driver *inside* WSL** — WSL inherits the Windows driver via GPU passthrough; a second driver inside the distro breaks it.

Verify passthrough from the WSL shell:

```bash
nvidia-smi      # must list your 3090(s). If it errors, update the Windows driver and reboot.
```

## 3. Give WSL2 enough RAM — `.wslconfig` ⚠️

**The most common silent failure.** WSL2 defaults to **50% of host RAM**, and the model loader needs to hold the whole checkpoint in RAM. If it can't, vLLM disables auto-prefetch, falls back to a slow streaming path, and you get a *misleading* `Tried to allocate ~44 MiB` "GPU OOM" that's actually **host-RAM starvation** ([#32](https://github.com/noonghunna/club-3090/issues/32)).

Create `C:\Users\<You>\.wslconfig` (Windows side) — size `memory` above your checkpoint (the 27B INT4 is ~18 GB, so give ≥24 GB):

```ini
[wsl2]
memory=24GB
swap=8GB
```

Apply it from **PowerShell**, then verify in WSL:

```powershell
wsl --shutdown          # PowerShell — fully restarts the VM
```
```bash
free -h                 # WSL — "total" should now show ~24Gi
```

## 4. Docker + NVIDIA Container Toolkit (for vLLM)

Use **either** Docker Desktop (WSL2 backend, GPU enabled in Settings → Resources) **or** `docker-ce` + `nvidia-container-toolkit` installed inside the distro. Verify the GPU reaches a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If you only want the llama.cpp / ik_llama path you can skip Docker and use a native build — but Docker keeps you on the same commands as the rest of the docs.

## 5. Clone onto the ext4 filesystem — NOT `/mnt/c` ⚠️

Clone into your **WSL home** (`~`), which is the distro's native ext4 filesystem:

```bash
cd ~
git clone https://github.com/noonghunna/club-3090.git
cd club-3090
```

**Do not clone under `/mnt/c` or `/mnt/d`.** Those are the Windows drive mounted via DrvFs, which is **10–50× slower** for the many-small-file I/O that git and the scripts do, and it **doesn't preserve Unix file modes** — so the helper scripts lose their exec bit and you hit mysterious `permission denied` failures. Model *weights* are large-but-few files and can live on a Windows drive if you're short on space (see step 7); the repo itself must be on ext4.

## 6. Keep `.env` and scripts as LF — not CRLF ⚠️

If you create or edit `.env` (or any script) with a Windows editor, it may save with **CRLF** line endings. That breaks two things:

- **`docker compose`** reads `GPU_MEMORY_UTILIZATION=0.94\r` — the trailing `\r` becomes part of the *value*, producing baffling "no such file"/invalid-number errors.
- **Shell scripts** fail with `bad interpreter: /usr/bin/env bash^M`.

Prevent it before cloning, and fix any file that slipped through:

```bash
git config --global core.autocrlf input    # set BEFORE cloning
dos2unix .env                               # or: sed -i 's/\r$//' .env
```

In VS Code, set the file's EOL to **LF** (bottom-right status bar) and enable `"files.eol": "\n"`.

## 7. Download weights — `WEIGHTS` + `MODEL_DIR`

**Prerequisite — the `hf` CLI.** `setup.sh` fetches weights with Hugging Face's `hf` CLI. On a fresh Ubuntu 24.04 / WSL, the usual `pip install huggingface-hub` is blocked (`error: externally-managed-environment` — that's [PEP 668](https://peps.python.org/pep-0668/), not a bug on your end). If the CLI is missing, `setup.sh` will **offer to install it for you** — isolated, via `pipx`/`uv`, no changes to system Python — just answer **Y**. To install it yourself first:

```bash
sudo apt install -y pipx
pipx install 'huggingface-hub[hf_transfer]'
pipx ensurepath      # then restart the terminal so `hf` lands on PATH
```

> ⚠️ **Don't use a raw `python -m venv` for this.** Activating a venv rewrites your `PATH` and can hide `docker` and your model paths (a common WSL trip-up). `pipx` installs the `hf` CLI *isolated* but on your PATH — no `activate`, nothing shadowed.

For the **robust single-card path on a 24 GB card** (recommended on WSL2 — see step 10), fetch the **GGUF** weights for llama.cpp / ik_llama:

```bash
WEIGHTS=gguf bash scripts/setup.sh qwen3.6-27b      # Q4_K_M MTP GGUF + vision mmproj, SHA-verified
```

For the vLLM path, omit `WEIGHTS` (defaults to the AutoRound INT4):

```bash
bash scripts/setup.sh qwen3.6-27b
```

**Where weights live:** keep them on ext4 if you have room. If not, point `MODEL_DIR` at a Windows drive both OSes can see — weights are OS-agnostic and the DrvFs slowness barely matters for a few multi-GB files (unlike the repo in step 5):

```bash
export MODEL_DIR=/mnt/d/models      # from WSL; or D:\models from PowerShell
```

Set `MODEL_DIR` **consistently** — either `export` it in your shell *or* put it in the repo-root `.env`, then use it for both `setup.sh` and `launch.sh`. (Mixing the two sources can disagree; see [#187](https://github.com/noonghunna/club-3090/issues/187).)

## 8. Budget for the ~1.3 GiB WSL2 GPU overhead

WSL2's container CUDA context reserves **~1.3 GiB of VRAM that `nvidia-smi` doesn't show at idle** but is locked once a container starts — so the headless-Linux defaults can crash on boot. The fixes (don't repeat them here):

- **Single-card vLLM:** drop `GPU_MEMORY_UTILIZATION=0.94` into `models/qwen3.6-27b/vllm/compose/.env`.
- **Single-card llama.cpp / ik_llama:** lower the context (e.g. `CTX_SIZE=131072`), since these allocate by fixed size, not a ratio.

**Shrink the overhead (not just budget for it).** Part of the ~1.3 GiB is the WSL2 GPU-paravirtualization context itself — unavoidable while you're on WSL2 at all — but the **display/WDDM portion is reclaimable**, often most of it:

- **Don't let the 3090 drive the Windows desktop.** If the display runs off a second GPU or the CPU's integrated graphics, the 3090 is effectively headless and the WDDM/display reservation drops toward zero. **Biggest single lever.**
- **Close GPU-accelerated Windows apps before you boot** — hardware-accelerated browsers (Chrome/Edge), games, video editors, other CUDA jobs all hold VRAM. Confirm `nvidia-smi` in WSL shows near-idle first.
- **Keep the Windows NVIDIA driver current** — newer driver/WSL combos have trimmed the paravirt cost.
- **Disable WSLg if you never run Linux GUI apps** — add `guiApplications=false` under `[wsl2]` in `.wslconfig` (frees the small GPU + RAM the WSLg compositor reserves).
- **Want zero overhead? Dual-boot Linux** — sidesteps WSL2/WDDM (and TDR) entirely.

**Measure your real headroom:** `nvidia-smi` in WSL at idle, then again after boot — the jump above your idle baseline is exactly what you're budgeting for, and these tips shrink that idle baseline.

Full per-compose VRAM table + the combined `.env` template: [FAQ.md → Windows/WSL2](FAQ.md#does-this-work-on-windows--wsl2) and [HARDWARE.md → GPU memory budget on WSL2](HARDWARE.md#note-for-wsl2--windows-users).

## 9. Long-prompt + boot-crash gotchas (TDR, expandable_segments)

Two WSL2-specific failure modes, both fixed on the **Windows** side, both documented in [HARDWARE.md → WSL2/Windows](HARDWARE.md#note-for-wsl2--windows-users):

- **TDR timeout** — Windows force-resets the GPU after 2 s of kernel time; long-context prompts trip it (`CUDA driver error: device not ready`). Fix: raise `TdrDelay` to 60 via the registry + reboot.
- **`expandable_segments` boot crash** — `device not ready` at `gptq_marlin_repack` on some drivers. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (already exposed as a `.env` knob).

## 10. Boot it

Then it's the normal [Quick start](../README.md#quick-start). On a single 24 GB card under WSL2, the **llama.cpp / ik_llama** paths are the most forgiving (no prefill cliffs, smaller VRAM footprint):

```bash
bash scripts/launch.sh --variant ik-llama/iq4ks-mtp   # single-card, leanest VRAM — fits WSL2 at defaults
bash scripts/launch.sh --variant llamacpp/default     # single-card, cliff-immune (drop CTX_SIZE if tight)
bash scripts/launch.sh --variant vllm/dual            # 2 cards — WSL2 overhead is noise at TP=2
```

Sanity-check the endpoint (the launcher prints this curl too):

```bash
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}'
```

---

## 11. Expose the API to your local network (LAN)

By default the endpoint is only reachable **on the machine running WSL2** — even after you set `BIND_HOST=0.0.0.0`. Two things to know:

- **`launch.sh` printing `http://localhost:8020` is cosmetic** (a fixed display string), not the actual bind. Docker already publishes the port on `0.0.0.0`, so the server *is* listening on all interfaces **inside the WSL2 VM** — `BIND_HOST` doesn't change reachability here.
- **The real blocker is WSL2's NAT.** WSL2 runs in a VM with its own IP (`172.x.x.x`); the Windows host's LAN IP does **not** forward to it, so other machines can't reach `172.x.x.x`. Fix it on the **Windows side**, one of two ways:

**Option A — mirrored networking (cleanest; Windows 11 22H2+).** Edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then `wsl --shutdown` (PowerShell) and restart. WSL2 now shares the Windows host's network, so the `0.0.0.0` bind is directly reachable on the host's LAN IP. Allow the port if Windows Firewall prompts.

**Option B — `netsh portproxy` (any Windows version; NAT mode).** In an **admin** PowerShell:

```powershell
wsl hostname -I                 # the WSL2 IP, e.g. 172.20.x.x
netsh interface portproxy add v4tov4 listenport=8020 listenaddress=0.0.0.0 connectport=8020 connectaddress=<WSL2-IP>
New-NetFirewallRule -DisplayName "club3090-8020" -Direction Inbound -LocalPort 8020 -Protocol TCP -Action Allow
```

LAN clients then hit `http://<WINDOWS-host-LAN-IP>:8020/`. ⚠️ In NAT mode the WSL2 IP **changes on reboot** — re-run the `portproxy add` line (or script it). Option A avoids this entirely.

**Verify** from another machine: `curl http://<windows-lan-ip>:8020/v1/models` should list the model. (The `.env` `URL=` is the *client/bench* target — point it at the reachable address; it does **not** affect the server bind.)

---

## Native llama.cpp in WSL (no Docker)

Prefer to skip Docker — one fewer layer, or because you just don't want the daemon? (It's marginally leaner on VRAM, but the real overhead lever is headless + closed apps from [Step 8](#8-budget-for-the-13-gib-wsl2-gpu-overhead), not the engine.) llama.cpp / ik_llama run **natively** in the WSL distro. The GPU passthrough from **Step 2 is all you need**: a native process uses WSL's CUDA libraries (`/usr/lib/wsl/lib`) directly, so there's **no `nvidia-container-toolkit`** and you **skip Step 4** entirely. (vLLM has no native path here — it's Docker-only.)

You still do steps **1–3** (WSL + driver/passthrough + `.wslconfig` RAM) and **5–7** (ext4 clone, LF endings, `WEIGHTS=gguf` weights). Then, instead of `launch.sh` (which drives the Docker composes):

1. **Get a CUDA llama.cpp build** — build it in the distro:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
   ```
   …or grab a prebuilt CUDA binary. To match the MTP / spec-decode support the Docker image ships, track a recent build — the composes pin `ghcr.io/ggml-org/llama.cpp:server-cuda-b9967` (or newer).

2. **Run `llama-server` with the flags the compose uses.** The compose is the source of truth — lift them from [`models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml`](../models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml). The equivalent native invocation:
   ```bash
   ./build/bin/llama-server --host 0.0.0.0 --port 8020 \
     -m "$MODEL_DIR/qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf" \
     -c 200000 -ub 512 -ngl 99 -fa on \
     --cache-type-k q4_0 --cache-type-v q4_0 \
     --spec-type draft-mtp --spec-draft-n-max 2 \
     --jinja --temp 0.6 --top-p 0.95 --top-k 20
   ```
   On a tight WSL2 VRAM budget, drop the context per [Step 8](#8-budget-for-the-13-gib-wsl2-gpu-overhead) — e.g. `-c 131072`. The endpoint then answers the same sanity curl as Step 10.

**Caveat:** going native puts you **off the repo's scripted path** — `launch.sh` / `switch.sh` and the bench / verify / soak scripts all target the Docker containers. You own the `llama-server` process and its flags; treat the compose file as the canonical flag list and re-check it after pulling repo updates.

---

## Recommended config on WSL2

| Hardware | Recommended | Why |
|---|---|---|
| 1× 24 GB (3090/4090) | `ik-llama/iq4ks-mtp` (GGUF) | Leanest VRAM — fits at defaults despite the ~1.3 GiB overhead; no prefill cliffs |
| 1× 24 GB, want vLLM | `vllm/single` + `GPU_MEMORY_UTILIZATION=0.94` `.env` | Full feature stack; needs the WSL2 VRAM + TDR tuning (steps 8–9) |
| 2× 24 GB | `vllm/dual` | TP=2; the ~1.3 GiB overhead is noise at ~17 GB/card |

## Diagnostics on WSL2

Filing a bug or sharing cross-rig data? Run [`report.sh`](../README.md#diagnostics). On a minimal WSL2 distro, install `pciutils` first so the hardware section is complete (it's tiny and not bundled by default):

```bash
sudo apt install -y pciutils
```

⚠️ Expectation-setter: under WSL2, `lspci` lists the GPU as a paravirtualized **"Microsoft Basic Render Driver"**, not your real NVIDIA card — WSL exposes the GPU via `/dev/dxg` (GPU-PV), so the real PCIe link/gen/ACS topology isn't visible. That's normal; `report.sh` reads GPU topology from `nvidia-smi topo` instead. (The `lspci` PCIe/P2P detail only matters for **bare-metal multi-card** P2P diagnosis, not WSL2.)

## See also

- [FAQ.md → Does this work on Windows / WSL2?](FAQ.md#does-this-work-on-windows--wsl2) — runtime VRAM/ctx tuning
- [HARDWARE.md → Note for WSL2 / Windows users](HARDWARE.md#note-for-wsl2--windows-users) — TDR, expandable_segments, the VRAM-overhead formula
- [README → Quick start](../README.md#quick-start) · [SINGLE_CARD.md](SINGLE_CARD.md) — the single-card config detail
