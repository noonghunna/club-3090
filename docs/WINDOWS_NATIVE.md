# Running club-3090 models on Native Windows (no WSL2, no Docker)

This page documents a **validated production path** for running llama.cpp inference directly on native Windows — no WSL2, no Docker Desktop, no GPU passthrough layer. Just `llama-server.exe` + GGUF files.

> Contributed by @Isaac-opz. Validated 2026-08 on: RTX 3090 (sm_86, 24 GB), Ryzen 5 5600G, 32 GB DDR4, Windows 10, llama.cpp build b10435 (CUDA 13).

---

## Why this page exists

The main [`WSL_SETUP.md`](WSL_SETUP.md) states:

> *"Native Windows runs only the upstream llama.cpp binary — none of this repo's tooling."*

That was true for the **Docker Compose + bash script** tooling (which is Linux-only). But the underlying inference path — `llama-server.exe` serving a GGUF model with an OpenAI-compatible API — works **perfectly on native Windows** when you apply three critical flags. This page documents that path so Windows users don't have to install WSL2 + Docker Desktop + NVIDIA Container Toolkit just to run a local LLM.

**What you skip:**
- ❌ WSL2 installation + Ubuntu distro
- ❌ Docker Desktop (or docker-ce + nvidia-container-toolkit)
- ❌ GPU passthrough configuration
- ❌ `.wslconfig` RAM tuning
- ❌ ext4 filesystem cloning
- ❌ `wsl --shutdown` recovery cycles

**What you keep:**
- ✅ Full llama.cpp feature set (MTP, DFlash, vision, tool calling, metrics)
- ✅ OpenAI-compatible API on `localhost:18080`
- ✅ All the same models, quants, and context sizes as the WSL2 path
- ✅ Lower latency (no VM boundary for GPU access)

---

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU | NVIDIA RTX 3090 (24 GB) or similar | sm_86+ (Ampere). Driver 580.x+ recommended. |
| RAM | 32 GB DDR4/DDR5 | Model pages are pinned via `--mlock`. You need ≥ model size + ~4 GB headroom in physical RAM. |
| OS | Windows 10/11 x64 | No WSL2 required. |
| llama.cpp | Build b10435+ (CUDA 13) | Pre-built Windows binaries available from GitHub releases or community builds. |
| Disk | NVMe SSD recommended | Cold start pages in the model (~17 GB). HDD works but first request is very slow. |

---

## The three critical flags

These are the flags that make native Windows work where naive invocations fail:

### 1. `--mlock` (NON-NEGOTIABLE)

**Problem:** Windows' memory manager treats llama-server's RAM as reclaimable. When other processes need memory (browsers, IDEs, OS services), it evicts pages from the model. Since the model is fully loaded in VRAM but some tensors/activations touch system RAM, eviction causes:
- Decode speed drops from ~40 t/s to ~10-15 t/s
- Intermittent stalls (page faults)
- Performance degrades over time as more pages get evicted

**Fix:** `--mlock` pins all model pages in physical RAM. They cannot be swapped or evicted. Performance stays stable indefinitely.

**Cost:** The model's RAM footprint becomes permanent. On a 32 GB system with a ~17 GB model, you have ~11 GB for everything else. Fine for a dedicated inference machine; tight if you're also running Chrome + VS Code.

### 2. `--ubatch-size` sizing (THE CLIFF LEVER)

**Problem:** The default `ubatch_size=1024` (or `512` in some builds) allocates a per-pass activation buffer proportional to the ubatch size × model width. At high context (170K+) with MTP spec-decode, this buffer pushes total VRAM over 24 GB:
- Symptom: server boots fine, first request starts prefill, then decode collapses to ~8-10 t/s (VRAM thrashing)
- Or: outright OOM at prefill

**Fix:** Reduce `--ubatch-size` to match your context + speculative decoding load:

| Context | MTP | ubatch_size | Why |
|---------|-----|-------------|-----|
| 262K | off | **128** | Smallest activation buffer, fits in remaining VRAM after KV |
| 170K | on (n=2) | **512** | MTP draft adds ~1.5 GB; 512 keeps activations under control |
| 131K | off | **128-512** | Comfortable headroom either way |
| 64K | any | **512-1024** | Default is fine at low context |

This is the **same mechanism** as vLLM's "Cliff 2" (documented in [`CLIFFS.md`](CLIFFS.md)) — per-pass activation peak exceeding VRAM budget — but in llama.cpp it's a simple CLI flag rather than requiring a kernel-level streaming refactor.

### 3. `--gpu-layers 99` (NOT `auto`)

**Problem:** With `--gpu-layers auto` + `--fit on`, the fit logic logs:
```
failed to fit params ... n_gpu_layers already set by user to 99
```
and skips the fit calculation. The server may boot with an unexpected layer split, or the fit message is confusing and makes you think something is wrong.

**Fix:** Hardcode `--gpu-layers 99` (all layers on GPU). On a 24 GB card with a Q4 quant (~17 GB), all layers fit. Don't let the auto-logic second-guess you.

---

## Additional Windows-specific notes

### `--parallel 1` always

The default `--parallel auto` resolves to **4** concurrent slots. Each slot gets its own KV cache allocation. At 262K context, that's 4× the KV pool → instant OOM. On a single 24 GB card, you want exactly one slot:

```
--parallel 1
```

Concurrent requests queue (fIFO). For single-user/agent use, this is invisible.

### Cold start behavior

The first request after boot is **significantly slower** than subsequent ones. Windows needs to page in the model from disk into RAM (even with `--mlock`, the initial load goes through the page cache). On an NVMe SSD: ~5-15 seconds for a 17 GB model. On HDD: minutes.

This is NOT a bug. Don't benchmark on the first token. Wait for one warm request, then measure.

### `--image-min-tokens` at high context

The default `--image-min-tokens 1024` reserves VRAM for image processing even when no images are sent. At 170K+ context where you're already at ~24 GB, this reservation can be the difference between fitting and not fitting. If you're pushing context and don't use vision:

```
--image-min-tokens 512
```

Or omit `--mmproj` entirely if you don't need vision.

### Port binding

`--host 0.0.0.0` binds to all interfaces (useful for Tailscale/LAN access). `--host 127.0.0.1` restricts to localhost. For a home setup with Tailscale, `0.0.0.0` + firewall rules is the standard pattern.

### Process management

On native Windows, there's no `setsid` or `< /dev/null` equivalent for daemonizing. Options:
- **PowerShell wrapper** (recommended): a `.ps1` script that kills any existing instance, starts the server, logs to file
- **NSSM** (Non-Sucking Service Manager): wraps the exe as a Windows service with auto-restart
- **Task Scheduler**: run at logon, restart on failure

Example PowerShell wrapper pattern:
```powershell
# Kill existing
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

# Start
& "C:\path\to\llama-server.exe" `
  --model "Z:\Models\qwen38\Qwen3.8-27B-UD-Q4_K_XL.gguf" `
  --mmproj "Z:\Models\qwen38\mmproj-Qwen3.8-27B-Q8_0.gguf" `
  --alias qwen3.8-27b-mtp-q4xl-170k `
  --host 0.0.0.0 --port 18080 `
  --ctx-size 174080 --gpu-layers 99 `
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 `
  --batch-size 2048 --ubatch-size 512 `
  --mlock --parallel 1 `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --reasoning off `
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 `
  --timeout 3600 --metrics --slots --no-ui
```

---

## What DOESN'T work on native Windows (yet)

| Feature | Status | Workaround |
|---------|--------|------------|
| Docker Compose configs | ❌ Linux-only | Use the reference YAMLs in `models/*/llama-cpp/compose/single/` as documentation, invoke manually |
| `scripts/setup.sh`, `launch.sh`, `switch.sh` | ❌ Bash + Docker | Manual invocation or PowerShell wrappers |
| vLLM / SGLang | ❌ Linux + CUDA only | llama.cpp is the path (which is fine — it's the cliff-immune engine anyway) |
| `c3` TUI cockpit | ❌ Python + Linux | Use your own monitor/dashboard or the `/metrics` endpoint |
| Genesis patches | ❌ vLLM-specific | N/A for llama.cpp path |

---

## Performance comparison: Native Windows vs. WSL2

Measured on identical hardware (RTX 3090, same model, same config):

| Metric | Native Windows | WSL2 + Docker |
|--------|---------------|---------------|
| Decode TPS (warm) | ~40-44 t/s | ~40-44 t/s (identical) |
| Prefill TPS | ~900-1000 t/s | ~850-950 t/s (slight VM overhead) |
| Cold start (first request) | ~5-15s (NVMe page-in) | ~10-20s (VM + Docker startup) |
| VRAM usage | Identical | Identical |
| RAM overhead | ~0 (no VM) | ~2-4 GB (WSL2 VM baseline) |
| Complexity | 1 exe + model files | WSL2 + Ubuntu + Docker + NVIDIA toolkit |

**Verdict:** No performance penalty for native Windows. Slightly faster prefill (no VM boundary). Significantly simpler setup. The only reason to use WSL2 is if you need the bash script tooling or want to run vLLM/SGLang (which are Linux-only).

---

## Recommended setup for a dedicated inference machine

If this PC exists primarily to serve local LLMs:

1. **Install llama.cpp** — pre-built Windows CUDA binary (b10435+). Put it in a stable path like `C:\llama\`.
2. **Download model files** — put GGUF + mmproj on your fastest drive (NVMe). `Z:\Models\` or `D:\models\`.
3. **Write a PowerShell wrapper** — one `.ps1` per profile (see examples in `models/qwen3.8-27b/`).
4. **Pin the process** — `--mlock` is in every config. Ensure your RAM budget allows it.
5. **Add to startup** — Task Scheduler or a simple `.bat` in `shell:startup`.
6. **Monitor** — hit `http://localhost:18080/metrics` (Prometheus) or `/v1/models` (health check).

Total setup time: ~10 minutes. No WSL2, no Docker, no Linux knowledge required.

---

## FAQ

**Q: Can I run multiple models simultaneously?**
A: On a single 24 GB card, no. One model at a time. You can have multiple `.ps1` wrappers and switch between them (kill one, start another). On dual 3090s, you could run two instances on separate GPUs with `--gpu 0` / `--gpu 1`.

**Q: Does `--mlock` work if I have a RAM disk or fast NVMe?**
A: `--mlock` pins in **physical RAM**, not on disk. If you don't have enough physical RAM for the model + OS, `--mlock` will fail at boot (the process can't allocate). On 32 GB with a 17 GB model, you're fine. On 16 GB, you're not.

**Q: Why not just use LM Studio or Ollama?**
A: You can! They wrap the same llama.cpp engine. But they don't expose all the flags (MTP spec-decode, custom ubatch sizing, `--mlock` control). For production agentic workloads where you need precise VRAM management, direct `llama-server.exe` gives you full control. LM Studio is great for casual use.

**Q: Does this work on RTX 4090 / 5090?**
A: Yes. Same flags, same approach. The 4090 has tighter idle VRAM (display output reserves ~300 MB), so you may need slightly lower context or `--image-min-tokens 512`. The 5090 (32 GB) gives you more headroom — you can push higher context or enable MTP at 262K.
