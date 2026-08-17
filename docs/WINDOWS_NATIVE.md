# llama.cpp on Native Windows (no WSL2, no Docker)

Reference notes for running `llama-server.exe` directly on native Windows. Covers the flags that differ from the Linux/WSL2 path and the operational patterns that work.

> Single-rig data: RTX 3090 (sm_86), Ryzen 5 5600G, 32 GB DDR4, Windows 10, llama.cpp build b10435 (CUDA 13). See [`models/qwen3.8-27b/`](../models/qwen3.8-27b/README.md) for a full worked example.

---

## Flags that matter on native Windows

### `--mlock`

Pins model pages in physical RAM so the OS cannot evict them under memory pressure. Without it, sustained mixed usage (browser + IDE + inference) causes decode to drop from ~40 t/s to ~10–15 t/s as pages get paged out. With `--mlock`, performance stays stable across extended sessions.

Cost: the model's RAM footprint becomes permanent. On 32 GB with a ~17 GB model, you have ~11 GB for everything else. Fine for a dedicated inference box; tight if you're also running Chrome + VS Code.

Less critical on Linux/WSL2 where the page cache and OOM killer behave differently.

### `--ubatch-size` sizing

The per-pass activation buffer is proportional to ubatch size × model width. At high context with MTP spec-decode, the default (1024) can push total VRAM over 24 GB:

| Context | MTP | ubatch_size | Rationale |
|---------|-----|-------------|-----------|
| 262K | off | 128 | Smallest activation buffer, fits after KV |
| 170K | on (n=2) | 512 | MTP draft adds ~1.5 GB; 512 keeps activations under control |
| 131K | off | 128–512 | Comfortable headroom either way |
| ≤64K | any | 512–1024 | Default is fine at low context |

Symptom of wrong sizing: server boots, prefill starts, then decode collapses to ~8–10 t/s (VRAM thrashing) or outright OOM. Same mechanism as vLLM "Cliff 2" in [`CLIFFS.md`](CLIFFS.md), but tunable via `-ub` in llama.cpp.

### `--gpu-layers 99` (hardcoded)

With `auto` + `--fit on`, the fit logic logs:
```
failed to fit params ... n_gpu_layers already set by user to 99
```
and skips its calculation. Hardcoding 99 avoids the ambiguity. On 24 GB with a Q4 quant (~17 GB), all layers fit.

### `--parallel 1`

The default (`auto`) resolves to 4 concurrent slots, each with its own KV cache allocation. At high context on 24 GB, that's instant OOM. Single-slot queues requests (FIFO) — invisible for single-user/agent use.

---

## Cold start

First request after boot is significantly slower than subsequent ones. Windows pages the model in from disk into RAM (~5–15 s on NVMe for a 17 GB file; minutes on HDD). This is expected, not a bug. Don't benchmark on the first token.

---

## `--image-min-tokens` at high context

The default (1024) reserves VRAM for image processing even when no images are sent. At 170K+ where you're already near the ceiling, this reservation can be the difference between fitting and not fitting. Options:
- `--image-min-tokens 512` if you use vision but want to reclaim a little VRAM
- Omit `--mmproj` entirely if you don't need vision

---

## Process management

No `setsid` or `< /dev/null` equivalent on native Windows. Options:

**PowerShell wrapper (simplest):**
```powershell
# Kill existing
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

# Start
& "C:\llama\llama-server.exe" `
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

**NSSM** (Non-Sucking Service Manager): wraps the exe as a Windows service with auto-restart on crash.

**Task Scheduler:** run at logon, restart on failure.

---

## What doesn't work on native Windows

| Feature | Status | Alternative |
|---------|--------|-------------|
| Docker Compose configs | Linux-only | Use the reference YAMLs in `models/*/llama-cpp/` as documentation; invoke manually |
| `scripts/setup.sh`, `launch.sh`, `switch.sh` | Bash + Docker | Manual invocation or PowerShell wrappers |
| vLLM / SGLang | Linux + CUDA only | llama.cpp is the path (which is fine — it's the cliff-immune engine) |
| `c3` TUI cockpit | Python + Linux | Use `/metrics` endpoint or your own dashboard |
| Genesis patches | vLLM-specific | N/A for llama.cpp path |

---

## Port binding

- `--host 0.0.0.0`: all interfaces (Tailscale/LAN access). Standard for home setups with Tailscale + firewall rules.
- `--host 127.0.0.1`: localhost only. Safer default if you don't need remote access.

---

## FAQ

**Can I run multiple models simultaneously?**
On a single 24 GB card, no. One model at a time. Multiple `.ps1` wrappers let you switch (kill one, start another). On dual 3090s, two instances on separate GPUs with `--gpu 0` / `--gpu 1`.

**Does `--mlock` work if I don't have enough RAM?**
No. It pins in physical RAM. If you can't allocate the model + OS in physical memory, `--mlock` fails at boot. On 32 GB with a 17 GB model: fine. On 16 GB: not.

**Why not LM Studio or Ollama?**
They wrap the same llama.cpp engine but don't expose all flags (MTP spec-decode, custom ubatch sizing, `--mlock` control). For production agentic workloads where you need precise VRAM management, direct `llama-server.exe` gives full control. LM Studio is fine for casual use.

**Does this work on RTX 4090 / 5090?**
Same flags, same approach. The 4090 has tighter idle VRAM (display output reserves ~300 MB), so you may need slightly lower context or `--image-min-tokens 512`. The 5090 (32 GB) gives more headroom — you can push higher context or enable MTP at 262K.
