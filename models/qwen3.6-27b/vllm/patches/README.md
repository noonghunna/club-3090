# Patches for Qwen3.6-27B on vLLM

This directory contains the model + engine-specific patches that different compose variants apply at boot. Each patch is opt-in — composes mount only what they need.

| File | Used by | Purpose |
|---|---|---|
| `patch_tolist_cudagraph.py` | single-card default + dual-turbo | CUDA graph capture fix for TurboQuant continuation prefill |
| `patch_pr40798_workspace.py` | (none — research artifact) | Negative-result reproducer for vllm#40798 |
| `genesis/` | single-card default + dual-turbo | Sandermage's Genesis v7.14 patch tree (gitignored; fetched by setup.sh) |
| External: `/opt/ai/vllm-src/` (Marlin pad fork) | all 4 dual-card composes | vLLM PR #40361 patched source, not a file in this repo |

---

## When you need each patch

- **Single-card default** (`docker-compose.yml`) — uses TurboQuant 3-bit KV + Genesis v7.14 P65 + tolist patch. Fetched by `setup.sh qwen3.6-27b`.
- **Single-card fast-chat / no-genesis-mtp / minimal** — fp8 KV, no patches needed.
- **Single-card tools-text** — fp8 KV + Genesis (for the qwen3-coder tool parser); no tolist patch (fp8 doesn't trip the bug).
- **Dual-card default + DFlash variants** — fp8 / fp16 KV. Need only the Marlin pad fork (no Genesis, no tolist).
- **Dual-card turbo** — TQ KV + Genesis v7.14 + tolist + Marlin pad fork.

---

## vLLM PR #40361 — Marlin pad-sub-tile-n (dual-card requirement)

**What it fixes:** Marlin's `GPTQ_MARLIN_MIN_THREAD_N=64` blocks any W4A16 shard where per-rank out-dim falls below 64. Hits on Ampere SM 8.6 with AutoRound INT4 quants under TP=2.

**Status:** PR open at https://github.com/vllm-project/vllm/pull/40361, labeled `bug`, awaiting maintainer review.

**Setup:** all 4 dual-card composes volume-mount the patched source from `/opt/ai/vllm-src/`. Clone the fork once before booting any dual-card compose:

```bash
sudo mkdir -p /opt/ai && sudo chown $USER /opt/ai
git clone -b marlin-pad-sub-tile-n https://github.com/noonghunna/vllm.git /opt/ai/vllm-src
```

The compose then mounts two specific files over the nightly image's copies — no rebuild needed:

```yaml
volumes:
  - /opt/ai/vllm-src/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:ro
  - /opt/ai/vllm-src/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:ro
```

When PR #40361 lands, drop both mounts and the dual composes just use upstream nightly.

### Brittleness note

The Marlin patch is a **file override** (volume-mount the entire patched `marlin.py` and `MPLinearKernel.py` over the container's copies), not an anchor-based disk-edit. If upstream refactors those files in nightly while #40361 is still open, our patched versions could fall out of sync with the rest of vLLM's import graph and crash at load time with `ImportError` or `AttributeError`.

If that happens:
1. Pull the latest patched files from https://github.com/noonghunna/vllm/tree/marlin-pad-sub-tile-n into `/opt/ai/vllm-src/`.
2. If the fork is also out of date, rebase it on current main and re-apply the pad-sub-tile-n change.
3. Pin the image to the last-known-good digest as a fallback while you sort it out.

---

## `patch_tolist_cudagraph.py` (single-card default + dual-turbo)

**What it fixes:** A `.tolist()` GPU→CPU sync in TurboQuant continuation-prefill that's illegal during CUDA graph capture. Trips when `--speculative-config` + `--enable-chunked-prefill` + `turboquant_*` KV are combined. Without this patch, vLLM crashes during engine warmup with:

```
turboquant_attn.py:570  qsl = query_start_loc.tolist()
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

**How:** disk-edit at container startup. Wraps both `.tolist()` sites with `torch.cuda.is_current_stream_capturing()` guards. Idempotent — re-running it on already-patched files is a no-op.

**Why this lives here, not Genesis:** the equivalent functionality also ships as Genesis P78 (since v7.14, with attribution). Both running together is harmless. We keep our standalone version because it's pinned in this repo's git history, while Genesis is fetched fresh per setup.

---

## `patch_pr40798_workspace.py` (research artifact)

**What this is NOT:** a fix.

**What this IS:** a backport of vLLM PR #40798 (workspace-manager refactor) that we hypothesized would close [#40880](https://github.com/vllm-project/vllm/issues/40880) (TurboQuant × spec-decode × cudagraph corruption). Probe 8 in our 9-probe forensics ladder — see [INTERNALS.md](../INTERNALS.md) for full context. The backport was clean; the bug persisted.

We keep this file in the repo for reproducibility of the negative result. No compose mounts it.

---

## Genesis tree (`genesis/`)

Sandermage's [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) — a runtime monkey-patcher for vLLM that fixes several Qwen3-Next architectural bugs (hybrid-attention TurboQuant gate, MTP head loading, cudagraph spec-decode downgrade for #40880, etc).

Setup:
- Cloned by `bash scripts/setup.sh qwen3.6-27b` at the pinned tag (currently `v7.51-stable-2026-04-27`)
- Override the pin via `GENESIS_PIN=<tag-or-commit>` env var
- Gitignored from this repo (we don't vendor someone else's tree)

Mounted by composes that need TurboQuant KV (single-default + dual-turbo) or the Qwen3 tool-parser fixes.

---

## Genesis env-opts (per-patch toggles)

Genesis v7.14+ ships several patches as opt-in env flags. Each compose enables only the subset relevant to its config:

| Env var | What it does | Used by |
|---|---|---|
| `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1` | Forces cudagraph PIECEWISE for spec-decode (closes #40880) | single-default, dual-turbo |
| `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1` | Filters cudagraph capture sizes for spec-decode divisibility | single-default, dual-turbo |
| `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` | Streaming MTP tool-call edge case fix | single-default, fast-chat, tools-text, dual-turbo |
| `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1` | Long-ctx tool-format adherence | single-default, fast-chat, tools-text, dual-turbo |
| `GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1` | Long-ctx tool-format reminder | single-default, fast-chat, tools-text, dual-turbo |

Composes that don't load Genesis (no-genesis-mtp, minimal, dual-default, dual-dflash, dual-dflash-noviz) ignore these env vars.
