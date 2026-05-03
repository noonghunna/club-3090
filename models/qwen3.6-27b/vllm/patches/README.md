# Patches for Qwen3.6-27B on vLLM

This directory contains the model + engine-specific patches that different compose variants apply at boot. Each patch is opt-in — composes mount only what they need.

| File | Used by | Purpose |
|---|---|---|
| `patch_tolist_cudagraph.py` | single-card default + dual-turbo | CUDA graph capture fix for TurboQuant continuation prefill |
| `patch_pn12_ffn_pool_anchor.py` | long-text | Local PN12 anchor repair for the SiluAndMul FFN intermediate pool on dev205+ |
| `patch_fa_max_seqlen_clamp.py` | long-text | Local P104-style FA2 `max_seqlen_k` runtime clamp |
| `patch_pr40798_workspace.py` | (none — research artifact) | Negative-result reproducer for vllm#40798 |
| `genesis/` | single-card default + dual-turbo | Sandermage's Genesis v7.14 patch tree (gitignored; fetched by setup.sh) |
| `vllm-marlin-pad/marlin.py` + `MPLinearKernel.py` | all 4 dual-card composes | vLLM PR #40361 patched source, vendored in-repo |

---

## When you need each patch

- **Single-card default** (`docker-compose.yml`) — uses TurboQuant 3-bit KV + Genesis v7.14 P65 + tolist patch. Fetched by `setup.sh qwen3.6-27b`.
- **Single-card minimal** — fp8 KV, no patches needed (escape hatch if `setup.sh` Genesis clone fails).
- **Single-card tools-text** — fp8 KV + Genesis (P64 qwen3-coder tool parser fix + PN8 memory savings); no tolist patch (fp8 doesn't trip the bug).
- **Dual-card default + DFlash variants** — fp8 / fp16 KV. Need only the Marlin pad fork (no Genesis, no tolist).
- **Dual-card turbo** — TQ KV + Genesis v7.14 + tolist + Marlin pad fork.

---

## vLLM PR #40361 — Marlin pad-sub-tile-n (dual-card requirement)

**What it fixes:** Marlin's `GPTQ_MARLIN_MIN_THREAD_N=64` blocks any W4A16 shard where per-rank out-dim falls below 64. Hits on Ampere SM 8.6 with AutoRound INT4 quants under TP=2.

**Status:** PR open at https://github.com/vllm-project/vllm/pull/40361, labeled `bug`, awaiting maintainer review.

**Setup:** ✅ Nothing. The patched files are vendored in this repo at `vllm-marlin-pad/`. All 4 dual-card composes mount them via repo-relative paths — no host filesystem dependency, no manual `git clone` step. (Previous design required cloning the fork to `/opt/ai/vllm-src/`; vendored on 2026-05-03 to fix [club-3090#37](https://github.com/noonghunna/club-3090/issues/37).)

The compose mounts look like:

```yaml
volumes:
  - ../patches/vllm-marlin-pad/marlin.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:ro
  - ../patches/vllm-marlin-pad/MPLinearKernel.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:ro
```

When PR #40361 lands upstream, the entire `vllm-marlin-pad/` directory + the four compose mount lines get deleted, and dual composes just use upstream nightly.

### Brittleness note

The Marlin patch is a **file override** (mount the entire patched `marlin.py` and `MPLinearKernel.py` over the container's copies), not an anchor-based disk-edit. If upstream refactors those files in nightly while #40361 is still open, our vendored versions could fall out of sync with the rest of vLLM's import graph and crash at load time with `ImportError` or `AttributeError`.

If that happens:
1. Re-sync the vendored files: see `vllm-marlin-pad/README.md` for the procedure (verify upstream files haven't changed since fork base, re-copy from a fresh clone of the patched fork).
2. If the fork is also out of date, rebase it on current main and re-apply the pad-sub-tile-n change before re-copying.
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

## `patch_pn12_ffn_pool_anchor.py` (long-text Cliff 1 attack)

**What it fixes:** Genesis PN12 can silently no-op on vLLM dev205+ because its text anchor expects `SiluAndMul.forward_cuda` to be followed directly by `@CustomOp.register("silu_and_mul_with_clamp")`. Current `activation.py` has the `MulAndSilu` section there instead, so the Genesis dispatcher can say PN12 applied while the live method still allocates with `torch.empty(output_shape, ...)`.

**How:** disk-edit at container startup, after Genesis. Replaces only `SiluAndMul.forward_cuda` with PN12's env-gated pooled-output body, anchored within the current `SiluAndMul` class. Runtime pooling still requires `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1`.

**Why this lives here, not Genesis:** this is a local Cliff 1 diagnostic/fix path. Do not push it to Sandermage's tree unless we decide on a PR after the local stack is proven.

---

## `patch_fa_max_seqlen_clamp.py` (long-text Cliff 1 attack)

**What it fixes:** FlashAttention 2 varlen allocates `softmax_lse` from `max_seqlen_k`, not the actual `cu_seqlens_k` span. Long-context TurboQuant metadata can pass a conservative upper bound, so a much shorter chunk pays the full max-context workspace cost.

**How:** disk-edit at container startup, after Genesis. Replaces TurboQuant's `_flash_attn_varlen` helper so runtime calls clamp `max_seqlen_k` to `max(actual cu_seqlens_k span, max_seqlen_q)`. It skips CUDA graph capture and still requires `GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP=1`.

**Why this lives here, not Genesis:** this is the held P104 work converted into a local sidecar. Keep it local until the Cliff 1 stack is proven and we explicitly decide PR strategy.

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
| `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` | Streaming MTP tool-call edge case fix (kotori-yan vllm#39598 backport) | single-default, tools-text, dual-turbo |
| `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1` | MTP draft online-quant propagation (vllm#40849 backport) — frees ~800 MiB on FP8+MTP | tools-text |
| ~~`GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1`~~ | ~~Long-ctx tool-format adherence~~ — **disabled 2026-04-29**, breaks IDE agents (see [club-3090#2](https://github.com/noonghunna/club-3090/issues/2#issuecomment-4346740245)) | (none — opt-in only) |
| ~~`GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1`~~ | ~~Long-ctx tool-format reminder~~ — **disabled 2026-04-29**, same reason | (none — opt-in only) |

Composes that don't load Genesis (minimal, dual-default, dual-dflash, dual-dflash-noviz) ignore these env vars.
