# Patches

This stack tracks `vllm:nightly` (latest dev branch) and adds patches
only when something actively breaks for our TP=2 + Lorbus + fp8 + MTP
configuration.

Currently we have **one** patch dependency.

## vLLM PR #40361 — Marlin pad-sub-tile-n

**What it fixes:** Marlin's `GPTQ_MARLIN_MIN_THREAD_N=64` blocks any W4A16
shard where per-rank out-dim falls below 64. Hits on Ampere sm_86 with
AutoRound INT4 quants under TP=2 — Qwen3.6-27B's DeltaNet `linear_attn`
projections are one such case (similar in shape to the Qwen3.5 GDN
`in_proj_ba` issue tracked in #35924). No stock Ampere fallback kernel
works (Machete/CutlassW4A8 are Hopper-only, AllSpark needs `group_size=-1`).

**Status:** PR open at https://github.com/vllm-project/vllm/pull/40361,
labeled `bug`, awaiting maintainer `ready` label.

**How we apply it:** the patched source lives at `/opt/ai/vllm-src/`
on branch `marlin-pad-sub-tile-n` (fork: `noonghunna/vllm`). The
compose volume-mounts the two patched files over the nightly image's
copies — no rebuild needed.

```yaml
volumes:
  - /opt/ai/vllm-src/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:ro
  - /opt/ai/vllm-src/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:ro
```

When the PR merges, drop both mounts and the compose just uses upstream
nightly.

## What we DO NOT need (and why)

- **Genesis patches** (Sandermage/genesis-vllm-patches) — only needed
  when using TurboQuant KV on hybrid models. We're using fp8_e5m2 KV,
  which goes through vLLM's stock attention backends. No hybrid-gate
  bypass required.
- **`patch_tolist_cudagraph.py`** (from the single-card project) — the
  bug it fixes is in `turboquant_attn.py`'s continuation-prefill path.
  Doesn't fire on fp8 KV.
- **`cudagraph_mode=NONE` workaround for #40831** — the bug is in the
  TurboQuant attention backend. Doesn't fire on fp8 KV. We get full
  cudagraph + torch.compile speed.
- **A pinned image digest** — the single-card project pins
  `vllm/vllm-openai@sha256:9bba4628...` because Genesis's anchor-text
  patches need a specific upstream layout. We don't run Genesis here,
  so we ride `vllm:nightly` and rebuild dependencies only when nightly
  actively breaks our path.

So this setup is meaningfully simpler than the single-card project:
just one upstream patch dependency (Marlin pad), and even that drops
out when #40361 lands.

## Brittleness note

The Marlin patch is a **file override** (volume-mount the entire
patched `marlin.py` and `MPLinearKernel.py` over the container's copies),
not an anchor-based disk-edit. If upstream refactors those files in
nightly while #40361 is still open, our patched versions could fall
out of sync with the rest of vLLM's import graph and crash at load
time with `ImportError` or `AttributeError`.

If that happens:

1. Pull the latest patched files from
   https://github.com/noonghunna/vllm/tree/marlin-pad-sub-tile-n into
   `/opt/ai/vllm-src/`.
2. If the fork is also out of date, rebase it on current main and
   re-apply the pad-sub-tile-n change.
3. Pin the image to the last-known-good digest as a fallback while
   you sort it out.

This is the price we pay for "ride latest nightly" + a still-open patch.
If/when #40361 lands upstream, this entire concern disappears.
