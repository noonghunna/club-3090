# Single 3090 — history, forensics & watch-list

This is the **Tier-2 archive** for [`SINGLE_CARD.md`](SINGLE_CARD.md) — the failure forensics, superseded
configs, and not-yet-recommended drafter tracking that used to clutter the recipe page. Nothing here is
needed to *run* a single-card config; it's the rationale trail for "why X isn't recommended" and "what we
already tried." The cliff mechanism itself is single-sourced in [`CLIFFS.md`](CLIFFS.md).

---

## Other single-card variants in the repo (not recommended for shipping)

These exist for troubleshooting, niche workloads, or historical comparison — the recommended recipes in
[`SINGLE_CARD.md`](SINGLE_CARD.md) cover their use cases:

- **`tq3-mtp.yml`** — 48K + TQ3 + vision, mem-util 0.92. The "below both cliffs by definition" baseline
  (engine HTTP-400-rejects requests >48K, so Cliff 2 is unreachable). Useful for bulletproof error behavior
  on a small-ctx workload, or as a fast-boot diagnostic. Most users should pick `long-vision` or `llamacpp/default`.
- **`tools-text.yml`** — 75K + FP8 KV + PN8. Was the only Cliff-1-safe single-card path before the PN12 anchor
  fix landed. FP8 KV is closer to FP16 quality than TQ3, so kept for accuracy-sensitive comparisons.
- **`minimal.yml`** — 32K + FP8 + no Genesis + no spec-decode. Stripped-down stack for isolating "is this a
  Genesis bug?" questions. Half the throughput of any other variant. (Also the `@stiggy2k16` small-context
  vLLM-speed escape hatch — Genesis-free; run on a current image via `VLLM_IMAGE=vllm/vllm-openai:latest`
  until its pin is bumped, [#167](https://github.com/noonghunna/club-3090/issues/167).)

---

## Superseded vLLM single-card detail (blocked on [#167](https://github.com/noonghunna/club-3090/issues/167))

The TQ3 + Genesis single-card vLLM configs (`long-vision` / `long-text` / `long-text-no-mtp` /
`bounded-thinking`) are pinned to a purged vLLM nightly and won't boot until the next Genesis-compatible pin
lands. They remain in the repo for when the pin returns. The full measured detail:

- **`long-vision.yml`** — 145K + vision tower + TQ3 KV + DS layout + Genesis MTP n=3 + full v7.69 patch stack
  (PN12 + PN17 + PN25 + PN30 part3 + PN26b + P38B + P15B + PN33 + PN32 GDN chunked-prefill) at mem-util 0.95.
  verify-stress 7/7 on text-only paths post-v7.69; vision tower's ~1 GB tightens the single-prompt envelope.
  Code 66 / narr 50 TPS (n=5, CV 2–4%), AL 3.40–3.56.
- **`long-text.yml`** (Balanced MTP) — 180K + TQ3 KV + DS layout + MTP K=3 + the v7.69 stack + local
  vllm#35975 backport at mem-util 0.93. **60K single-prompt PASS @ 623s wall** (HTTP 200, recall correct,
  AL=4.00). Code 67 / narr 50 TPS (n=5, CV 2.6%). Was the steady-state default.
- **`long-text-no-mtp.yml`** (Max-context) — 200K + TQ3 KV + DS layout + MTP **off** + the v7.69 stack +
  vllm#35975 backport at mem-util 0.95. **60K single-prompt PASS @ 537s wall**. Decode ~33 narr / ~40 code TPS
  (no spec-decode). Use when KV headroom beats steady-state TPS.

---

## The two cliffs — single-card forensics

Full mechanism + cross-engine picture: [`CLIFFS.md`](CLIFFS.md). Single-card-specific notes:

### Cliff 2a — single-prompt OOM (mostly closed on v7.69) ✅
**Pre-v7.69:** vLLM single-card variants crashed on single prompts >~50K tokens. **Post-v7.69 + vllm#35975 +
0.93 mem-util (Balanced MTP) / MTP-off + 0.95 (Max-context):** 60K single-prompt passes cleanly (HTTP 200,
recall correct, AL=4.00 on Balanced MTP). 90K is past wall-clock-feasible on this hardware. The fix combines
[vllm#35975](https://github.com/vllm-project/vllm/pull/35975) (skip `inputs_embeds` GPU buffer for text-only
models, frees ~444 MiB at boot) with mem-util tuning for the late-stage 50 MiB allocation.

### Cliff 2b — accumulated-context OOM under multi-turn agent traffic (NOT closed) ❌
Same kernel, different trigger. The 50 MiB `chunk_fwd_o → torch.empty_like(v)` allocation also fails when
accumulated multi-turn KV + GDN forward live-tensor cascade peaks above 24 GiB. Hits at **~21–26K accumulated
tokens** across 4–5 turns of hermes/openhands/OpenCode/Cline/OpenClaw. Validated 2026-05-03 across all six
shipped single-card vLLM composes; only `vllm/dual` (TP=2) and `llamacpp/default` survive. The simultaneous
live-set of `chunk_gated_delta_rule_fwd` is ~500 MiB at T=4128 (q/k/v/u/v_new/o/w/A/Ai/h all alive at once);
adding KV + model + workspace + MTP draft exceeds 24 GiB. Kernel-level fix only — not a config knob.

**2026-05-05 — Genesis v7.72.2's PN59 streaming-GDN doesn't close it.** Sander shipped PN59 as the structural
Cliff 2b fix, but its eligibility check rejects calls with `chunk_indices`/`chunk_offsets` populated — which
vLLM's mandatory `--max-num-batched-tokens 4128` always sets on 24 GB single-card. PN59 falls through to
vanilla code which OOMs at the same site. Filed at
[Sandermage/genesis-vllm-patches#22](https://github.com/Sandermage/genesis-vllm-patches/issues/22) with a
reproducer + 4 fix proposals; pending review. The two safe paths (dual / llama.cpp) remain the recommendation.

### Cliff 1 mech B (closed) ✅
An inductor compile-path FFN intermediate-buffer leak ([#16](https://github.com/noonghunna/club-3090/issues/16))
that crashed long-* variants on real IDE-agent prompts. **Closed 2026-05-02** via Genesis PN25 (Inductor-safe
`silu_and_mul` opaque op) + PN30 (DS conv state dst-shaped temp fix). Both ship by default; no user action needed.

---

## Watch list — Luce DFlash + PFlash (not a recommendation)

Re-tested **2026-05-20** against [`Luce-Org/lucebox-hub`](https://github.com/Luce-Org/lucebox-hub) HEAD
`248e191` on Qwen3.6-27B Q4_K_M target + the released
[`Lucebox/Qwen3.6-27B-DFlash-GGUF`](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF) draft. The 3.6
draft has shipped and PFlash now exists in the binary — **but Luce still loses to `llamacpp/mtp` on every
realistic workload except greedy-only code:**

| Workload | Luce DFlash @ temp=0.0 (greedy) | Luce DFlash @ temp=0.6 (realistic) | **llamacpp/mtp (any temp)** |
|---|---|---|---|
| Narrative essay | 37–47 TPS (mean ~40) | ~20 TPS (DDTree silent → AR-equivalent) | **51 TPS** ⭐ |
| Code (heap/LRU/AST) | 63–76 TPS (mean ~72) | ~20 TPS | **60 TPS** |
| AL (code) | 5.9–7.1 | n/a (DDTree off) | 3.4 (MTP) |

**What works at HEAD 248e191:** ✅ tool calls via native Qwen3 path (no `server_tools.py` patch); ✅ streaming
SSE with `reasoning_content` deltas; ✅ daemon mode with cache-reuse; ✅ PFlash compresses at small source ctx
(12K source → 1011 tokens kept (8.2%) → 3.3s response; drafter Qwen3-0.6B BF16 loads when max-ctx ≤ 16K).

**What keeps it off the recommended list (re-confirmed 2026-05-20):**
- ❌ **Greedy only** — at temp > 0, DDTree silently disables and decode collapses to ~20 TPS (AR-equivalent).
  Kills chat / IDE-agent / creative workloads. **THE shipping gate.**
- ❌ **Loses to `llamacpp/mtp` at realistic temperatures** (20 t/s vs 51–60), and `llamacpp/mtp` ships clean
  via `ghcr.io/ggml-org/llama.cpp:server-cuda` (no source build, no Python+CUDA toolchain).
- ❌ **PFlash on single 3090 is structurally OOM-constrained via the C++ HTTP server** — drafter (1.43 GB) +
  target + KV + DDTree workspace OOMs at max-ctx ≥ 32K. Working ceiling max-ctx=16K caps source prompts far
  below the 128K headline (Luce's 128K bench is via Python `server.py` / `test_dflash`, not the production
  HTTP server). Dual-GPU may unlock the C++ HTTP path at long source ctx — untested.
- ❌ **verify-stress prompt format triggers premature EOS at long-ctx + greedy** (Luce upstream
  [#121](https://github.com/Luce-Org/lucebox-hub/issues/121) shape, extends to the non-PFlash + FP16-KV path).
- ❌ **No vision tower.**
- ❌ **`enable_thinking` chat_template_kwargs handled differently** than vLLM — verify-full check 6 fails.
- ❌ **Daemon-mode hang at long source ctx** — first ≥50K request can lock the GPU for minutes; cleared by restart.
- ❌ **Build fragility** — fresh clone needs `git submodule update --init` +
  `-DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_ENABLE_BSA=ON`; no upstream docker image.

**Re-test triggers (any one would change the verdict):** Luce adds non-greedy DDTree sampling (the single
biggest unblock); Luce-Org publishes an official docker image; the C++ HTTP server validates source length
*after* PFlash compression; or we add a dual-GPU compose target and confirm the PFlash drafter fits. Track in
[`UPSTREAM.md`](UPSTREAM.md#luce-dflash-luce-orglucebox-hub).
