# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-08-21 — Incubating DFLASH15 fast-target compose

Added `vllm/qwen38-27b-dual-dflash15-fast`: a one-link HF copy of the optimized W4A16 target plus the external DFlash2 W4A16 drafter. It carries lookup-augmented DFlash2, split-KV FlashAttention, hybrid KV/CUDAGraph sizing, Ninja/CUDA-header mounts, the WSL2 UVA fallback and the max-context/single-stream envelope. The canonical defaults are 244,320 tokens at GPU util `.85`, `MAX_NUM_SEQS=1`, a 9,000,000,000-byte KV pool, CUDAGraph reserve 1900 MiB and capture size 16. The architectural ceiling is 262K and would likely need `.90` utilization, but one GPU drives the desktop on this rig; `.90` OOMs under soak, so 244,320 is the reproducible ceiling here. The v1 checkpoint intentionally retains MTP artifacts for exact reproduction; a headless v2 is a separate follow-up. Local evidence is approximately 105.94 narrative / 188.06 code decode TPS on 2x RTX 3090 WSL2; the upstream-image re-run measured ~93.69 / ~163.82 across two canonical benches, passed verify-full/stress and continuous soak, and is recorded in `BENCHMARKS.md`. Vision remains enabled (`language_model_only=false`); the post-change verify-full/stress runs passed, and a real base64 PNG request returned a coherent image description (HTTP 200). Full quality with thinking force-on scored **126/150 medium** and **131/150 low** at pass@1 (**133/150** and **138/150** at pass@3).

## 2026-08-21 — MAX_NUM_SEQS=2 and scheduling controls (experimental)

The DFLASH15 compose now exposes the upstream HOL controls (`--long-prefill-token-threshold=4096`, `--enable-chunked-prefill`) and the Qwen3.8 default chat/sampler controls (`enable_thinking=false`, `reasoning_effort=low`, instruct sampler override). A separate `MAX_NUM_SEQS=2` run on the same `.85`/244K pool measured **92.30 / 161.29 decode TPS**. The concurrency probe passed 5 rounds at 2×16K with **81.0 tok/s per stream**, 99.1% retention and 0 MiB post-warm growth (peak 41,958 MiB). The agentic ramp had 0 tool-call misses but showed the expected GDN/SSM recurrent-prefill growth: TTFT 1.31s → 9.26s from 1.5K → 35.4K accumulated tokens. The registry/default remains MAX_NUM_SEQS=1; this variant is not promoted to the full-context default.

## 2026-08-21 — DFlash2 super/ultra tiers benched (full matrix); iq4xs single-card slug; HOL flag

**The DFlash2 tier hierarchy is measured.** All six dual slugs benched fresh, same session (canonical 3 warm + 5 measured, stock `vllm/vllm-openai:v0.27.1` + the vendored [`vllm-dflash2-backport`](vllm/patches/vllm-dflash2-backport/README.md) of [vllm#52816](https://github.com/vllm-project/vllm/pull/52816)). Decode TPS, narrative / **code**:

| Slug | Drafter · KV / attn | Ctx | narr / **code** | vs base |
|---|---|--:|--:|--:|
| `dual-fast` | MTP n=4 · fp8 | 262K | 73 / **100** | — |
| `dual-superfast` | DFlash2 · fp8 / FlashInfer | 262K | 78 / **141** | **+41%** |
| `dual-ultrafast` | DFlash2 · bf16 / FA2 | ~200K | 128 / **231** | **+131%** |
| `dual-max` | MTP n=3 · fp8 | 262K | 69 / **87** | — |
| `dual-supermax` | DFlash2 · fp8 / FlashInfer | 144K | 68 / **130** | **+49%** |
| `dual-ultramax` | DFlash2 · bf16 / FA2 | 64K | 90 / **172** | **+98%** |

Two structural findings: (1) the **FA2 ⊕ fp8-KV mutual exclusion on Ampere** is what splits `super` (fp8 KV → FlashInfer → keeps 262K, ~40% slower decode) from `ultra` (bf16 KV → FA2 → fastest decode, but 2× KV so context drops); (2) the **fidelity (fp8-weight) series decodes slower than the speed (int4) series** — Ampere has no native fp8 compute, so fp8 weights upcast to fp16. Full numbers + prefill/VRAM in [`../../BENCHMARKS.md`](../../) (stack) and `learnings/qwen3.8-27b.md`.

**Local WSL reproduction audit (Cristian, 2026-08-21):** the shipped W4A8 `dual-superfast` path reached no valid bench after a Marlin post-load `device not ready`; the adjusted W4A16 path measured **70.17 / 158.37 decode TPS** at 262K. The shipped 204.8K `dual-ultrafast` envelope did not fit our `.85` policy (KV sizing estimated 174,528 max); an adjusted 170K run measured **80.94 / 179.52**. `dual-supermax` and `dual-ultramax` reached Marlin but produced no bench because dynamic FP8/W8A8 was rejected on RTX 3090 (`Marlin W8A8 is not supported`). These are reproduction boundaries on this WSL rig, not replacements for the published tier table. The direct Club `dual-fast` baseline on the same rig was **60.18 / 80.30**; repeated boots varied materially, so cross-session values are not a matched A/B. The local `dual-faster` PR candidate measured **74.64 / 106.15**, but is not a shipped Club compose and is excluded from the catalog comparison.

**Single-card slug swapped `iq4nl` → `iq4xs`.** unsloth removed `Qwen3.8-27B-IQ4_NL.gguf` from the repo (2026-08-19), so `llamacpp/qwen38-27b-single-iq4nl` is dead. Replacement `llamacpp/qwen38-27b-single-iq4xs` (unsloth UD-IQ4_XS, 14.3 GB) ships **q4_0 KV at the full 262144** (halving the KV clears the q8 build's 131072 ceiling) **+ F16 vision** (mmproj-F16, `WITH_VISION=1`). Decode 61.6 narr / 71.3 code; NIAH-clean to 240,635 (91% of n_ctx); one correct image recognition. ⚠️ **q4_0 KV is below the stack serving floor** — a max-ctx / vision *exhibit*, not serving-grade; `KV_TYPE=q8_0 CTX_SIZE=131072` restores the q8@131K config. **🐣 Incubating.** (PR [#1068](https://github.com/noonghunna/club-3090/pull/1068).)

**HOL flag defaulted.** `--long-prefill-token-threshold 4096` added to all 20 vLLM composes (env `LONG_PREFILL_TOKEN_THRESHOLD`, 0 disables). Inert at the shipped `max_num_seqs=1` — preparatory for concurrency (Zylone's tip; qwen3.6 dual-fast precedent).

**Fast-tier weights swap** `Avuja` → `Frozenlock` (both AutoRound INT4) landed earlier — see [vllm#52873](https://github.com/vllm-project/vllm/issues/52873): the permanent MTP acceptance-collapse was checkpoint-specific to Avuja, not a vLLM bug (PR [#1070](https://github.com/noonghunna/club-3090/pull/1070)).

**Port hygiene.** Resolved 3 cross-model `default_port` collisions by moving the qwen3.8 side: `single-iq4xs` 8086→8090, `dual-fast` 8095→8113, `multi4-fast` 8096→8114 (nemotron/inkling/agents-a1 keep their ports). New gate `scripts/tests/test-compose-port-conflicts.sh` fails on any cross-model port overlap (aliases + same-model variants allowed); 8020/8032 allowlisted pending a separate hygiene PR.

## 2026-08-14/16 — onboarding: 5 incubating slugs + v0.27.1 pin + W4A8 default

Qwen3.8-27B onboarded as 🐣 Incubating across llama.cpp + vLLM (1/2/4/8 cards). Served from the official [FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and unsloth's [dynamic GGUFs](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF); AutoRound INT4 fast tier ships **W4A8** (int8 activations) by default. Numbers deliberately withheld at onboarding after a bench ran on a silently-degraded (pipeline-parallelism-off) config — see discussion [#993](https://github.com/noonghunna/club-3090/discussions/993) for the slugs, the sampler rows, and the traps. Same Qwen3-Next hybrid-GDN architecture as 3.5/3.6, so it inherits the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) MTP crash exposure (mitigate `SPEC=off`).
