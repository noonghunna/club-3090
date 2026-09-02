# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-09-02 — dual-max: `ASYNC_SCHED=off` wired — vllm#50021 mitigation (1), drafter kept at ~0% cost

The `dual/fp8/mtp.yml` header has documented two mitigations for the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) MTP × hybrid-GDN wild write since the 2026-08-19 production crash (#1059): (1) `ASYNC_SCHED=off` → `--no-async-scheduling`, which keeps the drafter, and (2) `SPEC=off`, which drops it. Only (2) was actually plumbed — `ASYNC_SCHED` was header prose with no env passthrough and no entrypoint branch, so anyone following the header got the drafter-off path by default and paid for it in decode. This wires (1) the same way `SPEC` is wired: env passthrough + entrypoint `case`, off by default, appended to both `exec vllm serve` branches.

**Measured on-rig (2026-09-02, ref 2×3090 PCIe, v0.27.1, canonical 3 warm + 5 measured, same boot family):**

| Config | narr / **code** decode | prefill @10K / @90K | TTFT | MTP AL |
|---|--:|--:|--:|--:|
| `SPEC=off` (mitigation 2) | 44.2 / **45.1** | 1353 / 1097 | 77–84 ms | — |
| `ASYNC_SCHED=off`, MTP n=3 (mitigation 1) | **67.8 / 88.1** | 1316 / 1058 | 98–113 ms | 3.0–3.3 |
| BENCHMARKS row 2026-08-17 (async ON, MTP n=3) | 67.4 / 85.8 | 1166 / 942 | 152 ms | 2.62 |

Mitigation (1) restores **+53% prose / +95% code** decode over the drafter-off path and is at parity or better with the async-ON catalog row on every column — the header's ~0% cost claim holds. Decode CV 2.1% / 3.3%; VRAM 22.3 GB/card, 0 MiB leak; no Xid / CUDA-error signatures across bench + 8-pack + verify-full on the new config. ⚠️ This bounds the crash *mechanism* the maintainer identified (MTP + prefix-cache + async), not #50021 itself, which is still open — a 10-minute bench is not 44 h of agent traffic. If Xid 31 recurs, `SPEC=off` remains the fallback.

## 2026-08-23 — dual-fast: FlashInfer decode-buffer unpin merged (#1051) — MTP concurrency unlocked to C=32

Merged **[#1051](https://github.com/noonghunna/club-3090/pull/1051)** (thanks **@A1RM4X** — reproduced independently on-rig before promoting). New patch [`vllm-flashinfer-decode-pin`](vllm/patches/vllm-flashinfer-decode-pin/README.md) flips `pin_memory=True→False` on the `flashinfer/decode.py` workspace-buffer allocs, forcing a synchronous plan copy per step and closing the stale-plan async-copy race ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756)) that crashed `vllm/qwen38-27b-dual-fast` (W4A8 + MTP n=4 + fp8 KV) with an Xid 31 VIRT_READ under concurrency. Idempotent, marker-gated, no-ops without FlashInfer, hard-fails on drift (boot-refused). Wired on `mtp.yml` alongside `vllm-gdn-mtp-async-spec-order` (the two fix **distinct** bugs: async wild-write vs decode-plan race).

**On-rig validation (2026-08-23, ref 2×3090, v0.27.1, real delivery path — switch.sh → compose mount → entrypoint install.sh):** the fix holds far past the PR's conservative c=4 claim. With `MAX_NUM_SEQS=32` it survived the full concurrency ladder **clean to C=32** (spec-ON MTP n=4): agg 71→241 tok/s (C=1→32), per-stream decode 84→18, drafter accepting (mean accept 2.85), zero crashes. Unpatched, the same slug crashes at C≥8. Distinct from DFlash2 (dual-superfast), which OOMs at C=8 regardless (VRAM, not this race). Full A/B + all four arms: `learnings/qwen3.8-27b.md` 2026-08-23.

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

**Single-card slug swapped `iq4nl` → `iq4xs`.** unsloth removed `Qwen3.8-27B-IQ4_NL.gguf` from the repo (2026-08-19), so `llamacpp/qwen38-27b-single-iq4nl` is dead. Replacement `llamacpp/qwen38-27b-single-iq4xs` (unsloth UD-IQ4_XS, 14.3 GB) ships **q4_0 KV at the full 262144** (halving the KV clears the q8 build's 131072 ceiling) **+ F16 vision** (mmproj-F16, `WITH_VISION=1`). Decode 61.6 narr / 71.3 code; NIAH-clean to 240,635 (91% of n_ctx); one correct image recognition. ⚠️ **q4_0 KV is below the stack serving floor** — a max-ctx / vision *exhibit*, not serving-grade; `KV_TYPE=q8_0 CTX_SIZE=131072` restores the q8@131K config. **🐣 Incubating.** (PR [#1068](https://github.com/noonghunna/club-3090/pull/1068).)

**HOL flag defaulted.** `--long-prefill-token-threshold 4096` added to all 20 vLLM composes (env `LONG_PREFILL_TOKEN_THRESHOLD`, 0 disables). Inert at the shipped `max_num_seqs=1` — preparatory for concurrency (Zylone's tip; qwen3.6 dual-fast precedent).

**Fast-tier weights swap** `Avuja` → `Frozenlock` (both AutoRound INT4) landed earlier — see [vllm#52873](https://github.com/vllm-project/vllm/issues/52873): the permanent MTP acceptance-collapse was checkpoint-specific to Avuja, not a vLLM bug (PR [#1070](https://github.com/noonghunna/club-3090/pull/1070)).

**Port hygiene.** Resolved 3 cross-model `default_port` collisions by moving the qwen3.8 side: `single-iq4xs` 8086→8090, `dual-fast` 8095→8113, `multi4-fast` 8096→8114 (nemotron/inkling/agents-a1 keep their ports). New gate `scripts/tests/test-compose-port-conflicts.sh` fails on any cross-model port overlap (aliases + same-model variants allowed); 8020/8032 allowlisted pending a separate hygiene PR.

## 2026-08-14/16 — onboarding: 5 incubating slugs + v0.27.1 pin + W4A8 default

Qwen3.8-27B onboarded as 🐣 Incubating across llama.cpp + vLLM (1/2/4/8 cards). Served from the official [FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and unsloth's [dynamic GGUFs](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF); AutoRound INT4 fast tier ships **W4A8** (int8 activations) by default. Numbers deliberately withheld at onboarding after a bench ran on a silently-degraded (pipeline-parallelism-off) config — see discussion [#993](https://github.com/noonghunna/club-3090/discussions/993) for the slugs, the sampler rows, and the traps. Same Qwen3-Next hybrid-GDN architecture as 3.5/3.6, so it inherits the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) MTP crash exposure (mitigate `SPEC=off`).
