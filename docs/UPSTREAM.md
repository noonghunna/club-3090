# Upstream tracker

Issues and PRs in upstream repos that affect this stack — what we depend on, what we've filed, what unblocks for us when each lands.

This file is the **single source of truth** for upstream status. When you file or notice an upstream issue / PR / commit relevant to club-3090, add a row here. When status changes (closed, merged, propagated), update it. Don't scatter the same link across multiple docs without coming back here first.

If you're adding a new compose that depends on an unmerged upstream patch (volume-mount of a fork, monkey-patch script), it MUST link to a row in this file so future readers know when the workaround can drop.

---

## How rows work

Each row covers one upstream link with: **title • status • our dependency / impact • workaround (if any)**.

**Status vocabulary:**
- 🟢 **Landed** — merged upstream + propagated to our pinned versions (pin-bump done)
- 🔵 **Merged, awaiting propagation** — merged upstream but our nightly / commit pin hasn't picked it up yet
- 🟡 **Open / in review** — PR open, no merge yet; we depend on it landing
- 🟠 **Open / blocked or stalled** — PR exists but progress stalled
- 🔴 **Open, no PR yet** — issue acknowledged but no fix in progress (us or upstream)
- ⚫ **Workaround locally, no plan to merge** — fixed in our patches, upstream not pursuing
- ✅ **Resolved** — closed and resolved (kept for historical context)
- ❌ **Closed without fix** — closed, won't fix, kept for context

---

## vLLM (`vllm-project/vllm`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#40361](https://github.com/vllm-project/vllm/pull/40361) — Marlin pad-sub-tile-n | 🟡 Open, mergeable | All 4 dual-card composes mount our patched fork from `/opt/ai/vllm-src/`. Drops out as a setup dependency when this merges + propagates. | Volume-mount: see [`models/qwen3.6-27b/vllm/patches/README.md`](../models/qwen3.6-27b/vllm/patches/README.md#vllm-pr-40361). |
| [#40807](https://github.com/vllm-project/vllm/issues/40807) — `.tolist()` cudagraph crash on continuation-prefill | ⚫ Local workaround | Single-card TQ3 + spec-decode + chunked-prefill blocked without it. We ship a file-edit patch. | `patch_tolist_cudagraph.py` runs in `setup.sh`. Drop when upstream fixes the sync. |
| [#40849](https://github.com/vllm-project/vllm/pull/40849) — MTP draft online-quant propagation | 🟡 Open / Genesis backport active | Closes Cliff 1 on FP8+MTP path (`tools-text.yml`). | Genesis PN8 backport: `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1`. |
| [#40914](https://github.com/vllm-project/vllm/pull/40914) — Sandermage K+1 verify routing | 🟡 Open | Recovers ~22 TPS narrative regression on TQ3 + spec-decode (P65 cost). When it lands, default compose's narrative TPS catches up to ampersandru's pre-P65 numbers. | None local — wait for merge. |
| [#40334](https://github.com/vllm-project/vllm/pull/40334) — DFlash `combine_hidden_states` dtype mismatch | 🟡 Open | All `dual-dflash*.yml` need `--dtype bfloat16` flag to work around. | Composes set `--dtype bfloat16`. Drop when this lands. |
| [#40382](https://github.com/vllm-project/vllm/issues/40382) — Gemma-4 + DFlash unservable on Ampere | 🟠 Open, no fix in progress | Blocks DFlash on Gemma-4 family. Not directly our problem (we serve Qwen3.6) but tracked because future model adds may hit it. | None — different attention backend selection. |
| [#40354](https://github.com/vllm-project/vllm/issues/40354) — Marlin TP=2 W4A16 < 64 | ✅ Same root-cause as #40361 | Our PR #40361 resolves this. | See #40361 row. |
| [#39931](https://github.com/vllm-project/vllm/issues/39931) — DeltaNet rollback support | 🔴 Open, architectural | Blocks **all** spec-decode (EAGLE / DFlash) on Qwen3-Next family across engines. The reason "speculative decoding doesn't work" on this stack. | Use MTP (no rollback needed) until this lands. |
| [#40124](https://github.com/vllm-project/vllm/issues/40124) — related architectural | 🔴 Open | Pairs with #39931 for DeltaNet rollback. | Same as above. |
| [#40880](https://github.com/vllm-project/vllm/issues/40880) — MTP × TQ × cudagraph cascade | ✅ Closed (P65 fix) | Fixed by Genesis P65 (cudagraph PIECEWISE downgrade for spec-decode). | Auto-on in default compose. |
| [#40831](https://github.com/vllm-project/vllm/issues/40831) — TQ × spec-decode corruption | ✅ Closed | Resolved by P65 (MTP) + ngram-mod `prompt_lookup_min=8` (ngram). | See P65 / ngram routing. |
| [#40798](https://github.com/vllm-project/vllm/pull/40798) — workspace-manager refactor | ❌ Negative result | Hypothesized fix for #40831 / #40880; backporting it (Probe 8) didn't resolve the bug. Kept for context — saved future time on the same dead end. | n/a |
| [#40875](https://github.com/vllm-project/vllm/issues/40875) — ngram + MTP coexistence | ✅ Closed | Routed via `prompt_lookup_min=8` flag. | Set in compose where applicable. |
| [#41142](https://github.com/vllm-project/vllm/pull/41142) — Quentin-M streaming tool-call IndexError | 🟡 Open / Genesis backport active | Closes a streaming tool-call crash on Hermes / similar templates. | Genesis PN11 backport (auto-enabled where REC). |
| [#39598](https://github.com/vllm-project/vllm/pull/39598) — kotori-yan qwen3coder MTP streaming early-return | 🟡 Open / Genesis backport active | Empty `tool_calls[]` when MTP bundles last param + `</function>` in same delta. | Genesis P64 backport: `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` (default-on in our composes). |

---

## Genesis (`Sandermage/genesis-vllm-patches`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#5](https://github.com/Sandermage/genesis-vllm-patches/issues/5) — P8 ImportError on vLLM v0.20.0 | 🟡 Open | Blocks Genesis on the vLLM v0.20.0 release (we pin nightly so unaffected). | Stay on dev205+ pin; revisit when v0.20.x is the recommended pin. |
| [#6](https://github.com/Sandermage/genesis-vllm-patches/issues/6) — P65 PIECEWISE cost quantified | ✅ Closed | We characterized the +22 TPS narrative cost of P65 on Qwen3.6-27B + MTP. Sandermage acknowledged. Will recover when vllm#40914 lands. | Accept the cost on substrate-current; ampersandru's pre-P65 stack avoids it. |
| [#7](https://github.com/Sandermage/genesis-vllm-patches/issues/7) — P67 Triton CompilationError on Qwen3.6-27B | 🟡 Open | P67 (multi-query verify kernel, +25–35% TPS) crashes on GQA != power-of-2. Affects 27B specifically. | P67 left off by default on 27B configs. |
| [#9](https://github.com/Sandermage/genesis-vllm-patches/issues/9) — P68/P69 8000-char threshold breaks IDE agents | 🔴 Open (we filed 2026-04-29) | P68 silently rewrites `tool_choice: auto → required`; P69 injects "must use a tool" hint. Both fire at default 8000-char threshold — every IDE agent context exceeds that. | All shipped composes have `P68/P69` commented out. See [`docs/FAQ.md`](FAQ.md#will-this-work-with-vs-code-github-copilot-llm-gateway). |

---

## flash-linear-attention (`fla-org/flash-linear-attention`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Cliff 2 — DeltaNet GDN forward OOM at 50–60K single-prompt** | 🔴 Open, **no upstream issue filed yet** | The `chunk_gated_delta_rule_fwd` kernel allocates intermediate buffers proportional to `seq_len`. Fires regardless of mem-util. Sandermage explicitly punted (genesis-vllm-patches issue #1: *"can't fix this short of multi-GPU TP=2 or upstream fla.ops changes"*). | None on single-card vLLM. Use `tools-text.yml` (75K cap), `dual.yml` (TP=2 splits per-card seq_len), or `llamacpp/default` (262K, different engine, no DeltaNet OOM). |

---

## FlashQLA (`QwenLM/FlashQLA`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Ampere SM 8.6 / Ada SM 8.9 port** | 🔴 No issue filed; tweet to @QwenLM drafted but not yet posted | FlashQLA is QwenLM's TileLang DeltaNet kernels — would fix Cliff 2 if it ran on Ampere. Currently SM90+ only. | None. Watch the repo for Ampere support; revisit when an issue is filed and a port is on the roadmap. |

---

## llama.cpp (`ggml-org/llama.cpp`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [PR #21089](https://github.com/ggerganov/llama.cpp/pull/21089) — TurboQuant KV mainline | 🟡 Open (CPU first, CUDA follow-on) | When CUDA path lands, `turbo3` becomes a first-class option on llama.cpp. Naming will migrate from `turbo3` → `tbq3_0`. | Use Tom's fork for now: [llama-cpp-turboquant](https://github.com/tdraxl/llama-cpp-turboquant). |
| **Q3_K_XL TPS regression (28.5 TPS @ 262K → 21 TPS today)** | 🔴 Suspected, no upstream issue filed | Measured 2026-04-23 vs 2026-04-28: same model, same hardware, 28.5 TPS dropped to 21 TPS between commits `9ab47e7d8` and `0d0764dfd`. Bisect or file. | None — we're on the slower commit. Tracked in [club-3090 TODO](https://github.com/noonghunna/club-3090) (private). |

---

## transformers (`huggingface/transformers`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#45283](https://github.com/huggingface/transformers/issues/45283) — Qwen3.5 GGUF support | 🟡 Open | Together with vllm#38140 / vllm#37797, would unblock Qwen3.5 GGUF on vLLM/SGLang. llama.cpp already works. | llama.cpp path. |

---

## SGLang (`sgl-project/sglang`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Same Marlin pad-sub-tile-n bug as vllm#40361** | 🔴 Not filed; same kernel-line fix applies | Blocks Lorbus INT4 + EAGLE on SGLang. We haven't filed an SGLang PR. | None on SGLang. Use vLLM (with our patched fork) or wait for SGLang to pick up the upstream Marlin fix. |
| **DeltaNet KV rollback (vllm#39931 cross-engine)** | 🔴 Same architectural issue | Blocks EAGLE on Qwen3-Next family in SGLang too. | None — see vllm#39931. |

---

## Filing conventions

When you file or learn of a new upstream issue:

1. **Add a row** to the appropriate section of this file. Include the link, status emoji, one-line "why it matters," and the local workaround (if any).
2. **Cross-link** from any code, compose comment, or doc that depends on the workaround back to the row in this file (e.g., `# See docs/UPSTREAM.md — vllm#40361`).
3. **Update the row** when status changes — closed, merged, propagated, replaced. Don't delete; if a row is no longer load-bearing, mark it ✅ Resolved or ❌ Closed without fix and leave it as historical context.
4. **Bump the relevant pin** when an upstream lands (Genesis commit, vLLM nightly, llama.cpp commit). Add a CHANGELOG entry citing the upstream PR.

When you file an issue against an upstream repo from this work, **link back to club-3090** in the body so the upstream maintainer can see the affected user surface and re-test if needed.

---

## Related reading

- [`models/qwen3.6-27b/INTERNALS.md`](../models/qwen3.6-27b/INTERNALS.md) — model-specific deep dives (DFlash forensics, MTP head, AutoRound rationale)
- [`models/qwen3.6-27b/vllm/patches/README.md`](../models/qwen3.6-27b/vllm/patches/README.md) — local patches (tolist, Marlin pad fork, Genesis env-var matrix)
- [`AGENTS.md`](../AGENTS.md) — repo-wide conventions, including the rule that this file is the upstream-tracking single source of truth
