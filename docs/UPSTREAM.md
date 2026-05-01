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
| [#40961](https://github.com/vllm-project/vllm/pull/40961) — Preserve max_seq_len in ubatch metadata during CUDA graph capture | 🟡 Open PR | Confirms the cap-leak pattern: cudagraph capture passes `max_model_len` as `max_seq_len` through ubatch metadata. PR is *fixing a missing pass-through* for SWA models (where seqlen=1 at capture broke kernel selection) — by establishing that `max_model_len` is what gets carried through capture metadata, it cements the source of Cliff 1's max-ctx-dependent FA2 workspace sizing. | Stay at `default` 48K — see FA2 #1011 row + INTERNALS.md Cliff 1 mechanism. |
| [#40069](https://github.com/vllm-project/vllm/issues/40069) — [Tracking] TurboQuant / HIGGS Attention follow-ups | 🟡 Open tracker | Umbrella tracking for TurboQuant + attention backend issues on our stack class. | Watch for cross-references when Cliff 1/2 work lands upstream. |
| [#25543](https://github.com/vllm-project/vllm/pull/25543) — [V0 Deprecation] Remove `max_seq_len_to_capture` | ✅ Merged 2025-09-24 | Important to know: the `--max-seq-len-to-capture` flag (commonly suggested as a Cliff 1 mitigation) **does not exist in V1**. Don't recommend it. | n/a — flag removed. |
| [#39226](https://github.com/vllm-project/vllm/pull/39226) — workspace-resize GPU memory leak fix | 🔵 Merged into v0.20.0 | Strict `WorkspaceManager.lock()` semantics — after lock, any allocation that grows the workspace raises `AssertionError`. **Blocks our v0.20 upgrade across the entire TurboQuant + MTP single-card fleet** (long-text / long-vision / bounded-thinking / dual-turbo). `TurboQuant._decode_attention` requests 0.76 MB at runtime but `profile_run` doesn't exercise the path on our config so workspace gets locked at 0.00 MB; the grow request fails. Doesn't surface for Sandermage because his PROD configs (35B-A3B-FP8, 27B Lorbus fp8_e5m2) use fp8 KV — TQ decode path inactive. | Stay on dev205 pin for prod composes. Genesis **P98** (`GENESIS_ENABLE_P98=1`) is documented as "REQUIRED for TQ k8v4 on hybrid (WorkspaceManager fix vs vllm#40941)" in Sandermage's `bare_metal_27b_int4_TQ_k8v4.sh` — open question whether P98 also covers vllm#39226's lock semantics on our path. `v0.20-experimental` branch holds the test compose for re-evaluation. |
| [#40092](https://github.com/vllm-project/vllm/pull/40092) — TurboQuant FA3/FA4 prefill paths | 🔵 Merged into v0.20.0 | TQ + flash-attention 3/4 prefill support. Relevant if/when v0.20 unblocks for us — the FA varlen workspace allocator behavior may change under FA3/FA4 vs the FA2 path we currently hit. | Track. Re-evaluate the `flash_attn_interface.py:300` cliff (Genesis #15) once v0.20 unblocks since FA3/FA4 may have different workspace semantics. FA3/FA4 not enabled on Ampere SM 8.6 anyway (Hopper+ only). |
| [#40941](https://github.com/vllm-project/vllm/pull/40941) — TurboQuant share buffers | 🔵 Merged into v0.20.0 | Sandermage's bare_metal_27b_int4_TQ_k8v4.sh comments call out P98 as the workaround for "WorkspaceManager fix vs vllm#40941". Same WorkspaceManager class that vllm#39226 made strict. | Sandermage's **P98** is the workaround — required for TQ k8v4 on hybrid. Worth a focused investigation: enable P98 on the v0.20-experimental compose to see if it also unblocks vllm#39226's path. |

---

## Genesis (`Sandermage/genesis-vllm-patches`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#5](https://github.com/Sandermage/genesis-vllm-patches/issues/5) — P8 ImportError on vLLM v0.20.0 | 🟡 Open (different blocker now) | Originally about P8 ImportError on the v0.20.0 GA tag. Since superseded by vllm#39226 workspace-lock blocker — that's the actual reason our config can't move to v0.20 today. | Stay on dev205+ pin. See vllm#39226 row above. |
| [#6](https://github.com/Sandermage/genesis-vllm-patches/issues/6) — P65 PIECEWISE cost quantified | ✅ Closed | We characterized the +22 TPS narrative cost of P65 on Qwen3.6-27B + MTP. Sandermage acknowledged. Will recover when vllm#40914 lands. | Accept the cost on substrate-current; ampersandru's pre-P65 stack avoids it. |
| [#7](https://github.com/Sandermage/genesis-vllm-patches/issues/7) — P67 Triton CompilationError on Qwen3.6-27B | ✅ Closed | Resolved in v7.64 — P67 generalized to non-power-of-2 GQA via `BLOCK_QH = triton.next_power_of_2(HEADS_PER_KV)` + lane_valid mask. Tool-call 0/5 → 7/7 on 2× A5000 validation. | Now safe to enable on 27B configs with v7.64+. |
| [#9](https://github.com/Sandermage/genesis-vllm-patches/issues/9) — P68/P69 8000-char threshold breaks IDE agents | 🟠 Roadmap (Sandermage marked v7.65) | P68 silently rewrites `tool_choice: auto → required`; P69 injects "must use a tool" hint. Both fire at default 8000-char threshold — every IDE agent context exceeds that. v7.65 dev branch already raises threshold to 32000 ([commit d73fa9d](https://github.com/Sandermage/genesis-vllm-patches/commit/d73fa9d)). | All shipped composes have `P68/P69` commented out until v7.65 lands. See [`docs/FAQ.md`](FAQ.md#will-this-work-with-vs-code-github-copilot-llm-gateway). |
| [#11](https://github.com/Sandermage/genesis-vllm-patches/issues/11) — Cliff 1 mech A FA2 softmax_lse clamp request | ✅ Closed (we filed 2026-04-29; resolved in v7.64 via PN17) | Sandermage's PN17 in v7.64 lands the clamp at `flash_attn.py` (Sandermage's Python wrapper). PN17 is enabled in `long-text.yml`, `long-vision.yml`, `bounded-thinking.yml`. Our local P104 sidecar still mounted as defensive coverage of the TQ wrapper layer (different file, separate insurance). | PN17 enabled by default. P104 sidecar kept as backup but `GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP=1` only on long-text. |
| **Local P104 FA max_seqlen_k runtime clamp** | ✅ Superseded by PN17 in v7.64 | Built 2026-04-30 as `patch_fa_max_seqlen_clamp.py`. Sandermage's PN17 (anchored Genesis version) replaces it at the env level. P104 stays mounted on long-text as defensive coverage of the TQ wrapper layer that PN17 doesn't reach (`turboquant_attn.py:_flash_attn_varlen` line 394 vs PN17's `vllm/v1/attention/backends/flash_attn.py`). | n/a — both layers covered. |
| [PR #12](https://github.com/Sandermage/genesis-vllm-patches/pull/12) — P101 anchor drift fix | ✅ Closed by Sandermage (didn't reproduce on his pin) | P101 was silently no-op on vLLM dev205+ — anchor `_arange_cache[...]` no longer matches upstream `torch.arange(...)` form. Sandermage closed because it didn't reproduce on his `0.20.1rc1.dev16` pin. | Drift is dev205-specific. Local sidecar `patch_pn12_ffn_pool_anchor.py` covers PN12 on dev205; P101 not currently load-bearing on shipped configs. |
| [PR #13](https://github.com/Sandermage/genesis-vllm-patches/pull/13) — PN12 anchor drift fix | ✅ Closed by Sandermage (same — drift is dev205-specific) | PN12's FFN intermediate pool was silently no-op'd on dev205+. Sandermage closed because anchors match on his pin. | Local `patch_pn12_ffn_pool_anchor.py` carries the dev205-specific fix until we move pins. |
| [#14](https://github.com/Sandermage/genesis-vllm-patches/issues/14) — P38 silently no-op'd on TurboQuant KV path (we filed 2026-05-01) | 🟡 Fix on Sandermage `dev`, pending v7.65 | Sandermage shipped **P38B** in response — text-patches `turboquant_attn.py` source to inject a delegate hook at the start of `_continuation_prefill` body. Source-edit survives `aot_compile_fullgraph` capture (compiler reads the modified source at engine init). Different mechanism from our PN12-style `torch.library.custom_op` route but reaches the same compile-time visibility. Composes with existing P38 (eager-mode rebind) — both share the same Genesis impl. | When v7.65 ships: enable `GENESIS_ENABLE_P38B_COMPILE_SAFE=1` alongside the existing `GENESIS_ENABLE_P37=1`. Drop our line-903 cliff workaround. Empirically, the cliff also doesn't fire on v0.20 (cross-validated 2026-05-01 PM) — once we pin-bump to v0.20 the issue closes from two directions. |
| [#15](https://github.com/Sandermage/genesis-vllm-patches/issues/15) — FA varlen kernel workspace cliff at flash_attn_interface.py:300 (we filed 2026-05-01) | 🟡 Fix on Sandermage `dev`, pending v7.65 | Sandermage shipped **P15B** — direct backport of our suggestion path 1. Text-patches `turboquant_attn.py:_flash_attn_varlen` to clamp `max_seqlen_k` from actual `cu_seqlens_k` before the FA wrapper invocation. Trade-off: ONE GPU→CPU sync per call (acceptable on continuation-prefill path which is infrequent). batch=1 takes a fast-path tensor access; batch>1 takes diff().max() reduction. | When v7.65 ships: enable `GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP=1`. Drop our long-vision context backoff. Same as #14: empirically also not reproducing on v0.20 (vllm#40092 changes the workspace allocator behavior). |
| **PN25 — Inductor-safe `silu_and_mul` pool** | 🟡 On Sandermage `dev` branch, pending v7.65 | Sandermage's `torch.library.custom_op` for `forward_native` (sister of our local `patch_pn12_compile_safe_custom_op.py`). Independent convergence on the same fix. Currently at [`2b239a8`](https://github.com/Sandermage/genesis-vllm-patches/commit/2b239a8) on the dev branch. | When v7.65 ships: enable `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1`. Drop our local sidecar. |
| **P98 marker false-positive on v0.20** (we filed 2026-05-01 in [#9 thread](https://github.com/Sandermage/genesis-vllm-patches/issues/9#issuecomment-4359541875)) | 🟡 Side-noted to Sandermage; awaiting his call on fix | P98's drift detection auto-skips on v0.20 (`upstream marker 'UNIFORM_SINGLE_TOKEN_DECODE' detected — patch obsolete`) but the strict workspace lock from vllm#39226 still fires the path P98 was supposed to revert. Marker check needs to be aware that post-vllm#39226 WorkspaceManager class is in scope. | Local `patch_workspace_lock_disable.py` sidecar (v0.20-experimental branch only) relaxes the strict assertion to a one-shot WARNING. Drop when Sandermage ships either a marker fix or P98 with explicit env-override. |

---

## FlashAttention 2 (`Dao-AILab/flash-attention`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [#1011](https://github.com/Dao-AILab/flash-attention/issues/1011) — Variable memory allocation with varlen kernels | 🔴 Open since 2024, no fix | **Cliff 1 root cause.** `softmax_lse` is allocated as `[num_seqs, num_heads, max_seqlen]` — sized by `max_seqlen` parameter, NOT actual `cu_seqlens`. So a 25K-token chunked-prefill at `max_model_len=86K` allocates softmax_lse for 86K, not 25K. This is why Cliff 1 fires harder at higher max-ctx even when the actual prompt is the same. | None. Stay at `default` 48K (or `tools-text` 75K with PN8 mitigation). FA2 redesign of softmax_lse format would be the upstream fix. |

## flash-linear-attention (`fla-org/flash-linear-attention`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Cliff 2 — DeltaNet GDN forward OOM at 50–60K single-prompt** | 🔴 Open, **no upstream issue filed yet**. **Confirmed cleared on dual TP=2** (this rig, 2026-04-29 — see DUAL_CARD.md "237K single-prompt verified"). | The `chunk_gated_delta_rule_fwd` kernel allocates intermediate buffers proportional to `seq_len`. Fires on single-card regardless of mem-util. On dual TP=2 the activation memory splits across cards and the cliff doesn't fire — verified at 237K single-prompt prefill on `dual.yml` (~830 tok/s prefill, matches Sandermage's 262K @ 311s on 2× A5000). Sandermage explicitly punted on the single-card fix (genesis-vllm-patches issue #1: *"can't fix this short of multi-GPU TP=2 or upstream fla.ops changes"*). Likely the same architectural pattern as FA#1011 — recurrent state buffer pre-allocated by max_seq_len. | Single-card: use `tools-text.yml` (75K cap) or `llamacpp/default` (262K, different engine). Dual: `dual.yml` clears at ≥237K. |

---

## FlashQLA (`QwenLM/FlashQLA`)

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| **Ampere SM 8.6 / Ada SM 8.9 port** | 🔴 No issue filed; tweet to @QwenLM drafted but not yet posted | FlashQLA is QwenLM's TileLang DeltaNet kernels — would fix Cliff 2 if it ran on Ampere. Currently SM90+ only. | None. Watch the repo for Ampere support; revisit when an issue is filed and a port is on the roadmap. |

---

## Luce DFlash (`Luce-Org/lucebox-hub`)

Watch list — promising on code-heavy single-stream workloads but not yet a club-3090 shipping option. Re-benched 2026-04-30 PM on Qwen3.6-27B Q4_K_M + matched z-lab/Qwen3.6-27B-DFlash draft (under training).

| Issue / PR | Status | Why it matters | Workaround |
|---|---|---|---|
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) — draft model still under training | 🟡 Snapshot 2026-04-26 | Narrative AL ~3.7, code AL ~7.0. When training finishes, expected to climb toward Qwen3.5 reference (8.31 HE, 7.04 Math). Today: code 72 TPS / narr 40 TPS on our 3090 vs vLLM's 66/50. | Re-test when z-lab tags training-complete. |
| **Build fragility on `dflash` main HEAD** | 🔴 Reproducible 2026-04-30 PM | `cmake --build` errors with `ggml_turbo_wht` and `GGML_TYPE_TQ3_0` undefined. Required submodule commit `b6ffab4a9` not auto-fetched. Cross-rig signal — fresh clone fails. | After clone: `cd dflash/deps/llama.cpp && git fetch origin && cd ../../.. && git submodule update --init`. |
| **Daemon-mode "empty prompt" regression** | 🔴 Reproducible 2026-04-30 PM | After streaming requests, subsequent requests return `"empty prompt"` from the test_dflash daemon. Server keeps accepting requests but generates 0 tokens. Forces restart. | Restart server between request flavors; avoid mixing streaming + non-streaming. |
| **`enable_thinking` chat_template_kwargs honored differently than vLLM** | 🟡 Behavioural difference | Test sends `enable_thinking=true` and expects `reasoning_content` populated. Luce returns `content` directly. Not a missing feature, but breaks our `verify-full.sh` check 6. | Don't treat the thinking-mode test as a Luce-correctness signal until the chat-template path is documented. |
| **Greedy only** | 🟡 Documented limitation | `temperature` / `top_p` accepted but ignored. Real downside for creative-writing workloads. | Use vLLM long-text/long-vision when sampling matters. |
| **Prefill OOM in `fattn-chunked.cu` on 25K+ prompts at Q8_0 KV** | 🟡 Open (configuration trade) | Chunked flash-attention CUDA OOMs on large prefill at default Q8_0. **TQ3 KV (`DFLASH27B_KV_TQ3=1`) closes it** at max_ctx=65K — verify-stress passes 791 chars / finish=stop. Higher max_ctx (131K) reopens it. | Always set `DFLASH27B_KV_TQ3=1` for stress-test-passing config. Cap max_ctx at ~65K. |

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
