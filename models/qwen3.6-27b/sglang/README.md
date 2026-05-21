# Qwen3.6-27B on SGLang — PARKED, kept for archival only

> **⚠ PARKED 2026-05-21 — not a recommended path on this stack.** SGLang + EAGLE-3 on Qwen3-Next was investigated as a way to unlock external-drafter spec-decode (which neither vLLM nor mainline llama.cpp provides for this model family). The investigation reached boot + coherent output on dual 3090 with two vendored patches, but **MTP wins on every axis** (TPS, complexity, hardware coverage). The compose files + patches stay in the tree for archival reference; production paths remain `vllm/dual/turbo.yml` (TP=2 + MTP + Genesis TQ3) and `llamacpp/mtp.yml` (single-card + MTP).

For engine-level pros/cons, KV cache options, Ampere quirks, see [`../../../docs/engines/SGLANG.md`](../../../docs/engines/SGLANG.md).

## Why parked

Three independent findings, each sufficient on its own:

1. **EAGLE-3 is sub-MTP for Qwen3-Next, even on the architecture where it works.** Ex0bit (the drafter author) publishes their own numbers on the [PRISM-PRO-DQ model card](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ): native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)** on Blackwell. Qwen3-Next has a strong built-in MTP head; routing through an external drafter is structurally slower for this model family.
2. **CUTE_DSL capture-hang on Ampere.** SGLang v0.5.12's CUTE_DSL `get_version()` calls `pkgutil.walk_packages` during cuda-graph capture. The walk hits `cutlass.cute.experimental` (which raises `NotImplementedError` on CUDA toolkit < 13.1) and deadlocks against the locked capture stream. Workaround `--disable-cuda-graph` boots but caps decode at ~15-18 TPS (vs vLLM-MTP-dual ~85 TPS). Three patch iterations (pre-import, `sys.modules` stub at engine init, per-process `sys.modules` stub at `sglang/__init__.py`) all failed — the walk re-fires during capture regardless of cache state.
3. **vLLM-MTP-dual already beats SGLang+EAGLE3+cuda-graph-off on the same rig.** Even if we ignored point 2 and accepted the cuda-graph penalty, `vllm/dual/turbo.yml` (TP=2 + MTP + Genesis TQ3) gives ~85 TPS on this dual-3090 setup. This SGLang path tops out around 15-18 TPS with the workaround.

The artifacts in this tree document the SGLang state for the next person who comes to this question. If/when upstream lands a CUTE_DSL Ampere fallback OR an SGLang-specific MTP path materializes for Qwen3-Next, the [Re-test triggers](#re-test-triggers) section below tells you what to verify.

**One-line summary:** SGLang AutoRound on Qwen3-Next is a working *engineering* path but an unattractive *production* path. MTP (vLLM or llama.cpp) is strictly better for this model.

---

## TL;DR

| Variant | Path | Status |
|---|---|---|
| Dual 3090 (TP=2) EAGLE-3 + AutoRound INT4 | [`compose/dual/eagle3-experimental.yml`](compose/dual/eagle3-experimental.yml) | ⚠ PARKED — boots + coherent output validated, but ~15-18 TPS with `--disable-cuda-graph` workaround vs ~85 TPS on `vllm/dual/turbo.yml` |
| Single 3090 EAGLE-3 + AutoRound INT4 | [`compose/single/eagle3-experimental.yml`](compose/single/eagle3-experimental.yml) | ❌ PARKED — blocked on SGLang OffloaderV1 tied-weights bug; never reached first forward pass |
| Other quants (FenomAI AWQ-INT4, INT8 PTH) | TARGET_DIR override in dual YAML | ⚠ PARKED — untested, would inherit the same EAGLE-3 < MTP structural disadvantage |

---

## Quick recipe (dual 3090)

```bash
# 1. Prereqs:
#    a. Pull the SGLang image (pinned to v0.5.12)
docker pull lmsysorg/sglang:v0.5.12

#    b. Have AutoRound target weights at $MODEL_DIR/qwen3.6-27b-autoround-int4/
#       (this is the Lorbus path — same path our vLLM default uses)

#    c. Have EAGLE-3 drafter at $MODEL_DIR/qwen3.6-27b-prism-eagle3/
hf download Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 \
  --include 'compressed/*' --include 'patch_sglang_eagle3.py' --include 'README.md' \
  --local-dir $MODEL_DIR/qwen3.6-27b-prism-eagle3

# 2. Boot the dual compose
cd <repo>/models/qwen3.6-27b/sglang/compose/dual
MODEL_DIR=/your/models/dir docker compose -f eagle3-experimental.yml up -d

# 3. Wait ~75s for boot + drafter load
# 4. Verify
curl -s http://localhost:8041/v1/models | python3 -m json.tool
curl -s http://localhost:8041/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-eagle3-dual",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":50,"temperature":0.6}'
```

---

## Why dual, not single

We have a single-card compose ([`compose/single/eagle3-experimental.yml`](compose/single/eagle3-experimental.yml)) that progresses through every layer of the boot but **fails at first forward pass** because the only knob that fits everything in 24 GB is `--cpu-offload-gb`, and SGLang's OffloaderV1 hits a tied-weights bug on Qwen3-Next:

```
ValueError: functional_call got multiple values for keys
  ['linear_attn.attn.dt_bias', 'linear_attn.dt_bias'],
  which are tied. Consider using tie_weights=False
```

Without CPU offload, the target + EAGLE-3 drafter + Mamba state + KV cache + activations don't fit on a single 24 GB card with workable spec-decode batch size.

The dual compose sidesteps this entirely by splitting the target weights across both cards (TP=2). VRAM footprint becomes ~22 GB/card, no offload needed, no tied-weights crash.

When SGLang's OffloaderV1 ships handling for tied weights (track upstream), we'd revisit single-card.

---

## Vendored patches

Two patches apply at container startup, bind-mounted from [`patches/`](patches/):

### 1. `patch_sglang_eagle3.py` (EAGLE-3 capture hook)

Provided by [`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3) (the drafter author). Adds `set_eagle3_layers_to_capture` to `Qwen3_5ForConditionalGeneration` so EAGLE-3 can capture auxiliary hidden states from the target. The base SGLang Qwen3.5 model class doesn't ship this hook — SGLang's spec-decode infrastructure assumes a different model class hierarchy than Qwen3.6 actually has.

### 2. `patch_sglang_autoround_fused_bf16.py` (Marlin name-mapping fix)

Our local fix for the AutoRound + Qwen3-Next loader bug. See full mechanics in [`patches/patch_sglang_autoround_fused_bf16.md`](patches/patch_sglang_autoround_fused_bf16.md). One-paragraph summary:

> SGLang's auto-round loader drops `packed_modules_mapping`. Qwen3-Next's DeltaNet `linear_attn.in_proj_a` + `.in_proj_b` are checkpoint tensors that AutoRound keeps at BF16, but SGLang fuses them into `linear_attn.in_proj_ba` — and without `packed_modules_mapping`, the fused module doesn't know it should consult the split BF16-keep entries. Result: SGLang treats `in_proj_ba` as INT4-quantized → routes to Marlin → trips `size_n=96 not divisible by tile_n_size=64`. The patch preserves the mapping so the fused module correctly stays BF16.

This is an **AutoRound-specific** fix. The alternate AWQ checkpoint (`FenomAI/Qwen3.6-27B-AWQ-INT4`) routes through SGLang's `compressed-tensors` loader, which already preserves `packed_modules_mapping` — no patch needed for that route.

---

## What's validated vs not

**Verified 2026-05-20 (single boot, dual 3090 + AutoRound INT4 + EAGLE-3):**

- ✅ Both target shards load: `Qwen3_5ForConditionalGeneration, quant=auto-round, bits=4`, 8.80 GB/card
- ✅ FP8 KV cache allocated: 600,075 tokens, 4.58 GB K + 4.58 GB V per card
- ✅ Mamba cache: `max_mamba_cache_size: 8`, 0.63 GB ssm_state per card
- ✅ EAGLE-3 drafter loads: `LlamaForCausalLMEagle3`, 2.11 GB/card
- ✅ `/v1/models` returns `qwen3.6-27b-eagle3-dual` with `max_model_len=32768`
- ✅ Chat completion produces coherent output (40-token sample showed Qwen3.6's thinking-mode structured response)
- ✅ No Marlin assertion fired
- ✅ No tied-weight crash (because TP=2, no CPU offload needed)
- ✅ "Spec v2 is enabled by default for eagle/eagle3/standalone speculative decoding" — V2 scheduler engaged
- ✅ VRAM footprint stable at ~22 GB/card

**Pending (next session):**

- ⚠️ Decode TPS — need a longer completion (200+ tokens) for stable measurement
- ⚠️ Accept rate / spec-decode confirmation — need a long enough decode to flush stats
- ⚠️ Quality 8-pack (`scripts/quality-test.sh --full`) — need to confirm coherent output isn't just luck
- ⚠️ verify-stress.sh — boundary checks at 8K/32K context
- ⚠️ Soak test for stability over sustained context
- ⚠️ aider-polyglot-30 — code-agent eval

The dual compose passes the basic substrate gate. Real-workload validation is the next step.

---

## Knob reference (dual compose)

The compose ships with these knobs. Most are mandatory for Qwen3-Next on Ampere; a few are tunable:

| Flag | Default in compose | Tunable? |
|---|---|---|
| `--tp-size 2` | 2 | No (architectural) |
| `--disable-custom-all-reduce` | (set) | No (PCIe-only Ampere) |
| `--speculative-algorithm EAGLE3` | (set) | Yes — swap to drop spec-decode if you want a baseline |
| `--speculative-num-steps 3` | 3 | Yes — try 2 or 4 for accept-rate A/B |
| `--speculative-eagle-topk 1` | 1 | Yes — model card recommends chain (1), not tree (4) on hybrid GatedDeltaNet |
| `--speculative-num-draft-tokens 4` | 4 | Yes — affects verify-batch size |
| `--speculative-draft-model-quantization unquant` | (set) | **No** — required for BF16 drafter + INT4 target |
| `--mamba-scheduler-strategy extra_buffer` | (set) | Yes — `no_buffer` is more aggressive (use if VRAM-tight) |
| `--mm-attention-backend sdpa` | (set) | No on Ampere (fa4 is sm_10x/11x only) |
| `--quantization auto-round` | env-parametrized via `QUANTIZATION` | Yes — `compressed-tensors` for AWQ target |
| `--kv-cache-dtype fp8_e5m2` | env-parametrized via `KV_CACHE_DTYPE` | Yes — `fp8_e4m3` for accuracy (per SGLang docs recommendation) |
| `--disable-cuda-graph` | (set) | **No on Ampere** — CUTE_DSL hangs at capture |
| `--max-running-requests 4` | 4 | Yes — bigger if you have batch needs + VRAM headroom |
| `--max-mamba-cache-size 8` | 8 | Yes — proportional to max-running-requests |
| `--context-length 32768` | 32K | Yes — push higher if KV pool allows |
| `--mem-fraction-static 0.85` | 0.85 | Yes — bigger if tight, smaller if you need more cushion |

### Target variant overrides

The dual compose accepts env-var overrides so you can swap the target without YAML changes:

```bash
# Default (AutoRound INT4)
TARGET_DIR=qwen3.6-27b-autoround-int4 QUANTIZATION=auto-round  # (compose default)

# AWQ INT4 alternative (no AutoRound patch needed — compressed-tensors path)
TARGET_DIR=qwen3.6-27b-awq-int4 QUANTIZATION=compressed-tensors

# INT8 W8A16 (untested, would need ~36 GB download of TheHouseOfTheDude/Qwen3.6-27B-INT8)
TARGET_DIR=qwen3.6-27b-int8 QUANTIZATION=compressed-tensors
```

---

## Re-test triggers (post-parking watch list)

The path is parked because EAGLE-3 < MTP for Qwen3-Next (point 1 above) is a structural finding, not a fixable bug. The other two points are addressable upstream. Triggers to revisit:

| Trigger | What it unblocks |
|---|---|
| **SGLang ships an MTP path for Qwen3-Next** (i.e. uses the model's built-in head, like vLLM Genesis P67 does) | The structural reason for parking disappears — SGLang then competes with vLLM on engine merits, not spec-decode shape |
| **SGLang upstream lands a CUTE_DSL Ampere fallback** (gates the `cutlass.cute.experimental` walk on `SM >= 10.0`, or registers a stub before capture, or refactors `get_version()` not to walk during capture) | cuda-graph re-enables, recovers the ~80% decode TPS we lose to `--disable-cuda-graph` — moves the comparison from "vLLM 85 TPS vs SGLang 15-18 TPS" to something closer |
| **SGLang `OffloaderV1` handles tied weights** | Single 3090 EAGLE-3 becomes a viable VRAM-tight target — but still pinned by point 1 above |
| **SGLang adds asymmetric K/V or sub-FP8 INT KV** | Closes the KV-density gap vs vLLM TurboQuant; potentially ~262K context at TP=2 — independently useful regardless of the EAGLE-3 question |
| [`sgl-project/sglang#20370`](https://github.com/sgl-project/sglang/pulls/20370) merges OR upstream lands an equivalent | We can drop `patch_sglang_autoround_fused_bf16.py` and follow rolling upstream tags (this fix is broadly useful — anyone running SGLang AutoRound on Qwen3-Next hits the same crash) |

---

## Cross-links

- [`../../../docs/engines/SGLANG.md`](../../../docs/engines/SGLANG.md) — engine-level pros/cons, KV cache options, Ampere quirks
- [`patches/patch_sglang_autoround_fused_bf16.md`](patches/patch_sglang_autoround_fused_bf16.md) — full Marlin name-mapper fix mechanics + boot caveats
- [`../vllm/`](../vllm/) — vLLM track (production-best)
- [`../llama-cpp/`](../llama-cpp/) — llama.cpp track (single-card robustness + max ctx)
- [`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3) — the EAGLE-3 drafter
- [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — the target weights
- [`sgl-project/sglang#19406`](https://github.com/sgl-project/sglang/issues/19406) — upstream bug filing (the Marlin name-mapper crash)
