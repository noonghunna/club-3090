# SGLang — Qwen3-Next EAGLE-3 path PARKED; no shipped variant on this stack

SGLang is a strong alternative to vLLM for high-throughput multi-tenant serving — RadixAttention prefix sharing, structured-output-aware scheduling. We investigated it as a way to unlock **EAGLE-3 external-drafter spec-decode** for Qwen3-Next family (which vLLM doesn't support — blocked by DeltaNet KV rollback). The path reached boot + coherent output on dual 3090 with two vendored patches.

**Status (2026-05-21): PARKED.** Not currently a shipped variant on this stack for Qwen3-Next. Three independent findings drove the decision:

1. **EAGLE-3 is sub-MTP for Qwen3-Next, even on Blackwell where it works.** Ex0bit's own published numbers on the [PRISM-PRO-DQ model card](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ) report native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)**. The model family has a strong built-in MTP head; routing through an external drafter is structurally slower.
2. **CUTE_DSL capture-hang on Ampere.** SGLang v0.5.12's CUTE_DSL `get_version()` does `pkgutil.walk_packages` during cuda-graph capture, hits `cutlass.cute.experimental` which raises `NotImplementedError` on CUDA toolkit < 13.1, and deadlocks against the locked capture stream. Workaround `--disable-cuda-graph` caps decode at ~15-18 TPS. Three patch iterations (pre-import; sys.modules stub at engine init; per-process sys.modules stub at `sglang/__init__.py`) all failed — the walk re-fires during capture regardless of cache state.
3. **vLLM-MTP-dual already beats this path on the same rig.** `vllm/dual/turbo.yml` (TP=2 + MTP + Genesis TQ3) delivers ~85 TPS on this dual-3090 setup vs ~15-18 TPS for SGLang+EAGLE3 with the cuda-graph workaround.

The artifacts under [`models/qwen3.6-27b/sglang/`](../../models/qwen3.6-27b/sglang/) stay in the tree for archival reference; the README in that subtree has the full re-test triggers.

**Verdict:** for Qwen3-Next on consumer Ampere, use vLLM MTP (`vllm/dual/turbo.yml`) or llama.cpp MTP (`llamacpp/mtp.yml`). SGLang may still be worth revisiting for OTHER model families where its RadixAttention or structured-output features are the headline — that's a per-model decision.

---

## TL;DR

| What | Status |
|---|---|
| Dual 3090 (TP=2) + AutoRound INT4 + EAGLE-3 — **boots + serves coherent output** | ✅ Validated 2026-05-20 |
| Single 3090 + AutoRound INT4 + EAGLE-3 | ❌ Hits SGLang OffloaderV1 tied-weights bug on Qwen3-Next |
| Marlin alignment crash on AutoRound INT4 | ✅ Fixed by vendored patch (root cause: name-mapping, not kernel) |
| EAGLE-3 spec-decode capture hook on `Qwen3_5ForConditionalGeneration` | ✅ Vendored second patch |
| BF16 EAGLE-3 drafter loading | ✅ With `--speculative-draft-model-quantization unquant` flag |
| cuda-graph capture on Ampere | ❌ Hangs (CUTLASS CUTE Hopper-oriented) — must use `--disable-cuda-graph` |
| Sub-FP8 KV cache on Ampere | ❌ Hard ceiling at 8 bits/token (FP4 falls back to slow un-fused dequant) |
| TPS / accept-rate / quality | ⚠️ Pending bench session |

**One-line summary:** SGLang on club-3090 = "the EAGLE-3-on-Qwen3-Next-quantized path" that vLLM doesn't have, but only practical at TP=2 and with two vendored patches.

---

## Why pick SGLang over vLLM / llama.cpp here?

| Path | When SGLang wins |
|---|---|
| vs **vLLM** | You want EAGLE-3 external-drafter spec-decode on Qwen3-Next + a quantized target. vLLM's spec-decode is blocked by DeltaNet KV rollback (MTP works but is decoder-internal, not Ex0bit-style EAGLE-3). |
| vs **llama.cpp** | You want multi-tenant serving with RadixAttention prefix sharing, OR you want SGLang's V2 scheduler for hybrid Mamba models, OR you want SGLang's structured-output scheduler. |
| For everything else | vLLM (production-best on this stack) or llama.cpp (single-card robustness). |

**Where SGLang loses on club-3090:**
- Smaller KV-density ceiling than vLLM (8 bits/token vs vLLM's 3 bits with TurboQuant)
- Patches required for AutoRound INT4 + Qwen3-Next (we vendor them locally)
- cuda-graph disabled on Ampere → decode TPS will be lower than ideal
- Single-card EAGLE-3 not workable today (CPU-offload broken on Qwen3-Next tied weights)

---

## Pros

| Pro | Detail |
|---|---|
| **EAGLE-3 spec-decode on Qwen3-Next + quantized target** | The only validated external-drafter spec-decode path for the Qwen3-Next family on Ampere consumer hardware. |
| **SPEC_V2 scheduler handles hybrid GatedDeltaNet** | `SGLANG_ENABLE_SPEC_V2=1` is purpose-built for hybrid Mamba + radix cache. Unblocks what DeltaNet rollback blocked in vLLM's spec-decode. |
| **RadixAttention prefix sharing** | Strong fit for multi-tenant workloads with shared system prompts. |
| **Multiple quant loader paths** | 25 quantization methods supported in v0.5.12 (auto-round, compressed-tensors, awq, gptq, gptq_marlin, bitsandbytes, gguf, torchao int4wo-XX, etc). |
| **OpenAI-compatible API** | Drop-in API parity with vLLM/llama.cpp's OpenAI endpoint. |

## Cons (real)

| Con | Detail |
|---|---|
| **Vendored patches required for AutoRound + Qwen3-Next** | Two startup patches: `patch_sglang_eagle3.py` (EAGLE-3 capture hook) + `patch_sglang_autoround_fused_bf16.py` (preserves AutoRound's packed_modules_mapping so BF16-keep layers aren't routed to Marlin). Image is pinned to `v0.5.12` per AGENTS.md engine-image-pinning policy. |
| **KV cache density ceiling at FP8 (8 bits/token)** | SGLang's `--kv-cache-dtype` options are `auto / bf16 / fp8_e5m2 / fp8_e4m3 / fp4_e2m1`. FP4 falls back to un-fused dequant on Ampere → likely slow. No INT4 KV, no TurboQuant-equivalent, no asymmetric K/V (single dtype for both). Gap vs vLLM's `turboquant_3bit_nc` is 2.7× KV density. |
| **cuda-graph capture hangs on Ampere** | CUTLASS CUTE backend is Hopper-oriented; on Ampere it can hang at "Capture cuda graph bs [1]" indefinitely. Must use `--disable-cuda-graph`. Costs decode TPS. |
| **Single-3090 EAGLE-3 blocked** | At 24 GB the target (~17 GB) + EAGLE-3 drafter (~3 GB) + Mamba state + KV cache leaves ~0-2 GB headroom. CPU offload "fixes" the budget but hits SGLang's OffloaderV1 tied-weights bug on Qwen3-Next (`ValueError: functional_call got multiple values for keys ['linear_attn.attn.dt_bias', 'linear_attn.dt_bias']`). |
| **Multi-arch image is 47 GB extracted** | The `lmsysorg/sglang:v0.5.12` image bundles every CUDA arch's compiled kernels. Stripping to Ampere-only would need a custom Dockerfile build. |
| **Cookbook lacks consumer Ampere coverage** | [SGLang's Qwen3.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6) documents only BF16/FP8 on H100/H200/B200. Our path is community-pioneered. |

---

## What we ship

Two experimental composes under `models/qwen3.6-27b/sglang/compose/`:

| Compose | Topology | Status |
|---|---|---|
| `single/eagle3-experimental.yml` | 1× 3090 | ⚠️ Blocked on SGLang OffloaderV1 tied-weights bug. Kept as reference for re-test when SGLang's offloader handles tied DeltaNet params. |
| `dual/eagle3-experimental.yml` | 2× 3090 (TP=2) | ✅ Boots, serves coherent output. TPS/accept-rate pending bench. |

Both composes apply two patches at startup:

1. **`patch_sglang_eagle3.py`** — provided by [`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3) — adds `set_eagle3_layers_to_capture` hook to `Qwen3_5ForConditionalGeneration` so EAGLE-3's auxiliary hidden capture works.

2. **`patch_sglang_autoround_fused_bf16.py`** — our local fix — preserves AutoRound's `packed_modules_mapping` so SGLang's auto-round loader correctly keeps `linear_attn.in_proj_a` / `linear_attn.in_proj_b` (fused as `in_proj_ba`) at BF16 instead of incorrectly routing them through GPTQ-Marlin. See `models/qwen3.6-27b/sglang/patches/patch_sglang_autoround_fused_bf16.md` for the full mechanic.

The patches are idempotent + AST-validated + write `.bak` files. Both apply at container start; bind-mounted from the repo, no Dockerfile changes.

---

## Recipe — Dual 3090 EAGLE-3 on AutoRound INT4

```bash
# Prereqs: AutoRound target + EAGLE-3 drafter on disk
# (drafter download is described in models/qwen3.6-27b/sglang/README.md)

cd <repo>/models/qwen3.6-27b/sglang/compose/dual
MODEL_DIR=/your/models/dir docker compose -f eagle3-experimental.yml up -d

# Wait ~75s for boot, then probe:
curl -s http://localhost:8041/v1/models | python3 -m json.tool
curl -s http://localhost:8041/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-eagle3-dual",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":50,"temperature":0.6}'
```

**The dual compose ships with these knobs** (gleaned from Codex's single-card exploration, then relaxed for dual VRAM headroom):

| Flag | Value | Why |
|---|---|---|
| `--tp-size 2` | 2 | Split target across 2× 3090 |
| `--disable-custom-all-reduce` | (set) | PCIe-only Ampere has no NVLink; custom all-reduce must be off |
| `--speculative-algorithm EAGLE3` | (set) | Engages SPEC_V2 scheduler |
| `--speculative-draft-model-quantization unquant` | (set) | **Critical** — BF16 drafter must opt out of target's AutoRound quant |
| `--kv-cache-dtype fp8_e5m2` | (set) | ~50% KV savings vs BF16 (FP8 is the practical Ampere ceiling) |
| `--disable-cuda-graph` | (set) | **Critical** — CUTE_DSL hangs at capture on Ampere |
| `--max-running-requests 4` | 4 | Reasonable for dual; single needed 1 |
| `--max-mamba-cache-size 8` | 8 | Bigger than single-card 1, room for batch |
| `--mamba-scheduler-strategy extra_buffer` | (set) | Dual has headroom; no need for the `no_buffer` aggression single needed |
| `--mem-fraction-static 0.85` | 0.85 | SGLang default is 0.88; 0.85 leaves more cushion |
| `--context-length 32768` | 32K | Conservative; bumpable on dual |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | (env) | Allocator hygiene |

---

## Tuning levers + Ampere gotchas

### KV cache type — the biggest single lever

SGLang's `--kv-cache-dtype` choices and behavior on Ampere sm_86:

| Format | Bits/token | Practical on 3090? |
|---|---|---|
| `auto` (= BF16) | 16 | ✅ Default, no compression |
| `bf16` / `bfloat16` | 16 | ✅ Same as auto |
| `fp8_e5m2` | 8 | ✅ Storage compact, dequant fused with attention → ~50% KV savings, negligible decode overhead. **Our default.** |
| `fp8_e4m3` | 8 | ✅ Same path, slightly different precision (higher mantissa). SGLang docs recommend for accuracy when scales are calibrated. |
| `fp4_e2m1` | 4 | ⚠️ FlashInfer FP4 kernels are Blackwell-fast-path; on Ampere falls back to "pure tensor ops" — likely slow enough to erase the savings |

Not available in SGLang:
- INT4 KV
- INT8 KV (W8A16 path, but not as KV cache type)
- Asymmetric K/V (single dtype for both)
- Per-channel or per-token scaling (per-tensor only)
- TurboQuant equivalent (vLLM's smallest is 3 bits via `turboquant_3bit_nc`; SGLang has no equivalent)

### `--disable-cuda-graph` on Ampere

Without this flag, SGLang's cuda-graph capture hangs at "Capture cuda graph bs [1]" indefinitely, with `CUTE_DSL - WARNING - [handle_import_error]` for `cutlass.cute.experimental`. CUTLASS CUTE is the Hopper-targeted CUTLASS path; SGLang's Ampere fallback at graph-capture time can lock up.

Cost: decode TPS hit (cuda-graphs eliminate launch-overhead per token). On Ampere with this engine + model + quant combo, the trade-off is "boots vs. doesn't."

### `--speculative-draft-model-quantization unquant`

Without this, the BF16 EAGLE-3 drafter silently inherits the target's `--quantization auto-round` flag and fails to load (it's a BF16 model, no AutoRound config). The fix is to explicitly opt the drafter out via `unquant`. Mandatory for any external-BF16-drafter + quantized-target combo.

### Mamba memory pool caps

For hybrid Mamba models like Qwen3-Next, SGLang reserves a Mamba state pool sized as roughly `n_mamba_layers × max_running_requests × per-layer-state-size`. On tight single-card VRAM, the default 48-request reserve can starve the KV pool. On dual you have headroom; we cap at `--max-running-requests 4` + `--max-mamba-cache-size 8` for safety.

---

## Watch list — what would change the picture

| Trigger | Impact |
|---|---|
| SGLang upstream merges the AutoRound name-mapper fix (track [`sgl-project/sglang#19406`](https://github.com/sgl-project/sglang/issues/19406) + [`#20370`](https://github.com/sgl-project/sglang/pulls/20370)) | We can drop our `patch_sglang_autoround_fused_bf16.py` vendor. |
| SGLang upstream merges the EAGLE-3 capture hook for `Qwen3_5ForConditionalGeneration` | We can drop the Ex0bit `patch_sglang_eagle3.py` vendor (or it ships baked into the drafter). |
| SGLang's OffloaderV1 handles tied weights (Qwen3-Next `linear_attn.attn.dt_bias` / `linear_attn.dt_bias`) | Single-3090 EAGLE-3 becomes viable via CPU offload. |
| SGLang adds asymmetric K/V or sub-FP8 INT KV path | Closes the KV-density gap vs vLLM TurboQuant. Would unlock context lengths comparable to our `dual/turbo.yml` (262K). |
| CUTLASS CUTE adds Ampere kernels OR SGLang routes around it on sm_86 | We can re-enable cuda-graph and recover decode TPS. |

---

## When to use SGLang on this stack

- ✅ You want **EAGLE-3 spec-decode** on Qwen3-Next + a quantized target. Nothing else on club-3090 offers this today.
- ✅ You're testing **multi-tenant RadixAttention** workloads.
- ✅ You want to validate that **SPEC_V2 hybrid Mamba scheduling** is working for your application.

## When to use something else

- ❌ You want **max single-card context length** at any cost → use `llamacpp/default` (262K @ q4_0 KV) or `vllm/long-text.yml` (180K @ TQ3 KV).
- ❌ You want **production-stable serving with proven multi-week soak** → vLLM is our default.
- ❌ You want **no source-level patches** to maintain → use llama.cpp (no patches) or vLLM (patches but battle-tested).
- ❌ You're on **single 3090** and need EAGLE-3 → wait for SGLang OffloaderV1 fix, or use dual-GPU.

---

## See also

- [VLLM.md](VLLM.md) — current production-default path
- [LLAMA_CPP.md](LLAMA_CPP.md) — single-card robustness + max-context path
- [`models/qwen3.6-27b/sglang/`](../../models/qwen3.6-27b/sglang/) — the Qwen3.6 SGLang composes + patches
- [`models/qwen3.6-27b/sglang/patches/patch_sglang_autoround_fused_bf16.md`](../../models/qwen3.6-27b/sglang/patches/patch_sglang_autoround_fused_bf16.md) — full mechanics of the Marlin name-mapper fix
- [SGLang official docs](https://docs.sglang.io/) — the upstream documentation (cookbook lacks consumer Ampere coverage)
