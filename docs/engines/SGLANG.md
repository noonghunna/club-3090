# SGLang — currently blocked, watch list

SGLang is a strong alternative to vLLM for high-throughput multi-tenant serving — RadixAttention prefix sharing, structured-output-aware scheduling, batch structured decoding. It often beats vLLM by 10-30% on multi-tenant aggregate throughput **when both work**.

**Currently SGLang doesn't run cleanly on this stack** (Qwen3.6-27B-int4-AutoRound + 3090). Below: what's blocked, why, and what would unblock it.

---

## TL;DR

- ❌ Blocked by the same Marlin pad-sub-tile-n bug we hit on vLLM TP=2 (same kernel-line fix applies).
- ❌ EAGLE spec-decode (their MTP equivalent) blocked separately by DeltaNet/GDN hybrid layer not supporting KV rollback.
- ✅ Will likely unblock when (a) Marlin pad lands on SGLang, and (b) DeltaNet KV rollback support lands upstream (vllm#39931 / issue #40124 cross-engine).
- ⚠️ TBD recipe — we don't ship a working SGLang config yet. We'll add one when the blockers clear.

---

## Pros (when it works)

| Pro | Detail |
|---|---|
| **High-throughput multi-tenant serving** | RadixAttention shares prefix KV across requests automatically. Multi-tenant aggregate often beats vLLM by 10-30%. |
| **Structured-output-aware scheduling** | Prioritizes batched constraint-decoding requests for better GPU utilization. |
| **First-class OpenAI API** | Same level of API parity as vLLM. |
| **Active development** | Smaller community than vLLM but engaged maintainers. |

## Cons (right now, on this model class)

| Con | Detail |
|---|---|
| **Marlin pad-sub-tile-n bug** | Same INT4 kernel issue we filed [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) for. SGLang's Marlin call site has the equivalent constraint and would crash on Lorbus's quant + TP=2. We haven't filed an SGLang PR — would need someone to port the fix or wait for SGLang to pick up the upstream Marlin fix. |
| **EAGLE spec-decode blocked on hybrid attention** | Qwen3-Next's DeltaNet layers don't support KV rollback the way standard attention does. EAGLE (and any speculative-decode method that needs rollback) breaks. This is architectural; needs upstream `flash-linear-attention` to add rollback support, then SGLang to integrate. |
| **Smaller community than vLLM** | Fewer eyes on Qwen3-Next bugs. When something breaks here, we may be on our own. |

---

## Watch list — what would unblock SGLang on this stack

### 1. Marlin pad-sub-tile-n landed on SGLang

Two paths:
- **Upstream Marlin landing** — if vllm-project's Marlin gets the fix and SGLang picks it up via shared upstream code, we get it for free.
- **SGLang-side patch** — same kernel line check + pad logic, applied in SGLang's Marlin call site. Could be filed as an SGLang PR (we haven't yet — vLLM was the priority).

Track: our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361). When it lands and propagates, SGLang can pick it up.

### 2. DeltaNet KV rollback support for EAGLE / spec-decode

EAGLE (SGLang's MTP equivalent) requires the model to support rolling back the KV cache when speculative tokens are rejected. Standard attention layers do this trivially — KV cache is just discarded for the rejected positions. DeltaNet layers maintain a recurrent state that doesn't roll back cleanly.

Track:
- [vllm#39931](https://github.com/vllm-project/vllm/issues/39931) — Qwen3-Next hybrid attention rollback support
- [vllm#40124](https://github.com/vllm-project/vllm/issues/40124) — related upstream issue
- [`flash-linear-attention` library](https://github.com/fla-org/flash-linear-attention) — the underlying linear-attention impl needs rollback hooks added; this is the architectural change

When this lands, EAGLE on Qwen3-Next becomes possible across all engines (vLLM, SGLang, etc).

### 3. (Optional) FlashKDA Hopper kernels for Ampere

A separate research direction — Kimi Delta Attention (KDA) / FlashKDA brings prefill-speedup CUTLASS kernels for DeltaNet-family models, but they're targeting Hopper today. Not usable on Ampere. If/when an Ampere port appears or vLLM/SGLang adopt the `flash-linear-attention` backend, this would substantially reduce the GDN forward cost on this stack. Watch list, not unblocker.

---

## Recipe — TBD

We don't have a working SGLang config yet for this model. Here's what one would look like (untested, just as a placeholder):

```bash
# IF the Marlin pad fix were already in SGLang:
docker run --gpus all --shm-size 16g \
  -v /mnt/models/huggingface/qwen3.6-27b-autoround-int4:/model \
  -p 8020:30000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path /model \
    --quantization awq_marlin \
    --tp-size 1 \
    --mem-fraction-static 0.92 \
    --context-length 65536 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <eagle-draft-path> \
    --speculative-num-steps 3
```

**Don't run this** — it'll crash on the Marlin pad bug. Listed here as the shape of the recipe we'd ship once unblocked.

---

## When to revisit

We'll add a working recipe (and lift this page from "blocked" to "validated alternative") when **both**:
- A Marlin pad-equivalent fix lands on SGLang (upstream or via PR), AND
- DeltaNet KV rollback support lands upstream (or we accept running without spec-decode)

Until then, the [vLLM path](VLLM.md) is the validated option for serious local use.

---

## See also

- [VLLM.md](VLLM.md) — current validated path
- [LLAMA_CPP.md](LLAMA_CPP.md) — alternative for lighter setups
- [SGLang docs](https://sgl-project.github.io/) — official documentation
- Cross-engine architecture issue tracking: [vllm#39931](https://github.com/vllm-project/vllm/issues/39931) (DeltaNet rollback)
