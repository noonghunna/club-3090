# Qwen3.6-27B on SGLang — currently blocked

SGLang is a strong alternative to vLLM for high-throughput multi-tenant serving — RadixAttention prefix sharing, structured-output-aware scheduling. It often beats vLLM by 10-30% on aggregate throughput **when both work**.

**Currently SGLang doesn't run cleanly on this stack** (Qwen3.6-27B-int4-AutoRound + 3090). Below: what's blocked, why, and what would unblock it.

---

## TL;DR

- ❌ Blocked by the same Marlin pad-sub-tile-n bug we filed [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) for. Same kernel-line fix applies to SGLang's Marlin call site.
- ❌ EAGLE spec-decode (their MTP equivalent) blocked separately by DeltaNet/GDN hybrid layer not supporting KV rollback.
- ✅ Will likely unblock when (a) Marlin pad lands on SGLang, and (b) DeltaNet KV rollback support lands upstream.
- ⚠️ TBD recipe — we don't ship a working SGLang config yet. We'll add one when the blockers clear.

For full pros/cons + the TBD recipe shape, see [`/docs/engines/SGLANG.md`](../../../docs/engines/SGLANG.md).

---

## What would unblock SGLang on this model

### 1. Marlin pad-sub-tile-n landing on SGLang

Two paths:
- **Upstream Marlin landing** — if vllm-project's Marlin gets the fix and SGLang picks it up via shared upstream code, we get it for free.
- **SGLang-side patch** — same kernel line check + pad logic, applied in SGLang's Marlin call site. Could be filed as an SGLang PR.

Track: our [vllm#40361](https://github.com/vllm-project/vllm/pull/40361). When it lands and propagates, SGLang should pick it up.

### 2. DeltaNet KV rollback support for EAGLE / spec-decode

EAGLE requires the model to support rolling back the KV cache when speculative tokens are rejected. DeltaNet layers maintain a recurrent state that doesn't roll back cleanly.

Track:
- [vllm#39931](https://github.com/vllm-project/vllm/issues/39931) — Qwen3-Next hybrid attention rollback support
- [vllm#40124](https://github.com/vllm-project/vllm/issues/40124) — related upstream issue
- [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) — needs rollback hooks

When this lands, EAGLE on Qwen3-Next becomes possible across all engines (vLLM, SGLang, etc).

---

## When to revisit

We'll lift this from "blocked" to "validated alternative" when **both**:
- A Marlin pad-equivalent fix lands on SGLang (upstream or via PR), AND
- DeltaNet KV rollback support lands upstream (or we accept running without spec-decode)

Until then: [`../vllm/`](../vllm/) is the validated option for serious local use.
