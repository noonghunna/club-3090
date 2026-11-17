# vLLM — the validated path (this repo)

This is what the repo's [Quick start](../../README.md#quick-start) ships. Everything in the main README is the vLLM recipe — this page is a brief recap + tuning levers + when to deviate from defaults.

---

## TL;DR

- ✅ Validated, production-grade
- ✅ Full feature set: vision, tools, streaming, thinking, MTP n=3, TurboQuant 3-bit KV
- ✅ Full OpenAI API parity
- 50-53 narr / 66-70 code TPS on a single 3090; 58 narr / 76 code on dual-card (TP=2)
- 48K default ctx · 75K IDE-agent · **145K with vision · 180K Balanced MTP · 200K Max-context** (single 3090) · 262K dual-card. **Cliff 2 closed at 60K** since 2026-05-02 PM via Genesis v7.69 + vllm#35975 backport — see [docs/CLIFFS.md](../CLIFFS.md). >60K single-prompt still hits the 24 GB hardware-physical wall on single-card.

---

## What's in the box

- vLLM nightly (pinned to `vllm/vllm-openai:nightly-7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8` = `0.20.1rc1.dev16+g7a1eb8ac2`)
- Sandermage's [Genesis v7.69 dev tip patches](https://github.com/Sandermage/genesis-vllm-patches) (commit `2db18df`, mounted into vLLM's site-packages at boot)
- `patch_inputs_embeds_optional.py` sidecar — backport of [vllm#35975](https://github.com/vllm-project/vllm/pull/35975), skips the text-only `inputs_embeds` GPU buffer (~444 MiB freed at boot on our 180K + MTP K=3 path). Required for the Cliff 2 60K closure recipe.
- Our [`patch_tolist_cudagraph.py`](../../patches/patch_tolist_cudagraph.py) (CUDA graph capture fix for TurboQuant continuation prefill)
- 6 compose variants with different KV/ctx/feature trade-offs (see [README Status](../../README.md#status-at-a-glance))

---

## Quick recipe

The README's Quick start is the canonical recipe. Reproduced here:

```bash
git clone https://github.com/noonghunna/qwen36-27b-single-3090.git
cd qwen36-27b-single-3090
bash scripts/setup.sh
cd compose && docker compose up -d
docker logs -f vllm-qwen36-27b      # wait for "Application startup complete"
curl -sf http://localhost:8020/v1/models
```

Verify:
```bash
bash scripts/verify-full.sh         # 10 functional checks
bash scripts/bench.sh               # 3 warmups + 5 measured (narr + code)
```

---

## Pros (vs llama.cpp / SGLang)

| Pro | Detail |
|---|---|
| **Deepest Qwen3-Next feature support** | Vision tower, MTP head, all attention variants supported upstream. |
| **TurboQuant 3-bit KV** | Lets us reach 198K + vision or 214K text-only on 24 GB single-card (262K on dual-card). No equivalent in llama.cpp; SGLang has it but blocked by other bugs. |
| **MTP speculative decoding** | Works out of the box on the Lorbus quant; mainline llama.cpp doesn't expose MTP. |
| **Active development** | Bugs we hit get triaged within days. We've contributed back. |
| **Full OpenAI API parity** | Tools, streaming, vision-in-message, reasoning-mode, structured output — everything works. |

## Cons (vs llama.cpp / SGLang)

| Con | Detail |
|---|---|
| **Heavyweight** | Docker image is ~9 GB. NVIDIA-only. |
| **Longer cold start** | ~2 min for compile + cudagraph capture. |
| **Sensitive to upstream API drift** | We pin to a specific nightly SHA (`7a1eb8ac` = `0.20.1rc1.dev16`) to avoid this. Bumping the pin needs re-validation across all five main variants. |
| **Frontier features can ship with bugs** | TurboQuant × spec-decode × cudagraph corruption (the whole reason this repo's patches exist). |

---

## Tuning levers

### `--max-model-len` and `--gpu-memory-utilization`

Control context vs activation headroom. See the [Activation-memory caveat](../../README.md#activation-memory-caveat-read-this-before-raising---max-model-len) — this is the most consequential tuning lever.

**From vLLM v0.34.0 our composes also pass `--no-enable-extensible-kv-cache`.** v0.34.0's new default,
the extensible KV cache, sizes the pool after warmup and keeps the larger of the activation peak and
torch's reservation growth free in full, plus a margin of at least 1 GiB, on top of
`--gpu-memory-utilization`. On 24 GB cards that stops `vllm/qwen38-27b-dual-max` at 262,144 context
(3.25 GiB of KV against the 4.46 it needs) and shrinks `dual-fast`'s pool from 605K to 466K tokens.
The flag returns to v0.33.0's estimate-based sizing: 283,527 and 602,678 tokens. What it gives up is
the extensible cache's protection against memory spikes the profiling run misses; `verify-stress`
fills `dual-max` to 240,663 tokens without an OOM. Composes that set `--kv-cache-memory-bytes` don't
need it: that turns the extensible cache off by itself. Upstream: vllm#68184 in
[UPSTREAM.md](../UPSTREAM.md).

### KV cache type (`--kv-cache-dtype`)

| Type | Per-token bytes | 24 GB ceiling | Notes |
|---|---|---|---|
| BF16 (default) | ~55 KB | ~8K | Don't use on this hardware |
| `fp8_e5m2` | ~28 KB | ~32K | Fast-chat / Tools-text variants |
| `turboquant_4bit_nc` | ~23 KB | ~84K | Untested by us — should work |
| `turboquant_3bit_nc` ⭐ | ~17 KB | ~125K | Default v7.14 variant |

Lower bytes/token = more context, but more dequant scratch + activation pressure. The 3-bit variant is what makes the 198K + vision and 214K text-only tiers reachable on a single 24 GB card.

### Spec-decode (`--speculative-config`)

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

n=3 is the empirical sweet spot. n=4 nominally hits higher TPS on code but 4th-position acceptance collapses to ~21%. Don't push higher.

### Power cap

```bash
sudo nvidia-smi -pm 1            # one-time: enable persistence mode
sudo nvidia-smi -pl 290 -i 0     # air-cooled default — peak TPS/W (per 10W-resolution sweep)
sudo nvidia-smi -pl 330 -i 0     # water-cooled default (per @syangsao 3-cap data)
sudo nvidia-smi -pl 230 -i 0     # thermal-constrained / quiet — NOT a sweet spot, see note
```

Sweet spot: **290W (air) / 330W (water)** — peak TPS/W efficiency on 3090s and only ~5-7% TPS loss vs unrestricted stock. Past the sweet spot: diminishing returns (SM clocks saturate near 1.9 GHz). Stock TDP is *less* efficient than the sweet-spot cap on Qwen3.6's GDN-attention kernels.

**Note**: 230W is NOT the sweet spot — 290W is. The "230W sweet spot" lore traces back to coarse 3-cap-resolution data; the dense 10W sweep on this rig shows 230W costs ~16% efficiency vs 290W (air-cooled). 230W is a low-power / quiet cap, not a perf-per-watt one. On llama.cpp + Qwen3.6 it costs ~34% TPS (cross-rig data from [@syangsao](https://github.com/noonghunna/club-3090/issues/58#issuecomment-4388766174)) because the GDN forward kernel is genuinely compute-bound. On vLLM the penalty is smaller (~10-15%) because the kernel mix is GEMM-dominated, but 290W is still the better default. See [docs/HARDWARE.md#power](../HARDWARE.md#power) for the full cross-rig table.

### Genesis env-opt-in patches

The default compose enables P64/P65/P66 by default. Optional patches (env-gated):

```yaml
- GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1            # long-ctx tool adherence
- GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1
```

If you're running long-ctx tool flows (50K+ tokens with multiple tools active), these help with format compliance. They're already enabled in the default v7.14 compose.

---

## When to deviate from defaults

| Workload | Compose | Why |
|---|---|---|
| IDE agents (Cline / Cursor / Copilot) + long prompts | `single/autoround-int4/tools-text.yml` | fp8 + 75K + no vision; PN8 closes Cliff 1 |
| No spec-decode (debugging) | `single/autoround-int4/minimal.yml` | 32K + fp8 + no MTP — simplest stack |
| Minimal (no Genesis, no spec-decode) | `single/autoround-int4/minimal.yml` | 32K + fp8 — escape hatch if Genesis clone fails |

See [SINGLE_CARD.md](../SINGLE_CARD.md) and [DUAL_CARD.md](../DUAL_CARD.md) for full per-workload guidance.

---

## When to switch engines

You don't need to switch unless:
- You need lighter cold start / smaller footprint → [llama.cpp](LLAMA_CPP.md)
- You need to run on AMD / Intel / Apple Silicon → [llama.cpp](LLAMA_CPP.md)
- You're building a high-throughput multi-tenant service and want SGLang's RadixAttention scheduling → [SGLang](SGLANG.md) (currently blocked, see watch list)
- **You want max context (262K) on a single 3090** → [llama.cpp + Q4_K_M + q4_0 KV](LLAMA_CPP.md#going-to-262k-full-qwen36-context-on-a-single-3090). Honest trade-off: AutoRound (~19 GB) leaves only ~3-4 GB for KV at full ctx; a smaller GGUF leaves ~8-10 GB. You lose MTP spec-decode (~30% TPS) but gain ~5× the usable context.

For most local-LLM use cases, vLLM is the right pick on this model class. Other engines exist for legitimate reasons; this stack is just optimized for the vLLM path.
