# vLLM — the validated path (this repo)

This is what the repo's [Quick start](../../README.md#quick-start) ships. Everything in the main README is the vLLM recipe — this page is a brief recap + tuning levers + when to deviate from defaults.

---

## TL;DR

- ✅ Validated, production-grade
- ✅ Full feature set: vision, tools, streaming, thinking, MTP n=3, TurboQuant 3-bit KV
- ✅ Full OpenAI API parity
- 51-55 narr / 67-70 code TPS on a single 3090
- 48K default ctx (opt-in to 192K-205K with caveats)

---

## What's in the box

- vLLM nightly (pinned to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` = `dev205+g07351e088`, Sandermage's reference target)
- Sandermage's [Genesis v7.14 patches](https://github.com/Sandermage/genesis-vllm-patches) (mounted into vLLM's site-packages at boot)
- Our [`patch_tolist_cudagraph.py`](../../patches/patch_tolist_cudagraph.py) (CUDA graph capture fix for TurboQuant continuation prefill)
- 5 compose variants with different KV/ctx/feature trade-offs (see [README Status](../../README.md#status-at-a-glance))

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
| **TurboQuant 3-bit KV** | Lets us reach 192K-205K context on 24 GB. No equivalent in llama.cpp; SGLang has it but blocked by other bugs. |
| **MTP speculative decoding** | Works out of the box on the Lorbus quant; mainline llama.cpp doesn't expose MTP. |
| **Active development** | Bugs we hit get triaged within days. We've contributed back. |
| **Full OpenAI API parity** | Tools, streaming, vision-in-message, reasoning-mode, structured output — everything works. |

## Cons (vs llama.cpp / SGLang)

| Con | Detail |
|---|---|
| **Heavyweight** | Docker image is ~9 GB. NVIDIA-only. |
| **Longer cold start** | ~2 min for compile + cudagraph capture. |
| **Sensitive to upstream API drift** | We pin to dev205 to avoid this. Bumping pinned image needs re-validation. |
| **Frontier features can ship with bugs** | TurboQuant × spec-decode × cudagraph corruption (the whole reason this repo's patches exist). |

---

## Tuning levers

### `--max-model-len` and `--gpu-memory-utilization`

Control context vs activation headroom. See the [Activation-memory caveat](../../README.md#activation-memory-caveat-read-this-before-raising---max-model-len) — this is the most consequential tuning lever.

### KV cache type (`--kv-cache-dtype`)

| Type | Per-token bytes | 24 GB ceiling | Notes |
|---|---|---|---|
| BF16 (default) | ~55 KB | ~8K | Don't use on this hardware |
| `fp8_e5m2` | ~28 KB | ~32K | Fast-chat / Tools-text variants |
| `turboquant_4bit_nc` | ~23 KB | ~84K | Untested by us — should work |
| `turboquant_3bit_nc` ⭐ | ~17 KB | ~125K | Default v7.14 variant |

Lower bytes/token = more context, but more dequant scratch + activation pressure. The 3-bit variant is what enables the 192K opt-in tier.

### Spec-decode (`--speculative-config`)

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

n=3 is the empirical sweet spot. n=4 nominally hits higher TPS on code but 4th-position acceptance collapses to ~21%. Don't push higher.

### Power cap

```bash
sudo nvidia-smi -pl 230 -i 0    # production default — quiet, cool
sudo nvidia-smi -pl 330 -i 0    # ~+10% mean TPS during heavy sessions
```

Past 330W: diminishing returns (SM clocks saturate near 1.9 GHz on 3090s).

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
| Fast chat ≤20K | `docker-compose.fast-chat.yml` | fp8 + 20K + ~5% faster TPS |
| Long single prompts (60K+) | `docker-compose.tools-text.yml` | fp8 + 75K + no vision; avoids GDN cliff |
| No spec-decode (debugging) | `docker-compose.minimal.yml` | 32K + fp8 + no MTP — simplest stack |
| No Genesis patches | `docker-compose.no-genesis-mtp.yml` | 20K + fp8 + MTP — control variant |

See [USE_CASES.md](../USE_CASES.md) for full per-workload guidance.

---

## When to switch engines

You don't need to switch unless:
- You need lighter cold start / smaller footprint → [llama.cpp](LLAMA_CPP.md)
- You need to run on AMD / Intel / Apple Silicon → [llama.cpp](LLAMA_CPP.md)
- You're building a high-throughput multi-tenant service and want SGLang's RadixAttention scheduling → [SGLang](SGLANG.md) (currently blocked, see watch list)
- **You want max context (262K) on a single 3090** → [llama.cpp + Q4_K_M + q4_0 KV](LLAMA_CPP.md#going-to-262k-full-qwen36-context-on-a-single-3090). Honest trade-off: AutoRound (~19 GB) leaves only ~3-4 GB for KV at full ctx; a smaller GGUF leaves ~8-10 GB. You lose MTP spec-decode (~30% TPS) but gain ~5× the usable context.

For most local-LLM use cases, vLLM is the right pick on this model class. Other engines exist for legitimate reasons; this stack is just optimized for the vLLM path.
