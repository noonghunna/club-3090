# SGLang on this stack


> ⚠️ **Pin bumped v0.5.19 → v0.5.20 on 2026-09-19.** Performance is a measured NULL
> (dual-fast, 2 interleaved boots/arm: narrative −2.4%, code +0.4%, prefill −0.3%, all
> inside a boot-to-boot floor of 3.5–11%). The bump buys the `qwen4_exp` + `glm5_next`
> architectures, neither shippable on sm_86 yet. Statements below marked **RE-CHECK**
> were verified on v0.5.19 only.

**Current state (2026-09-14): SGLang IS shipped** — 13 composes for Qwen3.8-27B across
dual/multi4/multi8, 11 registered `sgl/` slugs, all `🧪 experimental`. Stock
`lmsysorg/sglang:v0.5.20`, no engine patches (the W4A8 overlay is opt-in and off by default).

⚠️ This page was previously titled *"EAGLE-3 path PARKED; no shipped variant on this stack"* and
described the 2026-05 Qwen3.6-27B investigation. That is now [archived below](#archive--the-2026-05-qwen3627b-eagle-3-investigation-superseded).

## TL;DR

| What | State |
|---|---|
| Shipped composes | 13 (Qwen3.8-27B), 11 registered slugs, all experimental |
| Engine | stock `v0.5.20`, no patches required |
| Tiers | `fast` (MTP n=4) · `superfast` (DFlash2) · `max`/`supermax` (fp8 weights) |
| cuda-graph on Ampere | ✅ **works** — captures decode to bs=24 (the 2026-05 hang is gone) |
| Concurrency | `--max-running-requests 1` shipped; the engine clamps to `K // r` regardless |
| HiCache | ✅ **works on v0.5.20 with `--mamba-max-states-per-path 1`** (host SSM-pool overflow otherwise) — opt-in `KV_OFFLOAD_GB` on `sgl/qwen38-27b-dual-fast` only |
| W4A8 | opt-in, vendored (#1226/#1248), off by default |

---

## ⭐ The GDN state pool — the thing that actually governs this engine

On a hybrid GDN model (Qwen3.8, Qwen3.6-35B-A3B …) SGLang keeps **two** device pools: the KV cache
and a **mamba/GDN state pool**. The state pool, not KV, is usually what binds.

⛔ **The old mental model on this page was wrong.** It said the pool is sized as
`n_mamba_layers × max_running_requests × per-layer-state-size` and recommended
`--max-mamba-cache-size 8`. Both are wrong: the pool is **auto-fit by a solver**, its size does
**not** depend on `max_running_requests`, and a hardcoded slot count is meaningless across tiers —
the five dual composes measured **55 / 34 / 33 / 21 / 8** on the same rig.

### Read your own boot line

```
Mamba Cache is allocated. max_mamba_cache_size: N
```

Everything below is computed from that `N` (call it `K`). ⚠️ It differs per weight tier, per
drafter tier and per topology. Do not carry a number between composes — that mistake shipped a
wrong constant to five files (club-3090#1319).

### Slots consumed per running request

```
usable working slots = K − k × max_running_requests
```

`k` is set by `--mamba-radix-cache-strategy` (`mem_cache/common.py:26-29`):

| strategy | `k` (slots/request) |
|---|--:|
| `extra_buffer` (default) | 3 |
| `extra_buffer_lazy` | 2 |
| `no_buffer` | 1 |

⚠️ **There is no `−1` for a sink slot.** `MambaSlotAllocator.clear()` builds `arange(1, K+1)` with
slot 0 reserved separately, so `K` is already net of it.

### ⭐ The concurrency clamp uses a DIFFERENT ratio

`_calculate_mamba_ratio()` returns **r = 5** (`extra_buffer` + overlap), **4** (lazy), **3**
(`no_buffer`), and `resolve_max_num_reqs()` caps concurrency at **`K // r`**. This is not `k`.

| K | 55 | 34 | 33 | 21 | 8 |
|---|--:|--:|--:|--:|--:|
| effective cap at r=5 | 11 | 6 | **6** | 4 | **1** |

Measured clamps match exactly: a request for 8 resolved to 8 on K=55 and to **6** on K=33.
⚠️ A pool of K=8 can run **one** request whatever you configure.

⚠️ **`server_args=` echoes the REQUEST, not the effective value.** The line that carries what took
effect is:

```
max_total_num_tokens=… chunked_prefill_size=… max_running_requests=N … context_len=…
```

Or read `/server_info` → `internal_states[0].effective_max_running_requests_per_dp` (the
**top-level** `max_mamba_cache_size` field reads `null`).

### ⚠️ `extra_buffer_lazy` is not a free saving

The auto-fit solve is `K = floor((B − p(1+D)) / (p(1+D/r)))` and uses **r**, so moving 5→4 raises
the denominator and can *shrink* K. Lower per-request cost, smaller pool. Measure both together.

### What concurrency actually costs

The slot count is invariant to `max_running_requests` (verified 1 vs 8 on two tiers). What
concurrency spends is **KV tokens** — roughly −20% going C=1→8 — because the intermediate state
caches grow with it.

---

## HiCache — works on v0.5.20; the trap is host SSM-pool capacity

On the v0.5.20 pin, stock `--enable-hierarchical-cache` serves prefix hits back from host RAM on hybrid
GDN. 2026-09-25, `sgl/qwen38-27b-dual-fast`: a 40K prompt evicted from the GPU came back in 0.19 s vs
26 s cold (`cached_tokens_total{cache_source="host"}`, `load_back_tokens` kv + mamba). What breaks it is
**capacity on the host SSM side**:

- stock SGLang backs up one SSM checkpoint (~37 MB per GPU) per ~2,048 prompt tokens;
- `--hicache-size` is split between host KV and host SSM **in proportion to the device pools**, so at a
  large device pool (dual-fast at the old auto-fit K=63: 451,633 tokens) the host SSM side holds only
  ~240 checkpoints, about 6
  long sessions;
- past that, LRU drops the oldest session's SSM state while its KV stays on host, and that prefix can
  never match again (80K prompts, 2026-09-25: 0 host hits; first misread as #33713).

`--mamba-max-states-per-path 1` cuts a session to ~3 host checkpoints: on the shipped 451K pool, 12 × 40K
sessions pushed off the GPU all came back from host (0.3-3.1 s vs 26 s cold). `slru` keeps re-used
prefixes ahead of one-shot prompts. `--hicache-io-backend direct` / `--hicache-mem-layout
page_first_direct` are not needed; nor are a patch or `--enable-mixed-chunk`. Flag set credit: @A1RM4X
(#1340).

Shipped as the opt-in `KV_OFFLOAD_GB` knob (plus `KV_OFFLOAD_DISK` / `KV_OFFLOAD_DIR` /
`KV_OFFLOAD_DISK_GB`) on `sgl/qwen38-27b-dual-fast` only. The other sgl slugs are not probed yet.

| trap | detail |
|---|---|
| `--hicache-size` is GB (1e9) **per rank** | the knob takes GiB total across both GPUs, like vLLM's `--kv-offloading-size`, and converts |
| host RAM runs above the setting | `KV_OFFLOAD_GB=64` → 74 GiB used (pinned pools plus engine overhead) |
| shrinking the device pool to make a probe fast | flips the split so host KV fills first, and the overflow failure disappears. A negative control must overflow the host SSM side first |
| host hit ≠ bigger GPU pool | HiCache decides how many idle sessions come back warm; it does not raise concurrency |
| a cached prefix ≠ a cold prefill, token for token | a host hit reproduces the GPU state exactly (6/6 identical text and first-token logprobs, host vs device), but ANY cached prefix (device or host) flips greedy near-ties vs a cold run after 0-20 tokens. Compare cached against cached, not against cold |

**Disk tier** (`KV_OFFLOAD_DISK=1` → `--hicache-storage-backend file` at `/kv-offload`). 2026-09-25, dual-fast: a
40K session written before a `docker restart` came back from disk in 4.74 s vs 26.70 s cold
(`cache_source="storage"`, prefetch hit 100%). `KV_OFFLOAD_DISK_GB` caps it (split per GPU; measured 1.43 GiB
per GPU under a 1.5 GiB cap after 3 sessions, LRU eviction). ⚠️ About **one file per token per GPU** (a 40K session
= ~89K files), all root-owned; the cap is per model; a 20 GiB free-space floor is always set.

**GPU pool (`MAX_MAMBA_CACHE_SIZE`, dual-fast pair).** Default K=20 since 2026-09-25 (was auto-fit 63): 548,520
tokens holds two full 262,144-token sessions (measured: two concurrent ~261.9K prompts, both needles recalled).
MTP accept len 3.12, decode and prefill flat, headroom unchanged (K only re-splits the budget). Don't go below ~15.
⛔ sgl#38147 (share the MTP draft's embed/lm_head) is not a pool lever at 0.95: the duplicate it reclaims is
the engine's runtime scratch, and reclaiming it OOMs on the first prefill's Triton autotune.

⚠️ The 2026-09-14 v0.5.19 write-only result (the repro we added to #33713) was not re-examined; don't
retro-attribute it to overflow.

---

## ⛔ `--enable-mixed-chunk` corrupts checkpoints on hybrid GDN

**[sgl#39342](https://github.com/sgl-project/sglang/issues/39342)** (open). With
`extra_buffer`/`extra_buffer_lazy` on a hybrid GDN model, `merge_batch` drops
`mamba_track_indices`, the checkpoint write is skipped, and the finished request donates an
**unwritten** slot to the radix cache — later requests restore stale SSM state. Exact match
0.91 → 0.84-0.86; p50 latency 200 ms → 3-7 s.

⚠️⚠️ **It fires only when prefill co-batches with decode**, so it passes verify-full, verify-stress,
soak (single-stream by design) and any c=1 bench. Our entire operational gate is blind to it. We
ship the flag nowhere; do not add it.

---

## Tuning levers + Ampere gotchas

### KV cache dtype
`fp8_e4m3` is the shipped default and halves bytes/token vs bf16. Sub-8-bit KV is not available on
this engine.

### ✅ cuda-graph on Ampere — the 2026-05 hang is GONE
The archived section below says capture hangs and mandates `--disable-cuda-graph`. On v0.5.19 the  ⚠️ RE-CHECK: verified on v0.5.19; pin moved to v0.5.20 2026-09-19 and this was NOT re-tested.
shipped composes run `cuda graph: True` and capture decode graphs for bs `[1..24]`. Do not carry
that advice forward.

### `--speculative-draft-model-quantization unquant`
Still correct: a BF16 external drafter otherwise inherits the target's `--quantization auto-round`
and fails to load. Mandatory for BF16-drafter + quantized-target.

### `--mamba-ssm-dtype bfloat16`
Shipped default. Halves per-slot state bytes, which **buys slots** (measured 26 → 55 on one tier).
A,B,A,B repeat-boot found ~+8% code decode; prefill ~−1%.

### Prefix-cache policy (opt-in, off by default)
`SESSION_RADIX_CACHE=1` sets `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` **and**
`--enable-session-radix-cache` together (upstream requires both; one alone is inert). It is also
inert unless clients send a `session_id` — no OpenAI-compatible client does by default.
`RADIX_EVICTION_POLICY` accepts `lru` (default) / `lfu` / `slru` / `priority`.
⛔ `--radix-eviction-policy-config` is documented upstream but **does not exist in v0.5.19** —  ⚠️ RE-CHECK: verified on v0.5.19; pin moved to v0.5.20 2026-09-19 and this was NOT re-tested.
passing it hard-fails the boot.

---

## ⚠️ Measurement traps on this engine

These each produced a wrong number for us; they are cheap to avoid and expensive to discover.

| trap | consequence |
|---|---|
| `cache_type`-labelled counters are emitted **per TP rank with identical labels** and summed by the multiprocess collector | `hicache_backup_tokens_total`, `load_back_tokens_total`, `evicted_tokens_total`, `hicache_dropped_tokens_total` read at **tp_size × truth**. The `_bytes_total` counters are genuinely additive. |
| `cached_tokens_total` carries `cache_source=device\|host\|storage` | aggregating without that label collapses the device-vs-host split |
| the device/host split is **not** in `usage` | it needs `return_cached_tokens_details: true` and arrives in a separate `sglext` chunk with `choices: []`, before the usage chunk |
| `usage.prompt_tokens_details` is `null` at 0 | with `--enable-cache-report` on, null means **zero-known**, not unknown |
| reusable prefixes clamp to a **128-token grid** | a prompt shorter than the grid can never insert or hit |
| `/tokenize` returns nothing on this build, and raw-body counts miss the chat template (~10K on a 32K prompt) | size any cache-pressure trace from `usage.prompt_tokens` measured on the wire |
| `evicted_tokens_total` counts **full-KV only** | mamba evictions are invisible in it |
| `mamba_*_tokens` gauges are **state slots**, not tokens | the name lies |
| `mamba_used` **excludes** evictable entries | `K = used + available + evictable` |

Detail and dates: [`learnings/sglang-engine.md`](../../../learnings/sglang-engine.md) 2026-09-14.

---

## Archive — the 2026-05 Qwen3.6-27B EAGLE-3 investigation (SUPERSEDED)

⚠️ Everything below is the **2026-05-21** state and describes a different model
(Qwen3.6-27B) and a different goal (EAGLE-3 external drafters). It is kept for its
re-test triggers and patch mechanics. Its headline — *"no shipped variant on this
stack"* — has been false since 2026-09-11.

### (archived) SGLang — Qwen3-Next EAGLE-3 path PARKED

SGLang is a strong alternative to vLLM for high-throughput multi-tenant serving — RadixAttention prefix sharing, structured-output-aware scheduling. We investigated it as a way to unlock **EAGLE-3 external-drafter spec-decode** for Qwen3-Next family (which vLLM doesn't support — blocked by DeltaNet KV rollback). The path reached boot + coherent output on dual 3090 with two vendored patches.

**Status (2026-05-21): PARKED.** Not currently a shipped variant on this stack for Qwen3-Next. Three independent findings drove the decision:

1. **EAGLE-3 is sub-MTP for Qwen3-Next, even on Blackwell where it works.** Ex0bit's own published numbers on the [PRISM-PRO-DQ model card](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ) report native MTP = **121 TPS (1.51×)** vs EAGLE-3 chain = **111 TPS (1.39×)**. The model family has a strong built-in MTP head; routing through an external drafter is structurally slower.
2. **CUTE_DSL capture-hang on Ampere.** SGLang v0.5.12's CUTE_DSL `get_version()` does `pkgutil.walk_packages` during cuda-graph capture, hits `cutlass.cute.experimental` which raises `NotImplementedError` on CUDA toolkit < 13.1, and deadlocks against the locked capture stream. Workaround `--disable-cuda-graph` caps decode at ~15-18 TPS. Three patch iterations (pre-import; sys.modules stub at engine init; per-process sys.modules stub at `sglang/__init__.py`) all failed — the walk re-fires during capture regardless of cache state.
3. **vLLM-MTP-dual already beats this path on the same rig.** `vllm/dual/autoround-int4/turbo.yml` (TP=2 + MTP + Genesis TQ3) delivers ~85 TPS on this dual-3090 setup vs ~15-18 TPS for SGLang+EAGLE3 with the cuda-graph workaround.

The artifacts under [`models/qwen3.6-27b/sglang/`](../../models/qwen3.6-27b/sglang/) stay in the tree for archival reference; the README in that subtree has the full re-test triggers.

**Verdict:** for Qwen3-Next on consumer Ampere, use vLLM MTP (`vllm/dual/autoround-int4/turbo.yml`) or llama.cpp MTP (`llamacpp/mtp.yml`). SGLang may still be worth revisiting for OTHER model families where its RadixAttention or structured-output features are the headline — that's a per-model decision.

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
| `single/autoround-int4/eagle3-experimental.yml` | 1× 3090 | ⚠️ Blocked on SGLang OffloaderV1 tied-weights bug. Kept as reference for re-test when SGLang's offloader handles tied DeltaNet params. |
| `dual/autoround-int4/eagle3-experimental.yml` | 2× 3090 (TP=2) | ✅ Boots, serves coherent output. TPS/accept-rate pending bench. |

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
| SGLang adds asymmetric K/V or sub-FP8 INT KV path | Closes the KV-density gap vs vLLM TurboQuant. Would unlock context lengths comparable to our `dual/autoround-int4/turbo.yml` (262K). |
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
