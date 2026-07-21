# ThinkingCap-Qwen3.6-27B — on 2× RTX 3090

**Run [LeaderboardModel1's ThinkingCap-Qwen3.6-27B](https://huggingface.co/LeaderboardModel1/ThinkingCap-Qwen3.6-27B-AutoRound-W4A16-Tuning) — a reasoning fine-tune of Qwen3.6-27B — served as W4A8 (int8 activations) with built-in MTP, full 262K context, on 2× RTX 3090s.**

> **Top-of-class 27B W4 quality.** 8-pack **113/150 think-off · 120/150 think-on** — beats Tess-4-27B W4A16 (108/115) and the Qwen fast tier (109) on *both* legs, strongest on reasoning (InstructFollow 15/15, BugFind 15/15 think-on).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven picks. TL;DR:

| Config | Max ctx | Decode TPS (narr/code) | Best for |
|--------|---------|------------------------|----------|
| `vllm/thinkingcap-dual-w4a8` (default) ⭐ | **262K** | 73.3 / 106.1 | Reasoning + agentic on 2× 24 GB — top 27B W4 quality at Tess-tied speed |

Run via:
```bash
bash scripts/launch.sh --variant vllm/thinkingcap-dual-w4a8
```

> ⚠️ **Caveats (⚠️ Production w/ caveats):**
> - **Thin VRAM margin at the 262K ceiling with MTP** (~69 MB free — MTP's draft KV eats ~1.2 GB of headroom). NIAH recall is clean to 240K, but for sustained agents at *max* context, size below 262K or lower `--gpu-memory-utilization`.
> - **Vision** tower is resident but **untested on this serve** (shipped text-only).
> - **Single-card** W4A8 tops out ~76K (the vision tower + weights leave too little KV headroom for 262K on one 24 GB card) — dual is the config for full context.

---

## Models

- **Target:** [`LeaderboardModel1/ThinkingCap-Qwen3.6-27B-AutoRound-W4A16-Tuning`](https://huggingface.co/LeaderboardModel1/ThinkingCap-Qwen3.6-27B-AutoRound-W4A16-Tuning) (~18.5 GB, auto-round INT4 g128 sym, vision + built-in MTP preserved)
- **Drafter:** built-in MTP head — no separate download (`mtp.fc`@BF16; n=5 knee)

## Key details

| Aspect | Notes |
|--------|-------|
| **Arch** | `qwen3_5` hybrid — 48 linear-attention + 16 full-attention layers (64 total); same family as Tess-4-27B (`qwen35-dense`) |
| **Quant** | AutoRound W4A16, served **W4A8** — int8 activations via `VLLM_MARLIN_INPUT_DTYPE=int8` + the shared `qwen-w4a8-int8-act` patches (AutoRound's ~50% negative scales are folded at boot; without them the toggle silently no-ops then corrupts) |
| **KV** | `fp8_e4m3` |
| **Drafter** | built-in **MTP n=5** (accept 5.0 / 80% warm; code decode +64% vs no-spec; n=6/8 regress) |
| **Vision** | base is VL-capable; shipped **text-only** (untested on this serve) |
| **Tools** | ✅ `--tool-call-parser qwen3_coder` (native chat template — no override needed) |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- The W4A8 int8-activation path depends on two vLLM bugs we filed + vendored-fix: [vllm#48904](https://github.com/vllm-project/vllm/issues/48904) (INC route drops `input_dtype`) and [vllm#48905](https://github.com/vllm-project/vllm/issues/48905) (Marlin reads int16 group scales unsigned). See [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md); the patches drop when both merge + land in the pinned release.
