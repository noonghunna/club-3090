# club-3090

**Recipes for serving LLMs locally on RTX 3090s** (and other NVIDIA cards): working compose configs,
patches and measured numbers, across vLLM, SGLang, llama.cpp, ik_llama and exllamav3. Pick a model
for your card count, launch it with one command, and get an OpenAI-compatible API.

> 📣 **The newest slugs, numbers and caveats are posted in [Announcements](https://github.com/noonghunna/club-3090/discussions/categories/announcements) first.** The repo docs catch up afterwards, so when a page here and an announcement disagree, the announcement is newer. The card pages below link each slug to its announcement thread.

## Pick your path

| You have / want | Start here |
|---|---|
| **1× RTX 3090** | [`docs/SINGLE_CARD.md`](docs/SINGLE_CARD.md): every single-card slug, what to pick, what to watch for |
| **2× RTX 3090** | [`docs/DUAL_CARD.md`](docs/DUAL_CARD.md): every dual-card slug, what to pick, what to watch for |
| **3 or more GPUs** | [`docs/MULTI_CARD.md`](docs/MULTI_CARD.md): the 4- and 8-card slugs, several copies vs one split |
| **New here** | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md): clone to first reply in about five minutes |
| **New to local AI** | [`docs/LOCAL_AI_PRIMER.md`](docs/LOCAL_AI_PRIMER.md): hardware, engines, model sizes and quants in plain English |
| **Your own model** | [`docs/BRING_YOUR_OWN.md`](docs/BRING_YOUR_OWN.md) (serve and validate it) · [`docs/PULL.md`](docs/PULL.md) (check any Hugging Face repo against your VRAM first) |
| **To measure a slug on your rig** | [`docs/RUN_EVALS.md`](docs/RUN_EVALS.md): quality and health checks, and how to post the results |
| **Something's broken** | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) and the [`FAQ`](docs/FAQ.md) |
| **Self-host vs cloud APIs** | [`docs/COMPARISONS.md`](docs/COMPARISONS.md) |

Every measured result lives in [`BENCHMARKS.md`](BENCHMARKS.md). Terms like TPS, KV cache and MTP are
explained in the [`GLOSSARY`](docs/GLOSSARY.md); engines in
[`INFERENCE_ENGINES.md`](docs/INFERENCE_ENGINES.md); quant names in
[`QUANTIZATION.md`](docs/QUANTIZATION.md). The full docs index is [`docs/README.md`](docs/README.md).

## Quick start

Needs Linux (on Windows, set up [WSL2](docs/WSL_SETUP.md) first), Docker with the NVIDIA Container
Toolkit, and room on disk for the weights.

```bash
git clone https://github.com/noonghunna/club-3090.git && cd club-3090

bash scripts/setup.sh           # pick a model; downloads and SHA-verifies the weights
bash scripts/launch.sh          # pick a config for your GPUs; boots it and runs a quick check

# the launcher prints a test request for what you booted; for the Qwen3.6-27B dual default:
curl -sf http://localhost:8010/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}'

bash scripts/switch.sh --list   # every config this machine can run (--all adds retired ones)
bash scripts/switch.sh <slug>   # switch to another one (🧪 slugs need --force)
bash scripts/update.sh          # later: pull the latest stack and re-run setup
bash scripts/settings.sh show   # your saved settings (model dir, default pins, …) and where each comes from
```

Settings you save (the weights dir, your default config) live in `~/.config/club-3090/`, shared by every
checkout: [where settings live](docs/FAQ.md#where-are-my-settings-saved-and-how-do-i-change-one).

Prefer a screen to the CLI? **`c3`** is a terminal cockpit for the same flow: browse the catalog,
serve with one key, watch GPUs and containers, run health checks. Install it with
`uv pip install -e tools/serve-cockpit` and run `c3`; details in
[`tools/serve-cockpit/`](tools/serve-cockpit/).

## Other hardware

The configs are hardware-class aware, and contributors have run them on 4090s, 5090s, A-series
cards and more: [Can I use a 4090?](docs/FAQ.md#can-i-use-a-4090-instead-of-a-3090) ·
[Can I use a 5090?](docs/FAQ.md#can-i-use-a-5090) · [`HARDWARE.md`](docs/HARDWARE.md) (card
classes, power limits, NVLink, PCIe).

## More than chat

**[Club 3090 AI Studio](docs/ai-studio/README.md)** adds open-weight image, video and audio
generation alongside a chat model, driven from Open WebUI. The image bundle is one command:
`bash scripts/setup-image-studio.sh`.

## Where things live

```
models/<model>/<engine>/compose/<topology>/<quant>/<serving>.yml   one compose per slug
models/<model>/<engine>/patches/                                     vendored fixes for that engine
scripts/          setup, launch, switch, verify, bench, quality, report, update
docs/             the guides linked above
tools/            the c3 cockpit, kv-calc, chart and docs generators
BENCHMARKS.md     every measured row · CHANGELOG.md  what changed, dated
```

Engines and hardware are documented once, under `docs/`; everything specific to a model (quants,
quirks, recipes) sits under `models/<name>/`.

## Community

- 💬 **[Discord](https://discord.gg/gzdfjhj5yN)** — casual chat, hardware questions, share what you're running. Use for synchronous Q&A.
- 📋 **[GitHub Discussions](https://github.com/noonghunna/club-3090/discussions)** — async, searchable. Best for cross-rig benchmark drops, "should I tune X" type threads, and anything you want others to find via search.
- 🐛 **[GitHub Issues](https://github.com/noonghunna/club-3090/issues)** — bug reports, regression repros, concrete asks. Triage ladder in [FAQ](docs/FAQ.md#troubleshooting-ladder--boot-the-simplest-stack-first) before filing.

## Community projects

Projects in the club-3090 ecosystem maintained outside this repo:

- **[VykosX/club-3090-server](https://github.com/VykosX/club-3090-server)** — single-file installer adding a server-management layer on top of club-3090: browser admin panel on `:8008/admin`, OpenAI-compatible reverse proxy on `:8009` with multi-backend routing, GPU-aware multi-instance orchestration, fan/power controls, audit logs, and per-user API auth/quota. Headless Arch + Debian/Ubuntu friendly. Started 2026-05-05, AGPL-3.0; see [discussion #108](https://github.com/noonghunna/club-3090/discussions/108) for the announcement and current WIP status. **Not yet officially adopted** — listed here as a community pointer until it converges on a stable surface area.

If you've built something that integrates with club-3090 and you'd like a pointer added here, open a discussion.

---

## Credits

The stack stands on a lot of shoulders:

- **Qwen team** ([@Alibaba_Qwen](https://huggingface.co/Qwen)) — for the base models and the MTP head architecture
- **[Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)** — for the AutoRound INT4 quant with preserved BF16 `mtp.fc` (the model this whole stack runs on)
- **[Sandermage](https://github.com/Sandermage)** — root-caused vllm#40880 (MTP × TurboQuant cudagraph capture) and shipped the fix that made spec-decode work on consumer Ampere. The patch tree it shipped in is no longer used here, but the diagnosis stands and is upstream at [vllm#40914](https://github.com/vllm-project/vllm/pull/40914).
- **[vibhavagarwal5](https://github.com/vllm-project/vllm/pull/38479)** — TurboQuant landing PR + tracking issue #40069
- **[vLLM project](https://github.com/vllm-project/vllm)** — the engine + active maintenance
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — the alternative engine path
- **[Luce z-lab](https://github.com/luce-spec)** — DFlash N=5 draft model for Qwen3.6-27B
- **Intel AutoRound** — quantization framework
- **[@paulp83](https://github.com/paulp83)** — the Blackwell reference rig. First full gate on native FP4 (Qwen3.6-27B NVFP4 TP=2 @262K: 168.0 / 215.7 decode, 91% NIAH at 240K, soak PASS), first NVFP4 validation on the 35B-A3B MoE, first Blackwell DiffusionGemma and Nemotron-75B runs, and the only **mixed-architecture** numbers anyone has contributed (5090 + 3090 Ti in one box) — a rig class we could not otherwise characterise.
- **[@henrykrinkle01](https://github.com/henrykrinkle01)** — raised the bar on how we *report* quality, not just what we measure. The [Results Card](docs/RESULTS_CARD.md) v2 quality table — per-pack dispersion and p50/p95 latency — is his format, adopted wholesale from [#770](https://github.com/noonghunna/club-3090/issues/770). Also the first cross-rig 8-pack on Qwen3.6-27B and still the only one running **both reasoning modes on one rig, one day, with repeats**, the finding that cli-40 variance is mode-dependent, and the custom-all-reduce-ON measurement on a working-P2P rig.
- **All cross-rig contributors** — [@ampersandru](https://github.com/ampersandru), [@walmis](https://github.com/walmis), [@3dluvr](https://github.com/3dluvr), and the Reddit / X local-LLM community for benchmark data and bug reports.

---

## License

Apache 2.0. Do what you want with it. If you get better numbers on your rig — open an issue. If you add a new model with working configs — open a PR.
