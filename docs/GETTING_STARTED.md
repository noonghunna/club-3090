# Getting started — zero to `curl` in 5 minutes

The fastest path from `git clone` to serving your first response. No decisions, no menus — just commands.

> New to local AI and the terms below feel like jargon? Read [LOCAL_AI_PRIMER.md](LOCAL_AI_PRIMER.md) first — how hardware, engines, model sizes, and quants fit together in plain English.

```bash
# 1. Clone
git clone https://github.com/noonghunna/club-3090.git
cd club-3090

# 2. Download the model (Qwen3.6-27B, ~18 GB)
bash scripts/setup.sh qwen3.6-27b

# 3. Boot the default config for this model on your hardware
#    (auto-picks: single-card → ik-llama/iq4ks-mtp; dual → vllm/dual)
bash scripts/launch.sh --variant qwen3.6-27b/default

# 4. Test it
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}'
```

If you see `Paris` in the response, you're up and running.

> After the boot finishes, `launch.sh` asks **"Make `<slug>` your default for `qwen3.6-27b`? [y/N]"**. Say `y` and a bare `bash scripts/launch.sh` next time goes straight to that exact config — no flags, one keypress to launch. (Change or clear it anytime: `bash scripts/switch.sh --set-default <slug>` / `--clear-default qwen3.6-27b`.)

---

## Next steps

| You want | Go here |
|----------|---------|
| **Pick a config by workload** (long context, vision, dual-card, etc.) | [`docs/SINGLE_CARD.md`](SINGLE_CARD.md) or [`docs/DUAL_CARD.md`](DUAL_CARD.md) |
| **Understand the jargon** (TPS, KV, MTP, TP) | [`docs/GLOSSARY.md`](GLOSSARY.md) |
| **Client code snippets** (Python, curl, IDE setup) | [`docs/EXAMPLES.md`](EXAMPLES.md) |
| **Run the canonical benchmark** | `bash scripts/bench.sh` |
| **Update to the latest** | `bash scripts/update.sh` |
| **Hardware questions** (power caps, NVLink, 4090/5090) | [`docs/HARDWARE.md`](HARDWARE.md) |
| **File an issue or share bench numbers** | `bash scripts/report.sh --full > my-rig.md` and open an issue |
