# ThinkingCap-Qwen3.8-27B — Changelog

Dated history for ThinkingCap-Qwen3.8-27B configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-09-24 — Onboard ThinkingCap-Qwen3.8-27B: 29 incubating replicas of the Qwen3.8 INT4 and FP8 slugs

bottlecapai released ThinkingCap-Qwen3.8-27B, a reasoning fine-tune of Qwen3.8-27B, on 2026-09-23. Its
`text_config` and chat template are identical to Qwen3.8-27B's, so the model serves from copies of the
existing Qwen3.8 composes.

Upstream publishes FP8, NVFP4 and GGUF, but no INT4 AutoRound, which the W4A8 tiers need. We built one
with the Tess "shot 2" recipe: W4A16 int4 g128 sym, `mtp.fc` in BF16, 200 iterations, 512 samples,
alg_ext on. Its packed layout matches Frozenlock's Qwen3.8 INT4 tensor for tensor. It is published,
gated, at `wasifb/ThinkingCap-Qwen3.8-27B-AutoRound-W4A16`. The FP8 tiers use bottlecap's own FP8,
which keeps the MTP head in BF16 (+0.37 GB over the official Qwen FP8).

Each of the 29 slugs (`thinkingcap38-27b-<topology>-<tier>`) is a copy of its `qwen38-27b-…` sibling.
Only the weights path, served name (`thinkingcap38-27b`), service and container names, and port differ.
The patch mounts point at the Qwen3.8 tree. Every slug starts 🐣 incubating, with no gateway route.

Booted on 2× 3090 at 230 W with vLLM v0.30.0, one fresh boot each:
- `vllm/thinkingcap38-27b-dual-fast`: 74.0 / 101.0 tok/s against Frozenlock's 75.5 / 106.5. MTP acceptance held at
  2.7–4.0 through a 20K-token forced generation. Through `switch.sh`, verify-full scored 9/10. The one failure is
  the 2+2 reasoning-length heuristic: ThinkingCap reasons in 41 characters, under the check's 50-character floor.
- `vllm/thinkingcap38-27b-dual-superfast`: verify-full passed, including through `switch.sh`; 80.0 / 157.6 tok/s against 84.0 / 154.0
  for the Frozenlock sibling in the same session. The base-trained DFlash2 drafter converts on the fine-tune.
- `vllm/thinkingcap38-27b-dual-ultrafast`, booted through `switch.sh` on the registered slug: verify-full
  passed; 108.3 / 198.9 tok/s against Frozenlock's 108.8 / 191.5 and 111.1 / 196.6; the FA2 fp8-KV plugin verified.

The other 26 slugs, including all SGLang and multi-card ones, have not been booted. The 8-pack quality
run is pending.

**License:** PolyForm Small Business 1.0.0 plus BottleCap's personal-use grant. This is not Apache-2.0,
unlike ThinkingCap on Qwen3.6. Commercial use by an organization that is not a small business needs a
license from BottleCap.
