# ThinkingCap-Qwen3.8-27B — Changelog

Dated history for ThinkingCap-Qwen3.8-27B configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-09-27 — SGLang slugs: `--sleep-on-idle` (idle CPU ~2 cores → under half a core)

Every SGLang ThinkingCap-Qwen3.8 compose now passes `--sleep-on-idle`. Without it each scheduler
rank busy-polls its event loop while nothing is being served: measured on
`sgl/qwen38-27b-dual-fast` (TP=2, v0.5.20), **192 % of a CPU core at idle** —
about one core per GPU, all day — against 3 % for the vLLM dual-fast slug. With it,
rank 0 blocks in a zmq poll until a request arrives: **42 %** at idle. Serving is
unchanged within run noise: TTFT after a 4 s idle gap 84 vs 80 ms (median of 8),
decode 63.4 vs 66.3 tok/s. The flag exists upstream but defaults off. The residual
~20 % per rank is the scheduler's idle bookkeeping waking ~16×/s (rank 0) and the
other ranks following it through the gloo broadcast.

## 2026-09-27 — tool schemas render with sorted keys (prefix cache)

The vendored `qwen38-reasoning-effort-template` now renders each tool schema with
`tojson(sort_keys=True)`. Qwen3.8 puts the tool definitions first in the prompt, so
a client, gateway or MCP server that re-serialised a schema with its keys in another
order used to change the prompt at its very start: on vLLM the next turn then missed
the prefix cache for the whole conversation (0 of 29.6K tokens cached, an 18 s
re-prefill). Sorted, the same tools always render the same bytes. Only key order
inside each schema changes; prompts without tools are byte-identical. Detail: the
patch's [`PROVENANCE.md`](../qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/PROVENANCE.md).

## 2026-09-27 — SGLang: a request's reasoning_effort is honoured again

On every SGLang ThinkingCap-Qwen3.8 slug the client's `reasoning_effort` was ignored: the
server default (`low`) rendered instead, whether the effort came top-level or in
`chat_template_kwargs` — SGLang ≤ 0.5.20 lets a `--default-chat-template-kwargs`
`reasoning_effort` override the request's own
([sglang#38104](https://github.com/sgl-project/sglang/issues/38104)). Measured on
`sgl/qwen38-27b-dual-fast`: `low`, `medium`, `xhigh` and `high` all rendered the
same 41-token prompt.

The SGLang composes now mount the vendored `qwen38-reasoning-effort-template` over
the checkpoint's own and set their server default as `default_reasoning_effort`,
which the template reads only when a request names no effort. Requests that send
nothing still get `low` (or `REASONING_EFFORT`); requests that send an effort now get
it, and `high` maps to `xhigh` as on the vLLM slugs. vLLM slugs render exactly as
before. Detail: the patch's
[`PROVENANCE.md`](../qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/PROVENANCE.md).

## 2026-09-24 — Onboard ThinkingCap-Qwen3.8-27B: 29 experimental replicas of the Qwen3.8 INT4 and FP8 slugs

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
The patch mounts point at the Qwen3.8 tree. Every slug ships 🧪 experimental, like its Qwen3.8 sibling, with no gateway route.

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
