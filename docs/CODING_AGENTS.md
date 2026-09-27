# Coding agents on club-3090 (omp, Claude Code)

Point a coding agent at the **LiteLLM gateway** (`:4000`), not at a model's own
port. The gateway's route set is re-rendered from whatever is actually serving on
every `switch.sh` launch and teardown (`scripts/lib/litellm-sync.sh`), so the agent
always sees the live model — no config edit when you switch slugs.

```
omp / claude ──► LiteLLM :4000 ──► whichever slug switch.sh booted (:8113, :8142, …)
```

## omp (oh-my-pi) — setup

```bash
bash scripts/omp-setup.sh                 # adds a `club` provider to ~/.omp/agent/models.yml
bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   # experimental slug: --force
omp models club                           # the live model, with its real context window
omp --config services/omp/omp-club.yml    # run with the local-GPU settings overlay
```

Always name models with the `club/` prefix. omp matches bare names fuzzily across
every provider it knows: `--model qwen3.8-27b` resolved to a *cloud* provider's
`qwen/qwen3.8-27b` in testing and sent the prompt there.

Scripted runs (`-p`) take the effort as a flag, not as a `:level` suffix on
`--model` (which `-p` reports as "model not found"), and need stdin closed —
omp waits for piped input otherwise:

```bash
omp -p --no-session --config services/omp/omp-club.yml \
    --model club/qwen3.8-27b --thinking low "…" </dev/null
```

`omp-setup.sh` writes one marked block (re-run it to refresh; other providers are
never touched, and the old file is backed up). The provider uses omp's
`discovery: litellm`, which reads the gateway's `/model_group/info`: each route
carries `model_info` rendered from the serving engine — `max_input_tokens` is the
`max_model_len` of *this* boot, `max_output_tokens` is 32768 — so omp budgets
compaction and replies against the real window instead of a generic default.

**Wire format.** omp's LiteLLM discovery talks the **OpenAI Responses API**
(`/v1/responses`) to any route whose LiteLLM provider is `openai` — which every
club route is, so the gateway forwards past reasoning untouched (see *Prefix
caching*). vLLM, SGLang and llama.cpp all serve `/v1/responses` natively; on
vLLM and SGLang it carried effort, tool calls and past reasoning correctly, and
on SGLang it reused the cached prefix exactly as chat completions does (vLLM's
Responses path wasn't cache-measured). Two things only exist
on the chat-completions wire and so don't reach the engine through the gateway:

- **`off`** — omp sends no reasoning setting, so the slug's server default applies
  (thinking on, `low`). Use `low` rather than `off` for the cheapest turns.
- **The thinking budget** (`thinking_token_budget: 8192` in the provider) — it
  only reaches a vLLM engine on the chat-completions wire (e.g. an omp provider
  of your own pointed straight at a vLLM port). The effort level is the lever
  over the gateway.

(On the chat-completions wire the provider sends thinking and effort inside
`chat_template_kwargs` — `thinkingFormat: qwen-chat-template` — which all three
engines read; omp's plain `qwen` format would send a top-level `enable_thinking`
that vLLM and SGLang ignore.)

The Qwen3.8 override also says `reasoning: true`: the gateway only marks a route
as reasoning-capable when its slug declares a thinking sampler profile, and omp
sends no effort at all to a model it thinks can't reason.

The overlay (`services/omp/omp-club.yml`) is loaded per run, so your own
`config.yml` stays as it is. What each setting is for:

| Setting | Why |
|---|---|
| explicit effort on every role (`default: …:medium`, `task`/`smol`: `…:low`, `plan`/`slow`: `…:xhigh`) | the role, not whichever slug is serving, decides how long the model thinks (club composes default to `low`; the checkpoint's own default is **xhigh**). On the vLLM dual-fast slug, two hard prompts took **10,395 / 16,000 (capped)** completion tokens at xhigh vs **5,916 / 7,280** at low and **4,474 / 6,455** at medium. |
| `maxTokens: 32768` (from the gateway) | the reply cap covers thinking *and* the answer — xhigh alone spent up to 16,000 tokens on one hard prompt, so a small cap cuts a file write off mid-file. |
| `provider.appendOnlyContext: on` | anything that rewrites the front of the prompt re-prefills the whole conversation. |
| `compaction.thresholdPercent: 80` | compaction swaps history for a summary and busts the cached prefix; a percentage tracks each model's real window. |
| `tools.artifactSpillThreshold: 10` (KB) | inlined 40K-character tool results are prefill on every later turn. |
| `task.maxConcurrency: 2` | one GPU pair has one prefill budget; parallel subagents split it. See the per-slug table below. |
| stream timeouts `900` s | a cold 200K-token prefill is ~2.5 min on dual-fast, before thinking. |

Effort on SGLang slugs: SGLang ≤ 0.5.20 let the server's default effort override
the request's ([sglang#38104](https://github.com/sgl-project/sglang/issues/38104));
the SGLang Qwen3.8 composes work around it from
[#1439](https://github.com/noonghunna/club-3090/pull/1439) on. Before
that, every role ran at `low` on SGLang whatever it asked for.

About the budget where it does apply (vLLM, chat completions): it caps the
*thinking* field, not the reply — with a 512 budget the model sometimes kept
reasoning in the visible answer until the reply cap. Keep it generous; SGLang
0.5.20 has no such field at all (sglang#36750 would add one).

## Which slug to serve for agent work

| Slug | Concurrent sequences | KV pool | Agent notes |
|---|--:|--:|---|
| **`vllm/qwen38-27b-dual-fast`** ⭐ | 8 | ~590K tokens | Recommended (🧪 experimental — launch with `--force`). Room for a long main session plus subagents; MTP. |
| `vllm/qwen38-27b-dual-max` | 2 | ~271K | Pool holds about one full-length session — add `KV_OFFLOAD_GB=64` so evicted sessions come back from host RAM. |
| `sgl/qwen38-27b-dual-fast` | 2 | ~548K | Fine for one agent + 1 subagent. Honours the requested effort from #1439. |
| `sgl/qwen38-27b-dual-max` | 1 | ~183K (160K ctx) | Set `task.maxConcurrency: 1`. |
| DFlash2 tiers (`superfast`, `ultrafast`, `supermax`, `ultramax`) | 1 | — | Set `task.maxConcurrency: 1`; KV offload is write-only on DFlash. |

Requests beyond a slug's sequence count queue at the server — they don't fail, but
a subagent then waits for the whole main turn.

**Host-RAM KV tier.** Booting with `KV_OFFLOAD_GB=64` lets a conversation that was
pushed off the GPU come back from RAM instead of re-prefilling. Measured with three
interleaved 58K-token agent sessions pushed off the GPU: revisits took **7.7 s
(SGLang) / 8.6 s (vLLM)** against **~42 s** cold (#1419).

## Prefix caching — what breaks it

The engine reuses a cached prefix only if the new prompt matches it token for
token from the start. Verified on this stack:

- **The gateway is transparent**: a prompt sent through LiteLLM and then repeated
  directly to the engine hit the cache in full (29,616 of 29,637 tokens).
- **Tool-schema key order matters on vLLM.** The Qwen3.8 template renders tool
  definitions *first*; the same 29.6K-token prompt with one tool's JSON keys
  reordered got **0** cached tokens (a full 18 s re-prefill) on vLLM. SGLang
  normalises the schemas and still hit in full. omp's built-in tools are
  serialised the same way every turn; MCP servers that rebuild schemas from a
  map can reorder keys between turns.
- **A request must repeat the same `tools`.** Qwen's template puts the tool block
  at the top of the system turn, so a turn sent without the tools shares only a
  few tokens with the cached conversation (the cause of the false FAIL in #1435's
  first probe run).
- **Turn to turn it holds, on both wires.** With a 28K-token system prompt, the
  next agent turn reused 27,968 of 28,075 tokens over `/v1/responses` and 28,928
  of 28,973 over chat completions on SGLang (~0.5 s vs ~18 s cold).
- **Past reasoning is kept, and costs one prefill.** Qwen3.8's template re-renders
  the model's earlier reasoning, omp sends it back, and the gateway forwards it.
  vLLM does not reuse the tokens it *generated*, only earlier prompts, so each
  turn's reasoning is prefilled once on the next turn: after a 2,673-token xhigh
  turn, the next turn prefilled 2,727 tokens (3.2 s) with the reasoning kept vs
  836 (1.3 s) with it stripped. The previous turn's prompt was reused either way.

## Claude Code

A contributor runs Claude Code against the same gateway through its Anthropic-compatible `/v1/messages` endpoint (not re-verified on the reference rig):

```json
"env": {
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
  "ANTHROPIC_API_KEY": "sk-litellm-master-key",
  "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"
}
```

## Troubleshooting a session

**Is the prefix cache doing its job?** Read the engine's own counters — nothing
is sent to the model:

```bash
bash scripts/cache-share.sh              # every serving engine: prompt tokens taken from cache since it booted
bash scripts/cache-share.sh --watch 30   # then one line per 30 s while you work (Ctrl-C stops)
```

```
:8113  vllm  qwen3.8-27b
  since boot      144,055 prompt tok ·  49.8% from cache (GPU 49.8%) · 72,247 prefilled

every 10s — tokens since the previous line:
  12:19:52       18,693 prompt tok ·  40.0% from cache (GPU 40.0%) · 11,221 prefilled
  12:20:02       12,878 prompt tok ·  86.7% from cache (GPU 86.7%) · 1,710 prefilled
  12:20:12       44,729 prompt tok ·  62.6% from cache (GPU 62.6%) · 16,745 prefilled
  12:20:22       41,781 prompt tok ·  85.5% from cache (GPU 85.5%) · 6,053 prefilled
```

*(An omp task on `vllm/qwen38-27b-dual-fast` that fans out to three subagents.
"Since boot" includes every cold first turn since the engine started; the watch
lines are the session.)*

A long agent session should sit at 90 %+ once it is past its first turns, and a
sudden drop means something rewrote the *front* of the prompt — compaction, a
changed system prompt, reordered tool schemas (see *Prefix caching* above) — so
the engine re-read the conversation. **Subagents pull the aggregate down without
anything being wrong:** in that run the main session reused 91–98 % per request
from its third turn on, while subagents started cold — two launched at the same
instant could not share with each other, the third reused 32 % on vLLM where the
same task on SGLang reused 91 % (omp puts per-subagent text partway through each
subagent's system prompt), and omp's small title calls (~340 tokens) were never
reused. vLLM and SGLang report the share split into the GPU cache and the
host-RAM tier (`KV_OFFLOAD_GB`); llama.cpp has no cached-token counter.

**What did the gateway actually send?** Request logging on the LiteLLM gateway is
**off by default** — one access line per request, no content. To see each request
exactly as it was forwarded to the engine (URL, every parameter, the messages) and
the engine's raw reply:

```bash
bash scripts/litellm-log.sh on       # recreates the gateway (~10 s) with LITELLM_LOG=DEBUG
docker logs -f litellm               # read it
bash scripts/litellm-log.sh off      # back to the default
bash scripts/litellm-log.sh status
```

⚠️ With it on, full prompts and replies go into the container log (capped at
3 × 50 MB). It stays on across the route-sync restarts `switch.sh` does, and
`gpu-mode status` warns while it is; any fresh start of the gateway comes back off.

## Known limits

- **Route changes restart the gateway.** LiteLLM has no reload endpoint, so a
  `switch.sh` that changes the route set restarts the container — switch at a
  turn boundary.
- **An engine that stops without `switch.sh`** (a crash, a plain `docker stop`)
  stays advertised until the next sync; requests fail with a clean HTTP error.
  Re-sync with `bash scripts/lib/litellm-sync.sh`.
- **`qwen3_coder` tool parser** drops everything after a literal `<tool_call>` in
  a reply's prose ([#1191](https://github.com/noonghunna/club-3090/issues/1191)) —
  rare in practice, but agents that *talk about* tool calling can hit it.
