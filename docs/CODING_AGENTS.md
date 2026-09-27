# Coding agents on club-3090 (omp, Claude Code)

Point a coding agent at the **LiteLLM gateway** (`:4000`), not at a model's own
port. The gateway's route set is re-rendered from whatever is actually serving on
every `switch.sh` launch and teardown (`scripts/lib/litellm-sync.sh`), so the agent
always sees the live model — no config edit when you switch slugs.

```
omp / claude ──► LiteLLM :4000 ──► whichever slug switch.sh booted (:8113, :8142, …)
```

## What the gateway sets

The gateway serves `services/litellm/config.runtime.yaml` (gitignored), which
`scripts/lib/litellm-sync.sh` renders from the endpoints that answer `/v1/models`
— on every `switch.sh` launch and teardown, restarting the gateway only when the
route set changed. The tracked `config.yaml` is the registry catalog; the gateway
does not serve it. Each live route gets:

| Field | Value | Why |
|---|---|---|
| `model` | `openai/<served id>` | Forwards the messages untouched, including a past turn's `reasoning_content`, which `hosted_vllm/` drops. |
| `allowed_openai_params` | `[reasoning_effort]` | Without it the `openai` provider answers a top-level `reasoning_effort` with HTTP 400 before the request reaches the engine. |
| `model_info.max_input_tokens` | the booted server's context: vLLM/SGLang `max_model_len`, llama.cpp `n_ctx`; else the registry's configured context | Clients size compaction against the real window of *this* boot. |
| `model_info.max_output_tokens` | 32,768, or half the context if that is smaller | The reply cap (thinking + answer); half the window at most, so a long prompt still fits. |
| `model_info.supports_function_calling` | `true` | |
| `model_info.supports_reasoning` | set only where the slug declares a thinking sampler profile | Left out when unknown — a wrong `false` would make a client turn thinking off on a thinking model. |
| `model_info.supports_vision` | from the registry | |

Gateway-wide (`litellm_settings`): `request_timeout: 1800` — the client owns the
real timeout, and a long cold prefill plus a long reply outlasts short defaults —
and `num_retries: 0`, because a retry repeats the whole prefill. Request logging
is off by default (see *Troubleshooting*). Routes of your own — a cloud endpoint,
a private service — go in the gitignored `services/litellm/config.local.yaml`
(keys in `services/litellm/local.env`; see the `.example` files, #1446): the sync
serves them after the generated ones, and they never enter the tracked catalog.

What the gateway serves for one live slug — `config.runtime.yaml`, rendered by the
sync; you never write it:

```yaml
model_list:
  - model_name: qwen3.8-27b
    litellm_params:
      model: openai/qwen3.8-27b
      api_base: http://host.docker.internal:8142/v1
      api_key: EMPTY
      allowed_openai_params: [reasoning_effort]
    model_info:
      mode: chat
      supports_function_calling: true
      max_input_tokens: 262144
      max_output_tokens: 32768
litellm_settings:
  request_timeout: 1800
  num_retries: 0
```

The one gateway file you do write, and only for routes of your own —
`services/litellm/config.local.yaml` (apply with `bash scripts/lib/litellm-sync.sh`):

```yaml
model_list:
  - model_name: my-cloud-model
    litellm_params:
      model: openai/<provider-model-id>
      api_base: https://<your-endpoint>/v1
      api_key: os.environ/MY_CLOUD_API_KEY   # MY_CLOUD_API_KEY=… in services/litellm/local.env
```

## omp (oh-my-pi) — setup

```bash
bash scripts/omp-setup.sh                 # adds a `club` provider to ~/.omp/agent/models.yml
bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   # experimental slug: --force
omp models club                           # the live model, with its real context window
cp services/omp/extensions/tps-meter.ts ~/.omp/agent/extensions/   # optional: speed + cache share in the statusline
```

The last line installs the *Statusline meter (omp and pi)*.

Then put the settings below in your `~/.omp/agent/config.yml` — `omp-setup.sh`
never touches that file.

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
`discovery: litellm`, which reads each route's `model_info` from the gateway's
`/model_group/info` (see *What the gateway sets*), so omp sizes compaction and
replies against the real window of the serving slug. For the Qwen3.8 ids the
provider also pins `maxTokens: 32768`, which omp applies over the gateway's value:
when a route carries no `model_info`, omp otherwise falls back to its own catalog's
65,536 — the whole window of the 65K single-card slug. 32,768 is at most half the
window on every Qwen3.8 slug.

What it writes into `~/.omp/agent/models.yml` (`bash scripts/omp-setup.sh --print`
shows it without writing):

```yaml
providers:
  # >>> club-3090 local models (generated by club-3090 scripts/omp-setup.sh; re-run to refresh) >>>
  club:
    baseUrl: http://127.0.0.1:4000/v1        # the gateway, never an engine port
    apiKey: sk-litellm-master-key            # the gateway's LITELLM_MASTER_KEY
    api: openai-completions
    discovery:
      type: litellm                          # live models + model_info from /model_group/info
    compat:
      supportsDeveloperRole: false
      thinkingFormat: qwen-chat-template     # thinking on/off + effort in chat_template_kwargs
    modelOverrides:
      qwen3.8-27b:                           # same block for qwen3.8-27b-fp8,
        maxTokens: 32768                     # thinkingcap38-27b, thinkingcap38-27b-fp8
        reasoning: true
        thinking:
          mode: effort
          efforts: [low, medium, xhigh]
          defaultLevel: medium
        compat:
          supportsReasoningEffort: true
          qwenTemplateReasoningEffort: true
          extraBody:
            thinking_token_budget: 8192      # chat-completions wire only (see below)
  # <<< club-3090 local models <<<
```

The override keys are the ids the engines *serve* (what the gateway lists), not
the registry's model names; `test-omp-setup` checks them against the composes.

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

### Settings for `~/.omp/agent/config.yml`

The block from the tuning write-up (see *Background reading*), adapted to the
`club` provider:

```yaml
modelRoles:
  default: club/qwen3.8-27b:medium  # the main agent
  plan: club/qwen3.8-27b:xhigh      # plan mode, think hard once
  slow: club/qwen3.8-27b:xhigh      # reviewer
  task: club/qwen3.8-27b:low        # general subagents
  smol: club/qwen3.8-27b:low        # scouting, simple edits
  tiny: club/qwen3.8-27b:low        # titles, memory, background
  commit: club/qwen3.8-27b:low      # commit messages

defaultThinkingLevel: low  # requests no role covers

task:
  maxConcurrency: 2  # subagents at once — see "Which slug to serve"

tools:
  artifactSpillThreshold: 10  # KB, bigger results go to a file
  artifactHeadBytes: 10       # KB of the start kept inline
  artifactTailBytes: 10       # KB of the end kept inline

compaction:
  thresholdPercent: 80  # summarize only when nearly full, on any slug's window

provider:
  appendOnlyContext: "on"  # only add to the conversation

providers:
  streamFirstEventTimeoutSeconds: 900  # 15 min for first word
  streamIdleTimeoutSeconds: 900        # and for pauses in a reply

retry:
  fallbackChains:                            # only with OPENROUTER_API_KEY set
    club/*:                                  # when the local model is unreachable
      - openrouter/qwen/qwen3.8-27b:free     # the same model, hosted, free
      - openrouter/openrouter/free           # then any free model
```

⚠️ **Set `modelRoles` even if you change nothing else.** Without roles, omp picks a
model on its own from everything the gateway lists, and a route you can't use
can win — a contributor's empty `modelRoles` landed on a keyless cloud route and
got a 401. The ThinkingCap slugs serve `thinkingcap38-27b`, not `qwen3.8-27b`:
there, use `club/thinkingcap38-27b:<effort>` in the roles.

⚠️ **Start the slug before omp.** If the default role's model isn't being served
when omp starts, omp picks a model itself from *any* provider you hold a key for,
without asking — with only `OPENROUTER_API_KEY` set it started on
`openrouter/openai/gpt-5.5`, a paid model. Check the model omp shows before your
first prompt, or launch with `omp --model club/qwen3.8-27b`: that exits with an
error when the model isn't served ("Set an API key environment variable…" — it
means the model isn't up) and sends nothing.

Rather leave your `config.yml` alone? The same settings ship as an overlay you
load per run: `omp --config services/omp/omp-club.yml` (e.g. as an `omp-club`
alias).

What each setting is for:

| Setting | Why |
|---|---|
| explicit effort on every role (`default: …:medium`, `task`/`smol`/`tiny`/`commit`: `…:low`, `plan`/`slow`: `…:xhigh`) | the role, not whichever slug is serving, decides how long the model thinks (club composes default to `low`; the checkpoint's own default is **xhigh**). On the vLLM dual-fast slug, two hard prompts took **10,395 / 16,000 (capped)** completion tokens at xhigh vs **5,916 / 7,280** at low and **4,474 / 6,455** at medium. |
| `maxTokens: 32768` (the gateway's value, pinned by the provider) | the reply cap covers thinking *and* the answer — xhigh alone spent up to 16,000 tokens on one hard prompt, so a small cap cuts a file write off mid-file. |
| `provider.appendOnlyContext: on` | anything that rewrites the front of the prompt re-prefills the whole conversation. |
| `compaction.thresholdPercent: 80` | compaction swaps history for a summary and busts the cached prefix, so it should happen late. A percentage follows each slug's real window; the article's `thresholdTokens: 200000` would sit past the end of the 147K and 163K slugs' windows. |
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

### Cloud fallback to OpenRouter's free models — how it's set up

Optional, and only active if you set `OPENROUTER_API_KEY`. It lives entirely in
omp; the gateway and `models.yml` don't change. Three pieces:

1. **omp's built-in `openrouter` provider.** It needs nothing in `models.yml` —
   omp enables it when `OPENROUTER_API_KEY` is in its environment, and then lists
   OpenRouter's models (`omp models openrouter`), including the two used here.
2. **The chain**, in `config.yml` (it is part of the block above):

   ```yaml
   retry:
     fallbackChains:
       club/*:                                # any model of the club provider
         - openrouter/qwen/qwen3.8-27b:free   # the same model, hosted, free
         - openrouter/openrouter/free         # then OpenRouter's free-model router
   ```

   A bare entry inherits the failing turn's effort, and the first fallback is the
   same Qwen3.8-27B, so the effort ladder and tool-call format carry over.
   `openrouter/free` is OpenRouter's router over its free models (it picks one
   per request).
3. **omp's retry defaults** decide *when* (`retry.modelFallback: true`,
   `retry.maxRetries: 10`, `retry.baseDelayMs: 500`, `retry.fallbackRevertPolicy:
   cooldown-expiry` — none of these need setting).

What happens when a request to the club model fails, measured through the gateway
(SGLang dual-fast):

| What broke | The gateway answers | omp |
|---|---|---|
| The engine stopped or crashed; its route is still listed | 500 (`Cannot connect`) | retried for ~16 s, then sent the turn to OpenRouter — 20 s in all |
| The route is gone: a `switch.sh` in progress, or after `--down` | 400 (`Invalid model name`) | fell back at once, no retry |

The turn goes to `openrouter/qwen/qwen3.8-27b:free`; if that fails too — it was
rate-limited (429) in some runs here — to `openrouter/openrouter/free`. After the
cooldown omp goes back to the local model.

So **turns you send during a slug switch go to OpenRouter**: the old route is gone
before the new slug is up. Wait for `switch.sh` to finish, or set
`retry.modelFallback: false` if you'd rather such a turn failed.

Without `OPENROUTER_API_KEY` there is nothing to fall back to, and the turn fails
once omp's own retries run out. To see the fallback work: in a running omp session,
stop the slug (`bash scripts/switch.sh --down`) and send a message — the answer
comes from OpenRouter, and the session records the model change.

**Why in omp and not in the gateway:** a gateway fallback (LiteLLM's `fallbacks`)
would swap models silently for *every* client of the gateway — Claude Code, other
agents, quality runs pointed at it — and the tracked gateway catalog routes only
to local engines (#1446). In omp the fallback is per user, visible in the session, and
off unless you hold a key.

Before relying on it:

- **Your prompts leave the machine** on a fallback turn — code included — to
  whichever provider serves the free model, under its data terms.
- **The free tier is small:** 20 requests a minute and 50 a day (1,000 a day once
  you've bought $10 of credits). One small omp task took ~20 requests here.
- It covers an *unreachable* model, not a *stuck* one: omp only falls back on
  failed requests, never because a task is hard.
- It covers turns in a running session, not startup. If no club model is up when
  omp starts, the chain never runs and omp picks a model itself — see *Start the
  slug before omp* above.
- Opt out with `retry.modelFallback: false`, or leave the `retry` block out.

## Statusline meter (omp and pi)

An extension that shows the model's decode speed and how much of each prompt the
engine served from its prefix cache, after every reply, for the request and for
the session:

```
⚡ 66.3 tok/s · ttft 0.3s · out 131 · think 29 · cache 98% of 7.8K · Σ 66.1 tok/s · Σ cache 95% (n=12)
```

```bash
cp services/omp/extensions/tps-meter.ts ~/.omp/agent/extensions/   # omp
cp services/pi/extensions/tps-meter.ts  ~/.pi/agent/extensions/    # pi
```

The two files are the same program (only the package their type import names
differs; `test-agent-statusline-meter` keeps them in step). It loads with the next
session and needs nothing else — it reads the usage each reply already carries.

| Field | Meaning |
|---|---|
| `⚡ 66.3 tok/s` | decode speed of this reply: output tokens over the time from the first generated token to the end, so prefill is left out |
| `ttft 0.3s` | request start to first generated token (thinking or text) — prefill plus queueing |
| `out 131` · `think 29` | output tokens, and the reasoning tokens the provider reported (left out when it reports none) |
| `cache 98% of 7.8K` | share of this request's 7.8K-token prompt taken from the prefix cache: `cacheRead / (input + cacheRead + cacheWrite)` (both agents count only the *uncached* part as `input`) |
| `Σ … tok/s` · `Σ cache 95%` · `(n=12)` | the same over the session on this model; it resets when you switch models. `Σ cache` is the number to watch — 90 %+ past the first turns; a drop means something rewrote the front of the prompt (see *Prefix caching — what breaks it*) |

Cache figures appear only once the backend has reported a cache hit for the
model — a backend that reports no cached tokens would otherwise read as a false
0 %. While a reply streams, the status shows elapsed time and phase instead.
`scripts/cache-share.sh` reads the same share from the engine's own counters
(*Troubleshooting a session*).

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
- **Tool-schema key order no longer matters on the Qwen3.8 slugs.** The template
  renders tool definitions *first*, so before the fix the same 29.6K-token prompt
  with one tool's JSON keys reordered got **0** cached tokens (a full 18 s
  re-prefill) on vLLM — the way an MCP server that rebuilds schemas from a map
  busts every turn. The vendored template now renders each schema with sorted
  keys (SGLang already normalised them). Other models' templates don't: there,
  keep tool schemas serialised the same way every turn (omp's built-in tools are).
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

Claude Code talks to the same gateway through LiteLLM's Anthropic-compatible
`/v1/messages` endpoint. In `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_API_KEY": "sk-litellm-master-key",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"
  },
  "model": "qwen3.8-27b"
}
```

- `ANTHROPIC_BASE_URL` is the gateway itself (no `/v1`), never an engine port; the
  key is the gateway compose's `LITELLM_MASTER_KEY`.
- With discovery on, Claude Code lists every route the gateway serves. `model` must
  be a served id: `qwen3.8-27b` on every Qwen3.8 slug, `thinkingcap38-27b` on the
  ThinkingCap ones.

Checked through the gateway on the reference rig (`sgl/qwen38-27b-dual-fast`): a
`/v1/messages` request with tools came back as a `tool_use` block, and a
`thinking.budget_tokens` request reached the model as a different reasoning effort
(283 prompt tokens vs 309 at the server's default). The model's reasoning is
**not** returned as thinking blocks on this path — only the answer and tool calls.
A contributor runs Claude Code this way day to day (#1419).

Two differences from omp: there is no counterpart to omp's key-gated cloud
fallback — if the local model can't be reached, the request fails — and a slug
switch restarts the gateway (see *Known limits*), so switch between turns.

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

**The same numbers in your agent's statusline:** see *Statusline meter (omp and pi)*.

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
- **After updating the repo, the gateway keeps its old routes** until the next
  `switch.sh` re-renders them (or run `bash scripts/lib/litellm-sync.sh`). Routes
  rendered before #1438 carry no `model_info`, and omp then assumes its catalog's
  context window (262K) — past the end of the smaller slugs' windows.
- **An engine that stops without `switch.sh`** (a crash, a plain `docker stop`)
  stays advertised until the next sync; requests fail with a clean HTTP error.
  Re-sync with `bash scripts/lib/litellm-sync.sh`.
- **`qwen3_coder` tool parser** drops everything after a literal `<tool_call>` in
  a reply's prose ([#1191](https://github.com/noonghunna/club-3090/issues/1191)) —
  rare in practice, but agents that *talk about* tool calling can hit it.


## Background reading

doug.sh's write-ups of running a local Qwen3.8-27B coding agent on two RTX 3090s
are where most of this setup comes from:

- [Tuning a local coding agent (oh-my-pi)](https://doug.sh/posts/tuning-a-local-coding-agent-oh-my-pi/)
  — the omp settings the overlay adopts: effort per role, the 32K reply cap,
  artifact spill, append-only context, late compaction, subagent concurrency,
  stream timeouts.
- [oh-my-pi custom models](https://doug.sh/posts/oh-my-pi-custom-models/) —
  `models.yml` pitfalls: a provider name omp already uses, `qwenTemplateReasoningEffort`,
  fields silently inherited from omp's catalog.
- [vLLM KV cache for agents](https://doug.sh/posts/vllm-kv-cache-agents/) —
  prefix-cache forensics: tool-schema key order, live subagent status in the
  system prompt, compaction, and measuring the cached share rather than hit counts.

Where this setup differs, and why:

| Topic | Articles | Here |
|---|---|---|
| Thinking budget | `thinking_token_budget` via the provider's `extraBody` | Doesn't reach the engine through the gateway: omp talks the Responses API to `openai/` routes, and vLLM accepts the budget only on chat completions. Effort per role is the lever. |
| Compaction | `thresholdTokens: 200000` (on a 262K window) | `thresholdPercent: 80` — follows each slug's window, which runs from 65K to 262K here. |
| Fallback | between the author's two local machines | `club/*` → OpenRouter's free Qwen3.8-27B, then `openrouter/free` — only with `OPENROUTER_API_KEY` set. |
| Subagents | `task.maxConcurrency: 4` | 2 — vLLM dual-fast runs 8 sequences, SGLang dual-fast 2; see *Which slug to serve*. |
| Tool-schema key order | template fix `tojson(sort_keys=True)` | Shipped in the Qwen3.8 template (#1441). |
| Host-RAM KV tier on hybrid models | served ~1.5 % of what was asked | Revisits of evicted agent sessions took 7.7 s (SGLang) / 8.6 s (vLLM) vs ~42 s cold (#1419). |
