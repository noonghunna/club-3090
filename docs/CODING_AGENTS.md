# Coding agents on club-3090 (omp, pi, Hermes Agent, dsh, Claude Code)

Point a coding agent at the **LiteLLM gateway** (`:4000`), not at a model's own
port. The gateway's route set is re-rendered from whatever is actually serving on
every `switch.sh` launch and teardown (`scripts/lib/litellm-sync.sh`), so the agent
always sees the live model — no config edit when you switch slugs.

```
omp / pi / hermes / dsh / claude ──► LiteLLM :4000 ──► whichever slug switch.sh booted (:8113, :8142, …)
```

## Cautions — read before you start

Each of these can cost money, send your code off this machine, or leave the gateway
open. The detail is in the section named in the last column.

| What can happen | When | Avoid it |
|---|---|---|
| omp starts on a **paid cloud model** | no club model is being served when omp starts, and you hold a key for another provider — with only `OPENROUTER_API_KEY` set it picked `openrouter/openai/gpt-5.5` | Start the slug first, or launch `omp --model club/qwen3.8-27b`, which exits instead of choosing a model (*Start the slug before omp*, under *Settings for `~/.omp/agent/config.yml`*) |
| omp picks a route **you can't use** | `modelRoles` left empty | Set `modelRoles` (*Settings for `~/.omp/agent/config.yml`*) |
| **Prompts and code leave the machine** | omp's OpenRouter fallback fires — only with `OPENROUTER_API_KEY` set | Leave the `retry` block out, or set `retry.modelFallback: false` (*Cloud fallback to OpenRouter's free models*) |
| Hermes **resumes on your default model**, which may be a cloud one | `hermes chat --resume <id>` without a provider | Repeat `--provider custom:club -m qwen3.8-27b` (*Hermes Agent — setup*) |
| Hermes tools **reach outside the machine**, or `computer_use` **drives your desktop** | Hermes's default toolsets (`web`, `browser`, `x_search`, `image_gen`, `tts`, `connections`, `computer_use`) | Name a local set with `-t` (*Hermes Agent — setup*) |
| dsh sends its first request to **DeepSeek's paid API** | a dsh profile runs before it names a default model, and `DEEPSEEK_API_KEY` is set in your shell: dsh's built-in default is DeepSeek's own cloud model, and an old `~/.dsh/settings.yaml` is only read in after that first run has started | Create the profile with `--dump-config` and set `agent-default-model` before its first run (*dsh — setup*) |
| **Full prompts and replies in the gateway's log** | request logging is on (`scripts/litellm-log.sh on`) | Turn it off when done; `gpu-mode status` warns while it's on (*Troubleshooting a session*) |
| **Anyone on your network can use the gateway**, including the cloud routes in your `config.local.yaml` | no key of its own is stored (`setup.sh` stores one only on a fresh install), so the gateway runs on the public default, the same on every install; or it runs on no key at all. It listens on every interface (`4000:4000`) | `bash scripts/gateway-key.sh rotate --apply`: a key of your own, kept in `~/.config/club-3090/secrets.env`. Never remove the key. `400 No connected db.` means the client's key is wrong: fix the key, not the gateway (*The gateway key*) |

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
| `use_chat_completions_api` | `true` on a route whose engine has no `/v1/responses` (tabbyAPI; the sync checks with an empty-body request); left out elsewhere | omp's Responses calls and Claude Code's translated `/v1/messages` reach the engine as chat completions instead of a 404 — which would also lock every client out of the model for 5 s. Reasoning crosses the bridge both ways — see *Wire format*. |
| `model_info.max_input_tokens` | the booted server's context: vLLM/SGLang `max_model_len`, llama.cpp `n_ctx`; else the registry's configured context | Clients size compaction against the real window of *this* boot. |
| `model_info.max_output_tokens` | 32,768, or half the context if that is smaller | The reply cap (thinking + answer); half the window at most, so a long prompt still fits. |
| `model_info.supports_function_calling` | `true` | |
| `model_info.supports_reasoning` | set only where the slug declares a thinking sampler profile | Left out when unknown — a wrong `false` would make a client turn thinking off on a thinking model. |
| `model_info.supports_vision` | from the registry | |
| `model_info.supported_endpoints` | `["/v1/chat/completions", "/v1/responses", "/v1/messages"]` on a vLLM or SGLang route whose own `/v1/messages` keeps a mid-conversation `system` message in place (the sync checks with `count_tokens`); left out elsewhere | LiteLLM then forwards Claude Code's Anthropic requests to the engine's own endpoint untranslated, so the model's reasoning reaches Claude Code — see *Claude Code*. |

Gateway-wide (`litellm_settings`): `request_timeout: 1800` — the client owns the
real timeout, and a long cold prefill plus a long reply outlasts short defaults —
and `num_retries: 0`, because a retry repeats the whole prefill. Request logging
is off by default (see *Troubleshooting*). Routes of your own — a cloud endpoint,
a private service — go in `~/.config/club-3090/litellm/config.local.yaml`, with
your other settings (see *Routes of your own* below): the sync serves them after
the generated ones, and they never enter the tracked catalog.

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
      supported_endpoints: ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
litellm_settings:
  request_timeout: 1800
  num_retries: 0
```

### Routes of your own

The one gateway file you do write, and only for routes of your own, is
`litellm/config.local.yaml` in your club-3090 settings directory
(`~/.config/club-3090/`; `bash scripts/settings.sh path` shows where yours is). Start
from `services/litellm/config.local.yaml.example`:

```yaml
model_list:
  - model_name: my-cloud-model
    litellm_params:
      model: openai/<provider-model-id>
      api_base: https://<your-endpoint>/v1
      api_key: os.environ/MY_CLOUD_API_KEY
```

Save the key with your other secrets, not in the file:
`bash scripts/settings.sh set MY_CLOUD_API_KEY=<key>` puts it in `secrets.env`
(mode 0600). The gateway gets only the keys its routes name
(`os.environ/<NAME>`), not the whole `secrets.env`, which also holds your HF token:
gpu-mode, c3 and `litellm-log.sh` hand them over in a temporary 0600 file when they
start the gateway.

- A new or changed **route**: `bash scripts/lib/litellm-sync.sh` (switch.sh runs it
  on every launch too).
- A new or changed **key**, or a new route whose key the gateway hasn't been
  given yet: `bash scripts/gpu-mode.sh gateway`. A restart is not enough, because a
  container keeps the environment it was created with. `litellm-sync` warns when a
  route's saved key is missing from the running gateway.

**Older installs** kept these in the checkout: `services/litellm/config.local.yaml`
and its keys in `services/litellm/local.env`. Both still work: the checkout's
routes file is read while there is none in the settings directory, and the gateway
still loads `local.env`, with a key saved in your settings winning over it.
`bash scripts/settings.sh migrate` (`--dry-run` first shows the plan) copies the
routes file across, and the keys your routes use into `secrets.env`. It never
changes or deletes the repo files, and says which ones you can delete.

## The gateway key

Every client sends the gateway's key, `LITELLM_MASTER_KEY`. The omp, pi and Hermes
setup scripts write it into the agent's config for you; Claude Code, or anything
else you point at the gateway by hand, needs it pasted in.

Without a key of its own, the gateway runs on **the public default** —
`sk-litellm-master-key`, the fallback in `services/litellm/docker-compose.yml`, the
same on every club-3090 install. The gateway listens on every interface, so anyone
on your network who knows club-3090 can use your GPUs and the cloud routes in your
`config.local.yaml`. `gpu-mode` warns each time it starts the gateway on that key.

```bash
bash scripts/gateway-key.sh status          # public default or your own? (never prints the key)
bash scripts/gateway-key.sh rotate --apply  # a new random key, then steps 1 and 2 below
bash scripts/gateway-key.sh rotate          # the same new key, but only prints the steps
```

**A fresh install gets its own key.** `setup.sh` runs `gateway-key.sh init`, which
stores a random key, but only when nothing shows that this machine has used the
gateway, since a client may already hold the current key. It keeps the current key,
and says why, if any of these holds:

- `LITELLM_MASTER_KEY` is set in the shell, `club3090.env`, `secrets.env` or the
  repo `.env`, even empty;
- `services/litellm/config.runtime.yaml` exists in the checkout (gpu-mode and
  `switch.sh` render it whenever they start or sync the gateway; `gpu-mode off`
  removes the container, not this file);
- Docker has a `litellm` or `open-webui` container, running or stopped, or an Open
  WebUI data volume;
- Docker is installed but can't be asked (not running, or your user can't reach it
  without sudo), so none of that can be ruled out. Docker not installed counts as
  no container;
- something answers at the gateway's address;
- an omp, pi or Hermes config holds the provider its setup script wrote.

On an existing install nothing changes until you rotate.

`rotate` and `init` write the key to `~/.config/club-3090/secrets.env` (mode 0600;
`CLUB3090_CONFIG_DIR` moves the directory) and never print it. When you need to
paste it somewhere, it is the `LITELLM_MASTER_KEY=` line of that file. After a
rotate nothing changes until the gateway is recreated, so clients keep working until
then. The steps (`rotate --apply` does 1 and 2, in that order):

1. **Recreate the gateway** on the new key: `gpu-mode gateway` (needs sudo). It
   re-renders the gateway's routes, recreates only the `litellm` container with your
   settings, waits for it to answer and prints `gateway-key.sh status`. No model is
   started or stopped; a stopped gateway is started. A `docker restart` is not
   enough, because a container keeps the key it was created with. Request logging
   (`scripts/litellm-log.sh`) comes back off, as after any start.
2. **Re-run the agent setups** that point at this rig: `omp-setup.sh`, `pi-setup.sh`
   and `hermes-setup.sh` read the key from `secrets.env` (a `LITELLM_MASTER_KEY` set
   in the shell wins, for a gateway on another box). pi and Hermes read the
   gateway's routes with it, so they run after step 1. A setup pointed at another
   box's gateway is left out. If step 1 fails, `--apply` stops there: the key is
   stored and nothing else has changed.
3. **Update by hand** anything you pasted the old key into: Claude Code's
   `ANTHROPIC_API_KEY` (*Claude Code*) and an existing Open WebUI connection (below).

Until then those clients get `400 No connected db.`, LiteLLM's answer to a key it
doesn't know.

**Open WebUI.** Its connection to the gateway (`http://host.docker.internal:4000/v1`)
uses the gateway's key: `services/openwebui/docker-compose.yml` sets it to
`${LITELLM_MASTER_KEY:-sk-litellm-master-key}`, which gpu-mode passes to compose with
your other settings. Open WebUI (v0.11.4) copies that variable into its database
only when the database has no value yet, which in practice means a new
`open-webui-data` volume. After that the stored value is used. So on an existing
volume, after a rotate, set it yourself: **Admin → Settings → Connections →** the
`:4000` connection **→** key **→ Save**. (`ENABLE_PERSISTENT_CONFIG=false` would make
the variable win, but Open WebUI would then also forget every setting changed in its
UI, so the compose doesn't set it.) `OWUI_OPENAI_API_KEYS` still replaces the whole
key list. `setup-ai-studio.sh` adds a missing `:4000` connection to an older volume
with a placeholder key; set that one the same way.

⚠️ A plain `docker compose up` in `services/litellm` doesn't read your settings: it
brings the gateway back on the public default key, without the keys your own routes
use (only an older `services/litellm/local.env` is loaded). Use `gpu-mode gateway`
(or any `gpu-mode` mode), c3's Containers tab or `scripts/litellm-log.sh`.

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
    apiKey: sk-litellm-master-key            # your gateway key; shown here as the public default
    api: openai-completions
    discovery:
      type: litellm                          # live models + model_info from /model_group/info
    compat:
      supportsDeveloperRole: false
      thinkingFormat: qwen-chat-template     # thinking on/off + effort in chat_template_kwargs
      reasoningDisableMode: none-effort      # `off` over /v1/responses (see Wire format)
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
Responses path wasn't cache-measured). tabbyAPI (exllamav3) doesn't serve it, so
its routes carry `use_chat_completions_api: true`: LiteLLM converts the call to
chat completions, the reply's `reasoning_content` comes back as a reasoning item,
and past reasoning goes out as `reasoning_content` on the assistant turn. One
known gap, in LiteLLM: a *non-streaming* Claude Code request through this bridge
comes back empty; Claude Code's streamed turns are fine.

- **`off`** goes out as `reasoning: {effort: "none"}`, which vLLM and SGLang serve
  with no reasoning at all. That comes from the provider's `reasoningDisableMode:
  none-effort`, which `omp-setup.sh` writes from 2026-10-02 on. Without it omp sends
  no reasoning setting for `off`, and the slug's default (thinking on) applies, so
  **re-run `omp-setup.sh`** if your block predates it. Checked with omp 18.4.10
  through the gateway on SGLang dual-fast (no reasoning for `off`, `effort: "low"`
  for `low`), and with the same request sent straight to vLLM dual-fast. llama.cpp
  wasn't checked.
- **The thinking budget** (`thinking_token_budget: 8192` in the provider) exists
  only on the chat-completions wire, so it doesn't reach the engine through the
  gateway; it reaches a vLLM engine only from an omp provider of your own pointed
  straight at a vLLM port. Over the gateway, the effort level is the lever.

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

## pi — setup

pi is the agent omp is built on. It has no gateway discovery, so `pi-setup.sh`
writes a snapshot of what the gateway serves:

```bash
bash scripts/pi-setup.sh                  # adds a `club` provider to ~/.pi/agent/models.json
bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   # experimental slug: --force
pi --model club/qwen3.8-27b               # or /model in pi — Ctrl+S saves it as the default
cp services/pi/extensions/tps-meter.ts ~/.pi/agent/extensions/   # optional: speed + cache share in the statusline
```

The snapshot lists every route the gateway serves, each with the context window
the gateway reports, plus `qwen3.8-27b` — every Qwen3.8 slug serves that id, and it
gets 262,144 when nothing is up. **Re-run `pi-setup.sh` after switching to a slug
with a different window or model**: pi compacts against the window it was given.
Other providers in the file are left alone, the old file is backed up, a new file
is created `0600` (it holds keys), and `settings.json` is not touched.

What it writes with the vLLM dual-fast slug up:

```json
{
  "providers": {
    "club": {
      "name": "club-3090 local models (generated by club-3090 scripts/pi-setup.sh; re-run to refresh)",
      "baseUrl": "http://127.0.0.1:4000/v1",
      "api": "openai-completions",
      "apiKey": "sk-litellm-master-key",
      "compat": { "supportsDeveloperRole": false },
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "qwen3.8-27b (club-3090)",
          "contextWindow": 262144,
          "reasoning": true,
          "thinkingLevelMap": { "minimal": null, "low": "low", "medium": "medium", "high": null, "xhigh": "xhigh", "max": null },
          "maxTokens": 32768,
          "compat": {
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
              "enable_thinking": { "$var": "thinking.enabled" },
              "reasoning_effort": { "$var": "thinking.effort" }
            }
          }
        }
      ]
    }
  }
}
```

- `thinkingFormat: chat-template` sends pi's thinking switch and effort as template
  kwargs (`enable_thinking`, `reasoning_effort`), the field both engines hand to the
  Qwen3.8 template. `/thinking` offers off, low, medium and xhigh; a request for
  `high` goes out as `xhigh`.
- `supportsDeveloperRole: false`: the template has no developer role. pi also folds
  any later system message into the first one by default, so it does not hit the
  vLLM 400 that Claude Code does.
- `maxTokens: 32768` is the reply cap (thinking plus answer), as for omp. pi sends
  no sampling values, so each slug's server-side values apply (*Sampling*).
- `apiKey` is your gateway key, read from `~/.config/club-3090/secrets.env`; the
  block shows the public default, which is what an install without a key of its own
  gets (*The gateway key*).

Checked through the gateway on the reference rig (pi 0.87.1, SGLang and vLLM
dual-fast):

| Check | Result |
|---|---|
| Thinking levels | `off` sent `enable_thinking: false` and got no reasoning; `low` / `medium` / `xhigh` reached the template as `reasoning_effort` |
| Tools over three turns (`--continue`) | `bash`, then three parallel `read`s, then correct answers, on both engines |
| Prefix cache | 91–97 % of each prompt from cache on SGLang, 92–94 % on vLLM once warm |
| Past reasoning | sent back as `reasoning_content` on every assistant turn |
| Message roles | only `system`, `user`, `assistant`, `tool` — no 400 on vLLM |

Changing the thinking level in the middle of a session costs a full re-prefill —
see *Prefix caching — what breaks it*.

⚠️ **A role-router package overrides `--model` and the thinking level.** A pi
package that routes turns by role (such as `pi-fabric-role-router`) sends each turn
to its role's model, at that role's `thinking` level from its own config
(`fabric-routing.json`). That level wins over `--thinking`, a `:level` suffix on
`--model` and the saved default level. With the router's primary role at
`thinking: medium`, `--thinking low` and `--thinking off` both went out as
`reasoning_effort: "medium"`. Without the router, or with `--no-extensions`, `low`
went out as `low` and `off` as `enable_thinking: false`. This was checked on pi
1.0.0 and 0.99.2 with router 0.4.1, 2026-10-02. A level set inside the session
(`/thinking`) still applies (checked over pi's RPC mode, which sets it the same
way). To control thinking per run, set it per role in the router's config, use
`/thinking`, or start pi with `--no-extensions` and load the extensions you want
with `-e` (`-e ~/.pi/agent/extensions/tps-meter.ts` for the meter). Point the
router's roles at `club/qwen3.8-27b`.

Scripted runs (a role router overrides `--thinking`, see above):

```bash
pi -p --model club/qwen3.8-27b --thinking low "…" </dev/null
```

## Hermes Agent — setup

```bash
bash scripts/hermes-setup.sh              # adds a `club` provider to ~/.hermes/config.yaml
bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   # experimental slug: --force
hermes chat --provider custom:club -m qwen3.8-27b          # or `hermes model` to make it the default
```

Hermes can't read a context window from the gateway, so, as for pi, the script
writes a snapshot: every route the gateway serves with the window it reports, plus
`qwen3.8-27b` (262,144 when nothing is up). **Re-run it after switching to a slug
with a different window or model.** It writes through `hermes config set`, which
keeps the file's comments and layout; it leaves your default model and reasoning
effort alone, refuses a `club` provider it didn't write, and backs the file up.
In a session, `/model custom:club:qwen3.8-27b` switches to it.

What it writes with the vLLM dual-fast slug up:

```yaml
providers:
  club:
    name: club-3090 local models (scripts/hermes-setup.sh)
    api: http://127.0.0.1:4000/v1
    api_key: sk-litellm-master-key   # your gateway key; shown here as the public default
    transport: chat_completions
    default_model: qwen3.8-27b
    models:
      qwen3.8-27b:
        context_length: 262144
```

- Hermes sends its effort as a top-level `reasoning_effort` (`/reasoning`, or
  `agent.reasoning_effort` in the config). On the Qwen3.8 template `none` turns
  thinking off, `low` / `medium` / `xhigh` are the real rungs, `high` and `max` run
  as `xhigh`, and `minimal` as `low` — Hermes's own `ultra` goes out as `max`
  (`minimal` and `max` from #1458 on; before it they 400'd).
- It sends no reply cap to a custom endpoint, so the server's default applies.
- **Turn on `model.reasoning_echo`** (`hermes config set model.reasoning_echo true`).
  By default Hermes drops the model's earlier reasoning from the history it sends,
  so inside a tool loop the model sees its past tool calls but not the plan behind
  them; with it on, each assistant turn carries its `reasoning_content` (checked on
  the wire), which the Qwen3.8 template renders — the same as omp and pi do. The
  cost: each turn's reasoning is prefilled once on the next request; the prompt stays
  append-only, so the prefix cache holds either way. It is global to Hermes's `model:`
  block, and strict providers (Mistral, Groq, Cerebras, SambaNova) reject the field —
  turn it off before making one of those your main model.
- At startup it probes `/api/v1/models`, `/api/tags` and `/v1/props` for a context
  window; the gateway answers 404 to each, which is harmless — the window comes from
  the config.
- A fresh Hermes home installs its runtime (browser tools and more, ~2 GB) on the
  first `hermes` command, `hermes config set` included — run Hermes once before the
  script.
- **`--resume` doesn't keep the session's provider.** A resumed session runs on your
  *default* model, so repeat the provider unless club is your default:
  `hermes chat --resume <id> --provider custom:club -m qwen3.8-27b`.
- **One-shot runs cap subagents.** `hermes chat -q` / `--oneshot` may spawn at most
  `delegation.oneshot_max_children` subagents in total (default 2); ask for more and
  Hermes refuses the whole batch, and the model does the work itself. Interactive
  sessions run up to `delegation.max_concurrent_children` (default 10) at once — more
  than the slug has sequences for queue (see *Which slug to serve*).

**Tools.** Which tools the model gets comes from your Hermes toolsets
(`hermes tools list`), not from the `club` provider — the same set as for any model.
Some reach outside the machine even though the model runs locally (`web`, `browser`,
`x_search`, `image_gen`, `tts`, `connections`), and `computer_use` drives your
desktop. For a run that stays on this machine, name a local set:

```bash
hermes chat --provider custom:club -m qwen3.8-27b \
    -t terminal,file,code_execution,todo,memory,session_search,skills,clarify,delegation
```

**Speed and cache share are built into Hermes's status bar** — `cache_hit` and `tps`,
shown by default on a wide terminal (`display.status_bar.fields`) — so the omp/pi
meter isn't needed here. `cache_hit` works through the gateway: Hermes records the
engine's cached-token count exactly. `tps` is output tokens over the whole call time,
prefill included, so with Hermes's long prompts it reads well below the decode speed.

Checked through the gateway on the reference rig (Hermes Agent 0.21.5):

| Check | Result |
|---|---|
| Tool calls | `search_files` → `read_file` → `write_file` → `terminal` ×2 (write a script, run it, append to a file), results verified on disk — 29 s on sgl dual-fast |
| Prefix cache across resumed turns | 98 % of the prompt from cache on sgl dual-fast, for a plain turn and for one with a tool loop; 97 % on vllm dual-fast |
| Subagents | `delegate_task`, two children in parallel: both completed on `qwen3.8-27b`, every gateway request 200 — 22 s on sgl dual-fast |
| Tool task on vllm dual-fast | three files read and joined in 14 s; resumed follow-up in 4 s |

Hermes's system prompt with its default 33 tools is about 14.5K tokens.

## dsh (DeepSeek Harness) — setup

```bash
bash scripts/dsh-setup.sh                 # adds a `club` block to the headless + web profiles in ~/.dsh
bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   # experimental slug: --force
dsh headless "…"                          # or: dsh web — pick club/qwen3.8-27b and its effort in the model menu
```

dsh (`@deepseek-ai/dsh`) is DeepSeek's agent harness: `dsh web` serves a browser UI,
`dsh headless "task"` answers one task and exits. It reaches models through pi's model library,
so the settings mirror pi's. Its settings live in **profiles** under `~/.dsh/profiles/<name>/`,
each with a `cordis.patch.yml`. `dsh-setup.sh` writes one marked block into each profile you name
(`--profile`, default `headless` and `web`); re-run it to refresh. The rest of the file is never
touched, and the old file is backed up.

- **The first run.** dsh's built-in default model is DeepSeek's own cloud API, so a new profile's
  first run goes there unless the profile names a model first. The script creates a missing
  profile with `dsh --profile <name> --dump-config`, which runs no model, and writes
  `club/qwen3.8-27b` (effort `low`) as its default before dsh ever runs it. A profile that already
  names a default model of its own keeps it; the script adds only the provider. Your
  `DEEPSEEK_API_KEY` is left alone, so DeepSeek's models stay in the model menu.
- **An older `~/.dsh/settings.yaml`** that names a default model gets moved by dsh into the first
  profile it boots, replacing the club default there. The script warns about it and leaves it
  alone; rename it to keep the club default.
- **The key.** The provider only names its key (`apiKeyEnv`). The script stores the key in
  `~/.dsh/.env` as `CLUB3090_GATEWAY_KEY` (`0600`; the file's other lines are kept), which dsh
  reads for that name. A `CLUB3090_GATEWAY_KEY` exported in your shell wins over the file. Re-run
  the script after `gateway-key.sh rotate`.
- **The window.** dsh can't read a context window from the gateway, so the model list is a
  snapshot of what the gateway serves, as for pi and Hermes, plus `qwen3.8-27b` (262,144 when
  nothing is up). **Re-run it after switching to a slug with a different window or model.**
- **Refused:** a profile whose file already configures `llm-pi-ai` itself (a provider added by
  hand or in the web UI). Add the `club` provider below to that entry yourself.
- The web UI's model settings can still change the default model and add providers, since the
  block sits in each profile's own file. The same block in `~/.dsh/cordis.patch.yml` would apply to
  every profile but override them: per dsh's docs, the UI then can't change those entries.

What it writes into `cordis.patch.yml` (`bash scripts/dsh-setup.sh --print` shows it without
writing; this is the gateway-down snapshot):

```yaml
# >>> club-3090 local models (generated by club-3090 scripts/dsh-setup.sh; re-run to refresh) >>>
- id: llm-pi-ai
  config:
    providers:
      club:
        displayName: club-3090 local models
        apiKeyEnv: CLUB3090_GATEWAY_KEY   # the key itself is in $DSH_HOME/.env
        api: openai-completions
        baseURL: "http://127.0.0.1:4000/v1"
        compat:
          supportsDeveloperRole: false
        models:
          - id: "qwen3.8-27b"
            contextWindow: 262144
            maxTokens: 32768
            reasoningEfforts:
              "off":
              low: low
              medium: medium
              xhigh: xhigh
            compat:
              thinkingFormat: chat-template
              chatTemplateKwargs:
                enable_thinking: { $var: thinking.enabled }
                reasoning_effort: { $var: thinking.effort }
- id: agent-default-model
  config:
    provider: club
    model: qwen3.8-27b
    reasoningEffort: low
# <<< club-3090 local models <<<
```

- `thinkingFormat: chat-template` sends thinking on/off and the effort as template kwargs, the
  field both engines read. `off` is left empty on purpose: with `chat-template` it sends
  `enable_thinking: false`. The Qwen3.8 rungs are `low`, `medium` and `xhigh`; the model menu
  offers only the levels listed.
- `maxTokens` is the reply cap; dsh sends it as `max_completion_tokens`.
- To try a setup with no chance of a cloud call, prefix that one command with
  `env -u DEEPSEEK_API_KEY`: it drops the key for that command only, not from your shell. A
  scratch `DSH_HOME=` keeps the test away from your real profiles.

Checked through the gateway on the reference rig (dsh 0.2.0-rc.2, vLLM dual-fast), with a scratch
`DSH_HOME` and `DEEPSEEK_API_KEY` unset:

| Check | Result |
|---|---|
| Set up by `dsh-setup.sh` | a fresh profile created without running a model; the key read from `~/.dsh/.env`; `low` and the quoted `"off"` level reached the engine as below |
| Effort `off` | `chat_template_kwargs: {enable_thinking: false}`; no thinking in the run |
| Effort `low` | `{enable_thinking: true, reasoning_effort: "low"}`; the model thought, then answered |
| Session titles | dsh's 64-token title request goes out with thinking off at every effort, so it can't be eaten by reasoning |
| Tools | `glob` then `bash` on a three-file workspace; correct count |
| Continued session (`--session-id`) | correct follow-up answers; the system prompt and the 24 tools were identical on every turn, and one follow-up took 13,440 of its 13,692 prompt tokens (98 %) from the prefix cache |
| Sampling | none sent; the slug's server-side values apply (*Sampling*) |

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
| `vllm/qwen38-27b-dual-max` | 2 | ~271K | Pool holds about one full-length session — add `KV_OFFLOAD_GB=64` so evicted sessions come back from host RAM (it must fit in `/dev/shm`, below). |
| `sgl/qwen38-27b-dual-fast` | 2 | ~548K | Fine for one agent + 1 subagent. Honours the requested effort from #1439. |
| `sgl/qwen38-27b-dual-max` | 1 | ~183K (160K ctx) | Set `task.maxConcurrency: 1`. |
| DFlash2 tiers (`superfast`, `ultrafast`, `supermax`, `ultramax`) | 1 | — | Set `task.maxConcurrency: 1`; KV offload is write-only on DFlash. |

Requests beyond a slug's sequence count queue at the server — they don't fail, but
a subagent then waits for the whole main turn.

**Host-RAM KV tier.** Booting with `KV_OFFLOAD_GB=64` lets a conversation that was
pushed off the GPU come back from RAM instead of re-prefilling. Measured with three
interleaved 58K-token agent sessions pushed off the GPU: revisits took **7.7 s
(SGLang) / 8.6 s (vLLM)** against **~42 s** cold (#1419).

On vLLM the RAM tier lives in the host's `/dev/shm`, which is half of RAM by default, so on a
128 GB host 64 does not fit: keep it under the size `df -h /dev/shm` shows (48 there), or enlarge
`/dev/shm`. `switch.sh` refuses a tier larger than `/dev/shm` before it stops the running slug (#1503).

## Sampling

The gateway sets no sampling. Whatever an agent sends reaches the engine as sent,
and anything it leaves out falls back to the engine's values. The setup scripts
(`omp-setup.sh`, `pi-setup.sh`, `hermes-setup.sh`) don't write sampling settings
either.

For Qwen3.8-27B and ThinkingCap, every compose starts with thinking on, and the
engine already serves the model card's thinking values without the client sending
anything:

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|--:|--:|--:|--:|--:|
| Thinking on (every compose's default) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking off | 0.7 | 0.8 | 20 | 0.0 | **1.5** |

- **Thinking on:** nothing to send. vLLM applies the values from
  `--override-generation-config`, and SGLang reads them from the checkpoint's
  `generation_config.json`. `presence_penalty` 0.0 is the engines' own default, so
  it comes out right as well.
- **Thinking off** (omp `off`, pi `off`, Hermes Agent `/reasoning none`):
  the card's values are `temperature: 0.7`, `top_p: 0.8` and
  `presence_penalty: 1.5`, and **only the client can send them**. The server can't
  switch to them for you, for two reasons:
  - **The server's values are chosen at boot.** The compose picks them once, from
    its `ENABLE_THINKING` setting (on by default). A request that turns thinking
    off changes the chat template, not the sampling, so the engine keeps using the
    thinking values.
  - **`presence_penalty` can't be set on the server at all, on either engine.** The
    server-side defaults vLLM and SGLang accept are limited to temperature, top_p,
    top_k, min_p and repetition_penalty. Their chat endpoints also fill in
    `presence_penalty: 0.0` on every request that doesn't send one. The
    `PRESENCE_PENALTY` variable in the composes records the intended value and
    changes nothing at runtime.

**What each agent sends**, read from the gateway's request log
(`scripts/litellm-log.sh on`) on SGLang dual-fast, 2026-10-02:

| Agent | Sampling values | Thinking off reaches the engine as |
|---|---|---|
| omp 18.4.10 | none | `reasoning: {effort: "none"}`, once `omp-setup.sh` has been re-run (*Wire format*, under *omp*) |
| pi 1.0.0 | none | `enable_thinking: false` (`--thinking off` or `/thinking off`), unless a role-router package overrides the level (*pi — setup*) |
| Hermes Agent 0.21.5 | none on chat turns, from its source (not captured on the wire): a temperature goes out only when the provider's profile fixes one. Some side tasks, such as session titles, ask for 0.3 | `/reasoning none` (*Hermes Agent — setup*) |
| dsh 0.2.0-rc.2 (vLLM dual-fast) | none | `enable_thinking: false` at effort `off` (*dsh — setup*) |
| Claude Code 2.1.x (`claude -p`) | none | Claude Code sends no thinking setting to this model (*Claude Code*) |

So with any of these agents a thinking-off turn runs on the thinking values above,
with `presence_penalty` 0.0. We found no setting for sampling values in omp's or
pi's model config. omp's `extraBody` reaches the engine only on the
chat-completions wire, and would apply at every thinking level alike.

**Changing the server's values.** On **vLLM**, `TEMP`, `TOP_P`, `TOP_K` and `MIN_P`
change what a request that sends nothing gets. Set them for one launch with
`TEMP=0.7 TOP_P=0.8 bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast`, or
for every launch with `bash scripts/settings.sh set TEMP=0.7`, which applies to
every slug that reads `TEMP`. `switch.sh --set <slug>` doesn't take them: it only
saves the launch settings it catalogues. On **SGLang**
they have **no effect on chat requests**. Its `--preferred-sampling-params` flag
only applies on the native `/generate` endpoint (sglang#39096, see
[UPSTREAM.md](UPSTREAM.md)), and its chat requests take temperature, top_p and
top_k from `generation_config.json`. On SGLang, send the values per request.

`bench.sh` sends all four sampler values explicitly for this reason (see *Bench
protocol* in `AGENTS.md`).

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
- **Changing the thinking level mid-session re-prefills everything.** The Qwen3.8
  template writes the effort's instructions at the top of the system turn, so a new
  level changes the prompt's first tokens. In a pi session on vLLM, switching to
  xhigh on the third turn took the cached share from 94 % to 0 % (all 2,036 tokens
  prefilled again). The same holds for any client that changes the effort inside
  one conversation — in omp, a role with a different effort.

## Claude Code

Claude Code (2.1.x) sends its `# Environment` block as a `system` message after the
first user turn. The stock Qwen3.8 template refuses a system message that isn't
first, so every Claude Code request used to fail on the vLLM slugs with
`400 … System message must be at the beginning`. The vendored template renders it
where it sits, as its own system turn, which keeps the prompt append-only
([#1447](https://github.com/noonghunna/club-3090/discussions/1447)). Still seeing
that 400? The slug was started before the fix — relaunch it with `switch.sh`.

Claude Code talks to the same gateway through its Anthropic-compatible
`/v1/messages` endpoint. What happens next depends on the slug:

| Slug | The gateway… | The model's reasoning | Follow-up turns |
|---|---|---|---|
| Qwen3.8 on vLLM or SGLang | forwards the request untouched to the engine's own `/v1/messages` | comes back as thinking blocks; Claude Code sends them back on the next request | 97–99 % of the prompt from the prefix cache |
| any other slug | translates it (LiteLLM → the engine's Responses API) | not returned — only the answer and tool calls | — |

LiteLLM's translation builds thinking blocks only from a reasoning *summary*, which
neither engine produces, so on that path Claude Code never sees the reasoning and
has nothing to replay. The engines' own endpoints return it, but Claude Code sends a
`system` message every turn, and an endpoint that moves it to the front of the
prompt makes every turn re-read the whole conversation. vLLM v0.30.0 moves it
unless the server was started with `--chat-template` — it checks the flag, not the
template it loaded (vllm#58754, [UPSTREAM.md](UPSTREAM.md)) — so the Qwen3.8 vLLM
composes pass their mounted template that way too. The sync measures this per route
with `count_tokens` (`supported_endpoints` in *What the gateway sets*); nothing to
configure.

Claude Code (2.1.283) sends no thinking setting or effort for a model it doesn't
recognise, so the model reasons at the slug's default effort.

In `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_API_KEY": "<your gateway key>",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144"
  },
  "model": "qwen3.8-27b"
}
```

- `ANTHROPIC_BASE_URL` is the gateway itself (no `/v1`), never an engine port.
  `ANTHROPIC_API_KEY` is your gateway key: the `LITELLM_MASTER_KEY=` line of
  `~/.config/club-3090/secrets.env`, or the public default `sk-litellm-master-key`
  when none is stored there (`gateway-key.sh status` says which; *The gateway key*).
  Update it here after every rotate.
- With discovery on, Claude Code lists every route the gateway serves. `model` must
  be a served id: `qwen3.8-27b` on every Qwen3.8 slug, `thinkingcap38-27b` on the
  ThinkingCap ones.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the serving slug's context window — the
  route's `max_input_tokens` (see *What the gateway sets*); 262144 on the dual-fast
  slugs. Claude Code doesn't know the model, assumes 200K otherwise and compacts
  against that. Lower it when you serve a smaller slug, or a long session runs past
  the window.
- `400 No connected db.` on every request means the key Claude Code sends isn't
  the gateway's `LITELLM_MASTER_KEY` — after a rotate, most likely the old one:
  LiteLLM looks an unknown key up in a database the gateway doesn't have. Fix the
  key rather than removing the master key — the gateway listens on every
  interface, so without one anyone on your network can use it.

Checked on the reference rig with the Claude Code CLI (2.1.283) through the
gateway, direct path on both engines: a task with tool calls (find three files,
read each, answer), then three `--continue` follow-ups, one with another tool call.
On `vllm/qwen38-27b-dual-fast` that took 18 s, 2 s, 2 s and 5 s, on
`sgl/qwen38-27b-dual-fast` 20 s, 2 s, 4 s and 4 s, with 97–99 % of every follow-up
served from the prefix cache; every request carried all of the session's earlier
thinking blocks back to the model, and every answer was right.

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

- **Claude Code needs a vLLM or SGLang slug.** The llama.cpp Qwen3.8 slugs use the
  template inside the GGUF, which refuses the system message Claude Code sends after
  the first user turn; the vendored template that accepts it is mounted on the vLLM
  and SGLang slugs only.
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
