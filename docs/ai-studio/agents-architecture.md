# AI Studio — Agent Architecture

How the AI Studio "director" makes decisions: the one 4B model behind every lane, the **two
distinct agent shapes** it runs in, the Production Director's decision flow, and exactly how the
agent differs lane-by-lane.

> ## ⚠️ Status — the 🎬 Production Director is a work in progress
>
> **It is NOT recommended for production use.** The multi-turn chat interface is still **buggy and
> does not yet follow instructions reliably** — intent reads, mid-conversation stack changes, and
> confirms can misfire. Treat it as **experimental**: fun to play with, not something to depend on.
>
> The 12 single-shot lanes (image · video · audio) are simpler and further along, but still carry
> the rough edges listed in [Current challenges](#current-challenges--known-limitations). Everything
> below describes the **current design**, not a finished, dependable agent.

> **One model, many roles.** Every lane is powered by the **same** small director —
> `qwen3.5-4b-uncensored` (HauhauCS-Aggressive Q4_K_M) on llama.cpp, container `studio-director`
> @ `:8090`. It always runs **thinking-OFF** (see [Key decisions](#key-design-decisions)). What
> changes per lane is the **system prompt** and the **control flow wrapped around it** — not the
> model.

> **Where the agent lives.** The OWUI **Studio pipe** (`services/studio/build_studio_pipe.py` →
> baked into `studio_pipe.py`) is the agent's control plane. It runs *inside* the Open WebUI
> container with no repo access, so its persona + intent logic + workflows are **baked in at build
> time**. Edit the source, then `python3 services/studio/build_studio_pipe.py && bash
> services/studio/push-pipe-to-owui.sh`.

---

## Two agent shapes

The 13 lanes (`_LANES` in `build_studio_pipe.py`) split into two completely different control flows:

| Shape | Lanes | Flow | Backs onto |
|---|---|---|---|
| **A — Single-shot prompt-crafter** | the 12 generation lanes (Video ×4, Image ×5, Music, SFX, Voice) | one call: craft a prompt **or** decline-with-chat → render one asset | ComfyUI `:8188` / step-voice `:8193` |
| **B — Conversational plan-then-execute Director** | `🎬 Studio · Production` (1 lane) | multi-turn: intent → confirm → plan → multi-asset film | Production service `:8195` |

Shape A renders immediately. Shape B is a stateful agent that plans a whole short film and only
builds on an explicit **go**. The Production lane is gated on the `:8195` service health
(`/produce/health`) — it only appears in the picker when the Production server is running; the 12
single-shot lanes are always present when ComfyUI is up.

---

## Shape A — single-shot lanes (craft-or-decline)

These lanes are conversational but simple — there's no plan and no confirm. The director either
turns your message into a render prompt, or, if it isn't a real request, replies with a one-line
nudge:

```
You type in a single-shot lane (Image · Video · Music · SFX · Voice)
        │
        ▼   the director reads your message —
   ┌────────────────────────────────────┬──────────────────────────────────┐
   ▼                                     ▼
 NOT a real request                   a real request
 (greeting · question · too vague)    ("rain on a tin roof")
   │                                     │
   ▼                                     ▼
 a friendly one-line nudge,           it crafts a full prompt and
 NOTHING renders                      renders it right away → one asset
 ("Describe a sound…")
```

- **Greeting / question / too vague** → a nudge, **no GPU spend** (it won't render a film from "hi").
- **A real request** → rendered straight away — no "go" step.
- **A follow-up** ("add reverb", "at night") → it remembers the *previous* render's prompt and
  applies only your change. That's the lane's entire memory.
- The chat is a **task-keeper**: it points you back to describing what to make — it does *not*
  answer questions like "Chroma or HiDream?" (you'd get "tell me what to create").

---

## Shape B — the Production Director

A real conversational agent: it holds the conversation as state, **proposes** a plan, and only
**builds** the (expensive, irreversible) film when you say **go**. Two stages — how it responds to
you, then what "go" actually builds.

### Why this shape — "creative within bounds"

Two choices explain why the Director plans-then-executes rather than running a live tool-calling loop:

- **Serial is the unlock, not the limitation.** On a 2× 3090 PCIe rig (no NVLink), co-loading
  video+image+audio buys almost nothing — ComfyUI runs one workflow at a time and a diffusion step
  can't be cheaply split across cards, so you'd burn most of the 48 GB budget just to skip reloads.
  Going **serial** (load a lane for its step → render → evict → next) dissolves the VRAM problem and
  turns the hard part into *creative orchestration*, where an LLM actually adds value.
- **The model proposes, the executor disposes.** The 4B plans once, up front, into a validatable
  JSON plan (pure reasoning, no side effects); a deterministic executor then runs it shot-by-shot and
  owns everything safety-critical — valid lanes, durations, asset wiring, the VRAM mutex, the
  validators. The LLM is **never trusted to sequence GPU ops**. It's the rig's ethos: *deterministic
  control flow, the model only for the fuzzy bits.*

So the director is **creative in the film-making layer, never the control plane**: it chooses story
angle, tone, pacing, shot concepts, camera language, narration, when a keyframe or music helps — but
it *can't* choose invalid lanes, impossible durations, or arbitrary paths (the executor rejects those
and feeds back a bounded repair). The schema isn't there to make it less creative — it's there to
stop it spending creativity on things the machine can't execute.

### Stage 1 — how it responds to you

Every message, the director reads the **whole conversation so far** and works out what you want now:

```
You type a message in the 🎬 Production lane
        │
        ▼   the director reads the whole conversation and decides:
        │
  ┌─ just chatting / asking? ──────► it replies; nothing renders
  │    "hi" · "what models can I use?" · "what does continuity mean?"
  │
  ├─ described a film? ────────────► it sizes it + shows a PLAN CARD, then waits
  │    "a 1-minute documentary on the history of Pakistan"
  │       🎥 video model · 🖼️ image model · ⏱️ length → shots · ⚙️ est. render time
  │       🔎 documentary? it offers to research real web facts first
  │
  ├─ changing the plan? ───────────► it updates the card + shows it again
  │    "use LTX" · "30 seconds" · "no music" · "hidream keyframes" · "research"
  │
  └─ said "go"? ──────────────────► it builds the film  ▼ (Stage 2)
       the ONLY thing that starts a render
```

**Two guarantees that shape the behavior:**
- **Only "go" renders.** A film brief on its own never starts a render — the director shows the
  plan and waits. (Say "go" before describing anything and it asks for a brief first.) This safety
  latch exists because a render is expensive and irreversible.
- **It's truthful about what it can do.** It answers questions honestly, won't claim it has already
  started rendering, and won't pretend it can browse arbitrary web pages — it can only look up real
  facts to ground a *documentary* ([#523](https://github.com/noonghunna/club-3090/pull/523)).

### Stage 2 — what "go" builds

When you say **go**, the director plans and renders the whole film:

```
"go" → the Production Director:

  1. Documentary or story?   picks the format — a documentary is narrated with real
                             facts and NO invented main character; a story gets one
  2. Research (opt-in)       documentary + you asked → looks up real facts on the web
                             (SearXNG) to ground the script; skipped otherwise
  3. Cast + treatment        for a story, defines its recurring characters ONCE (a
                             "Character Bible" — fixed look + seed) so they stay
                             consistent shot-to-shot; writes a short treatment
  4. Shot-by-shot plan       turns it into N shots (~5s each), each naming which
                             characters appear and how the shot looks + sounds
  5. Self-check              re-reads the plan, fixes off-topic / repetitive shots once
                             before committing
  6. Render                  keyframes → video → narration → music → stitched into one
                             MP4, with live progress streamed into the chat
```

**Keeping a film coherent** is the Director's real job beyond a single clip — characters and style
*drifting* between independently-generated shots is **the** classic AI-film problem (the quality
ceiling, not a VRAM one). Two mechanisms attack it:

- **Character Bible** (stories only) — recurring characters are described **once** with a canonical
  look + a stable seed, then referenced by name in each shot, so the same character looks the same
  across shots ([#502](https://github.com/noonghunna/club-3090/pull/502)). Documentaries keep this
  **empty** — they narrate a real subject, no protagonist.
- **Continuity mode** (⚙️ valve) — how shots are kept visually consistent:

  | Mode | How it links shots | Best for |
  |---|---|---|
  | **storyboard** *(default)* | each shot gets its **own keyframe** from the image model, all sharing one **style bible** (palette/look); the shot's video is then animated (i2v) from that keyframe | most films — per-shot art direction **and** a film-wide consistent look |
  | **hero** | **one** shared keyframe seeds every shot | a single subject / scene throughout |
  | **chain** | each shot continues (i2v) from the **previous shot's last frame** | one flowing, continuous take |
  | **none** | independent text→video shots, no keyframes | fastest; loosest continuity |

> **Web research (SearXNG).** Step 2 leans on the rig's **SearXNG** (`:8088`, the same instance
> OWUI's web search uses). It's **opt-in** (say "research" / "dig", or accept the offer on the plan
> card), **documentary-only**, and **fails open** — if SearXNG is down or returns nothing, the plan
> just proceeds on the model's own knowledge. It's the only lane that does web research; the
> single-shot lanes never touch it.

> **Under the hood (for contributors).** Stage 1 is the OWUI pipe (`build_studio_pipe.py`, pure
> classifiers in `director_intent.py`); Stage 2 is the `:8195` server — `planner.py` (a treatment
> **0.8** → plan **0.4** → critic **0.0** temperature ladder, since the model runs thinking-OFF; the
> plan token budget scales with shot count), `critic.py`, `research.py`, `prompts.py`. Function- and
> line-level pointers are in the [system-prompt map](#system-prompt-map) + [See also](#see-also).

---

## How the agent differs per lane

| | 🎬 Production (Director) | 12 single-shot lanes |
|---|---|---|
| Agent shape | conversational plan-then-execute | one-shot craft-or-decline |
| Decides chat-vs-act | reads the whole conversation → a structured decision | a craft-or-chat reply from one model call |
| Chat reply | substantive — answers "what options do we have?" | one-line nudge back to "describe what to create" |
| Acts on a real request | proposes a plan, waits for **go** | renders immediately on send |
| Memory | full conversation (last ~10 turns) | only the previous render's prompt |
| Genre awareness | detects documentary vs story | none |
| Web research (SearXNG) | yes — grounds documentary scripts | no |
| Visual continuity | Character Bible + continuity modes (storyboard / hero / chain / none) | n/a — one asset |
| Output | a multi-asset film (keyframes→video→narration→music→assembly) | one asset (image / clip / audio) |
| Confirm gate | yes — render is double-gated; only a real "go" fires it | no |
| Backs onto | Production service `:8195` | ComfyUI `:8188` / step-voice `:8193` |

A consequence worth knowing: a documentary request typed into a **single-shot video lane** (e.g.
"a 30s video on the history of Pakistan" in the LTX lane) gets one cinematic prompt chained to the
duration — *not* a researched, narrated documentary. The genre-detection / research / narration /
shot-planning all live in the **Production** lane only. (Lane choice is currently manual — picked in
the OWUI model selector.)

---

## System-prompt map

All lanes use the same 4B, but **8 distinct system prompts** across **3 locations**. `AGENTS.md` is
**not** a studio-wide persona — it governs the Production lane's *conversation* only.

| Prompt | Drives | Source | Role |
|---|---|---|---|
| `DIRECTOR_PRODUCER_SYS` | 🎬 Production (chat) | **`director/AGENTS.md`** (baked) | conversational persona — greetings, questions, nudge to a brief |
| `build_controller_system()` | 🎬 Production (intent) | `director_intent.py` | the intent JSON `{intent, brief, confirm, …}` |
| treatment / plan / critic | 🎬 Production (planning) | `production/prompts.py` | server-side plan generation |
| `DIRECTOR_SYS` | LTX · Sulphur · 10Eros · Wan | `build_studio_pipe.py:277` | video prompt-crafter |
| `DIRECTOR_IMG_SYS` | Ideogram-4 | `:290` | image crafter (JSON caption) |
| `DIRECTOR_IMG_PROSE_SYS` | Chroma · Z-Image · Krea | `:314` | image crafter (prose prompt) |
| `DIRECTOR_HIDREAM_SYS` | HiDream-O1 | `:356` | image crafter (HiDream style) |
| `DIRECTOR_MUSIC_SYS` | ACE-Step | `:328` | music crafter (tags + lyrics) |
| `DIRECTOR_SFX_SYS` | Stable Audio | `:340` | sound-design crafter |

**To change behavior:**
- Production-lane *conversation* → edit [`director/AGENTS.md`](../../services/studio/director/AGENTS.md).
- Production-lane *intent parsing* → edit `build_controller_system()` in [`director_intent.py`](../../services/studio/director_intent.py).
- Production-lane *planning* → edit `production/prompts.py`.
- A single-shot lane's craft/chat → edit the matching `DIRECTOR_*_SYS` in [`build_studio_pipe.py`](../../services/studio/build_studio_pipe.py).

Then rebuild + push the pipe (above). The intent classifiers in `director_intent.py` are
stdlib-only and unit-tested offline (`services.studio.production.tests.test_director_intent`); the
bake injects their source verbatim so the deployed pipe and the tests can't drift.

---

## Key design decisions

These are settled (don't re-litigate without new evidence):

- **The director runs thinking-OFF, everywhere.** The HauhauCS-Aggressive 4B's `<think>` runs away
  — empty `content`, output routed to `reasoning_content`, `finish_reason=length` even at a 16k
  budget (~100 s). Tested thoroughly. **Creativity comes from per-stage temperature**, accuracy
  from the temp-0 critic — not from reasoning.
- **The render is double-gated (Production lane).** A build fires only when the latest turn carries
  a real go-word — either a bare confirm ("go", "ok do it") or the model's confirm *corroborated* by
  a go-word. It never fires on the model's say-so alone, because the render is expensive + irreversible.
- **The LLM owns open-ended brief extraction; keywords own the closed-vocab confirm.** The keyword
  floor is *not* consulted for the brief when the LLM is up (fixes "ok do it" becoming the film).
- **Documentary ≠ narrative**, decided up front, so a factual brief gets no invented protagonist —
  and stories carry a **Character Bible** (fixed look + seed per recurring character) so they stay
  consistent across shots.
- **The director is truthful about its capabilities.** It won't claim a render/search has already
  started, and won't pretend it can browse arbitrary web pages — only that it can look up real facts
  to ground a documentary (the #523 "honesty fix").
- **Transport-success ≠ real success.** The executor validates the *actual* output before assembly
  — ffprobe duration + audio presence, dimensions, placeholder / near-uniform-frame checks — and
  fails or re-renders the shot. A call that returned a file but produced an empty / placeholder /
  collapsed clip is a failure, not a success.
- **Single-shot lanes refuse to render on chit-chat** (the `CHAT:` gate) — no GPU spend on "hi".

History of the Production-lane hardening (genre detection, the semantic critic, SearXNG research,
the LLM-first intent controller): club-3090 PRs **#519–#524**.

---

## Current challenges / known limitations

**The Production Director is work-in-progress and not production-ready** (see the [status banner](#-status--the--production-director-is-a-work-in-progress) up top) — the chat does not yet follow instructions reliably. These are the known open issues as of the #519–#524 hardening, documented so they're tracked, not surprises.

| Challenge | What happens | Path forward |
|---|---|---|
| **No cross-lane routing** — lane choice is 100% manual (the OWUI model picker); only the Production lane reasons about intent. | A documentary brief typed into a single-shot video lane (e.g. "a 30s video on the history of Pakistan" → LTX) renders one scenic clip chained to the duration, **not** a researched/narrated documentary — that lane has none of the genre/research/narration machinery. | The parked **Q4 cross-lane router**: one front door that detects "make a film" / "documentary" / "read this aloud" intent and routes (or nudges) to the right lane instead of trusting the manual pick. |
| **The 4B is the capability ceiling.** It runs thinking-OFF (its `<think>` runs away), so planning leans on the temperature ladder, not real reasoning — and it's a borderline critic that occasionally garbles a researched fact. | Weak factual judgment in the critic; plans can drift on nuanced briefs; the researched script isn't always faithful to the facts. | The critic/treatment `call` is injectable → route those stages to the **27B/35B** for stronger factual judgment (parked). |
| **Single-shot lanes are shallow conversationalists.** Their "chat" is the craft-or-`CHAT:` gate, nothing more. | "Chroma or HiDream?" gets a generic *"tell me what to create"* nudge, not an answer; no genre/research awareness; refinement remembers only the **previous render's** prompt (`prior_spec`), not the conversation. | Give the single-shot `CHAT:` branch `_produce_reply`-style latitude (small change), and/or fold lane Q&A into the cross-lane router. |
| **Duration parsing mis-reads decades.** `_target_seconds`'s `\d+\s*s\b` regex matches "1980s" / "90s" / "2000s" as **seconds** (verified). | "a film set in the **1980s**" → parsed as 1980 s → capped to 120 s → a multi-segment long clip nobody asked for; "**90s** synthwave" → a 90 s track. | Tighten the regex — require a separating space or exclude decade-style `\d{2,4}s`, drop bare-`s` matching, add a test. Small, self-contained fix. |
| **Fragmented agent instructions.** 8 system prompts across `AGENTS.md` + inline `DIRECTOR_*_SYS` + `director_intent.py` + `prompts.py` (see the [system-prompt map](#system-prompt-map)). | No single studio persona/voice; a cross-lane tone change is several edits, not one. | If a consistent voice becomes a goal: a shared persona preamble + per-lane task suffix. |
| **Not user-validated end-to-end since #524.** The LLM-first intent + research + critic chain passes its offline unit tests + dry-runs, but hasn't had a live OWUI run by a user since the last change. | Unknown real-world rough edges — e.g. the 4B's JSON-decision quality under genuinely messy multi-turn chat. | The parked **live validation**: re-run the search → dig → "ok do it" flow in OWUI on the rig. |

---

## Diagnosing & troubleshooting

The Director spans three moving parts — the **OWUI pipe** (Stage 1), the **4B model** (`:8090`), and
the **planning server** (`:8195`) — so most problems localize to one of them. Work top-down.

### 1. Is everything up?

| Check | Command | Healthy = |
|---|---|---|
| Planning server | `curl -s localhost:8195/produce/health` | `{"ok": true, …}` — **if this fails the 🎬 lane won't even appear in OWUI** (it's health-gated) |
| 4B director | `sudo docker ps --filter name=studio-director` · `sudo docker logs --tail 50 studio-director` | container **Up (healthy)**, serving `:8090` |
| Research backend | `curl -sf localhost:8088 >/dev/null && echo ok` | SearXNG up (research fails *open*, so this only matters for documentary grounding) |
| The scene | `gpu-mode status` | `ai-studio` active |

Start the planning server if it's down: `nohup python3 -m services.studio.production.server &`
(restart it after a code change — it bakes nothing, but it holds the loaded config).

### 2. Reproduce a plan offline — the key diagnostic

To see **what the Director plans** for a brief without burning a GPU render, run the CLI with the
**synthetic** backend (ffmpeg stand-ins, no ComfyUI):

```bash
python3 -m services.studio.production.run \
  --brief "a 1-minute documentary on the history of Pakistan" \
  --backend synthetic
```

It runs the whole chain — format detection → (research) → treatment → plan → critic → assemble — and
writes the **plan JSON** + a stand-in MP4 under the productions dir. This is how you debug "the plan
is wrong" (off-topic shots, a documentary that invents a protagonist, bad sizing) in seconds. Add
`--continuity` / `--keyframe-lane` / `--shots` to match the stack you're chasing, or `--backend live`
to render for real. (Stage-1 *intent* isn't exercised here — `run.py` takes the brief directly; for
intent bugs see §4 + §5.)

### 3. Inspect a live render

A `/produce` call returns a `job_id`; poll it:

```bash
curl -s localhost:8195/job/<job_id>     # → {status, phase, frac, title, error?}
```

`status: error` carries the failure in `error`; `done` carries the gallery URL. One film at a time —
a second `/produce` while one runs returns **409**.

Every run also writes a self-contained **`productions/<job_id>/`** folder under the ComfyUI output
tree — the plan, per-shot assets, audio, the final MP4, logs, and a typed **`manifest.json`** that
records run-level provenance (registry + workflow versions, every seed, validator results, and the
**exact ffmpeg assembly command**) plus per-artifact records. So a finished *or failed* production is
fully reproducible and any single shot is re-runnable in isolation — read the manifest to see exactly
what was generated, with which seed, and whether each validator passed.

### 4. Test the logic offline (no rig)

The intent / critic / research / stack logic is pure and unit-tested — run these to localize a
behavioral bug without the GPU:

```bash
python3 -m unittest \
  services.studio.production.tests.test_director_intent \
  services.studio.production.tests.test_critic \
  services.studio.production.tests.test_research \
  services.studio.production.tests.test_stack
```

### 5. Symptom → where to look

| Symptom | Likely cause | Where to look |
|---|---|---|
| 🎬 Production lane missing from the OWUI picker | `:8195` server down (the lane is health-gated) | start the server (§1) |
| "Could not reach the Production service" mid-chat | `:8195` went down | restart the server |
| Chat misreads you — renders without "go", ignores a change, treats a brief as chat | **Stage 1 intent** (the 4B's JSON decision) | `sudo docker logs studio-director` (the controller request/response); prompt is `build_controller_system()` in `director_intent.py` |
| Plan is off-topic, or a documentary invents a character | **Stage 2 planner** | reproduce with `run.py --backend synthetic` (§2); check `detect_format` + the treatment/plan prompts in `prompts.py` |
| Director hangs ~100 s then fails / empty reply | the 4B's **thinking ran away** (it must stay thinking-OFF) or the model is overloaded | `sudo docker logs studio-director` for `finish_reason=length` / empty `content` |
| Research never grounds the script | SearXNG down — it **fails open silently** | `curl :8088`; documentary briefs only |
| Wrong segment count / unexpected long clip | duration mis-parse (e.g. "1980s" → 1980 s — see [Challenges](#current-challenges--known-limitations)) | the brief's wording; `_target_seconds` |

---

## See also

- [`README.md`](README.md) — the AI Studio overview + the 12-lane table
- [`image.md`](image.md) · [`video.md`](video.md) · [`audio.md`](audio.md) — per-modality lane deep-dives
- [`services/studio/director/AGENTS.md`](../../services/studio/director/AGENTS.md) — the editable Production-director persona (the conversation spec baked into the pipe)
- [`services/studio/director_intent.py`](../../services/studio/director_intent.py) — the pure, tested intent classifiers
- [`services/studio/production/`](../../services/studio/production/) — the `:8195` planning server ([`planner.py`](../../services/studio/production/planner.py) · [`critic.py`](../../services/studio/production/critic.py) · [`research.py`](../../services/studio/production/research.py) · [`prompts.py`](../../services/studio/production/prompts.py))
