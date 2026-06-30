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

**Keeping a film coherent** is the Director's real job beyond a single clip:

- **Character Bible** (stories only) — recurring characters are described **once** with a canonical
  look + a stable seed, then referenced by name in each shot, so the same character looks the same
  across shots ([#502](https://github.com/noonghunna/club-3090/pull/502)). Documentaries keep this
  **empty** — they narrate a real subject, no protagonist.
- **Continuity mode** (⚙️ valve) — how shots are visually linked: **storyboard** (per-shot keyframes
  + a shared style bible, *default*) · **hero** (one shared keyframe) · **chain** (each shot
  continues from the previous frame) · **none** (independent shots).

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

## See also

- [`README.md`](README.md) — the AI Studio overview + the 12-lane table
- [`image.md`](image.md) · [`video.md`](video.md) · [`audio.md`](audio.md) — per-modality lane deep-dives
- [`services/studio/director/AGENTS.md`](../../services/studio/director/AGENTS.md) — the editable Production-director persona (the conversation spec baked into the pipe)
- [`services/studio/director_intent.py`](../../services/studio/director_intent.py) — the pure, tested intent classifiers
- [`services/studio/production/`](../../services/studio/production/) — the `:8195` planning server ([`planner.py`](../../services/studio/production/planner.py) · [`critic.py`](../../services/studio/production/critic.py) · [`research.py`](../../services/studio/production/research.py) · [`prompts.py`](../../services/studio/production/prompts.py))
