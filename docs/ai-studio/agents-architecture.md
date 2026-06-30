# AI Studio — Agent Architecture

How the AI Studio "director" makes decisions: the one 4B model behind every lane, the **two
distinct agent shapes** it runs in, the Production Director's decision flow, and exactly how the
agent differs lane-by-lane.

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

There is **no separate intent classifier**. Each lane makes **one** 4B call (`_enhance`,
`build_studio_pipe.py:432`) whose system prompt does double duty — it returns either a crafted
render prompt **or** a `CHAT:`-prefixed reply, and `_chat_gate` (`:463`) branches on the prefix:

```
user turn (only the latest, + prior render's prompt for refinement)
        │
        ▼
   _enhance → 4B, lane-specific persona, think-OFF
        │
        │  system prompt says: "if NOT a real <sound/image/video> request —
        │  a greeting, small talk, a question, or too vague — reply exactly
        │  `CHAT: <one friendly sentence>` and nothing else."
        │
   ┌────┴───────────────────────────────┐
   ▼                                     ▼
 "CHAT: <nudge>"                  "<a crafted render prompt>"
   │ _chat_gate strips "CHAT:"          │ _chat_gate → None
   ▼                                    ▼
 show reply, NO render            submit ONE ComfyUI/voice job → media link
```

- **Greeting / question / too-vague** → `CHAT: …` → shown as chat, **no GPU spend**.
- **A real request** → crafted into a full prompt → rendered immediately (no confirm step).
- **A follow-up** ("add reverb", "at night") → `_prior_spec(body)` pulls the previous render's
  prompt from history; the 4B applies *only* the requested change (`:446`). This is the lanes'
  entire "memory".

The chat reply is intentionally a **one-line nudge back to the task** — it does not deeply answer
questions (e.g. "Chroma or HiDream?" gets "tell me what to create", not a comparison).

---

## Shape B — the Production Director

Two stages: a **conversation controller** in the pipe (decides what to do with the turn), then —
only on **go** — a **planning pipeline** on the `:8195` server.

### Stage 1 — conversation controller (`pipe()`, `:794`)

The **LLM is the driver; keyword heuristics are the floor; confirm is a safety latch.**

```
user turn (model contains "production")
        │
        ▼  Task-prompt guard: OWUI internal "### Task:" / autocomplete → return ""   :784
        ▼  gather ALL user turns; last = newest; none → help line
        │
        ▼  FLOOR (deterministic fallback, build_studio_pipe.py)                       :853
        │    brief_kw   = pick_brief(users)        first brief-shaped turn
        │    ov         = Σ _overrides(turns)      explicit toggles typed (lane/music/secs/research)
        │    confirm_kw = is_confirm(last)         exact / compound "ok do it"
        │
        ▼  CONTROLLER — the 4B reads the WHOLE convo → ONE JSON decision  _classify   :873
        │    {intent, brief, stack_patch, confirm, reply}
        │      LLM up  → brief = LLM's brief   (LLM OWNS extraction)
        │                ov   += LLM lanes     (typed toggles still win)
        │                confirmed = confirm_kw OR (LLM.confirm AND go-word) ◄── latch :886
        │      LLM down → fall back to brief_kw / confirm_kw / keyword intent
        │
        ▼  resolve stack (ov → ⚙️ valves → auto): video·keyframe·continuity·music·    :898
        │    narration · secs→shots(⌈/5⌉, ≤24) · est render · research · is_documentary
        │
        ▼  decide_action(brief, confirmed, intent) → ONE action       director_intent.py:262
        │
   ┌────┼────────────────┬──────────────────┬─────────────────────┐
   ▼    ▼                ▼                  ▼                     ▼
 CHAT  NEED_BRIEF      PROPOSAL            BUILD ────────────────► Stage 2
 LLM   "give me a      Plan card:          POST /produce, poll
 reply  brief"         models·length·      job, stream progress,
 /chat                 est·🔎research       return the MP4         :975
```

`decide_action` truth table (`director_intent.py:262`):

| condition | action |
|---|---|
| confirmed **and** brief | **build** |
| confirmed **and** no brief | **need_brief** |
| no brief | **chat** |
| brief **and** intent ∈ {question, smalltalk, cancel} | **chat** |
| brief **and** intent ∈ {brief, revise, stack} | **proposal** |

### Stage 2 — planning pipeline (`plan_from_brief`, `:8195`, only after BUILD)

```
/produce(brief, stack, research)
   │
   ▼  detect_format(brief) ─────────────► documentary | narrative      prompts.py:33
   │  research_notes(brief)  IF (research AND documentary)              research.py:62
   │     └─ SearXNG :8088, top results → notes   (FAILS OPEN)
   ▼  TREATMENT   director_call temp 0.8, think-OFF   (creative)        planner.py:260
   ▼  PLAN JSON   director_call temp 0.4, think-OFF   (structured)      planner.py:272
   ▼  SCHEMA-REPAIR loop   parse + validate; feed exact error back  ×≤3 planner.py:290
   ▼  SEMANTIC CRITIC   deterministic checks + per-shot off_topic LLM,
   │                    temp 0, think-OFF, fail-open  ×1 round           critic.py:152
   ▼  ship valid plan → deterministic executor:
        keyframes → video → narration → music → assembly → one MP4
```

The temperature ladder (treatment **0.8** → plan **0.4** → critic **0.0**) *is* the
creativity/accuracy lever, since the model runs thinking-OFF throughout.

---

## How the agent differs per lane

| | 🎬 Production (Director) | 12 single-shot lanes |
|---|---|---|
| Agent shape | conversational plan-then-execute | one-shot craft-or-decline |
| Decides chat-vs-act via | structured controller `_classify` → JSON `{intent,…}` + `decide_action` | the `CHAT:` prefix convention inside `_enhance` |
| Chat reply quality | substantive — `_produce_reply` *answers* "what options do we have?" | one-line nudge back to "describe what to create" |
| Acts on a real request | proposes a plan, waits for **go** (confirm latch) | renders immediately on send |
| Multi-turn memory | full conversation (last 10 turns) | only `prior_spec` (the previous render's prompt) |
| Genre awareness | detects documentary vs narrative | none |
| Web research (SearXNG) | yes — grounds documentary scripts | no |
| Output | a multi-asset film (keyframes→video→narration→music→assembly) | one asset (image / clip / audio) |
| Confirm gate | yes — render is double-gated (`confirm_kw OR (LLM.confirm AND go-word)`) | no |
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
- **The render is double-gated (Production lane).** `confirmed = confirm_kw OR (LLM.confirm AND
  has_confirm_word(last))`. The irreversible build never fires on the LLM's say-so alone — a real
  go-word must be present.
- **The LLM owns open-ended brief extraction; keywords own the closed-vocab confirm.** The keyword
  floor is *not* consulted for the brief when the LLM is up (fixes "ok do it" becoming the film).
- **Documentary ≠ narrative**, decided up front, so a factual brief gets no invented protagonist.
- **Single-shot lanes refuse to render on chit-chat** (the `CHAT:` gate) — no GPU spend on "hi".

History of the Production-lane hardening (genre detection, the semantic critic, SearXNG research,
the LLM-first intent controller): club-3090 PRs **#519–#524**.

---

## See also

- [`README.md`](README.md) — the AI Studio overview + the 12-lane table
- [`image.md`](image.md) · [`video.md`](video.md) · [`audio.md`](audio.md) — per-modality lane deep-dives
- [`services/studio/director/AGENTS.md`](../../services/studio/director/AGENTS.md) — the editable Production-director persona (the conversation spec baked into the pipe)
- [`services/studio/director_intent.py`](../../services/studio/director_intent.py) — the pure, tested intent classifiers
- [`services/studio/production/`](../../services/studio/production/) — the `:8195` planning server ([`planner.py`](../../services/studio/production/planner.py) · [`critic.py`](../../services/studio/production/critic.py) · [`research.py`](../../services/studio/production/research.py) · [`prompts.py`](../../services/studio/production/prompts.py))
