"""Pure conversation-intent classifiers for the 🎬 Studio Production lane.

This is the SOURCE OF TRUTH for how the director reads a user turn (greeting? confirm?
question? a film brief?). It is a standalone, stdlib-only module so it can be unit-tested
offline — and `build_studio_pipe.py` INJECTS its source verbatim into the generated
`studio_pipe.py` (the OWUI pipe runs in-container with no repo access, the same reason the
director persona + workflows are baked in). Edit here; the bake keeps the deployed pipe and
these tests from drifting.

Keep it PURE: stdlib only, no `self`, no valves, no network. Anything needing valve state
(e.g. clamping a duration to `max_seconds`) stays in the pipe and is passed in as an arg.

The bug this fixes (Codex F1, 2026-06-29): the old gate classified "can you make a 30s noir
short?" as a question and dropped it, so no brief was ever captured. A GENERATION REQUEST now
takes precedence over question-shape, so a creation ask phrased as a question is still a brief.
"""
import json
import re

# Exact-match small-talk + confirmation vocabularies (matched on a normalized turn).
GREET = ("hi", "hello", "hey", "yo", "sup", "hiya", "hello there", "hola", "thanks",
         "thank you", "cool", "nice", "test", "testing", "ping", "help", "?")
CONFIRM = ("go", "yes", "y", "start", "render", "render it", "proceed", "do it", "ok",
           "okay", "ok go", "yep", "yeah", "sure", "build", "build it", "make it",
           "let's go", "go ahead", "confirm", "\U0001F44D")
# Leading interrogatives → treat as a question (chat), UNLESS it's a generation request.
QWORDS = ("what", "how", "why", "which", "can", "could", "should", "do", "does",
          "is", "are", "who", "when", "where", "will", "would", "tell")

# A CREATION ask — a verb of making + a film/media noun close by — even when phrased as a
# question ("can you make a 30s noir short?") or a polite request ("give me a 1-min doc").
_GEN_VERB = (r"(?:make|create|render|build|generate|produce|do|design|put\s+together|whip\s+up"
             r"|give\s+me|i\s+want|i'?d\s+like|i\s+need|let'?s\s+(?:make|do|create|build))")
_GEN_NOUN = (r"(?:films?|movies?|shorts?|videos?|clips?|documentar\w*|docu\w*|trailers?|montages?"
             r"|reels?|animations?|teasers?|promos?|adverts?|ads?|scenes?|stor(?:y|ies)|pieces?)")
_GEN_RE = re.compile(_GEN_VERB + r"\b.{0,40}?\b" + _GEN_NOUN, re.I)


# Tokens that, on their own, only ever mean "yes, proceed" — so a SHORT turn made up entirely of
# them is a confirm even if the exact phrase isn't in CONFIRM (e.g. "ok do it", "yes go ahead").
# This stops a compound confirm from leaking into brief detection (the "ok do it" became the film
# brief bug, 2026-06-29).
_CONFIRM_TOKENS = {"go", "yes", "y", "ya", "yeah", "yep", "yup", "sure", "ok", "okay", "k",
                   "do", "it", "that", "start", "render", "build", "proceed", "confirm",
                   "please", "now", "lets", "let's", "ahead", "on", "make", "run", "begin"}


def _norm(t):
    return (t or "").strip().lower().strip(" .!?")


def is_confirm(t):
    norm = _norm(t)
    if norm in CONFIRM:
        return True
    toks = [w.strip(".,!?") for w in norm.split() if w]
    return bool(toks) and len(toks) <= 4 and all(w in _CONFIRM_TOKENS for w in toks)


def is_greeting(t):
    return _norm(t) in GREET


def is_generation_request(t):
    """A turn that asks to CREATE a film/clip — the strongest brief signal."""
    return bool(_GEN_RE.search(t or ""))


# Documentary/factual signal — MIRRORS production/prompts.py detect_format (guarded by
# test_looks_documentary_matches_detect_format). Used in the pipe to OFFER web research on a
# documentary brief (the pipe can't import the server-side detect_format).
_DOC_SIGNALS = re.compile(
    r"document\w*"
    r"|\bhistory of\b|\bthe history\b|\bhistories of\b"
    r"|\bexplainer\b|\bexplain(?:ed|s|ing)?\b"
    r"|\beducational\b|\bbiograph\w*"
    r"|\bguide to\b|\boverview of\b|\btimeline of\b|\bfacts? about\b"
    r"|\btutorial\b|\bhow to\b|\bhow .{0,30}?works?\b|\bcase study\b",
    re.I,
)


def looks_documentary(brief):
    """True if a brief reads as documentary/factual (→ offer web research)."""
    return bool(_DOC_SIGNALS.search(brief or ""))


def is_question(t):
    """Question-shaped (leading interrogative or trailing '?'). Pure to its name —
    callers that want creation-asks excluded use is_brief_candidate (gen wins there)."""
    tl = (t or "").strip().lower()
    if not tl:
        return False
    first = tl.split()[:1]
    return tl.endswith("?") or (bool(first) and first[0] in QWORDS)


def is_brief_candidate(t, pure_override=False):
    """Should this turn be treated as the film brief?

    A generation request ALWAYS qualifies (even if question-shaped) — that's the F1 fix.
    Otherwise it's a brief only if it isn't a greeting / confirm / pure stack-tweak / question.
    `pure_override` is supplied by the caller (it needs valve/seconds context to compute).
    """
    # A creation ask OR a named factual subject ("dig history of pakistan?", "the history of jazz")
    # is a brief even when question-shaped — the old gate dropped "dig history of pakistan?" as a
    # question and the confirm phrase after it became the brief (2026-06-29).
    if is_generation_request(t) or looks_documentary(t):
        return True
    return not (is_confirm(t) or is_greeting(t) or pure_override or is_question(t))


def pick_brief(turns, pure_override_of=None):
    """First turn that reads as a film brief, else ''. `pure_override_of(t)->bool` optional."""
    po = pure_override_of or (lambda _t: False)
    for t in turns:
        if is_brief_candidate(t, po(t)):
            return t
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Batch 2 — the structured LLM intent controller (Codex F1/F2/F3).
#
# The keyword classifiers above are the deterministic FLOOR/FALLBACK. The controller
# asks the 4B to read the whole conversation and emit ONE small JSON decision —
# {intent, brief, stack_patch, confirm, reply} — which the pipe merges ON TOP of the
# floor (LLM brief/stack win when present; the floor fills the gaps). The pure parse +
# validation below is table-tested; the HTTP call + fallback orchestration live in the
# pipe (`_classify`). A render only starts when the decision says confirm AND the latest
# turn carries a real confirm word (has_confirm_word) — the irreversible action is never
# triggered on the LLM's say-so alone.
# ─────────────────────────────────────────────────────────────────────────────

# Valid slot values — MUST mirror the wired lanes in production/stack.py (guarded by
# test_controller_lane_constants_match_stack). The pipe can't import stack.py (it runs
# in the OWUI container), so these are duplicated here, same as the pipe's own lane maps.
VIDEO_LANES_VALID = ("wan", "ltx", "sulphur", "10eros")
KEYFRAME_LANES_VALID = ("chroma", "zimage", "krea", "hidream")
CONTINUITY_VALID = ("storyboard", "hero", "chain", "none")
INTENTS = ("brief", "revise", "stack", "question", "confirm", "smalltalk", "cancel")

# A turn that signals "start rendering NOW" — used to CORROBORATE the LLM's confirm flag
# so an expensive render never fires on a hallucinated confirm. Looser than is_confirm
# (exact-match) so compound asks like "go with LTX" still count.
_CONFIRM_WORD_RE = re.compile(r"\b(go|yes|yeah|yep|sure|start|render|build|proceed|confirm|okay?)\b", re.I)
_CONFIRM_PHRASES = ("do it", "go ahead", "let's go", "lets go", "ok go", "render it",
                    "build it", "make it", "ship it", "send it")


def has_confirm_word(t):
    """Does this turn carry an explicit 'start now' signal? (corroborates the LLM confirm)."""
    tl = (t or "").lower()
    return bool(_CONFIRM_WORD_RE.search(tl)) or any(p in tl for p in _CONFIRM_PHRASES)


def build_controller_system():
    """The 4B's intent-parser system prompt — read the conversation, emit ONE JSON decision."""
    return (
        "You are the intent parser for a film studio's chat assistant. Read the conversation and "
        "output ONE JSON object describing what the user wants RIGHT NOW. OUTPUT ONLY THE JSON — no "
        "prose, no markdown fences.\n\n"
        "{\n"
        '  "intent": "brief|revise|stack|question|confirm|smalltalk|cancel",\n'
        '  "brief": "<the ONE-LINE film the user currently wants, or empty string>",\n'
        '  "stack_patch": {"video_lane": "", "keyframe_lane": "", "continuity": "", "music": true, "narration": true, "seconds": 0},\n'
        '  "confirm": false,\n'
        '  "reply": "<a warm, concise 1-2 sentence reply to the user>"\n'
        "}\n\n"
        "Rules:\n"
        "- brief: the SUBJECT the user wants a film about — INFER it even from indirect phrasing: a topic "
        "they asked you to research / dig / search (\"dig history of pakistan\" -> \"the history of pakistan\"), "
        "a bare topic (\"history of jazz\"), or a changed subject (\"actually make it a bookstore promo\" -> use "
        "the NEW one). If they named ANY subject to film or research, THAT subject is the brief. Leave it empty "
        "ONLY if they named no topic at all (pure greetings, or questions about what you can do).\n"
        "- stack_patch: include ONLY settings the user explicitly asked for; OMIT keys they didn't mention. "
        "video_lane ∈ {wan, ltx, sulphur, 10eros}; keyframe_lane ∈ {chroma, zimage, krea, hidream}; "
        "continuity ∈ {storyboard, hero, chain, none}; music/narration are booleans; seconds is the requested length.\n"
        "- confirm: true ONLY if the user is telling you to START rendering now (\"go\", \"ok do it\", "
        "\"yes do it\", \"go with ltx\", \"render it\"). A change request like \"make it 30 seconds\" is NOT a confirm.\n"
        "- intent: brief = first/main film description; revise = changing the film; stack = changing a "
        "setting/model; question = asking something; confirm = start now; smalltalk = greeting/chit-chat; "
        "cancel = stop/never mind.\n"
        "- reply: what to say back, warm and concise. For a question, ANSWER it truthfully. You CAN research "
        "real facts on the web (SearXNG) to ground a DOCUMENTARY you're making — so \"can you search the web?\" "
        "is YES, for grounding a documentary — but you canNOT browse arbitrary pages or fetch live info to chat "
        "about. Never claim rendering or searching has already started (only \"go\" starts a render).\n"
        "Output ONLY the JSON object."
    )


def _extract_json(text):
    """First balanced {...} object out of the model's reply, or None."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t[:4].lower() == "json" else t
    i = t.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    return None
    return None


def parse_controller_json(raw):
    """Parse the controller's reply into a dict, or None (→ caller falls back to heuristics)."""
    d = _extract_json(raw)
    return d if isinstance(d, dict) else None


def normalize_decision(parsed):
    """Validate + clean a parsed controller dict → a usable decision, or None.

    Drops unknown lanes (never silently coerced — the heuristic floor fills them), coerces
    types, and clamps `seconds`. None only when there's nothing usable to act on.
    """
    if not isinstance(parsed, dict):
        return None
    brief = str(parsed.get("brief") or "").strip()

    # The controller owns ONLY the explicit MODEL/continuity picks. music/narration/seconds are
    # deliberately NOT taken from the LLM — the 4B over-reaches on them (e.g. it stripped music +
    # narration from a "bookstore promo" nobody asked to silence). Those stay keyword-driven in the
    # pipe (`_overrides`: "no music" / "30 seconds" are reliable + explicit). Unknown lanes are
    # DROPPED, never coerced — the keyword floor fills the gap.
    raw_patch = parsed.get("stack_patch") or {}
    patch = {}
    if isinstance(raw_patch, dict):
        v = str(raw_patch.get("video_lane") or "").strip().lower()
        if v in VIDEO_LANES_VALID:
            patch["video_lane"] = v
        k = str(raw_patch.get("keyframe_lane") or "").strip().lower()
        if k in KEYFRAME_LANES_VALID:
            patch["keyframe_lane"] = k
        c = str(raw_patch.get("continuity") or "").strip().lower()
        if c in CONTINUITY_VALID:
            patch["continuity"] = c

    intent = str(parsed.get("intent") or "").strip().lower()
    if intent not in INTENTS:
        intent = "brief" if brief else "smalltalk"
    reply = str(parsed.get("reply") or "").strip() or None
    return {
        "brief": brief,
        "stack_patch": patch,
        "confirm": bool(parsed.get("confirm")),
        "intent": intent,
        "reply": reply,
    }


def decide_action(brief, confirmed, intent):
    """Route a resolved turn to ONE pipe action. Pure — the pipe handles the rendering.

      build      → a brief is set AND the user confirmed (corroborated) → start the render
      need_brief → the user confirmed but there's nothing to build yet
      proposal   → a brief is set and the plan is new/changed (brief/revise/stack) → show the card
      chat       → answer the user naturally (no brief yet, or a question/smalltalk/cancel)
    """
    if confirmed:
        return "build" if brief else "need_brief"
    if not brief:
        return "chat"
    return "chat" if intent in ("question", "smalltalk", "cancel") else "proposal"
