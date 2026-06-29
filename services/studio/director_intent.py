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


def _norm(t):
    return (t or "").strip().lower().strip(" .!?")


def is_confirm(t):
    return _norm(t) in CONFIRM


def is_greeting(t):
    return _norm(t) in GREET


def is_generation_request(t):
    """A turn that asks to CREATE a film/clip — the strongest brief signal."""
    return bool(_GEN_RE.search(t or ""))


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
    if is_generation_request(t):
        return True
    return not (is_confirm(t) or is_greeting(t) or pure_override or is_question(t))


def pick_brief(turns, pure_override_of=None):
    """First turn that reads as a film brief, else ''. `pure_override_of(t)->bool` optional."""
    po = pure_override_of or (lambda _t: False)
    for t in turns:
        if is_brief_candidate(t, po(t)):
            return t
    return ""
