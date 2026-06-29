"""Semantic plan critic — 'is this a GOOD plan?', not just 'is it valid JSON?'.

Runs AFTER schema.validate() and BEFORE render, inside the planner's repair loop (Codex F5).
'Valid JSON' is not 'good film plan': the schema can't tell that a documentary wandered
off-topic, that shots repeat, or that the narration won't fit. Two layers:

  - DETERMINISTIC checks (no LLM, always run, reliable): narration that won't fit its shot
    window, near-duplicate shots, a documentary that smuggled in fictional characters.
  - an LLM SEMANTIC critique (conservative, thinking-ON, FAIL-OPEN): does every shot serve the
    brief? does the plan drift off-topic partway through? — the general form of the "documentary
    became a character drama about Arif" bug (the genre detector fixed one special case; this
    catches the class). It judges with reasoning on, and is parsed defensively.

critique() returns a list of blocking issue strings — empty = ship. It FAILS OPEN: a flaky or
unreachable critic must NEVER block a render, so any LLM error degrades to deterministic-only.
"""
from __future__ import annotations

import json

from .util import strip_reasoning

WORDS_PER_SEC = 2.5            # Kokoro ≈ 2.5 wps; matches the planner's narration budget
NARRATION_SLACK_WORDS = 3     # a little headroom before we flag
DUP_JACCARD = 0.85            # near-identical prompt_intent → a lazy duplicate shot


def _tokens(s):
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in (s or "")).split()
            if len(w) > 2}


def deterministic_issues(plan, fmt="narrative"):
    """Cheap, reliable checks that need no model. Returns a list of issue strings."""
    issues = []
    shots = list(plan.shots)

    # 1) narration that won't fit its shot window
    for s in shots:
        narr = (s.narration or "").strip()
        if not narr:
            continue
        budget = max(1, int(s.target_seconds * WORDS_PER_SEC) + NARRATION_SLACK_WORDS)
        wc = len(narr.split())
        if wc > budget:
            issues.append(f"shot {s.id}: narration is {wc} words but only ~{budget} fit in "
                          f"{s.target_seconds:.0f}s — shorten it to one short sentence")

    # 2) near-duplicate shots (lazy repetition)
    for i in range(len(shots)):
        ti = _tokens(shots[i].prompt_intent)
        if not ti:
            continue
        for j in range(i + 1, len(shots)):
            tj = _tokens(shots[j].prompt_intent)
            if tj and len(ti & tj) / len(ti | tj) >= DUP_JACCARD:
                issues.append(f"shots {shots[i].id} and {shots[j].id} are near-duplicates — "
                              f"make each shot visually distinct")
                break

    # 3) a documentary must carry no invented characters (belt-and-suspenders; normalize strips them)
    if fmt == "documentary" and getattr(plan, "characters", None):
        names = ", ".join(c.name for c in plan.characters)
        issues.append(f"this is a DOCUMENTARY but it invents characters ({names}) — "
                      f"a documentary has no fictional protagonist")

    return issues


# Per-shot RELEVANCE framing — a live A/B (2026-06-29) showed the 4B is far better at the bounded
# "is shot N about the brief's subject?" judgement than a holistic "is this a good plan?" verdict
# (which it rubber-stamped, missing obvious drift). It also showed thinking-ON returns EMPTY output
# on this model (the <think> budget eats the answer) — so the critic runs thinking-OFF.
CRITIC_SYS = (
    "You review a film's shot list against its BRIEF and FORMAT. Output ONLY JSON:\n"
    '{"off_topic": [shot numbers that do NOT serve the brief], "issues": ["other clear problems"]}\n'
    "Judge EACH numbered shot:\n"
    "- off_topic: list a shot's number if it does NOT depict the BRIEF's real subject. For a "
    "DOCUMENTARY / factual / historical brief, a generic modern person, an INVENTED character, or "
    "an unrelated everyday scene (office, park, shopping, a person's daily life) is OFF-TOPIC. For a "
    "NARRATIVE brief, a shot unrelated to the story is off-topic.\n"
    "- issues: ONLY other clear, important problems — the plan repeats the same shot, or a "
    "documentary uses vague poetic narration instead of factual narration (real names/places/dates).\n"
    "Do NOT nitpick style, word choice, camera angles, or creative taste. If every shot serves the "
    "brief and the plan is coherent, return {\"off_topic\": [], \"issues\": []}. Output ONLY the JSON."
)


def _plan_digest(plan, brief, fmt):
    lines = [f"BRIEF: {brief}", f"FORMAT: {fmt}", "SHOTS:"]
    for i, s in enumerate(plan.ordered_shots(), 1):
        narr = (s.narration or "").strip()
        lines.append(f"{i}. {s.prompt_intent.strip()}" + (f"  [narration: {narr}]" if narr else ""))
    return "\n".join(lines)


def parse_critique(raw):
    """Parse the LLM critique into {ok, issues}, or None on failure (after stripping reasoning)."""
    t = strip_reasoning(raw)
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
                    d = json.loads(t[i:j + 1])
                    return d if isinstance(d, dict) else None
                except Exception:
                    return None
    return None


def normalize_critique(parsed):
    """A parsed critique → a list of blocking issue strings ([] = on-topic / not actionable).

    Accepts the per-shot relevance shape {"off_topic": [shot numbers], "issues": [...]}.
    """
    if not isinstance(parsed, dict):
        return []
    out = []
    off = parsed.get("off_topic")
    if isinstance(off, list):
        for n in off:
            n = str(n).strip()
            if n:
                out.append(f"shot {n} is off-topic — replace it with one that depicts the brief's subject")
    issues = parsed.get("issues")
    if isinstance(issues, list):
        out += [str(x).strip() for x in issues if str(x).strip()]
    return out


def llm_issues(plan, brief, fmt, call, *, prov=None):
    """The semantic critique. FAILS OPEN: any error → [] (deterministic checks still apply).

    Runs thinking-OFF: a 2026-06-29 A/B showed thinking-on returns EMPTY output on the 4B
    director (the <think> budget eats the JSON), while thinking-off + the per-shot framing
    catches drift reliably. extract/parse still strip <think> defensively (F7)."""
    if call is None:
        return []
    user = _plan_digest(plan, brief, fmt) + "\n\nReview this plan. Output ONLY the JSON."
    try:
        # temperature 0 (greedy): a 2026-06-29 A/B showed temp 0.2 was noisy run-to-run (it
        # missed drift AND false-flagged a clean plan ~1/3 of the time); greedy decoding is
        # reliable — drift → every off-topic shot flagged, a clean plan → no issues.
        raw = call([{"role": "system", "content": CRITIC_SYS}, {"role": "user", "content": user}],
                   max_tokens=400, temperature=0.0, enable_thinking=False)
    except Exception:
        return []
    if prov is not None:
        prov.append(("critic", CRITIC_SYS, user, raw))
    return normalize_critique(parse_critique(raw))


def critique(plan, brief, fmt="narrative", *, call=None, prov=None):
    """All blocking issues (deterministic + LLM). Empty list = ship. Fails open on LLM error."""
    return deterministic_issues(plan, fmt) + llm_issues(plan, brief, fmt, call, prov=prov)
