"""The planner prompt pack — the 4B director's 'tight operating manual.'

Creative-within-bounds: the director is free in the film-making layer (angle, tone,
shot concepts, narration, mood) but the OUTPUT must be a valid ProductionPlanV1, and
if an idea exceeds a lane limit it must ADAPT the idea, never break the limit. The
executor/validator wins (the validator-repair loop enforces this).
"""
from __future__ import annotations

import re

from .registry import prompt_slice

# --- format detection: documentary/factual vs narrative/fiction --------------
# The brief's FORMAT decides whether the director invents a protagonist (narrative)
# or narrates a real subject with NO recurring person (documentary). Without this,
# the 4B latches onto the dominant CHARACTER BIBLE instruction and fictionalizes
# every brief — e.g. a "documentary on the history of Pakistan" became a character
# drama about an invented young architect named "Arif" (diagnosed 2026-06-29). Keep
# these signals tight: a false positive (a story mislabeled documentary) is cheap to
# correct ("make it a story"), and the strong signals below rarely occur in fiction.
_DOC_SIGNALS = re.compile(
    r"document\w*"                                   # documentary / documentory / documentry / documenting
    r"|\bhistory of\b|\bthe history\b|\bhistories of\b"
    r"|\bexplainer\b|\bexplain(?:ed|s|ing)?\b"
    r"|\beducational\b|\bbiograph\w*"
    r"|\bguide to\b|\boverview of\b|\btimeline of\b|\bfacts? about\b"
    r"|\btutorial\b|\bhow to\b|\bhow .{0,30}?works?\b|\bcase study\b",
    re.I,
)


def detect_format(brief: str) -> str:
    """Classify a brief as 'documentary' (non-fiction/factual) or 'narrative' (fiction)."""
    return "documentary" if _DOC_SIGNALS.search(brief or "") else "narrative"


def build_treatment_system(fmt: str = "narrative") -> str:
    """The stage-1 treatment system prompt, branched on format."""
    if fmt == "documentary":
        return (
            "You are a documentary director. Given a factual brief, write a SHORT treatment "
            "(4-6 sentences) for a NON-FICTION, observational/archival piece narrated by an "
            "unseen narrator. Do NOT invent a protagonist, a main character, or a personal "
            "story — there is no recurring person to follow. Cover the REAL subject faithfully "
            "(real places, events, eras, artifacts, figures relevant to the brief) in a sensible "
            "order. Give 3 concrete visual beats, each a real scene that could be ONE ~5-second "
            "shot. Plain prose only — no JSON, no markdown, no preamble."
        )
    return (
        "You are a production director. Given a brief, write a SHORT creative treatment "
        "(4-6 sentences): the angle, the tone, the structure, and 3 concrete visual beats "
        "— each something that could be ONE ~5-second shot. Be specific and cinematic. "
        "Plain prose only — no JSON, no markdown, no preamble."
    )


# Back-compat: the narrative treatment system prompt as a module constant.
TREATMENT_SYS = build_treatment_system("narrative")


def build_plan_system(reg: dict, n_shots: int = 3, video_lane: str = "wan",
                      fmt: str = "narrative") -> str:
    documentary = fmt == "documentary"

    if documentary:
        format_banner = (
            "FORMAT: This is a DOCUMENTARY — NON-FICTION, narrated by an unseen narrator.\n"
            "- Do NOT invent a protagonist, a named character, or a personal story. `characters` MUST be an empty list [], and NO shot may list `characters`.\n"
            "- Every shot is a real, ON-TOPIC observational or archival image of the ACTUAL subject (real places, events, eras, artifacts, or figures relevant to the brief). No recurring fictional person to follow.\n"
            "- narration is FACTUAL: real names, places, dates, and events — NOT poetic mood lines.\n"
            "- Stay ON TOPIC for the ENTIRE piece: every shot must clearly serve the brief's subject from first to last.\n\n"
        )
        characters_field = '  "characters": [],\n'
        shot_characters = ""
        char_guidance = (
            "- NO CHARACTERS: this is a documentary, so `characters` MUST be [] and no shot lists `characters`. "
            "Never invent a person to follow — depict the real subject."
        )
        intent_guidance = (
            "- prompt_intent: vivid, cinematic prose for ONE real, ON-TOPIC shot (imagery + action + camera) — "
            "a true place, event, era, or artifact relevant to the brief. Never a fictional protagonist."
        )
        narr_guidance = (
            "- narration: ONE FACTUAL spoken sentence per shot — real names, places, dates, events — in ENGLISH "
            "ONLY. Keep it short (~2.5 words/second, so a 5 s shot is AT MOST 10 words)."
        )
    else:
        format_banner = ""
        characters_field = (
            '  "characters": [\n'
            '    {"id": "det", "name": "Detective Marlowe", "role": "protagonist",\n'
            '      "description": "<canonical, reusable visual appearance: build, age, face, hair>",\n'
            '      "wardrobe": "<signature clothing>", "props": "<signature props>",\n'
            '      "negative": "<what NOT to render — drift to avoid>", "seed": 111}\n'
            "  ],\n"
        )
        shot_characters = '\n      "characters": ["det"],'
        char_guidance = (
            "- CHARACTER BIBLE: define every recurring on-screen character (a person/creature appearing in MORE "
            "THAN ONE shot) ONCE in `characters`, with a CANONICAL, detailed, REUSABLE appearance — build, age, "
            "face, hair, `wardrobe`, signature `props`, and a `negative` note of drift to avoid. Then in each shot, "
            "list the ids of the characters present via `characters` — do NOT restate their looks in prompt_intent "
            "(the executor injects the canonical block automatically, so the SAME character stays consistent). Give "
            "each character a stable `seed`. If the piece has NO recurring character, leave `characters` empty and "
            "omit the per-shot `characters`."
        )
        intent_guidance = (
            "- prompt_intent: vivid, cinematic prose for the video model (one shot's worth of imagery + action + "
            "camera), referring to characters by NAME but not re-describing their fixed appearance."
        )
        narr_guidance = (
            "- narration: ONE spoken sentence the viewer hears over that shot, in ENGLISH ONLY (no other-language "
            "words). Keep it short enough to fit (~2.5 words per second), so a 5 s shot is AT MOST 10 words."
        )

    return f"""You are the production director for an automated video studio. Turn the brief and treatment into ONE valid ProductionPlanV1 JSON object. OUTPUT ONLY THE JSON — no prose, no markdown fences, no comments.

{format_banner}{prompt_slice(reg, video_lane=video_lane)}

The video lane is PINNED by the operator to '{video_lane}'. Output project.video_lane = "{video_lane}" and EVERY shot's lane = "{video_lane}". Do NOT choose a different video lane.

ProductionPlanV1 shape:
{{
  "project": {{"title": str, "tone": str, "target_seconds": int, "video_lane": "{video_lane}"}},
{characters_field}  "shots": [
    {{"id": "s1", "lane": "{video_lane}", "mode": "t2v", "target_seconds": 5,
      "prompt_intent": "<vivid cinematic prose describing the shot>",{shot_characters}
      "narration": "<ONE short spoken sentence heard over this shot>", "seed": 12345}}
  ],
  "music": {{"lane": "ace-step", "tags": "<comma-separated mood tags>", "lyrics": "[instrumental]", "seed": 7}},
  "timeline": [
    {{"clip": "s1", "narration_offset": 0.0, "music_level_db": -18, "transition_in": "dissolve"}}
  ]
}}

Guidance (be inventive WITHIN these bounds):
- Aim for {n_shots} shots. Each shot is ONE video window (<= 5 s), so keep target_seconds 4-5.
{char_guidance}
{intent_guidance}
{narr_guidance}
- If an idea needs more than ~5 s, SPLIT it into two shots — never exceed the limit.
- Every shot needs a UNIQUE id. The timeline lists EVERY shot exactly once, in play order.
- OUTPUT ONLY THE JSON OBJECT."""


def repair_user(raw: str, error: str) -> str:
    return (
        "Your previous output was not a valid plan.\n"
        f"ERROR: {error}\n\n"
        f"PREVIOUS OUTPUT:\n{raw}\n\n"
        "Output a corrected ProductionPlanV1 JSON object ONLY (no prose, no fences) that fixes that error."
    )


def critic_repair_user(raw: str, issues: list) -> str:
    """Re-ask the plan model to fix CONTENT problems the critic found (the plan is valid JSON)."""
    bullets = "\n".join(f"- {i}" for i in issues)
    return (
        "Your plan is valid JSON but has these CONTENT problems:\n"
        f"{bullets}\n\n"
        f"PREVIOUS PLAN:\n{raw}\n\n"
        "Output a corrected ProductionPlanV1 JSON object ONLY (no prose, no fences) that fixes these "
        "problems. Keep the SAME format, the SAME pinned video lane, and the SAME number of shots."
    )
