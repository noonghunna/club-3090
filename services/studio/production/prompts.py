"""The planner prompt pack — the 4B director's 'tight operating manual.'

Creative-within-bounds: the director is free in the film-making layer (angle, tone,
shot concepts, narration, mood) but the OUTPUT must be a valid ProductionPlanV1, and
if an idea exceeds a lane limit it must ADAPT the idea, never break the limit. The
executor/validator wins (the validator-repair loop enforces this).
"""
from __future__ import annotations

from .registry import prompt_slice

TREATMENT_SYS = (
    "You are a production director. Given a brief, write a SHORT creative treatment "
    "(4-6 sentences): the angle, the tone, the structure, and 3 concrete visual beats "
    "— each something that could be ONE ~5-second shot. Be specific and cinematic. "
    "Plain prose only — no JSON, no markdown, no preamble."
)


def build_plan_system(reg: dict, n_shots: int = 3) -> str:
    return f"""You are the production director for an automated video studio. Turn the brief and treatment into ONE valid ProductionPlanV1 JSON object. OUTPUT ONLY THE JSON — no prose, no markdown fences, no comments.

{prompt_slice(reg)}

ProductionPlanV1 shape:
{{
  "project": {{"title": str, "tone": str, "target_seconds": int, "video_lane": "wan"}},
  "shots": [
    {{"id": "s1", "lane": "wan", "mode": "t2v", "target_seconds": 5,
      "prompt_intent": "<vivid cinematic prose describing the shot>",
      "narration": "<ONE short spoken sentence heard over this shot>", "seed": 12345}}
  ],
  "music": {{"lane": "ace-step", "tags": "<comma-separated mood tags>", "lyrics": "[instrumental]", "seed": 7}},
  "timeline": [
    {{"clip": "s1", "narration_offset": 0.0, "music_level_db": -18, "transition_in": "dissolve"}}
  ]
}}

Guidance (be inventive WITHIN these bounds):
- Aim for {n_shots} shots. Each shot is ONE Wan window (<= 5 s), so keep target_seconds 4-5.
- prompt_intent: vivid, cinematic prose for the video model (one shot's worth of imagery).
- narration: ONE spoken sentence the viewer hears over that shot, in ENGLISH ONLY (no other-language words). Keep it short enough to fit (~2.5 words per second), so a 5 s shot is AT MOST 10 words.
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
