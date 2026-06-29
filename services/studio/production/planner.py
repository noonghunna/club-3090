"""The 4B planner — brief -> validated ProductionPlanV1.

Two-stage, creative-within-bounds: (1) a free-form creative treatment, then
(2) a structured plan, then a **validator-repair loop** — parse the director's JSON,
`schema.validate()` it, and on any failure feed the exact error back asking for a
corrected JSON (up to `max_repairs`). The repair loop is the backbone (we do NOT
rely on server-side constrained output, which is flaky on this build).

The director call is injectable (`llm=`) so the whole planner — prompt construction,
normalization, and the repair loop — is unit-testable offline with a stub LLM, no
model required (mirrors the synthetic lane backend).
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request

from . import config, critic
from .manifest import Artifact
from .prompts import (build_plan_system, build_treatment_system, critic_repair_user,
                      detect_format, repair_user)
from .schema import PlanError, ProductionPlanV1
from .util import sha256_text, strip_reasoning


class PlannerError(RuntimeError):
    pass


# -- sizing: derive the shot count from the brief's requested duration --------
SECONDS_PER_SHOT = 5.0   # Wan native window ≈ 81 frames @ 16 fps ≈ 5 s
MAX_SHOTS = 24           # cap auto-derivation (~120 s) so a stray "10 minute" can't queue hours
DEFAULT_SHOTS = 4        # when the brief states no duration


def parse_duration_seconds(text: str):
    """Best-effort 'how long' from a brief: '1 minute' -> 60.0, '45s' -> 45.0. None if absent."""
    t = text or ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", t, re.I)
    if m:
        return float(m.group(1)) * 60.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", t, re.I)
    if m:
        return float(m.group(1))
    return None


def derive_shots(brief: str, *, default: int = DEFAULT_SHOTS):
    """Map a brief's requested duration to a shot count (~5 s/shot, capped at MAX_SHOTS).

    The agent SIZES the film from the request instead of a fixed test count —
    'a 1 minute video' -> ~12 shots. Returns (shots, requested_seconds | None).
    """
    secs = parse_duration_seconds(brief)
    if not secs or secs <= 0:
        return default, None
    shots = max(1, min(MAX_SHOTS, math.ceil(secs / SECONDS_PER_SHOT)))
    return shots, secs


# -- director call (the injectable boundary) ----------------------------------
def director_call(messages: list[dict], *, max_tokens: int, temperature: float,
                  base: str | None = None, timeout: int = 120,
                  enable_thinking: bool = False) -> str:
    """One /v1/chat/completions call to the llama.cpp director. Returns content.

    `enable_thinking` is off for the structured stages (treatment/plan/repair — a <think> block
    would pollute JSON extraction) and ON for the semantic critic, where reasoning improves the
    judgement; extract_json/parse_critique strip any <think> block defensively (F6/F7)."""
    payload = {
        "model": config.DIRECTOR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        (base or config.DIRECTOR_URL).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"]["content"].strip()


# -- robust JSON extraction (director emits free-form) ------------------------
def extract_json(text: str) -> dict:
    """Pull the first balanced {...} object out of the director's reply."""
    t = strip_reasoning(text)
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t[:4].lower() == "json" else t
        t = t.strip()
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object in director output")
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[i:j + 1])
    raise ValueError("unbalanced JSON in director output")


# -- lenient normalization (fill obvious defaults so the 4B can be terse) -----
def normalize(data: dict, video_lane: str = "wan", *, music: bool = True,
              fmt: str = "narrative") -> dict:
    data.setdefault("schema_version", "ProductionPlanV1")
    proj = data.setdefault("project", {})
    # The video lane is an operator/stack decision — FORCE it (the planner authors
    # shot content, not the lane). Whatever the 4B emitted is overridden by the pin.
    proj["video_lane"] = video_lane
    # Format is the planner's deterministic genre decision (documentary vs narrative) —
    # FORCE it so the manifest records how the piece was planned (and a documentary
    # can't carry an invented protagonist the model snuck in).
    proj["format"] = fmt
    proj.setdefault("title", "Untitled")
    documentary = fmt == "documentary"
    if documentary:
        # Hard guarantee: a documentary has NO recurring fictional person. Strip any
        # Character Bible the 4B emitted anyway, so the executor never stamps an invented
        # protagonist onto every shot of a factual piece (the "Arif" failure, 2026-06-29).
        data["characters"] = []
    shots = data.get("shots") or []
    for i, s in enumerate(shots):
        s.setdefault("id", f"s{i + 1}")
        s["lane"] = video_lane          # every shot uses the pinned lane
        s.setdefault("mode", "t2v")
        s.setdefault("prompt_intent", "")
        s["characters"] = [] if documentary else s.get("characters", [])  # Bible cast ids; [] = none
    data.setdefault("delivery", {"aspect": "16:9", "width": 832, "height": 480,
                                 "fps": 16, "loudness_lufs": -14.0})
    data.setdefault("assembly", {"transition_seconds": 0.6, "duck_db": 6})
    if not music:
        data["music"] = None            # operator opted out of a music bed
    elif not data.get("music"):
        data["music"] = {"lane": "ace-step", "tags": "ambient, cinematic, instrumental",
                         "lyrics": "[instrumental]", "seed": 7}
    if not data.get("timeline"):
        data["timeline"] = [
            {"clip": s["id"], "narration_offset": 0.0, "music_level_db": -18,
             "transition_in": "dissolve"} for s in shots
        ]
    return data


# -- continuity (control-plane: the planner wires it, the 4B stays creative) ---
def apply_continuity(data: dict, mode: str, *, image_lane: str = "chroma") -> dict:
    """Deterministically rewrite a plan into a continuity mode (v0b-images).

    The 4B produces creative shots; THIS wires the asset-DAG so consecutive clips
    flow. `chain` = each shot i2v from the previous last frame; `hero` = all shots
    i2v from one generated hero keyframe; `none` = leave as authored (t2v).
    """
    if mode not in ("none", "chain", "hero", "storyboard"):
        raise ValueError(f"unknown continuity {mode!r}")
    if mode == "none":
        return data
    shots = data.get("shots") or []
    proj = data.setdefault("project", {})
    proj["continuity"] = mode
    deliv = data.get("delivery", {})
    iw, ih = deliv.get("width", 832), deliv.get("height", 480)
    if mode == "storyboard":
        # a shared style bible + ONE deliberate keyframe per shot (each shot i2v from
        # its OWN keyframe). Shared style/palette across keyframes -> coherence; the
        # per-shot prompt -> deliberate per-shot subject. Threads between hero
        # (every shot orbits one image) and chain (drifts).
        tone = (proj.get("tone") or "").strip()
        bible = ", ".join(filter(None, [
            "cohesive cinematic style", "consistent palette and lighting",
            "soft film grade", tone]))
        proj["image_policy"] = {"storyboard_keyframe_lane": image_lane}
        tasks = []
        for s in shots:
            kf = f"kf_{s.get('id', 'x')}"
            tasks.append({
                "id": kf, "role": "storyboard_keyframe", "lane": image_lane,
                "prompt": f"{bible}. {s.get('prompt_intent', '')}".strip(),
                "seed": 7,            # shared seed across keyframes -> stronger style coherence
                "width": iw, "height": ih,
                "characters": list(s.get("characters", [])),   # cast → executor injects the bible
            })
            s["mode"] = "i2v"
            s["start_from"] = kf
        data["asset_tasks"] = tasks
        return data
    if mode == "chain":
        for i, s in enumerate(shots):
            if i == 0:
                s["mode"] = "t2v"
                s.pop("start_from", None)
            else:
                s["mode"] = "i2v"
                s["start_from"] = "prev_last_frame"
    elif mode == "hero":
        deliv = data.get("delivery", {})
        anchor = (shots[0].get("prompt_intent") if shots else "") \
            or data["project"].get("title", "establishing shot")
        data["project"]["image_policy"] = {"hero_keyframe_lane": image_lane}
        hero_cast = list(dict.fromkeys(c for s in shots for c in s.get("characters", [])))
        data["asset_tasks"] = [{
            "id": "hero", "role": "hero_keyframe", "lane": image_lane, "prompt": anchor,
            "seed": 7, "width": deliv.get("width", 832), "height": deliv.get("height", 480),
            "characters": hero_cast,   # the hero anchor depicts the whole recurring cast
        }]
        for s in shots:
            s["mode"] = "i2v"
            s["start_from"] = "hero"
    return data


# -- the planner --------------------------------------------------------------
MAX_CRITIC_ROUNDS = 1   # one semantic-critic fix attempt, then ship the valid plan (fail-open)


def plan_from_brief(brief: str, reg: dict, *, llm=None, max_repairs: int = 3,
                    n_shots: int = 3, continuity: str = "none", stack=None,
                    prompts_dir: str | None = None, use_critic: bool = False):
    """Brief -> (ProductionPlanV1, list[Artifact]). Raises PlannerError on failure.

    `llm(messages, *, max_tokens, temperature)` is the injectable director call.
    `stack` is a resolved ProductionStack (stack.py) pinning video/keyframe lanes,
    continuity, and narration/music. When None, the operator pinned nothing → the
    default stack (wan · chroma · storyboard · narration+music) is resolved from the
    `continuity=` kwarg, preserving the pre-stack call signature.

    The planner authors shot CONTENT within these bounds; it does NOT choose lanes —
    the lane pin is forced (see `normalize`), so the 4B can't silently pick Wan vs LTX.
    """
    from .stack import resolve_stack
    st = stack or resolve_stack(continuity=continuity)
    call = llm or director_call
    prov: list[tuple[str, str, str, str]] = []   # (role, system, user, response)

    # FORMAT first — documentary (non-fiction, no protagonist) vs narrative (fiction).
    # Decided deterministically from the brief and threaded through BOTH stages so the
    # 4B can't fictionalize a factual brief (it used to invent a protagonist for a
    # "documentary on the history of Pakistan"; diagnosed 2026-06-29).
    fmt = detect_format(brief)
    treatment_sys = build_treatment_system(fmt)

    # stage 1 — creative treatment (free-form)
    treatment = call([{"role": "system", "content": treatment_sys},
                      {"role": "user", "content": brief}],
                     max_tokens=400, temperature=0.8)
    prov.append(("treatment", treatment_sys, brief, treatment))

    # stage 2 — structured plan (the video lane is PINNED to the operator's choice).
    # Size the token budget to the shot count — a ProductionPlanV1 is ~150 tokens/shot
    # (prompt_intent + narration + timeline + characters); a fixed 900 truncated longer
    # films (e.g. a 30 s / 6-shot brief) into unbalanced JSON.
    plan_tokens = min(4096, 700 + n_shots * 180)
    plan_sys = build_plan_system(reg, n_shots=n_shots, video_lane=st.video_lane, fmt=fmt)
    plan_user = f"BRIEF: {brief}\n\nTREATMENT:\n{treatment}\n\nNow output the ProductionPlanV1 JSON only."
    raw = call([{"role": "system", "content": plan_sys}, {"role": "user", "content": plan_user}],
               max_tokens=plan_tokens, temperature=0.4)
    prov.append(("plan", plan_sys, plan_user, raw))

    def _build(raw_json: str) -> dict:
        data = apply_continuity(
            normalize(extract_json(raw_json), video_lane=st.video_lane, music=st.music, fmt=fmt),
            st.continuity, image_lane=st.keyframe_lane,
        )
        if not st.narration:                      # operator opted out of voiceover
            for s in data.get("shots", []):
                s["narration"] = ""
        return data

    # SCHEMA-repair → SEMANTIC-critic loop. Schema errors repair against max_repairs; once a plan
    # is structurally valid, the critic (deterministic + LLM, fail-open) gets MAX_CRITIC_ROUNDS to
    # fix CONTENT problems (off-topic drift, repetition, bad narration) before we ship. A critic
    # that finds nothing — or errors — never blocks: 'valid JSON' degrades to shipping the plan.
    last_err = None
    schema_repairs = 0
    critic_rounds = 0
    while True:
        try:
            plan = ProductionPlanV1.from_dict(_build(raw))   # parse + schema validate
        except (PlanError, ValueError, json.JSONDecodeError) as e:
            last_err = str(e)
            if schema_repairs >= max_repairs:
                _write_provenance(prov, prompts_dir)         # keep the failed trail for debugging
                raise PlannerError(
                    f"director could not produce a valid plan after {max_repairs} repairs: {last_err}"
                )
            schema_repairs += 1
            ru = repair_user(raw, last_err)
            raw = call([{"role": "system", "content": plan_sys}, {"role": "user", "content": ru}],
                       max_tokens=plan_tokens, temperature=0.2)
            prov.append((f"repair_{schema_repairs}", plan_sys, ru, raw))
            continue

        # structurally valid → critique CONTENT (only while budget remains, so we never re-critique
        # a plan we're about to ship anyway).
        issues = (critic.critique(plan, brief, fmt, call=call, prov=prov)
                  if (use_critic and critic_rounds < MAX_CRITIC_ROUNDS) else [])
        if issues:
            critic_rounds += 1
            ru = critic_repair_user(raw, issues)
            raw = call([{"role": "system", "content": plan_sys}, {"role": "user", "content": ru}],
                       max_tokens=plan_tokens, temperature=0.3)
            prov.append((f"critic_repair_{critic_rounds}", plan_sys, ru, raw))
            continue

        return plan, _write_provenance(prov, prompts_dir)


def _write_provenance(prov, prompts_dir) -> list[Artifact]:
    """Write each LLM step to prompts/<role>.json and return llm_prompt records."""
    arts: list[Artifact] = []
    if prompts_dir:
        os.makedirs(prompts_dir, exist_ok=True)
    for (role, system, user, resp) in prov:
        rel = f"prompts/{role}.json"
        if prompts_dir:
            with open(os.path.join(prompts_dir, f"{role}.json"), "w") as f:
                json.dump({"model": config.DIRECTOR_MODEL, "role": role,
                           "system": system, "user": user, "response": resp}, f, indent=2)
        arts.append(Artifact(
            id=f"prompt.{role}", type="llm_prompt", role=role, path=rel,
            lane="director", prompt_hash=sha256_text(system + "" + user),
            validation={"response_hash": sha256_text(resp)},
        ))
    return arts
