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
import os
import urllib.request

from . import config
from .manifest import Artifact
from .prompts import TREATMENT_SYS, build_plan_system, repair_user
from .schema import PlanError, ProductionPlanV1
from .util import sha256_text


class PlannerError(RuntimeError):
    pass


# -- director call (the injectable boundary) ----------------------------------
def director_call(messages: list[dict], *, max_tokens: int, temperature: float,
                  base: str | None = None, timeout: int = 120) -> str:
    """One /v1/chat/completions call to the llama.cpp director. Returns content."""
    payload = {
        "model": config.DIRECTOR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
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
    t = (text or "").strip()
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
def normalize(data: dict, video_lane: str = "wan", *, music: bool = True) -> dict:
    data.setdefault("schema_version", "ProductionPlanV1")
    proj = data.setdefault("project", {})
    # The video lane is an operator/stack decision — FORCE it (the planner authors
    # shot content, not the lane). Whatever the 4B emitted is overridden by the pin.
    proj["video_lane"] = video_lane
    proj.setdefault("title", "Untitled")
    shots = data.get("shots") or []
    for i, s in enumerate(shots):
        s.setdefault("id", f"s{i + 1}")
        s["lane"] = video_lane          # every shot uses the pinned lane
        s.setdefault("mode", "t2v")
        s.setdefault("prompt_intent", "")
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
        data["asset_tasks"] = [{
            "id": "hero", "role": "hero_keyframe", "lane": image_lane, "prompt": anchor,
            "seed": 7, "width": deliv.get("width", 832), "height": deliv.get("height", 480),
        }]
        for s in shots:
            s["mode"] = "i2v"
            s["start_from"] = "hero"
    return data


# -- the planner --------------------------------------------------------------
def plan_from_brief(brief: str, reg: dict, *, llm=None, max_repairs: int = 3,
                    n_shots: int = 3, continuity: str = "none", stack=None,
                    prompts_dir: str | None = None):
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

    # stage 1 — creative treatment (free-form)
    treatment = call([{"role": "system", "content": TREATMENT_SYS},
                      {"role": "user", "content": brief}],
                     max_tokens=400, temperature=0.8)
    prov.append(("treatment", TREATMENT_SYS, brief, treatment))

    # stage 2 — structured plan (the video lane is PINNED to the operator's choice)
    plan_sys = build_plan_system(reg, n_shots=n_shots, video_lane=st.video_lane)
    plan_user = f"BRIEF: {brief}\n\nTREATMENT:\n{treatment}\n\nNow output the ProductionPlanV1 JSON only."
    raw = call([{"role": "system", "content": plan_sys}, {"role": "user", "content": plan_user}],
               max_tokens=900, temperature=0.4)
    prov.append(("plan", plan_sys, plan_user, raw))

    def _build(raw_json: str) -> dict:
        data = apply_continuity(
            normalize(extract_json(raw_json), video_lane=st.video_lane, music=st.music),
            st.continuity, image_lane=st.keyframe_lane,
        )
        if not st.narration:                      # operator opted out of voiceover
            for s in data.get("shots", []):
                s["narration"] = ""
        return data

    # validator-repair loop
    last_err = None
    for attempt in range(max_repairs + 1):
        try:
            data = _build(raw)
            plan = ProductionPlanV1.from_dict(data)     # validates
            return plan, _write_provenance(prov, prompts_dir)
        except (PlanError, ValueError, json.JSONDecodeError) as e:
            last_err = str(e)
            if attempt == max_repairs:
                break
            ru = repair_user(raw, last_err)
            raw = call([{"role": "system", "content": plan_sys}, {"role": "user", "content": ru}],
                       max_tokens=900, temperature=0.2)
            prov.append((f"repair_{attempt + 1}", plan_sys, ru, raw))

    _write_provenance(prov, prompts_dir)   # keep the failed trail for debugging
    raise PlannerError(
        f"director could not produce a valid plan after {max_repairs} repairs: {last_err}"
    )


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
