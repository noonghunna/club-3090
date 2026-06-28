"""The deterministic executor — the four production phases for v0a.

v0a has no pre-production (no image lanes), so the loop is: video → audio → post.
The LLM proposes (here: a static plan); the executor validates, renders, stores
artifacts, and assembles. No retries/takes in v0a — a failure raises (and that
shows up as `zero_manual_restarts=False` in the summary).
"""
from __future__ import annotations

import dataclasses
import json
import os

from . import assemble as _assemble
from . import config, ensure, validators
from .lanes import get_backend
from .manifest import Artifact, Manifest
from .schema import ProductionPlanV1
from .util import last_frame, sha256_file, sha256_text

WAN_STEPS = 4          # distilled AllInOne schedule (studio_pipe default)
MUSIC_STEPS = 50
MUSIC_CFG = 5.0
VO_TOLERANCE_S = 1.0   # a shot's narration may run up to its length + this


def _workflow_versions() -> dict:
    out = {}
    for lane, fn in (("wan", "wan22_rapid.json"), ("ace-step", "ace_step_music.json")):
        p = os.path.join(config.WORKFLOW_DIR, fn)
        if os.path.isfile(p):
            out[lane] = sha256_file(p)
    return out


def run_production(
    plan: ProductionPlanV1,
    *,
    backend_name: str,
    job_id: str,
    now_iso: str,
    productions_dir: str = config.PRODUCTIONS_DIR,
    extra_artifacts: list = (),
    progress_cb=None,
    stack=None,
) -> dict:
    def _p(msg: str, frac: float):
        if progress_cb:
            try:
                progress_cb(msg, round(frac, 3))
            except Exception:
                pass
    backend = get_backend(backend_name)
    d = plan.delivery
    prod_dir = os.path.join(productions_dir, job_id)
    for sub in ("shots", "audio", "assembly", "logs"):
        os.makedirs(os.path.join(prod_dir, sub), exist_ok=True)

    def rel(p: str) -> str:
        return os.path.relpath(p, prod_dir)

    pairs = [(plan.shot_by_id(t.clip), t) for t in plan.timeline]   # (shot, timeline) in play order
    from .stack import stack_from_plan
    st = stack or stack_from_plan(plan)   # record the operator-chosen lanes in the manifest

    def _inject(base: str, pos: str) -> str:
        """Prepend the canonical Character-Bible block to a render prompt (identity first)."""
        return f"{pos}. {base}".strip(". ") if pos else base

    man = Manifest(
        job_id=job_id, title=plan.project.title, created_utc=now_iso, backend=backend.name,
        stack=st.to_dict(),
        characters=[{"id": c.id, "name": c.name, "role": c.role} for c in plan.characters],
        delivery={"aspect": d.aspect, "width": d.width, "height": d.height, "fps": d.fps,
                  "codec": d.codec, "loudness_lufs": d.loudness_lufs},
        workflow_versions=_workflow_versions(),
        seeds=[s.seed for s, _ in pairs] + ([plan.music.seed] if plan.music else []),
    )
    for a in (extra_artifacts or []):   # planner llm_prompt provenance (v0b)
        man.add(a)
    with open(os.path.join(prod_dir, "production.json"), "w") as f:
        json.dump(dataclasses.asdict(plan), f, indent=2)
    # Character Bible → a first-class production asset + manifest provenance entry.
    if plan.characters:
        bible_path = os.path.join(prod_dir, "character_bible.json")
        with open(bible_path, "w") as f:
            json.dump([dataclasses.asdict(c) for c in plan.characters], f, indent=2)
        man.add(Artifact(id="character_bible", type="provenance", role="character_bible",
                         path=rel(bible_path)))

    # -- Phase 0: pre-production (generate image assets, v0b-images) -------
    # The asset-DAG resolves here: anything a shot's start_from references must be
    # produced BEFORE the video that consumes it.
    asset_paths: dict[str, str] = {}
    _na = len(plan.asset_tasks)
    for ai, at in enumerate(plan.asset_tasks):
        _p(f"keyframe {ai + 1}/{_na} ({at.role})", 0.05 + 0.20 * ai / max(1, _na))
        ensure.ensure_lane(at.lane, backend=backend.name)
        cpos, cneg = plan.character_block(getattr(at, "characters", []))
        img = backend.generate_image(
            prompt=_inject(at.prompt, cpos), width=at.width, height=at.height, seed=at.seed,
            lane=at.lane, out_stem=os.path.join(prod_dir, "assets", at.id), negative=cneg,
        )
        asset_paths[at.id] = img
        ipr = validators.ffprobe(img)
        man.add(Artifact(
            id=f"asset.{at.id}", type="media", role=at.role, path=rel(img), lane=at.lane,
            seed=at.seed, prompt_hash=sha256_text(at.prompt),
            width=ipr.width, height=ipr.height,
            validation=validators.run_all(["non_empty", "has_video"], img),
        ))

    # -- Phase 1: video production ----------------------------------------
    vlane = plan.project.video_lane
    clip_paths: dict[str, str] = {}
    prev_clip = None
    _ns = len(pairs)
    for si, (shot, _t) in enumerate(pairs):   # play (timeline) order — prev_last_frame relies on it
        mode, start_image = "t2v", None
        if shot.start_from == "prev_last_frame":
            if prev_clip is None:
                raise RuntimeError(f"shot {shot.id}: start_from 'prev_last_frame' but no previous clip")
            start_image = last_frame(prev_clip, os.path.join(prod_dir, "assets", f"{shot.id}_start.png"))
            mode = "i2v"
        elif shot.start_from:
            start_image = asset_paths.get(shot.start_from)
            if not start_image:
                raise RuntimeError(f"shot {shot.id}: start_from {shot.start_from!r} was not generated")
            mode = "i2v"
        _p(f"shot {si + 1}/{_ns} ({mode}, {vlane})", 0.25 + 0.35 * si / max(1, _ns))
        ensure.ensure_lane(vlane, backend=backend.name)
        frames = max(1, round(shot.target_seconds * d.fps))
        spos, sneg = plan.character_block(shot.characters)
        path = backend.render_video(
            prompt=_inject(shot.prompt_intent, spos), width=d.width, height=d.height, frames=frames,
            steps=WAN_STEPS, seed=shot.seed, fps=d.fps, mode=mode, start_image=start_image,
            out_stem=os.path.join(prod_dir, "shots", shot.id), negative=sneg,
        )
        clip_paths[shot.id] = path
        prev_clip = path
        pr = validators.ffprobe(path)
        man.add(Artifact(
            id=f"shot.{shot.id}", type="media", role="shot", path=rel(path), lane=vlane,
            seed=shot.seed, prompt_hash=sha256_text(shot.prompt_intent),
            width=pr.width, height=pr.height, duration=pr.duration,
            validation=validators.run_all(shot.validators, path, target_seconds=shot.target_seconds),
        ))

    durations = [validators.ffprobe(clip_paths[s.id]).duration for s, _ in pairs]
    total = sum(durations)

    # -- Phase 2: audio production ----------------------------------------
    _p("narration + music bed", 0.62)
    ensure.ensure_tts(backend=backend.name)
    narration: dict[str, str] = {}
    for shot, _t in pairs:
        if shot.narration.strip():
            wav = backend.tts(text=shot.narration, voice="", speed=1.0,
                              out_stem=os.path.join(prod_dir, "audio", f"{shot.id}_vo"))
            narration[shot.id] = wav
            man.add(Artifact(
                id=f"narration.{shot.id}", type="media", role="narration", path=rel(wav),
                prompt_hash=sha256_text(shot.narration),
                duration=validators.ffprobe(wav).duration,
                validation=validators.run_all(["non_empty", "audio_present"], wav),
            ))

    bed = None
    if plan.music:
        ensure.ensure_lane("ace-step", backend=backend.name)
        secs = plan.music.seconds or total
        bed = backend.make_music(
            tags=plan.music.tags, lyrics=plan.music.lyrics, seconds=secs,
            steps=MUSIC_STEPS, cfg=MUSIC_CFG, seed=plan.music.seed,
            out_stem=os.path.join(prod_dir, "audio", "bed"),
        )
        man.add(Artifact(
            id="music.bed", type="media", role="music_bed", path=rel(bed), lane="ace-step",
            seed=plan.music.seed, duration=validators.ffprobe(bed).duration,
            validation=validators.run_all(["non_empty", "audio_present"], bed),
        ))

    # -- Phase 3: post-production (assemble) ------------------------------
    # narration as (path, shot_index, intra_offset) — assemble computes the
    # crossfade-aware absolute start from the shot index.
    narr_inputs = [
        (narration[shot.id], i, t.narration_offset)
        for i, (shot, t) in enumerate(pairs) if shot.id in narration
    ]
    # per-seam transitions — each governed by the transition_in of the clip the
    # seam leads INTO (dissolve default, cut available).
    transitions = [(t.transition_in, plan.assembly.transition_seconds) for (_s, t) in pairs[1:]]
    bed_level_db = pairs[0][1].music_level_db if pairs else -18.0

    final_path = os.path.join(prod_dir, "assembly", "final.mp4")
    cmd = _assemble.assemble(
        final_path, [clip_paths[s.id] for s, _ in pairs], durations, transitions, narr_inputs, bed,
        fps=d.fps, lufs=d.loudness_lufs, bed_level_db=bed_level_db, duck_db=plan.assembly.duck_db,
    )
    man.ffmpeg_cmds.append(cmd)
    fpr = validators.ffprobe(final_path)
    man.add(Artifact(
        id="final.mp4", type="media", role="final", path=rel(final_path),
        width=fpr.width, height=fpr.height, duration=fpr.duration,
        validation=validators.run_all(["non_empty", "audio_present", "duration"], final_path),
    ))
    man.final = rel(final_path)

    # -- exit criteria -----------------------------------------------------
    vo_ok = []
    for i, (shot, _t) in enumerate(pairs):
        if shot.id in narration:
            vo_dur = validators.ffprobe(narration[shot.id]).duration
            vo_ok.append({"shot": shot.id, "vo_s": round(vo_dur, 2), "shot_s": round(durations[i], 2),
                          "ok": vo_dur <= durations[i] + VO_TOLERANCE_S})
    man.exit_criteria = {
        # only MEDIA artifacts carry validators; llm_prompt provenance is excluded.
        "all_validators_pass": all(
            a.validation.get("ok", False) for a in man.artifacts if a.type == "media"
        ),
        "vo_within_tolerance": all(v["ok"] for v in vo_ok),
        "vo_detail": vo_ok,
        "final_has_audio": fpr.has_audio,
        "final_duration_s": round(fpr.duration, 2),
        "total_clip_s": round(total, 2),
        "better_than_lane_by_lane": None,   # human judgment — printed as a prompt
    }
    man.write(prod_dir)
    _p("done", 1.0)
    return {"prod_dir": prod_dir, "final": final_path, "manifest": man.to_dict()}
