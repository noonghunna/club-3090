"""The deterministic executor — the four production phases for v0a.

v0a has no pre-production (no image lanes), so the loop is: video → audio → post.
The LLM proposes (here: a static plan); the executor validates, renders, stores
artifacts, and assembles. No retries/takes in v0a — a failure raises (and that
shows up as `zero_manual_restarts=False` in the summary).
"""
from __future__ import annotations

import os

from . import assemble as _assemble
from . import config, ensure, validators
from .lanes import get_backend
from .manifest import Artifact, Manifest
from .schema import ProductionPlanV1
from .util import sha256_file, sha256_text

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
) -> dict:
    backend = get_backend(backend_name)
    d = plan.delivery
    prod_dir = os.path.join(productions_dir, job_id)
    for sub in ("shots", "audio", "assembly", "logs"):
        os.makedirs(os.path.join(prod_dir, sub), exist_ok=True)

    def rel(p: str) -> str:
        return os.path.relpath(p, prod_dir)

    pairs = [(plan.shot_by_id(t.clip), t) for t in plan.timeline]   # (shot, timeline) in play order
    man = Manifest(
        job_id=job_id, title=plan.project.title, created_utc=now_iso, backend=backend.name,
        delivery={"aspect": d.aspect, "width": d.width, "height": d.height, "fps": d.fps,
                  "codec": d.codec, "loudness_lufs": d.loudness_lufs},
        workflow_versions=_workflow_versions(),
        seeds=[s.seed for s, _ in pairs] + ([plan.music.seed] if plan.music else []),
    )

    # -- Phase 1: video production ----------------------------------------
    clip_paths: dict[str, str] = {}
    for shot, _t in pairs:
        ensure.ensure_lane("wan", backend=backend.name)
        frames = max(1, round(shot.target_seconds * d.fps))
        path = backend.render_video(
            prompt=shot.prompt_intent, width=d.width, height=d.height, frames=frames,
            steps=WAN_STEPS, seed=shot.seed, fps=d.fps,
            out_stem=os.path.join(prod_dir, "shots", shot.id),
        )
        clip_paths[shot.id] = path
        pr = validators.ffprobe(path)
        man.add(Artifact(
            id=f"shot.{shot.id}", type="media", role="shot", path=rel(path), lane="wan",
            seed=shot.seed, prompt_hash=sha256_text(shot.prompt_intent),
            width=pr.width, height=pr.height, duration=pr.duration,
            validation=validators.run_all(shot.validators, path, target_seconds=shot.target_seconds),
        ))

    durations = [validators.ffprobe(clip_paths[s.id]).duration for s, _ in pairs]
    total = sum(durations)
    starts = [sum(durations[:i]) for i in range(len(durations))]

    # -- Phase 2: audio production ----------------------------------------
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
    narr_inputs: list[tuple[str, int]] = []
    for i, (shot, t) in enumerate(pairs):
        if shot.id in narration:
            start_ms = max(0, int((starts[i] + t.narration_offset) * 1000))
            narr_inputs.append((narration[shot.id], start_ms))
    bed_level_db = pairs[0][1].music_level_db if pairs else -18.0

    final_path = os.path.join(prod_dir, "assembly", "final.mp4")
    cmd = _assemble.assemble(
        final_path, [clip_paths[s.id] for s, _ in pairs], narr_inputs, bed,
        fps=d.fps, lufs=d.loudness_lufs, bed_level_db=bed_level_db,
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
        "all_validators_pass": all(a.validation.get("ok") for a in man.artifacts),
        "vo_within_tolerance": all(v["ok"] for v in vo_ok),
        "vo_detail": vo_ok,
        "final_has_audio": fpr.has_audio,
        "final_duration_s": round(fpr.duration, 2),
        "total_clip_s": round(total, 2),
        "better_than_lane_by_lane": None,   # human judgment — printed as a prompt
    }
    man.write(prod_dir)
    return {"prod_dir": prod_dir, "final": final_path, "manifest": man.to_dict()}
