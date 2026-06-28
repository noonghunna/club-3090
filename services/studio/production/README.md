# Studio Production Agent — v0a

One static brief → a finished MP4, by driving the **existing** AI-Studio lanes
serially (Wan video · Kokoro narration · ACE-Step bed → ffmpeg mix). This is the
**v0a executor spike** from
[`/opt/ai/docs/ai-studio-production-agent-design.md`](../../../../docs/ai-studio-production-agent-design.md):
prove the executor + assembly can drive the lanes, collect + validate artifacts,
and assemble something watchable **without manual babysitting**.

**v0a scope (deliberately tight):** CLI/admin only · no OWUI lane · single-flight
(host file-lock) · **static** hand-authored `ProductionPlanV1` · one pinned video
lane (`wan`, t2v) · ffprobe validators · final ffmpeg mix at the delivery profile ·
typed/extensible manifest with run-level provenance.

**Deferred to v0b/v1:** the 4B planner, skills, image lanes / asset-DAG, takes +
rerender, Qdrant/SearXNG, durable queue, OWUI lane. (The manifest is already typed
so v0b prompt-provenance records slot in with no redesign.)

## Run

```bash
# from the repo root (/opt/ai/github/club-3090)

# offline — ffmpeg lavfi stand-ins, no GPU/ComfyUI/TTS (good for dev + CI):
python3 -m services.studio.production.run \
    services/studio/production/plans/lighthouse_3shot.json --backend synthetic

# live — drives ComfyUI (:8188) + Kokoro TTS (:8192); needs the ai-studio scene up:
gpu-mode ai-studio        # bring the scene up first
python3 -m services.studio.production.run \
    services/studio/production/plans/lighthouse_3shot.json --backend live
```

Output lands in `/mnt/models/comfyui/output/productions/<job_id>/`
(`shots/ audio/ assembly/final.mp4 manifest.json`). Override the base with
`--productions-dir`, endpoints with `COMFYUI_URL` / `TTS_URL` / `VOICE_URL`.

The run prints a summary against the **exit criteria** — all validators pass,
VO-within-tolerance, final-has-audio, durations — and ends on the decisive human
question: *is this better than picking lanes by hand?*

## Tests (offline, stdlib only — no pydantic/pytest/GPU)

```bash
python3 -m unittest services.studio.production.tests.test_v0a -v
```

The end-to-end test runs the **whole** pipeline on the synthetic backend and
asserts a real MP4 with an audio track + a well-formed manifest — so the logic is
verified before it ever touches the rig.

## Layout

| File | Role |
|---|---|
| `schema.py` | `ProductionPlanV1` (dataclasses + `validate()`) |
| `comfy.py` | minimal ComfyUI client (`submit` / `await_output` / `alive`) |
| `lanes.py` | `LiveBackend` (ComfyUI+Kokoro) · `SyntheticBackend` (ffmpeg lavfi) |
| `ensure.py` | `ensure_lane` — services-up + Step-Voice `/unload` before video |
| `validators.py` | ffprobe checks (duration / audio / dims / non-empty) |
| `assemble.py` | pure `build_mix_command` + `assemble` (concat + duck + loudnorm) |
| `manifest.py` | typed, extensible manifest + run-level provenance |
| `lock.py` | single-flight host file-lock |
| `executor.py` | the four-phase loop (video → audio → post) |
| `run.py` | CLI + exit-criteria summary |
| `plans/` | static `ProductionPlanV1` JSON |
