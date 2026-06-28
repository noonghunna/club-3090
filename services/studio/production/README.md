# Studio Production Agent — v0a + v0b-core + v0b-images

Brief → a finished MP4, by driving the **existing** AI-Studio lanes serially (Wan
video · Kokoro narration · ACE-Step bed → ffmpeg mix). From
[`/opt/ai/docs/ai-studio-production-agent-design.md`](../../../../docs/ai-studio-production-agent-design.md).

- **v0a** — a *static* hand-authored `ProductionPlanV1` → MP4 (the executor + assembly).
- **v0b-core** — the **4B director** (`qwen3.5-4b-uncensored` @ `:8090`) *plans* a
  one-line brief into a valid `ProductionPlanV1` — against the capability registry
  (`capabilities.yaml`) + a prompt pack, with a **validator-repair loop** (parse →
  `schema.validate()` → feed errors back) — then the v0a executor renders it.
  Prompt provenance is stored as `llm_prompt` manifest records under `prompts/`.
- **v0b-images** — **continuity** (the slideshow fix). Two modes via `--continuity`:
  **chain** (each shot Wan-i2v from the previous shot's last frame) and **hero** (all
  shots i2v from one generated **hero keyframe**, chroma image lane). Asset-DAG:
  `asset_tasks[]` (generated images) + `shot.start_from` (`prev_last_frame` | `<asset id>`),
  resolved in an executor **pre-production phase**; the 4B stays creative, the planner
  wires continuity deterministically (`apply_continuity`).

**Scope (deliberately tight):** CLI/admin only · single-flight · one pinned video
lane · ffprobe validators · ffmpeg mix at the delivery profile · typed manifest.

**Deferred to v1:** skills, takes + rerender, Qdrant/SearXNG, durable queue, OWUI
lane, the remaining image roles (reference / storyboard / title-card).

## Run

```bash
# from the repo root (/opt/ai/github/club-3090)

# offline — ffmpeg lavfi stand-ins, no GPU/ComfyUI/TTS (good for dev + CI):
python3 -m services.studio.production.run \
    services/studio/production/plans/lighthouse_3shot.json --backend synthetic

# v0b — the 4B director PLANS a brief, then renders it (needs the ai-studio scene up):
gpu-mode ai-studio
python3 -m services.studio.production.run \
    --brief "a 15-second calm documentary about lighthouses" --backend live --shots 3

# v0b-images — same, with CONTINUITY (chain = i2v from prev frame; hero = shared keyframe):
python3 -m services.studio.production.run \
    --brief "a 15-second calm documentary about lighthouses" --backend live --continuity chain
# or the static continuity experiment plans:
python3 -m services.studio.production.run \
    services/studio/production/plans/lighthouse_hero.json --backend live

# v0a live — a static plan; drives ComfyUI (:8188) + Kokoro TTS (:8192):
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
python3 -m unittest services.studio.production.tests.test_v0a \
                     services.studio.production.tests.test_v0b -v
```

`test_v0a` runs the **whole** render pipeline on the synthetic backend (real MP4 +
manifest). `test_v0b` drives the planner with a **stub LLM** — prompt construction,
JSON extraction, normalization, and the validator-repair loop — so both the executor
and the planner are verified before touching the rig.

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
| `capabilities.yaml` | lane capability registry — the planner's menu (wan/kokoro/ace) |
| `registry.py` | load the registry + compress it into the planner's prompt slice |
| `prompts.py` | the planner prompt pack (treatment + plan + repair) |
| `planner.py` | the 4B director: brief → plan, with the validator-repair loop |
| `run.py` | CLI (`PLAN.json` for v0a, `--brief "…"` for v0b) + exit-criteria summary |
| `plans/` | static `ProductionPlanV1` JSON (v0a) |
