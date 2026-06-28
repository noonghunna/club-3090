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
- **v0b-images** — **continuity** (the slideshow fix). Modes via `--continuity`:
  **storyboard** (DEFAULT — one deliberate keyframe per shot off a shared style bible,
  each shot i2v from its own keyframe), **hero** (all shots i2v from one generated
  keyframe), **chain** (each shot i2v from the previous shot's last frame), **none**
  (independent t2v). Asset-DAG: `asset_tasks[]` (generated images) + `shot.start_from`
  (`prev_last_frame` | `<asset id>`), resolved in an executor **pre-production phase**;
  the 4B stays creative, the planner wires continuity deterministically (`apply_continuity`).

## The production stack (operator-chosen, visible, overridable)

The director plans shot **content**; it does **not** silently pick the models. The
**stack** — video lane · keyframe lane · continuity · narration/music — is an operator
decision: defaulted-but-**visible** (echoed as a `Using stack:` block), **overridable**
(CLI flags / OWUI valves / `/produce` body), validated against what the executor can
actually render, and **recorded in the manifest** (`stack` field). The lane pin is
*forced* through the planner, so the 4B can't choose Wan-vs-LTX behind your back.

| Dimension | Default | Options (render **today**) | Roadmap (declared, not yet wired) |
|---|---|---|---|
| video lane | `wan` | `wan` | `ltx` · `sulphur` · `10eros` (LTX-family checkpoint-swap graph not yet ported to the executor) |
| keyframe lane | `chroma` | `chroma` · `zimage` · `krea` · `hidream` | `ideogram` (needs structured JSON, not prose → a title-card lane, not a continuity keyframe lane) |
| continuity | `storyboard` | `storyboard` · `hero` · `chain` · `none` | — |
| audio | narration + music | `--no-music` · `--no-narration` · `--voice <id>` | — |

A choice the executor can't honour (unknown or roadmap lane) **fails fast** with a clear
message + the renderable set — an operator is never handed a choice that errors at render.

## OWUI lane — 🎬 Studio · Production

Reach it from chat without the CLI: pick **🎬 Studio · Production** in OWUI, type a
brief, and the lane (a thin **plan-then-execute** client over the host `server.py`,
**not** a live tool-loop) streams progress and drops the finished MP4 in the gallery.
Pick the stack in the lane's ⚙️ valves (`production_video_lane` / `_keyframe_lane` /
`_continuity` / `_music`) — Auto by default, overridable. Bring the service up on the
host: `python3 -m services.studio.production.server` (`:8195`, gated by `/produce/health`).

**Scope (deliberately tight):** single-flight · ffprobe validators · ffmpeg mix at the
delivery profile · typed manifest.

**Deferred to v1:** skills, takes + rerender, Qdrant/SearXNG, durable queue, the
remaining image roles (reference / title-card), and wiring the **roadmap video lanes**
(LTX/Sulphur/10Eros) into the production executor.

## Run

```bash
# from the repo root (/opt/ai/github/club-3090)

# offline — ffmpeg lavfi stand-ins, no GPU/ComfyUI/TTS (good for dev + CI):
python3 -m services.studio.production.run \
    services/studio/production/plans/lighthouse_3shot.json --backend synthetic

# v0b — the 4B director PLANS a brief, then renders it (needs the ai-studio scene up).
# The default stack (wan · chroma · storyboard · narration+music) is echoed before rendering:
gpu-mode ai-studio
python3 -m services.studio.production.run \
    --brief "a 15-second calm documentary about lighthouses" --backend live --shots 3

# pick the stack explicitly (visible + overridable; --help lists every lane):
python3 -m services.studio.production.run \
    --brief "a 30s noir short about a detective" --backend live \
    --video-lane wan --keyframe-lane hidream --continuity storyboard --no-music
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
# from the repo root — discovers all production tests (58, stdlib unittest):
python3 -m unittest discover -s services/studio/production/tests -p "test_*.py" -t .
```

`test_v0a` runs the **whole** render pipeline on the synthetic backend (real MP4 +
manifest). `test_v0b` drives the planner with a **stub LLM** — prompt construction,
JSON extraction, normalization, and the validator-repair loop. `test_v0b_images`
covers the asset-DAG + every continuity mode. `test_stack` covers the operator-chosen
stack — resolution/validation, the no-silent-Wan rule, the keyframe-lane data-plane
wiring (each wired lane's patch lands in real workflow nodes), and the manifest record
— so executor, planner, and stack are all verified before touching the rig.

## Layout

| File | Role |
|---|---|
| `schema.py` | `ProductionPlanV1` (dataclasses + `validate()`) |
| `stack.py` | the operator-chosen stack — lane registries (wired flags), `resolve_stack`, `describe_stack`, `stack_from_plan` |
| `server.py` | host HTTP wrapper (`/produce` · `/job/<id>` · `/produce/health`) the OWUI lane drives |
| `comfy.py` | minimal ComfyUI client (`submit` / `await_output` / `alive`) |
| `lanes.py` | `LiveBackend` (ComfyUI+Kokoro, 4 keyframe lanes) · `SyntheticBackend` (ffmpeg lavfi) |
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
