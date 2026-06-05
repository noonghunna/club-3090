# Qwen3-Omni-30B-A3B on vLLM-Omni — 2× RTX 3090

Run Alibaba's **Qwen3-Omni-30B-A3B-Instruct** (omni-modal: text / image / audio / video **in**, text **+ speech out**) on two RTX 3090s via the **vLLM-Omni** framework.

> **Status: 🧪 Experimental.** The **text** path is validated on this rig (full 65 K context, needle-in-haystack recall clean at 60 K, coherent output). **Audio/speech output works in principle but is unvalidated here.** This model is **not** in the central registry (`compose_registry.py`) — launch it with **direct `docker compose`**, not `switch.sh` / `launch.sh`.

Compose: [`compose/dual/autoround-int4/omni.yml`](compose/dual/autoround-int4/omni.yml) (+ its mounted `qwen3_omni_3090.yaml` deploy-config).

---

## What you get (and what you don't)

| | |
|---|---|
| **Generates** | text ✅, speech ✅ (real-time TTS of its own reply) |
| **Understands (input)** | text, image, audio, video ✅ |
| **Does NOT generate** | images ❌, video ❌ — it's a conversation model, not a diffusion model |

Architecture = Qwen's 3-stage **Thinker → Talker → Code2Wav** pipeline:
- **Stage 0 — Thinker** (the 30B-A3B MoE text LLM): understands the prompt, generates the **text** reply. *This is all you need for text.*
- **Stage 1 — Talker**: turns the reply into audio codec tokens.
- **Stage 2 — Code2Wav**: vocoder, codec tokens → speech waveform.

On 2× 3090 the stages are **stage-parallel** (one engine per role), **not** tensor-parallel:
`thinker → GPU0` (~23 GB), `talker + code2wav → GPU1` (~16 GB).

---

## Why this exact image (don't "upgrade" it)

Use **`vllm/vllm-omni:v0.20.0`** (the **stable** release: omni 0.20.0 + its matching vLLM 0.20.0). The newer tags are **internally version-skewed and crash on load**:

| Image | Result |
|---|---|
| `vllm/vllm-omni:v0.20.0` | ✅ self-consistent — **use this** |
| `:latest` == `:v0.21.0rc1` | ❌ omni 0.21 + vLLM 0.20 → `ImportError: split_routed_experts` |
| `:v0.22.0rc1` (build) | ❌ needs symbols no released vLLM has → unbuildable |
| stock `vllm/vllm-openai` (any tag) | ❌ wrong tool — crashes profiling the omni MM-encoder (`cu_seqlens must be on CUDA`) |

Re-check only when vLLM-Omni publishes a **correctly-built 0.22 stable** image.

---

## Setup (step by step)

### 1. Get the weights (~25 GB, ungated)
```bash
hf download Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --local-dir "$MODEL_DIR/qwen3-omni-30b-a3b-instruct-int4-autoround"
```
`$MODEL_DIR` is your models root (the dir mounted as `/models`). int4 AutoRound is mandatory — bf16 is ~60 GB and won't fit 2× 24 GB.

### 2. Pull the engine image
```bash
docker pull vllm/vllm-omni:v0.20.0
```

### 3. Launch (both GPUs must be free)
```bash
cd models/qwen3-omni-30b-a3b/vllm-omni/compose/dual/autoround-int4
MODEL_DIR=/your/models/dir docker compose -f omni.yml up -d
```
First boot takes ~3 min (loads 3 stages + cudagraph capture). Watch it:
```bash
docker logs -f vllm-omni-qwen3-omni-30b   # wait for "Application startup complete"
curl http://localhost:8042/v1/models
```

### 4. Text request — **always pass `"modalities": ["text"]`**
```bash
curl localhost:8042/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/models/qwen3-omni-30b-a3b-instruct-int4-autoround",
  "modalities": ["text"],
  "messages": [{"role":"user","content":"In two sentences, what is an MoE model?"}],
  "max_tokens": 256
}'
```
> **`"modalities":["text"]` is not optional.** Without it the prompt is routed through the Talker (audio) stage, which hits a prefill shape bug and **takes the whole engine down**.

---

## Performance (measured on this rig)

- **Context:** full **65,536** (native max) on the single-card thinker via fp8 KV. NIAH recall PASS at 60 K (50 % & 90 % depth).
- **Decode:** ~12 tok/s single-stream (the multi-stage omni wrapper adds overhead vs a plain text model; this is expected).
- **Prefill:** ~1,800 tok/s (a 60 K prompt ≈ 34 s).
- **VRAM:** GPU0 ~23 GB (thinker), GPU1 ~16 GB (talker + code2wav).

For pure text speed/quality the rig's Qwen3.6 / Gemma models are better — Qwen3-Omni earns its place for **multimodal understanding + speech**, not plain text.

---

## Speech / audio output (experimental, unvalidated here)

The talker + code2wav stages are loaded and *can* produce speech, but we only smoke-tested them.
- **fp8 KV (the default) breaks the Code2Wav vocoder** (`snake_activation` dtype error). For audio output you must run **bf16 KV**:
  ```bash
  KV_CACHE_DTYPE=auto MODEL_DIR=/your/models/dir docker compose -f omni.yml up -d
  ```
  (bf16 KV → less context headroom on the thinker.)
- Omit `"modalities":["text"]` (or request audio) to exercise the speech path — expect to debug; the talker prefill path was fragile in our testing.

---

## Environment overrides

| Var | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `../../../../../../models-cache` | host models root → `/models` |
| `MODEL_SUBDIR` | `qwen3-omni-30b-a3b-instruct-int4-autoround` | weights subdir under `/models` |
| `PORT` | `8042` | host port (container serves on 8091) |
| `KV_CACHE_DTYPE` | `fp8` | `fp8` = full ctx text; `auto` = bf16 (needed for audio out) |
| `VLLM_OMNI_IMAGE` | `vllm/vllm-omni:v0.20.0` | engine image (keep pinned — see above) |

Tune per-stage VRAM/context in the mounted **`qwen3_omni_3090.yaml`** (thinker on GPU0, talker+code2wav on GPU1). If you raise the thinker's `max_model_len`, raise the **talker's too** — the prompt flows through every stage.

---

## Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `ImportError: split_routed_experts` / `_resolve_module_name` | Wrong image. Use `vllm/vllm-omni:v0.20.0`. |
| `cu_seqlens_q must be on CUDA` | You're on stock `vllm/vllm-openai`, not vLLM-Omni. Use the omni image. |
| Engine dies on a chat request; `_get_talker_assistant_parts` shape error | Missing `"modalities":["text"]` → request hit the talker. Add it. |
| `No available memory for the cache blocks` | Thinker gpu-util too low, or `max_model_len` too high for the KV pool. Keep thinker `gpu_memory_utilization ≥ 0.85`. |
| `snake_activation: expected fp32 got bf16` | fp8 KV + audio. Set `KV_CACHE_DTYPE=auto` for the speech path. |
| OOM on a long prompt | Both stages need `max_model_len ≥ prompt length` (prompt flows through all stages). |

---

## Notes for maintainers

- **Not registry-wired on purpose** — it's a custom-engine (vLLM-Omni) exploratory deploy. No `compose_registry.py` entry, no `switch.sh`/`launch.sh` integration; direct `docker compose` only.
- Co-locating an **image-generation** model on GPU1 alongside the audio stages was investigated and shelved: every quality image model drags an ~8 GB text encoder that blows the co-located budget. Image gen wants a dedicated card (time-share), not co-residence.
