# RunPod Template — Qwen3.6-27B on RTX 3090(s)

Custom Docker image with all club-3090 patches baked in, plus an auto-detecting entrypoint that picks the right vLLM config for your GPU count.

## What's baked into the image

- **vLLM nightly** (`dev205+g07351e088`) — the pinned nightly all composes are tested against
- **Genesis patches** (Sandermage/genesis-vllm-patches @ `917519b`) — v7.62.x, includes P64 (tool-call streaming fix) + PN8 (MTP draft online-quant, frees ~800 MiB on fp8+MTP)
- **Marlin pad-sub-tile-n** (vLLM PR #40361 fork) — required for TP=2 AutoRound W4A16 on Ampere
- **tolist cudagraph fix** (`patch_tolist_cudagraph.py`) — idempotent disk-edit for TQ3 KV + cudagraph capture crash
- **deps**: xxhash, pandas, scipy (needed by Genesis apply_all)

## Quick start

### 1. Build & push the image

```bash
cd club-3090
docker build -f runpod/Dockerfile -t <your-registry>/club-3090-vllm:latest .
docker push <your-registry>/club-3090-vllm:latest
```

### 2. Create RunPod templates

Create one template — it auto-detects GPU count at boot:

| Field | Value |
|---|---|
| Image | `<your-registry>/club-3090-vllm:latest` |
| GPU type | RTX 3090 |
| Container disk | 20 GB |
| Expose port | 8000 |
| `--shm-size` | 16g |
| Env vars | see below |

The same template works for 1× and 2×3090 pods. The entrypoint detects GPU count and picks the right config.

vLLM auto-downloads the model from HuggingFace on first boot to `HF_HOME=/workspace`, so it persists across pod restarts.

## Environment variables

### Required (set on template or as secrets)

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace token (only needed for gated models) | (empty) |

### Optional — override defaults

| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME_OR_PATH` | HF repo ID or local path to model | `Lorbus/Qwen3.6-27B-int4-AutoRound` |
| `HF_HOME` | HuggingFace cache dir (persistent storage) | `/workspace` |
| `MAX_MODEL_LEN` | Max context length | `75000` (1×) / `262144` (2×) |
| `GPU_MEMORY_UTILIZATION` | VRAM fraction for KV cache + weights | `0.97` (1×) / `0.92` (2×) |
| `MAX_NUM_SEQS` | Max concurrent requests | `1` (1×) / `2` (2×) |
| `MAX_NUM_BATCHED_TOKENS` | Max tokens per prefill batch | `2048` (1×) / `8192` (2×) |
| `TENSOR_PARALLEL_SIZE` | Override TP size (0 = auto-detect) | `0` |
| `VLLM_PORT` | vLLM listen port | `8000` |

### Genesis patches (single-card only — dual-card is Genesis-less)

These are set automatically by the entrypoint for 1×3090. Override if you want different behavior:

| Variable | Default (1×) | Effect |
|---|---|---|
| `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` | on | Streaming MTP tool-call edge case fix |
| `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1` | on | MTP draft online-quant (frees ~800 MiB on fp8+MTP) |

## What runs at boot

The entrypoint:
  1. Sets `HF_HOME=/workspace` (model downloads persist on network volume)
  2. Detects GPU count via `nvidia-smi`
3. **1 GPU**: applies Genesis patches + tolist fix, sets P64+PN8 env vars, launches tools-text config (75K, fp8 KV, text-only, MTP n=3)
4. **2 GPU**: skips Genesis (dual.yml is Genesis-less by design), launches dual config (262K, fp8 KV, vision + tools, MTP n=3, TP=2)

## Verification

After boot, test with:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":30}'
```

## What this doesn't cover

These compose variants are NOT in this RunPod image:

| Variant | Why not |
|---|---|
| `docker-compose.yml` (TQ3 default, 48K) | Tools-text (fp8 75K) is better for single-card on RunPod — no TQ3 cudagraph crash risk |
| `dual-turbo.yml` (TQ3 4-stream) | Needs Genesis P65/P66 which are TQ3-specific — dual.yml (fp8) is Genesis-less and safer |
| `dual-dflash.yml` (DFlash N=5) | DFlash needs non-causal attention path; YAGNI for most deployments |
| `long-vision.yml` / `long-text.yml` (192K/205K single) | Both fire prefill cliffs on tool-heavy workloads — tools-text (75K) is the safe ceiling |

If you need one of these, build from the repo's docker-compose files directly.

## Upstream dependencies

When these upstream PRs land, rebuild the image and drop the relevant patches:

| Upstream | What it unblocks |
|---|---|
| [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — Marlin pad-sub-tile-n | Drop the Marlin file overrides in Dockerfile |
| [vllm#40849](https://github.com/vllm-project/vllm/pull/40849) — MTP draft online-quant | Drop PN8 env var; upstream ships the fix |
| [vllm#39598](https://github.com/vllm-project/vllm/pull/39598) — qwen3coder MTP streaming | Drop P64 env var; upstream ships the fix |

Full upstream tracker: [`docs/UPSTREAM.md`](../docs/UPSTREAM.md)
