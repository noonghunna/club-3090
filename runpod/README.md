# club-3090 RunPod Template

One-click deploy Qwen3.6-27B (AutoRound INT4) on 1× or 2× RTX 3090. Same stack as the [club-3090](https://github.com/noonghunna/club-3090) docker-compose configs — all patches baked in, no setup scripts needed.

## Template configuration

| Field | Value |
|---|---|
| **Container image** | `abhinand5/club-3090-vllm:latest` |
| **Container disk** | 20 GB |
| **Volume disk** | 30 GB+ (model is ~14 GB; 30 GB leaves room) |
| **Volume mount path** | `/workspace` |
| **Expose HTTP port** | `8000` |
| **Extra HTTP ports** (optional) | `8080` — for `INTERACTIVE` debug mode |

## Environment variables

### Optional (set on template or as RunPod secrets)

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | (unset) | HuggingFace token for gated models |
| `API_KEY` | (unset) | If set, clients must pass `Authorization: Bearer <value>` |

### Model config

| Variable | Default |
|---|---|
| `MODEL_NAME_OR_PATH` | `Lorbus/Qwen3.6-27B-int4-AutoRound` |
| `SERVED_MODEL_NAME` | `Qwen3.6-27B-4bit` |
| `HF_HOME` | `/workspace/hf_home` |

### Performance (per-GPU defaults picked automatically)

| Variable | 1×3090 | 2×3090 |
|---|---|---|
| `TENSOR_PARALLEL_SIZE` | `1` | `2` |
| `MAX_MODEL_LEN` | `75000` | `262144` |
| `GPU_MEMORY_UTILIZATION` | `0.97` | `0.92` |
| `MAX_NUM_SEQS` | `1` | `2` |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | `8192` |
| `VLLM_PORT` | `8000` | `8000` |

### Feature toggles

| Variable | Default | Effect |
|---|---|---|
| `MULTIMODAL` | `0` | Set to `1` to enable vision (drops `--language-model-only` on 1×3090) |
| `INTERACTIVE` | `0` | Set to `1` to launch Jupyter Lab on port 8080 instead of vLLM |

## What runs at boot

| GPUs | Config | Context | KV | Vision | Spec decode | Genesis |
|---|---|---|---|---|---|---|
| 1 | tools-text | 75K | fp8 | text-only | MTP n=3 | P64 + PN8 |
| 2 | dual (default) | 262K | fp8 | vision | MTP n=3 | none |

First boot downloads the model from HuggingFace to `/workspace/hf_home` (~14 GB, ~5-10 min). Subsequent boots skip the download.

## Patches baked into the image

- **Genesis v7.62.x** (Sandermage/genesis-vllm-patches @ `917519b`) — P64 tool-call streaming fix, PN8 MTP draft online-quant (~800 MiB saved on fp8+MTP)
- **Marlin pad-sub-tile-n** (vLLM PR #40361) — required for TP=2 AutoRound W4A16 on Ampere
- **tolist cudagraph fix** — idempotent disk-edit for CUDA graph capture crashes

## Testing

```bash
# Sanity
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":30}'

# Tool call
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":100}'

# Streaming
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Write a haiku about GPUs"}],"max_tokens":50,"stream":true}'
```

## Common pitfalls

- **Community Cloud pods** may have broken GPU passthrough (`cuInit=999`, empty `/dev/nvidia-caps`). Switch to **Secure Cloud** if CUDA fails to init.
- If you hit OOM, dial `GPU_MEMORY_UTILIZATION` down by 0.03 and retry.
- On 1×3090 with `MULTIMODAL=1`, reduce `MAX_MODEL_LEN` to `48000` to free VRAM for the vision tower.

## Rebuilding

```bash
cd club-3090
docker build -f runpod/Dockerfile -t your-registry/club-3090-vllm:latest .
docker push your-registry/club-3090-vllm:latest
```

## Upstream tracker

See [`docs/UPSTREAM.md`](https://github.com/noonghunna/club-3090/blob/master/docs/UPSTREAM.md) — when these land, rebuild and drop the corresponding patch:

| Upstream | Unblocks |
|---|---|
| [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) — Marlin pad-sub-tile-n | Drop Marlin file overrides |
| [vllm#40849](https://github.com/vllm-project/vllm/pull/40849) — MTP online-quant | Drop PN8 env var |
| [vllm#39598](https://github.com/vllm-project/vllm/pull/39598) — qwen3coder streaming | Drop P64 env var |
