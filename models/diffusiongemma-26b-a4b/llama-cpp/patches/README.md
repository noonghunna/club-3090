# DiffusionGemma NDJSON server patch

`diffusion-ndjson-server.patch` adds an env-gated (`DG_NDJSON=1`) NDJSON request loop to
`llama-diffusion-cli`. It is **additive** — when `DG_NDJSON` is unset the binary behaves exactly
like upstream. The shim in [`../serve/`](../serve/) drives this mode.

> Status: ⏸️ **Upstream-gated.** Built against llama.cpp draft **PR #24423** (DiffusionGemma) at
> pinned sha `c84e85a`. See the PR #24423 row in [`docs/UPSTREAM.md`](../../../../docs/UPSTREAM.md).
> Drop this patch when upstream ships a diffusion HTTP server.

## What the patch does

- `getenv("DG_NDJSON")` → after the model/context load, run a request loop instead of the
  interactive/single-prompt path:
  - **in:** one JSON object per stdin line —
    `{"messages":[{"role","content"}...], "n_predict"?, "temperature"?, "seed"?, "enable_thinking"?}`
  - **out:** one JSON object per stdout line —
    `{"content","steps","blocks","time_ms"}` (or `{"error":...}`).
- Reuses the file's existing `apply_template()` + `run_turn()` — no new model/diffusion logic.
- `enable_thinking` defaults **false** in server mode (the template prefills an empty thought
  channel → compact answers, fewer denoising steps). The interactive CLI keeps upstream's `true`.
- Per-request `temperature`/`seed`/block-budget vary within the launch-time `n_ctx`/`n_ubatch`
  (context can't grow after creation — launch with `-c`/`-ub` sized for your max prompt).
- The step callback is silenced in server mode so stdout carries only response JSON.

## Delivery: custom Docker image (primary) — built from the existing PR branch

There is no stock image carrying `llama-diffusion-cli`, so the shipped delivery is a custom image
that **clones the existing draft-PR branch, applies this patch, and bakes in the shim**. The
Dockerfile (`../Dockerfile`) does this; the compose builds it:

```bash
docker compose -f ../compose/single/q4-k-m/openai-shim.yml build   # builds from PR #24423 @ c84e85a
docker compose -f ../compose/single/q4-k-m/openai-shim.yml up -d
curl http://localhost:8060/v1/models
```

The image pins `DG_REPO`/`DG_SHA` build-args to danielhanchen's branch — we rely on that branch
rather than vendoring a full source fork; only this small patch is ours. Re-pin the sha when the
PR advances. `CUDA_ARCH` build-arg = 86 (3090) / 89 (4090) / 120 (5090).

## Host build (no-Docker fallback — what was validated on the authoring rig)

```bash
# toolchain: CUDA toolkit + cmake (pip cmake works without sudo)
git clone https://github.com/danielhanchen/llama.cpp.git && cd llama.cpp
git checkout c84e85af61011f9fbfcf41479381d5ed1661a564          # PR #24423 head
git apply /path/to/club-3090/models/diffusiongemma-26b-a4b/llama-cpp/patches/diffusion-ndjson-server.patch

export PATH=/usr/local/cuda-12.8/bin:$PATH                      # adjust to your CUDA
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j --target llama-diffusion-cli

# point the shim at the binary
DG_CLI_BIN=$PWD/build/bin/llama-diffusion-cli bash \
  /path/to/club-3090/models/diffusiongemma-26b-a4b/llama-cpp/serve/serve.sh
```

`-DCMAKE_CUDA_ARCHITECTURES=86` = RTX 3090 (Ampere SM 8.6). We don't vendor the binary — it's a
~2 GB CUDA build of a draft PR; rebuild it locally and pin to the sha above.

## Smoke-test the patch directly (no shim)

```bash
printf '%s\n' '{"messages":[{"role":"user","content":"Write a Python is_prime(n). Code only."}],"n_predict":256}' \
  | DG_NDJSON=1 ./build/bin/llama-diffusion-cli -m diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
      -ngl 99 -c 3072 -ub 3072 -n 1024
# -> {"ready":...}  then  {"content":"```python\ndef is_prime...","steps":...,...}
```
