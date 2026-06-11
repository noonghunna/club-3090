# DiffusionGemma-26B-A4B — single-3090 serving design

**Date:** 2026-06-11
**Status:** ⏸️ Upstream-gated (depends on llama.cpp draft PR #24423 + our wrapper)
**Goal:** Serve `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M) on one RTX 3090 behind an
OpenAI-compatible `/v1/chat/completions` endpoint so it can be used from local coding harnesses,
and capture speed/VRAM numbers vs our existing autoregressive MoEs.

## Background / constraints (from Phase 1)

- DiffusionGemma is a **true diffusion LM** (non-autoregressive, block-diffusion canvas of 256).
  Standard `llama-server`/`vllm serve` cannot run it on Ampere:
  - vLLM path needs FP8/NVFP4 weights (Hopper/Blackwell) to fit 24 GB; bf16 ≈ 50 GB. Blocked.
  - llama.cpp support exists only in **draft PR #24423** (`danielhanchen`, AI-assisted, logit-verified).
    It adds the arch + `llama-diffusion-cli` (interactive) + a stdin/stdout **logits** microservice
    (`diffusion-gemma-server`, for a Python driver) — **no OpenAI HTTP server**.
- **Phase 1 measured (Q4_K_M, `-ngl 99`, this rig):**
  - ~61 tok/s effective (2 blocks / 30 steps / 8.4 s); ~40 tok/s (4 blocks / 98 steps / 25.3 s).
    Step count scales with content difficulty — not a flat rate.
  - Peak VRAM **22.3–23.0 GB of 24** even with short prompts → **long prompts risk OOM**.
  - Verdict: **not faster than our existing autoregressive MoEs on Ampere** (the diffusion
    speedups need FP8/Hopper). Value here is capability + experimentation, not speed.

## Architecture (3 units)

```
coding harness ──HTTP /v1──▶ FastAPI shim ──NDJSON over pipe──▶ llama-diffusion-cli (DG_NDJSON=1)
   (stateless,                (OpenAI shape,                       (model resident, prefix-KV-cache,
    full history)              channel parsing)                     entropy-bound denoise)
```

### Unit 1 — C++ patch: env-gated NDJSON server mode (`examples/diffusion/diffusion-cli.cpp`)

Minimal, additive, gated by `getenv("DG_NDJSON")`. Reuses the file's existing `apply_template()`,
`make_msg()`, and `run_turn()` — no new model/diffusion logic.

- After model+context setup, if `DG_NDJSON` is set, **skip** the interactive/single-prompt blocks and
  run a request loop instead:
  - Read one line of stdin = JSON request:
    `{"messages":[{"role","content"}...], "n_predict"?:int, "temperature"?:float, "seed"?:int}`.
  - Build `common_chat_msg` vector → `apply_template()` → call `run_turn()` directly
    (NOT `run_turn_reply`, which prints to stdout).
  - Per-request knobs **bounded by launch-time ctx/ubatch** (cannot grow ctx after creation):
    set `params.diffusion.blocks = ceil(n_predict / canvas_length)` (≤ launched max),
    `diff_params.temperature` / `eb_params.seed`.
  - Emit exactly one line of JSON to stdout:
    `{"content": "<raw detokenized>", "steps":int, "blocks":int, "time_ms":float, "n_input":int}`
    or `{"error":"..."}`. Use vendored `nlohmann/json.hpp` for encode/decode (handles newlines).
  - **stdout hygiene:** in server mode the step callback must NOT print (it currently `LOG_INF`s a
    progress bar). Add a `server_mode` flag to `callback_data`: when set, the callback only counts
    `steps_seen`/`blocks_seen` and returns — no printing. All diagnostics go to stderr.
- C++ stays "dumb": returns **raw** text + stats. No channel stripping, no OpenAI shaping in C++ —
  that lives in Python so we can iterate without rebuilds.

Delivered as a patch file `models/diffusiongemma-26b-a4b/llama-cpp/patches/diffusion-ndjson-server.patch`
against PR #24423's pinned sha (`c84e85a`), plus a build note. (We don't vendor a binary.)

### Unit 2 — FastAPI shim (`models/diffusiongemma-26b-a4b/llama-cpp/serve/diffusion_openai_server.py`)

- On startup spawn `llama-diffusion-cli` with `DG_NDJSON=1`, `-m <gguf> -ngl 99 -c <CTX> -ub <CTX>`,
  hold its stdin/stdout pipes; wait for the `READY`-equivalent (first successful request or a banner).
- Single `asyncio.Lock` (np=1 — one in-flight denoise; the model/ctx is single-slot).
- `POST /v1/chat/completions`:
  - Map OpenAI `messages` + `max_tokens`/`temperature`/`seed` → NDJSON request line.
  - Read the one response line; **parse the raw text**:
    1. Strip a leading `<|channel>thought ... <channel|>` block (optionally surface as `reasoning`).
    2. Strip trailing `<turn|>` / EOG markers.
    3. Remainder → `content`.
  - Return a standard `chat.completion` object (id, model, choices[0].message.content, usage best-effort
    from `n_input` + emitted token estimate, `finish_reason`).
  - `stream=true`: emit the final content as a single SSE chunk + `[DONE]` (diffusion is block-wise;
    true token streaming isn't meaningful). v1 may return 400 on stream and we add this next.
- `GET /v1/models`: report `diffusiongemma-26b-a4b`.
- Config via env: `DG_GGUF`, `DG_CLI_BIN`, `DG_PORT` (default e.g. 8060), `DG_CTX` (default sized to a
  safe prompt budget given VRAM — start ~3072 and document the OOM ceiling), `DG_NGL`.
- **Tool calls:** v1 returns assistant content only. Native `<|tool_call>` → OpenAI `tool_calls`
  parsing is a documented follow-up (many harnesses cope with content-only).

### Unit 3 — "serve like the other models" packaging

- **Today (no Docker in this WSL distro):** a host launcher `serve.sh` that starts the FastAPI server
  on `DG_PORT`, so a harness points at `http://localhost:<port>/v1` exactly like other endpoints.
- **For parity/future:** an `⏸️ Upstream-gated` compose
  `compose/single/q4-k-m/openai-shim.yml` + a `Dockerfile` that builds PR #24423 and bundles the shim,
  carrying the standard Profile (at-a-glance) header. Not wired into `compose_registry.py` DEFAULTS;
  not a production entry (can't pass `verify-full`/`bench` as a normal OpenAI server until upstream
  ships one). Kept discoverable + documented.

## Docs / tracking

- Add rows to `docs/UPSTREAM.md`: PR #24423 (llama.cpp diffusion-gemma) and PR #45163 (vLLM
  diffusion-gemma) — both ⏸️, with the Ampere-blocked rationale linked.
- A short findings note (numbers + "not faster on Ampere" conclusion + VRAM ceiling caveat) in the
  model dir, BENCHMARKS-style.
- Profile schema header `Genesis: N/A — Genesis is Qwen3-Next-specific`; `Status: ⏸️ Upstream-gated`;
  `Caveats:` listing draft-PR dependency + VRAM ceiling + thinking-channel parsing.

## Testing / acceptance

1. Rebuild `llama-diffusion-cli` with the patch; `DG_NDJSON=1` round-trips a single request → valid
   NDJSON, content matches a non-server run for the same prompt/seed.
2. FastAPI: `curl /v1/models` and a `/v1/chat/completions` for the quicksort prompt returns clean
   `content` (no `<channel>`/`<turn|>` leakage).
3. A realistic coding-harness-style request (system + multi-turn, ~1–2K-token prompt) returns within
   VRAM budget; **probe and document the prompt-length ceiling** (where it OOMs).
4. Point one local coding harness at the endpoint; confirm a basic edit/answer round-trip.

## Out of scope (v1)

- True per-token streaming, native tool-call JSON, multi-slot concurrency, vision inputs.
- Docker image as the primary path (host script is primary while Docker is unavailable here).
- Catalog DEFAULTS promotion / production status (gated on an upstream OpenAI server).

## Non-negotiables (CLAUDE.md)

- Don't claim TPS we didn't measure; label diffusion throughput as "effective tok/s" with method.
- Don't ship as production; keep ⏸️ Upstream-gated, untracked-until-validated where applicable.
- All upstream links go through `docs/UPSTREAM.md` rows, not inline scatter.

## Outcome (2026-06-11) — built & validated

All three units built and exercised on the rig:
- **C++ patch** (`llama-cpp/patches/diffusion-ndjson-server.patch`, 7 hunks) — `DG_NDJSON` mode +
  `enable_thinking` plumbing. Builds clean against PR #24423 sha `c84e85a` (SM 86).
- **FastAPI shim** (`llama-cpp/serve/`) — `/v1/chat/completions` + `/v1/models` + `/health` +
  single-chunk SSE. Strips `<|channel>thought…<channel|>` / `<turn|>`; thinking off by default;
  HTTP 413 on prompt-overflow (`steps==0`).
- **Host launcher** `serve.sh` (+ `requirements.txt`); container form sketched in
  `compose/single/q4-k-m/openai-shim.yml` (not built — no Docker here).

Validated: multi-turn + system, streaming, clean output (no marker leak), correct code for
quicksort/is_prime/fib/merge-sort/SQL, VRAM 22.3–23.0 GB, 413 guard, **~75 tok/s effective**
(thinking-off, coding). Findings + numbers in `models/diffusiongemma-26b-a4b/README.md`;
UPSTREAM rows added for PR #24423 (llama.cpp) and #45163 (vLLM).

Known quirks (documented, not blockers): trivial ultra-short prompts return empty; prompt ceiling
~2800 tok at `DG_CTX=3072`; no native tool-call JSON yet. Not promoted to the curated catalog.
