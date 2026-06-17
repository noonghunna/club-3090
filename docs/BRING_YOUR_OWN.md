# Bring your own model or compose

You don't need to touch the curated catalog to serve and validate your own model
on your rig. The same scripts that gate our shipped composes work on anything you
serve. The arc is **serve → tune → validate → share**: bring it up, dial it in
with the fast loops, then run the full gate once it's settled.

## 1. Serve it

Which path depends on what weights you have.

> **BYO models live outside the curated registry**, so the registry-driven
> wizards (`launch.sh`, `switch.sh`) can't list or boot them — those only resolve
> cataloged slugs. You boot a BYO compose **directly** with `docker compose -f
> <path> up -d` + env vars, and you drive the eval scripts by **endpoint**
> (`--url` / `MODEL=` / `URL=`), never by a registry slug. (The `switch.sh --list`
> below is only for *finding* a shipped compose to copy as a template.) Getting
> the wizards to recognise your model is the separate, heavier catalog path —
> [`ADDING_MODELS.md`](ADDING_MODELS.md).

### A — An HF safetensors repo  →  `scripts/pull.sh`

```bash
scripts/pull.sh <org/Model> --profile-like <registry-key> --dry-run             # evaluate
scripts/pull.sh <org/Model> --profile-like <registry-key> --out byo.yml --yes   # fetch + emit compose
docker compose -f byo.yml up -d                                                 # boot it directly
```

`pull.sh` writes a standalone compose (`--out`) and **does not** register the
model — you boot the emitted file directly, same as path B.

`--profile-like` is **required** — it borrows a curated config's *runtime shape*
(engine, KV format, tensor-parallel degree) to evaluate your model against our KV
math, and tells you whether it fits **before** downloading. Pick the key that
matches the engine + topology you want to test:

| Target | `--profile-like` key |
|---|---|
| vLLM, single GPU (TP=1) | `vllm/minimal` |
| vLLM, dual GPU (TP=2) | `vllm/dual` |

`pull.sh` is **safetensors-only** by design — GGUF / llama.cpp serving is a
separate cross-engine concern and out of scope here. A GGUF-only repo will
hard-stop; use path B. See [`PULL.md`](PULL.md) for the full gate taxonomy.

### B — A GGUF you already have  →  copy the closest compose

`pull.sh` doesn't serve GGUF, so for **ik-llama / llama.cpp / beellama** (single
*or* dual) you copy an existing compose and point it at your file. List every
shipped compose with its path and registry key:

```bash
bash scripts/switch.sh --list          # single + dual for this machine
bash scripts/switch.sh --list --all    # include multi-GPU
```

Copy the one closest to your target engine + topology as a starting point:

| Engine | Single-GPU starting point | Dual-GPU starting point |
|---|---|---|
| ik-llama | `ik-llama/iq4ks-mtp` | `ik-llama/apex-mtp-quality-dual` |
| llama.cpp | `llamacpp/mtp` | (multi-GPU via `--list --all`) |
| beellama | `beellama/dflash` | `beellama/qwen-dflash-dual` |

Then point it at your weights and boot directly — no registry or profile entry
needed:

```bash
env MODEL_DIR=/path/to/your/models \
    GGUF_FILE=relative/path/to/your-model.gguf \
    PORT=8062 CTX_SIZE=104000 \
    CUDA_VISIBLE_DEVICES=0 ESTATE_CONTAINER=byo-eval \
    docker compose -f models/<model>/<engine>/compose/<topology>/<quant>/<serving>.yml up -d
```

**Single vs dual** is encoded in the path/key, not a flag: for vLLM the
`--profile-like` key sets TP (`vllm/minimal` = TP=1, `vllm/dual` = TP=2); for GGUF
engines you copy a `single/` or `dual/` compose (dual splits layers / `-ts` across
cards). Custom all-reduce stays disabled on PCIe (no NVLink). See
[`SINGLE_CARD.md`](SINGLE_CARD.md) / [`DUAL_CARD.md`](DUAL_CARD.md).

### C — Swap a curated model for a fine-tune / abliterated variant  →  reuse its compose

If your model is the **same architecture** as one we already ship (e.g. an
abliterated or fine-tuned Qwen3.6-27B — config arch `Qwen3_5ForConditionalGeneration`),
`pull.sh --profile-like` will **refuse** it:

> `not generic-dense eligible (arch 'Qwen3_5ForConditionalGeneration') … NOTE: this arch is the curated 'qwen3.6-27b' …`

That's expected, not a bug — the generic deriver only fit-prices **generic-dense**
transformers, and Qwen3-Next / Gemma-4 are **hybrid / MoE** that need the curated
hand-tuned flags (mamba-cache-mode, MTP spec-config, the quant kernel). So **don't
derive — reuse the curated compose and swap the weights.** Three things to get right:

- **Artifact ↔ engine:** match the weights *format* to the engine. **GGUF goes to a
  `llama-cpp` / `ik-llama` / `beellama` compose, never a vLLM one** (vLLM serves
  safetensors — AutoRound / AWQ / GPTQ / fp8 — not GGUF). Pointing a vLLM compose at a
  `.gguf` is rejected at the `pull.sh` gate (`weight_format 'gguf' not loadable by
  engine … — serve GGUF with a llama.cpp/ik-llama/beellama compose`); the rows below
  already pair them correctly.
- **Quant:** the curated composes load a *quantized* checkpoint (AutoRound-int4 / AWQ
  / fp8). A **bf16 full-weight** repo won't fit 2×24 GB (~54 GB) — grab a pre-quantized
  variant whose format matches a compose's `--quantization`.
- **MTP:** the dual / "max" configs use a built-in MTP head for spec-dec. Use a
  **`-MTP` variant** (the head is embedded in the checkpoint), or drop
  `--speculative-config`.

Example — abliterated Qwen3.6-27B:

| You have | Reuse this curated compose | Notes |
|---|---|---|
| an **AWQ + MTP** variant (vLLM) | `models/qwen3.6-27b/vllm/compose/dual/awq-bf16-int4/int8.yml` | match `--quantization` (`awq` / `compressed-tensors`) to the repo |
| a **GGUF + MTP** variant (llama.cpp) | a `llama-cpp` compose (path **B** above) | self-contained — simplest |

```bash
# 1. download the quantized + MTP variant
hf download <org/Model-AWQ-MTP> --local-dir /mnt/models/huggingface/<your-name>
# 2. copy the matching curated compose; change --model to your weights path
#    (+ match --quantization; drop --speculative-config if there's no MTP head)
# 3. boot directly:
docker compose -f models/qwen3.6-27b/vllm/compose/dual/awq-bf16-int4/int8.yml up -d
```

It inherits everything from the curated config — TP, KV dtype, chat template, tool
parser, MTP, context — you only swap the weights. (Claude/Codex on the repo can do
the swap + quant-flag match for you.)

## 2. Tune it

Bring-up rarely lands on the best config first. Tune with the **fast loops** and
**change one variable at a time** — keep the long `rebench-full` run (§3) for the
end, it's a 2.5–3.5 hr gate, not a tuning loop:

- `verify-full.sh` — ~2 min functional smoke (boots, serves, tool-calls, streams).
- `verify-stress.sh` — ~5–10 min context / NIAH ladder.
- `bench.sh` — ~3–5 min TPS (3 warm + 5 measured).
- `quality-test.sh --full` — the 8-pack /150 quality read; or `--medium` (5
  deterministic packs /75) for a quick probe between config changes.

```bash
MODEL=<served-name> URL=http://localhost:<port> bash scripts/verify-full.sh
MODEL=<served-name> URL=http://localhost:<port> bash scripts/quality-test.sh --medium   # quick probe
MODEL=<served-name> URL=http://localhost:<port> bash scripts/quality-test.sh --full      # 8-pack /150
```

### Context size — find the *real* ceiling
The advertised max context is rarely the fillable ceiling on 24 GB: KV is
pre-allocated at boot, so a too-large `CTX_SIZE` leaves no headroom for the
prefill and a high-context request OOMs and wedges the server (we've watched a
"160K" compose die at ~125K).
- `verify-stress.sh` runs a staggered NIAH ladder to ~0.92 × n_ctx and reports
  the **fillable** ceiling + VRAM margin. An `HTTP 0` at a rung is an OOM/wedge,
  **not** a recall miss — drop `CTX_SIZE` and re-run.
- ik-llama: `--fit` auto-sizes context to free VRAM; otherwise step `CTX_SIZE`
  down until verify-stress passes with ≥1 GB margin.
- Watch the **cliffs** — the single-prompt prefill cliff (~50–60K on DeltaNet)
  and the accumulated-context Cliff 2b (~21–26K, soak-only). See
  [`CLIFFS.md`](CLIFFS.md); single-card long-context often belongs on dual.

### NIAH — confirm it actually *uses* the context
verify-stress's needle ladder requires **exact** recall at each depth (10K / 30K
/ 60K / 90K + the ceiling rungs). Allocating context isn't using it — a model
that allocates 200K but misses the needle at 90K is not a 200K model.

### KV-cache quant — context vs fidelity
Lower KV bits buy more context for a small quality cost — but **the lever differs
by engine**:
- **llama.cpp / ik-llama / beellama** expose **separate K and V** cache types, so
  you can quantise them *asymmetrically*: `-ctk q4_0 -ctv q4_0` for max context
  (cheapest KV), or `-ctk q8_0 -ctv q5_0` (or `q4_1`) + `-khad -vhad` for the
  **K-high / V-low** pattern (Anbeeld) — keep the precision-sensitive K accurate,
  quantise V harder, for tighter quality at slightly less context.
- **vLLM** applies **one KV format to both K and V** — there is **no per-stream
  K/V split** (no `-ctk`/`-ctv`, no `-khad/-vhad`). Stock `--kv-cache-dtype` is
  `auto` (bf16) / `fp8_e5m2` / `fp8_e4m3`, and on Ampere (sm_86) FP8 KV is
  **storage-only** (no native FP8 compute). The **INT8 KV** path this stack uses
  (INT8 per-token-head, "PTH") is **not** a stock dtype — it's a **vendored engine
  patch**, shipped via the `int8.yml` composes, so you only get it by running a
  patched compose, not by flipping a flag. Either way it's still one whole-cache
  format, not an asymmetric K/V knob.
- Re-run verify-stress + a quality `--medium` after any KV change. See the
  KV-cache entry in [`FAQ.md`](FAQ.md).

### Speculative decoding — sweep the draft depth (MTP / DFlash)
Spec-decode only wins if **net wall TPS** improves — high acceptance does *not*
guarantee a win (built-in MTP on an MoE can be net-*negative* despite 80%+
acceptance, because the draft forward re-runs expert routing). Always compare
against the no-spec baseline.
- **ik-llama built-in MTP**: `--multi-token-prediction --draft-max N
  --draft-p-min 0.0`. Sweep `N` (2 → 5) — more draft tokens isn't always faster
  as acceptance decays with depth. n=2 is a common mainline sweet spot; ik
  single-card often likes n=4–5.
- **mainline llama.cpp MTP**: `--spec-type draft-mtp` (a *different* flag from
  ik's `--multi-token-prediction`).
- **DFlash (beellama)**: external-drafter path, tool-grammar-neutral; n-sweep
  the same way.
- Read acceptance-length (AL) + per-position accept from the logs, but **judge on
  the bench delta**.

### Batch / ubatch + sampling
- `-b` (batch) / `-ub` (ubatch) are first-class levers — a smaller `-ub` (e.g.
  1024) can unblock high-context prefill that OOMs at the default.
- Qwen3.6 sampling defaults: `temp 0.6, top_p 0.95, top_k 20, min_p 0.0,
  repeat_penalty 1.0`. `thinking on/off` shifts both latency and quality —
  validate the mode you'll actually serve.

### A/B discipline
Give every arm an identical **docker restart + fixed settle** before benching,
**≥3 runs/arm**, compared on the same segment at **matched power** — a cold or
lower-power arm fakes a regression. Never trust a single run.

## 3. Validate it — the full gate

Once the config is settled, run the full pipeline. It chains everything in one
pass: bench → verify-stress → 8-pack quality (think-OFF **and** think-ON) → soak.

```bash
bash scripts/rebench-full.sh \
  --url http://localhost:<port> --model <served-name> \
  --engine vllm|llama-cpp|sglang|beellama --tag <your-tag>
```

> **Optional for your own use — mandatory to contribute.** Just serving it
> yourself? The tuning loops above are enough. But a contribution PR that adds a
> model to the **central registry must include a full `rebench-full` run** so its
> quality and stability are on record before it ships — and we reproduce those
> numbers on our own rig before promoting anything past `🧪`. See the gate list in
> [`CONTRIBUTING.md`](../CONTRIBUTING.md#submitting-a-new-compose-variant--full-gate-list).

Notes:
- It's a **2.5–3.5 hr** run — that's why you tune with the fast scripts first and
  run this once at the end. `--resume` skips completed steps; `--skip
  soak,quality-thinking` trims; artifacts land in `results/rebench/<tag>/`.
- Pass `MODEL=<served-name>` to every script — they default to a Qwen name and
  404 against a different endpoint. A **clean, no-slash** model id (e.g.
  `mymodel-q4`) also keeps the sandboxed packs (HermesAgent, BugFind) routing
  cleanly; the `quality-test` wrapper already sets the localhost-resolve env so
  those packs can reach a host model.

## 4. Share it / contribute it back

- Format your numbers with the [Results Card](RESULTS_CARD.md)
  (Serving · Quality · Takeaways).
- Contributing the compose upstream? **One model — or one feature/concern — per
  PR** (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)). Reproduce-before-promote: we
  re-run community numbers on our own rig before promoting past `🧪`.
- Your compose must carry the **Profile header** — the `# Profile (at-a-glance):`
  block with a `Status:` field (and a `Caveats:` line if `⚠️`/`👁️`/`⏸️`/`🗑️`).
  It's gate-tested (`test-compose-status-drift`); schema in
  [`CLAUDE.md`](../CLAUDE.md) → "Profile schema header."

## See also

- [`PULL.md`](PULL.md) — the safetensors evaluate-and-fetch gate in depth
- [`ADDING_MODELS.md`](ADDING_MODELS.md) — the heavier path: promoting a model into the curated catalog
- [`QUALITY_TEST.md`](QUALITY_TEST.md) — what the 8-pack quality harness measures
- [`CLIFFS.md`](CLIFFS.md) — the prefill / accumulated-context failure modes
- [`SINGLE_CARD.md`](SINGLE_CARD.md) / [`DUAL_CARD.md`](DUAL_CARD.md) — workload → config
