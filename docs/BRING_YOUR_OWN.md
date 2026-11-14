# Bring your own model or compose

You don't need to touch the curated catalog to serve and validate your own model
on your rig. The same scripts that gate our shipped composes work on anything you
serve. The arc is **serve → tune → validate → catalog → share**: bring it up, dial it in
with the fast loops, then run the full gate once it's settled.

> **Prefer a guided flow?** The `c3` cockpit's producer lane fronts this same
> path as **Bring & Validate** — a 5-stage funnel (① Bring → ② Serve → ③ Tune →
> ④ Measure vs the curated bar → ⑤ Promote to catalog), launched with
> `c3 --contribute` (see [`tools/serve-cockpit/README.md`](../tools/serve-cockpit/README.md)).
> Everything below is the same flow driven by hand — use the scripts when you
> want the individual loops, the cockpit when you want the rails; they gate on
> the same tests.

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
| ik-llama | ⚠️ none functional (all slugs `--force` only since 2026-08-12) | ⚠️ none functional |
| llama.cpp | ⚠️ none functional single-card since 2026-08-12 | `llamacpp/tess-dual-mtp` |
| vLLM | `vllm/minimal` | `vllm/dual` |
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

Examples — abliterated Qwen3.6-27B / Qwen3.8-27B:

| You have | Reuse this curated compose | Notes |
|---|---|---|
| an **AWQ + MTP** variant (vLLM) | `models/qwen3.6-27b/vllm/compose/dual/awq-bf16-int4/int8.yml` | match `--quantization` (`awq` / `compressed-tensors`) to the repo |
| a **GGUF + MTP** variant (llama.cpp) | a `llama-cpp` compose (path **B** above) | self-contained — simplest |
| a **Qwen3.8-27B FP8** variant with the MTP head (vLLM), e.g. `munekazu/Huihui-Qwen3.8-27B-abliterated-FP8` | `models/qwen3.8-27b/vllm/compose/dual/fp8/mtp.yml` | keeps `--quantization fp8`; Qwen3.8 reports the same config arch as Qwen3.6, so the `pull.sh` refusal naming `qwen3.6-27b` is expected |

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
  --engine vllm|llama-cpp|sglang|other --tag <your-tag>
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

## 4. Catalog it on your own rig — the LOCAL layer

Serving by hand (§1) gets you a running model but not a *first-class* one: no
`launch.sh` / `switch.sh --list` discovery, no VRAM projection, no guard coverage.
You can have all of that **without touching a single tracked file**.

**If you already have a working compose, that is all you need:**

```bash
bash scripts/catalog.sh register --compose ./my-compose.yml \
     --engine my-llamacpp --engine-type llama.cpp \
     --weights /path/to/model.gguf            # or an HF config.json
bash scripts/preflight-add-model.sh  my-llamacpp/<your-model>   # diagnose-profile + 9 catalog guards
```

`register` reads your compose and fills in what it can — engine, context, KV format,
tensor split, port, model id, weights path — then **prints every value it resolved,
marked `(given)` or `(derived)`, before writing anything**. A value your compose does not
state is marked `(defaulted)`: it is a placeholder, and the line after the list names the
`catalog.sh update --set …` that records what the model actually serves with once it is
up (an engine that sets its context or KV format at boot states neither in the compose).
`--dry-run` shows the whole plan and writes nothing. A compose is read mechanically: it
cannot know whether your `-ts 1,1` means a layer split or tensor parallelism in the
catalog's sense, so check the list rather than trusting it.

The copy in your layer keeps the compose's own place when it is filed the way the repo
files them, `…/compose/<topology>/<quant>/<serving>.yml`: registering
`multi4/qengine-int4/mtp.yml` files it as `multi4/<variant>/mtp.yml`, and when the compose
states no tensor-parallel size the directory supplies it (`multi4` is 4, marked
`(from its directory)`). A compose anywhere else is filed by its tensor-parallel size, as
`base.yml`.

Three things it will **refuse** rather than guess, because guessing wrong is silent:

| refusal | why |
|---|---|
| `--engine` | your image matches nothing we ship. Recording `engine: unknown` is worse than stopping. |
| `--weights` | a compose cannot state `hidden_size`; the dims come from your GGUF header or an HF `config.json`. Refused **before** writing, so a half-written layer never happens. |
| `--engine-type` | what your engine *behaves like* (`llama.cpp`, `vllm`, …). It drives drafter and feature logic, and a wrong value fails silently. |

**Running your own engine build is expected, not exceptional.** If `--engine` names an
engine the catalog does not know, a profile is written for it under
`profiles-local/engines.d/` from evidence only: your image, the KV format your compose
actually uses, and the compute capability of the card it demonstrably runs on. Capability
blocks it cannot verify are left **empty** — the stack makes no promises on your behalf.
Widen them once you have measured. This is exactly how our own fork is registered:
`llamacpp-club3090` is a distinct `id` with `type: llama.cpp`.

To register the engine first, from what you know rather than from one compose, use
`register-engine`, then register its slugs with `--engine <id>`:

```bash
bash scripts/catalog.sh register-engine --id vllm-myfork --type vllm \
     --image ghcr.io/you/vllm-myfork@sha256:… \
     --kv-formats fp8_e5m2 --families qwen3-next-hybrid --weight-formats autoround \
     --relation fork --base vllm --base-version 0.31.0 \
     --source https://github.com/you/vllm-myfork --source-ref <commit>
```

- **What's required:** `--type` is what the engine behaves like (`vllm`, `llama.cpp`,
  `sglang`, `exllamav3`), and `--image` is the image it runs.
- **What it refuses, before writing:** an id the catalog already has, a KV format no
  hardware profile lists, and an `--image-env` the launchers don't know. The last would
  stop every launch of the engine's slugs.
- **What it warns about:**
  - Without `--kv-formats`, `--families` or `--weight-formats`, `diagnose-profile` reports
    those checks failing for every slug on the engine.
  - Without the lineage flags, the engine registers but can't be exported for a PR yet.
- **Later changes:** `catalog.sh update-engine --id <id>` widens or narrows the lists once
  you've measured (`--add kv_formats=bf16`, `--remove …`), and sets lineage, the image or
  the display name (`--set relation=fork`, `--set image=…`). It validates the same way,
  refuses a curated engine, and puts the file back if the profiles stop loading.
- `unregister` keeps an engine registered this way when its last slug goes. Remove it
  with `catalog.sh unregister-engine --id <id>`, which refuses while any local slug
  still runs on it and names the `unregister` commands for those slugs.

To remove one again:

```bash
bash scripts/catalog.sh unregister --slug my-llamacpp/<your-model>   # --dry-run first if you like
```

It refuses any slug that is not in your local layer, so a curated entry is unreachable
from it — those are git-tracked and git is their removal tool.

<details><summary>Hand-authoring the spec instead</summary>

Still supported, and what `register` builds for you underneath:

```bash
python3 scripts/lib/profiles/promote.py --spec-file <spec>.json     # --layer local is the DEFAULT
```

</details>

That writes `scripts/lib/profiles-local/` — `models.d/<id>.yml`, `composes/<id>/…`
and `registry.local.json`. Slugs use the **same `<engine>/<name>` shape as curated
ones** — the layer decides where the *files* go, not what the slug is *called*, and
provenance lives in the entry's `origin` field rather than in the name. That is what
lets you name the engine you actually run (`my-llamacpp/my-model`, not just ours),
and it means publishing later flips a field instead of renaming a slug your scripts
and notes already point at. On an engine the catalog already uses, `register` takes
the catalog's prefix and directory for it (`vllm-stable` → `vllm/<model>`, filed under
`vllm/`), so the slug exports under the name it has. A local slug whose name collides with a shipped one is
**shadowed** — the curated row wins the lookup and yours is marked, not deleted.

> ⚠️ The old `local/<name>` namespace was removed (#1202). A `local/…` slug is now
> refused with the replacement spelled out; re-register it as `<engine>/<name>`.

### Seeing what you registered

Because local slugs now share the curated `<engine>/<name>` shape, they are marked
rather than named:

```bash
bash scripts/switch.sh --local          # only the models YOU registered
bash scripts/switch.sh --list --all     # everything; yours are tagged “· local”
bash scripts/catalog.sh list            # the whole local layer, and what in it needs attention
```

`catalog.sh list` covers more than slugs:
- **Contents:** engines (and whether `register-engine` or `register` wrote them), models,
  the variants attached to curated models, and slugs.
- **Problems:**
  - **broken:** a missing compose, engine, model or weights variant, or a file that doesn't parse;
  - **shadowed:** a curated slug now has your slug's name;
  - **unused:** an engine, model, attached variant or compose nothing uses.
- It reads the files directly, so it still answers when the layer is broken. `--json`
  prints the same as data.

`catalog.sh prune` removes what `list` reports as unused:
- **What goes:** attached variants and extension files no slug uses, composes no entry
  uses, engine profiles `register` wrote that no slug runs on, and local models with
  no slugs.
- **What stays:** an engine registered with `register-engine`, and anything broken or
  shadowed. Those need a decision, so they're listed and left alone.
- **Safety:** it prints every path first. Without a terminal it refuses unless `-y` is
  given, because git can't bring these files back. If the catalog stops loading
  afterwards, every file is put back.

```
  single   my-llamacpp/my-model    gguf/base.yml    (NA: 66K) · local
```

Or from the cockpit, which manages the layer without dropping to the CLI:

| Key | Where | Does |
|---|---|---|
| `ctrl+l` | anywhere in c3 | opens the local layer — every slug you registered, `shadowed` included |
| `r` | on a row | unregister it |
| `n` | on a row | rename it (moves the compose tree when the engine changes) |
| `e` | on a row | edit one field, `KEY=VALUE` |
| `esc` | | close |

`ctrl+l` is advertised in the footer only while the layer has something in it.
Every one of those actions is a repo write, so it goes through the same confirm
gate as a serve — you see the `catalog.sh` command before anything is touched,
and `Enter` commits it. The catalog re-reads itself afterwards; you do not need
to refresh. A curated slug is unreachable from this view by construction.

If a later `git pull` ships a curated slug under a name you already used, the
listing tells you so instead of quietly swallowing it:

```
  ⚠ shadowed local slug(s): vllm/minimal
    A curated entry now ships under that name, and core wins the lookup.
    Your registration is intact but unreachable by slug — rename it:
```

Core wins on purpose: a stack update must never silently change what one of our
slugs points at. Your files are untouched — rename it and it is reachable again:

```bash
bash scripts/catalog.sh rename --slug vllm/minimal --to my-llamacpp/minimal --dry-run
bash scripts/catalog.sh rename --slug vllm/minimal --to my-llamacpp/minimal
```

The slug's namespace **is** the engine, so renaming across engines moves the compose
tree and rewrites the entry's `engine` field with it — and is refused outright if the
target engine has no profile, rather than leaving you with a slug that claims an
engine which does not exist.

You can also edit an entry in place without re-registering:

```bash
bash scripts/catalog.sh update --slug my-llamacpp/my-model --set workload=fast-chat --set max_ctx=32768
bash scripts/catalog.sh update --slug my-llamacpp/my-model --set weights_variant=q8 --set "status_note=testing q8"
```

- **Editable fields:** `workload`, `max_ctx`, `max_num_seqs`, `mem_util`,
  `default_port`, `kv_format`, `tp`, `drafter`, `status`, `offload`, `spec_method`,
  `kvcalc_key`, `weights_variant`, `chat_template` and `status_note`.
- **Validated before writing:** the entry has to pass the same check the loader runs
  (an invalid `status` is refused), and `weights_variant` has to be one of the model's
  variants. If the catalog still fails to load afterwards, the file is put back.
- `status_note=` clears the note.

`origin` is not editable — it is stamped by the loader, and a local row able to call
itself `core` would hide from the very listings meant to mark it. `model` is not
editable either: it names the files on disk, so changing it is a move, not an edit.

The directory is
**gitignored** (everything except its README and `.gitignore`), so:

- `git pull` and branch switches can **never** conflict with or overwrite your models;
- nothing in the curated catalog is touched — delete the files to revert completely;
- `get_registry()` merges your entries into the catalog every launcher reads, so
  `switch.sh --list` and `launch.sh <slug>` just work;
- a local entry can never become a *curated* default, and a broken local layer
  fails loudly rather than silently shrinking the catalog.

> **Ports: your models live in the 202xx band.** The curated catalog occupies
> **8010–8299**, and it grows — so a local model parked in that range can collide
> with a slug that arrives in a later `git pull`, and it turns the repo's own
> `test-compose-port-conflicts` guard red on your checkout. `promote.py` refuses a
> local port that is already curated and tells you the deterministic replacement
> (`20200 + crc32(model_id) % 100`) — the same value c3's Promote scaffold assigns.
> Set your compose's `${PORT:-NNNN}` to match.

> ⚠️ **One command destroys this layer: `git clean -xdf`.** Your models here are
> *gitignored*, which is what makes them survive `git pull`, a branch switch, and even
> `git reset --hard` (all measured). But `-x` tells `git clean` to remove **ignored**
> files too, so the reflex "really clean the repo" command deletes every model you
> registered — silently, with no recovery. `git clean -fd` (no `-x`) is safe; so is
> `git stash -u`, though `git stash --all` is not.
>
> ⚠️ **It is not a backup either.** The layer lives inside this checkout and is resolved
> from the repo root it was imported from, so a second clone has its own empty layer and
> re-cloning loses everything. `export_pr.py --out <dir>` (§5) writes a complete portable
> bundle — the easiest way to keep a copy outside the tree.

> **Guided equivalent:** c3's Bring & Validate lane, stage **⑤ Promote** — the
> scaffold pre-fills every arch fact the deriver knows and you fill in
> `display_name` + `family`. Its key line names all three destinations:
> `⏎ Write LOCAL layer · C WRITE CORE REGISTRY (needs C3_ALLOW_CORE_PROMOTE=1) · E Export as PR bundle`.

## 5. Share it / contribute it back

**The local layer is a staging ground for a contribution, not a dead end.** What you
send depends on which case it is ([ADDING_MODELS.md → Which case is
this?](ADDING_MODELS.md#which-case-is-this)), and `export_pr.py` turns each one into a
ready-to-commit bundle:

| You validated | Case | In your local layer | Export spec | The bundle |
|---|---|---|---|---|
| a model the catalog doesn't have | 3 | `models.d/<id>.yml` (`catalog.sh register` or `promote.py` wrote it) | `{"model_id": "<id>"}` | `models/<id>.yml`, composes, `registry-entry.yaml`, `PR_BODY.md` |
| a new slug of a model we ship, on an engine that already serves it | 1 | `extends.d/<id>.yml`, a compose and an entry: `catalog.sh register --model <id>` ([below](#attaching-a-slug-to-a-model-we-ship)) | `{"model_id": "<id>"}` | composes, `registry-entry.yaml`, `weights-snippet-<id>.yaml`, `PR_BODY.md` |
| the same, on an engine that doesn't serve it yet | 2 | the same | the same | the same |
| an engine we don't ship | engine | `engines.d/<id>.yml` (`catalog.sh register-engine`, or `register` as a side effect) and its slugs | `{"engine_id": "<id>"}` | `scripts/lib/profiles/engines/<id>.yml`, its first slugs, `PR_BODY.md` |

A model or an engine we don't have starts with a proposal, and the PR waits for the
go-ahead: [model proposal](https://github.com/noonghunna/club-3090/issues/new?template=model-proposal.yml),
[engine proposal](https://github.com/noonghunna/club-3090/issues/new?template=engine-proposal.yml).
A new engine's PR carries its first slugs on a model we already serve; slugs on a model
we don't have go in a model PR after it, so the engine export leaves them out and says so.

```bash
B=~/pr-bundle-<id>                        # outside any checkout: the PR branch lives in another one
mkdir -p "$B" && echo '{"model_id": "<id>"}' > "$B.json"   # or {"engine_id": "<id>"}; "slugs": [...] sends only some
python3 scripts/lib/profiles/export_pr.py --spec-file "$B.json" --check --network            # validate, write nothing
python3 scripts/lib/profiles/export_pr.py --spec-file "$B.json" --network --out "$B"         # write the bundle
```

**What it checks.** It refuses (exit 3, nothing written) on anything a maintainer would
bounce the PR for:

- **Required fields:** a real `display_name` and `family` for a new model; `hf_repo`
  and `files:` on every new weights variant, plus `verify_glob` for GGUF; complete
  registry kwargs; a `kvcalc_key` on vLLM-type engines; the compose's `Status:` header.
- **The catalog rules** (`scripts/lib/profiles/catalog_rules.py`), the same ones
  `catalog.sh register` printed as advice:
  - the engine and container name read as the same engine family;
  - the internal port is one the serving scripts find;
  - the cache mount runs with the host group;
  - for an engine: lineage, and an image pinned by digest (or, for a stock or patched
    engine, a release tag).
- **The network rules**, with `--network`: `setup.sh` can download the weights from
  Hugging Face, and the engine image pulls with no registry login. A rule that couldn't
  run (no docker, a rate limit) counts as a failure there, never as a pass.
- **Overwrites:** a slug, compose or engine profile the catalog already has.

**What it does for you:**

- **Placement.** Composes go where core keeps that engine's composes (a compose
  `catalog.sh` filed under `vllm-stable/` lands in `vllm/`), and the slug keeps the
  prefix core uses there.
- **Mounts.** The local layer sits three directories deeper than `models/`, so a
  compose that boots locally reaches repo files with three more `../`.
  - The export re-points those mounts at the same targets.
  - It carries a patch or template that only your local layer has into the bundle.
  - It lists every such change in its plan and in `PR_BODY.md`.
- **Ports.** Your slug runs on a 202xx port, but curated slugs live on 8010–8299.
  - The export gives the slug the lowest curated port that no curated slug holds and none of the
    repo's own services (`services/`: Open WebUI 8080, ComfyUI 8188, ...) listens on.
  - It moves both the registry entry and the compose's `${PORT:-NNNN}`.
  - Your local layer keeps its 202xx port, so the slug you run doesn't move.
  - A slug already on a free curated port keeps it.
  - A compose with no `${PORT:-NNNN}`, or with several different ones, is refused.
- **The PR body.** `PR_BODY.md` is the repo's own
  [slug](../.github/PULL_REQUEST_TEMPLATE/slug.md) or
  [engine](../.github/PULL_REQUEST_TEMPLATE/engine.md) template:
  - the header is filled in and your case ticked;
  - the sections for other cases are dropped;
  - **Export check** holds the rules' output, and says whether the network rules ran.

  Run the final export with `--network`, so that section records them.

**Then, on a branch of your fork:**

1. Make the branch in a fresh `git worktree add` (or clone), not in this checkout.
   - A new worktree starts with an empty local layer.
   - Here, your local copy would collide with the curated one you're adding: an
     `extends.d` file that repeats a curated weights variant stops the profile loader.
   - Settings and compile caches are shared. So are your weights, if `MODEL_DIR` points outside
     the checkout (`bash scripts/settings.sh show`).
2. Copy `$B/models/` (and `$B/scripts/`, for an engine) into the repo: `cp -r "$B/models" .`
3. Merge `$B/registry-entry.yaml` with the command in its header.
4. Paste `$B/weights-snippet-<id>.yaml` under `weights:` in `scripts/lib/profiles/models/<id>.yml`, if the bundle has one.
5. Run `bash scripts/preflight-add-model.sh <slug>`. The slug is curated on that branch, so the rules are required now.
6. Run the full `scripts/tests` suite.
7. Open the PR with `gh pr create --body-file "$B/PR_BODY.md"`, then fill in its Evidence section.

Not coming from the local layer? Open the PR with the template (`?template=slug.md` or
`?template=engine.md` on the compare URL), and paste
`python3 scripts/lib/profiles/catalog_rules.py --slug <slug> --network` (or
`--engine <id>`) into its Export check section.

Writing the curated catalog **directly** is maintainer-only and double-gated
(`promote.py --layer core` **plus** `C3_ALLOW_CORE_PROMOTE=1`). `promote.py` reads
that gate from your shell or, failing that, from your saved settings
(`~/.config/club-3090/`, then a legacy repo-root `.env`), and says so when a file
set it. `export` it in your shell or pass it per-invocation; a saved one leaves
the gate open for every run.

- Format your numbers with the [Results Card](RESULTS_CARD.md)
  (Serving · Quality · Takeaways).
- Contributing the compose upstream? **One model — or one feature/concern — per
  PR** (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)). Reproduce-before-promote: we
  re-run community numbers on our own rig before promoting past `🧪`.
- Your compose must carry the **Profile header** — the `# Profile (at-a-glance):`
  block with a `Status:` field (and a `Caveats:` line if `⚠️`/`👁️`/`⏸️`/`🗑️`).
  It's gate-tested (`test-compose-status-drift`); schema in
  [`CLAUDE.md`](../CLAUDE.md) → "Profile schema header."

### Attaching a slug to a model we ship

Cases 1 and 2 serve a model the catalog already has, so the slug attaches to it rather
than inventing a new model id. `register` does that when `--model` names a curated model:

```bash
# reuse one of its weights variants
bash scripts/catalog.sh register --compose ./my.yml --engine vllm-stable \
     --model qwen3.6-27b --weights-variant autoround-int4
# or bring weights of your own, as a new variant
bash scripts/catalog.sh register --compose ./my.yml --engine vllm-stable \
     --model qwen3.6-27b --weights-variant my-awq --weights /path/to/weights --hf-repo org/repo
```

- **What it writes:**
  - `scripts/lib/profiles-local/extends.d/<id>.yml`, holding only the **new** variant (or
    no weights at all when you reuse a curated one) and any GPU count the curated
    profile doesn't list;
  - your compose;
  - the registry entry, whose `model` is the curated id.

  It writes no model profile.
- **The slug and the compose follow the layout the catalog already uses for that
  engine:** `vllm-stable` slugs are `vllm/…`, filed under `vllm/`. So the slug exports
  under the same name. The default slug is `<prefix>/<id>-<topology>-<variant>`;
  `--slug` names it yourself.
- **The extension is shared** by every slug attached to that model, and a broken one
  would stop the profile loader for everything. So the merged file is validated
  before anything is written, and if a check after the write fails, every file is
  put back.
- `--dry-run` shows the plan. `unregister` removes the slug and keeps the extension.
- **It prints what `export_pr.py` will still need:**
  - `files:` for a new variant, and its `hf_repo`, set with `catalog.sh update-weights --model <id>
    --variant <variant> --set "files=*.safetensors,*.json" --set hf_repo=<org/repo>`;
  - a `kvcalc_key` on a vLLM-type engine, set with `catalog.sh update --slug <slug> --set kvcalc_key=<id>:<profile>`.

The file format, for editing it by hand: [profiles-local
README](../scripts/lib/profiles-local/README.md), "Extending a core model".

## See also

- [`PULL.md`](PULL.md) — the safetensors evaluate-and-fetch gate in depth
- [`ADDING_MODELS.md`](ADDING_MODELS.md) — the heavier path: promoting a model into the curated catalog
- [`QUALITY_TEST.md`](QUALITY_TEST.md) — what the 8-pack quality harness measures
- [`CLIFFS.md`](CLIFFS.md) — the prefill / accumulated-context failure modes
- [`SINGLE_CARD.md`](SINGLE_CARD.md) / [`DUAL_CARD.md`](DUAL_CARD.md) — workload → config
