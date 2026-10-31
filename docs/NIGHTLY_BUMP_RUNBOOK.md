# Nightly / image pin bump runbook

This is the procedure for moving a club-3090 compose to a newer engine image
(vLLM nightly hash, llama.cpp tag, SGLang variant, etc.) — and especially for
**collapsing** pin drift, where one engine has multiple distinct tags pinned
across composes.

It exists because:

- Each pinned image is **~30 GB on disk** for vLLM nightly builds (and several
  GB for other engines). Three distinct vLLM nightly hashes = ~100 GB of cached
  images on this rig before any sandbox / one-off images.
- Each compose mounts **patches** that target specific files inside the engine
  image. Patches written against nightly A may break against nightly B — files
  move, function signatures change, hunks fail to apply.
- **Bumping blindly** can succeed at boot but produce incorrect output at
  runtime (e.g., a spec-decode patch silently degrading), so we don't.
- We bump by **process**, with a verify gate, so we catch regressions before
  they ship.

When to consider a bump:

- A new vLLM nightly fixes a bug we depend on (track via `docs/UPSTREAM.md`).
- We have **pin drift** — `scripts/maintenance/list-image-pins.sh` shows >1 tag
  for the same engine. Goal: collapse to one tag per engine.
- Disk pressure on the docker partition (`docker system df` shows multiple
  large nightlies cached with low reclaimable).

When NOT to bump:

- During an active bench / paper / cross-rig validation cycle. Wait until the
  cycle ends.
- If the current pin is a stable production target with known TPS / quality
  numbers and no upstream signal demanding the move.

---

## The procedure

### 0. Identify scope

```bash
bash scripts/maintenance/list-image-pins.sh
```

This prints the pin distribution + flags drift. Decide:

- **Which pin do you want to retire?** (e.g., the least-used one)
- **Which pin do you want to consolidate to?** (usually the newest, or the one
  required by an active bug fix)
- **Which composes are affected?** (the script lists them per pin)
- **What patches do those composes mount?** (drives migration cost)

Order the work from **lowest patch surface first** — composes with `patches=none`
or `patches=vllm-marlin-pad` (small) are quick wins; composes with `patches=
vllm-gemma4-dflash-int8` (~13 patched files) are higher-risk.

### 1. Branch + bump one compose

```bash
git checkout -b bump-<engine>-<short-new-tag>-<compose-name>
```

Edit the compose's `image:` line to the new tag. Don't bump multiple composes
in one branch — keep blast radius small.

### 2. Patch survival check

For each patch the compose mounts:

```bash
docker pull <new-image>
# For each patched file, fetch the upstream version and diff against our patch
docker run --rm <new-image> cat /path/to/patched/file > /tmp/upstream-new.py
diff -u /tmp/upstream-new.py <repo>/models/<model>/<engine>/patches/<patch-dir>/<file>
```

Verdict per patch:

- **Hunks land cleanly** → no rebase needed.
- **Context shifted, hunks land with offset** → no rebase needed but verify
  carefully (line numbers in our patch may now be off if anyone reads them).
- **Hunks fail / structural change** → manual rebase. Use Codex / agent help if
  the patch is large (e.g., DFlash with 13 files touching `gpu_model_runner`).

If any patch fails to rebase, **stop**: file an issue documenting which file
changed in upstream, why our patch broke, and link the upstream commit. Decide
whether to:

- Wait for upstream to absorb the patch (preferred — see `docs/UPSTREAM.md`).
- Rebase the patch against the new internals (effort cost).
- Stay on the old pin (status quo — document the reason in the tracker).

### 3. Boot the new pin

```bash
gpu-mode <appropriate-mode>
# or, manually for one compose. sudo drops your environment, so hand compose the
# resolved settings (shell > ~/.config/club-3090/ > repo .env) the way gpu-mode does:
envf="$(bash <repo>/scripts/settings.sh compose-env-file)"   # 0600 temp file; it holds your secrets too
sudo docker compose --env-file "$envf" -f <compose-file> up -d; rm -f "$envf"
docker logs -f <container-name>
```

Watch for tracebacks during model load. If anything looks off, **stop** and
revert the compose's `image:` line.

### 4. Verify gate

In order:

```bash
bash scripts/verify-full.sh        # ~1-2 min — reachability, tools, streaming, MTP AL
bash scripts/verify-stress.sh      # ~5-10 min — longctx needle ladder, tool-prefill OOM
```

Both must be GREEN. If either fails, **stop**: revert the bump, file an issue
linking the failing check + the new pin, document in `docs/UPSTREAM.md`.

### 5. Bench delta

Run the canonical 800-word essay bench and confirm TPS within ~5% of the
pre-bump pin:

```bash
bash scripts/bench.sh > /tmp/bench-after-bump.txt
# Compare to the row in BENCHMARKS.md for the same compose
```

If TPS drops >5% with no obvious cause (e.g., new nightly removed a Marlin
optimization), file an upstream issue and decide if the bump is still worth it.

### 6. Land the bump

Record the pin in the engine profile, beside `install.spec`: the old `# pin-current:` line becomes the first entry
of `# pin-earlier:` (it is the rollback), and a new `# pin-current:` says what is pinned now (the release, its source
commit, the date and what was booted). `docs/UPSTREAM.md`'s Pinned images table says why an image is pinned and
carries no pin, so leave it; `bash scripts/maintenance/list-image-pins.sh --pins` shows the result.

```bash
git add <compose-file>
git commit -m "bump <compose-name> to <engine>:<new-tag>"
# Open PR — link to verify-full + verify-stress run logs in description
```

### 7. Pin retirement (when ALL composes for one tag have moved)

When `list-image-pins.sh` shows that the old pin has zero composes:

1. Delete the now-cached image from local docker:
   ```bash
   docker image rm <old-image>:<old-tag>
   ```
2. Add an entry to `docs/UPSTREAM.md` under "Retired pins" documenting the
   retirement date + the reason (e.g., "consolidated to nightly-X after PR
   #41745 merged").

This is the "free 30 GB of disk" moment.

---

## Anti-patterns

- **Bumping all composes in one PR.** If anything breaks, you don't know which
  one. Bump one at a time.
- **Skipping the verify gate** because "it booted, looks fine." Spec-decode
  regressions are silent. Always run `verify-full.sh + verify-stress.sh`.
- **Bumping during an active bench cycle.** TPS deltas pollute your numbers.
- **Forgetting to update `docs/UPSTREAM.md`.** Future you will thank present
  you for documenting which pin was retired and why.

---

## Engine-specific notes

### vLLM

- Nightly tag format: `vllm/vllm-openai:nightly-<8-char-commit-hash>`.
- Most volatile internal: `vllm/v1/worker/gpu_model_runner.py`. DFlash and
  speculator patches both touch it.
- Marlin kernel (`vllm/model_executor/kernels/linear/mixed_precision/`) changes
  rarely but breaks completely when it does.
- Bumping a vLLM nightly often requires Genesis pin re-evaluation too — check
  `docs/UPSTREAM.md` for the current Genesis pin compatibility.

### llama.cpp

- Stable tag `ghcr.io/ggml-org/llama.cpp:server-cuda` shifts under us — no
  hash. To get a deterministic pin, capture the digest:
  ```bash
  docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
  docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/ggml-org/llama.cpp:server-cuda
  ```
  Use the `@sha256:...` form in the compose if reproducibility matters.

### SGLang

- Variant tags (`cu13`, `cu13-gemma4`, etc.) — different from version tags.
  Variant naming changes more often than version numbers. Document the variant
  reason in the compose header.

### bucko-vllm

- Pinned by digest to DominikBucko's official `ghcr.io/dominikbucko/qwen38-flash-next-2x3090` release image
  (`engines/bucko-vllm.yml`). Bump by moving the digest to a newer release; check the image's OCI labels
  (`revision`, `base.name`) and update `lineage:` with them.
- The volatile part for us is `vllm/sampling_params.py`: `bucko-default-thinking-budget` anchors on its
  `validate_thinking_token_budget` call and refuses the boot when that moves. The core
  `vllm-default-thinking-budget` does NOT load on bucko's base (no `trace_decode_token_ids`), so don't switch to it
  without checking the new base.
- The composes set bucko's own launcher environment (`MTP_DEPTH`, `CPU_OFFLOAD_GB`, `QWEN38_*`,
  `VLLM_WNA16_STATIC_HOT_CACHE_*`): re-read the release notes for renamed or retired variables.

### Strata and Project Maya: the bump rule

Both engines release often, so a bump of either one has the same short shape. The two sections after this one say
what is specific to each. `scripts/strata-pin-bump.sh <engine> <release>` runs steps 2–5 (Strata also takes
`--family ista|swift` or `--slug`) and prints the PR summary, with the releases the bump skipped; the steps say what it
does, and what to do by hand when it refuses.

1. **Take a release only when it touches us.** Read the release notes and, for Maya, the commits since our pin (it
   releases several times a day). Take the release when they touch Linux CUDA serving (Strata) or GLM (Maya), the
   server or the launch path, or a bug that affects us. Otherwise skip it, and name the skipped releases in the next
   bump's PR.
2. **In the pin worktree,** pull the image and pin its digest everywhere: `scripts/engine-pin-bump.sh <engine>
   <repo>@sha256:<digest>` (the spec and the compose defaults), `lineage.source_ref`, the engine profile's two `pin-`
   lines and, for Strata, the "Pinned" line of `docs/engines/STRATA.md`. `docs/UPSTREAM.md` carries no pin, so a
   bump never edits it. The
   launchers inject the profile's image, so a boot from that worktree runs the new digest. Before that, check in the
   new image what the launchers rely on: for Strata, that `strata launch --kv` still offers `KV_FORMATS`, the `--spec`
   help still says setup writes `SETUP_SPEC` (both in `launcher/strata_config.py`) and the entrypoint still reads the
   settings the launcher sets; for Maya, that `maya.py`'s `MODELS` still has the quants `maya_prepare.py` names.
3. **Boot one dual slug** through `switch.sh`. Every slug of an engine runs the same image, so one boot shows the
   image serves. Single-card and multi4 slugs aren't booted for a bump.
   - **Maya:** any dual slug. A dual runs the MTP draft across the split.
   - **Strata:** a dual slug of the model family the release notes touch (ISTA or Swift; setup and packing are per
     family). When they touch neither, the family the previous bump didn't boot.
4. **Send 3 warm-up requests, then run `verify-full`.** For Maya, add a long-prompt needle when the release changes
   how prompts are read (prefill).
5. **Never skip these:**
   - check that the running container's image is the pinned digest
     (`docker inspect <container> --format '{{.Config.Image}}'`);
   - re-record the settings evidence, `python3 scripts/lib/profiles/engine_config_reads.py verify <engine> --record`
     (`test-engine-config-reads` stays red until it's re-checked on the new image);
   - for Maya, the image build's GPU-free checks, which catch a flag the launcher passes being renamed.

### Strata

- Pinned by digest to upstream's own image, `ghcr.io/niko1221/strata:<release>-cuda13`, which Strata's
  `docker-images` workflow builds from the release's signed engine packages (from 0.2.1 on; until 2026-10-20 we built
  `strata-club3090` ourselves). The workflow is run by hand and was run several times for 0.2.1, so resolve the tag
  to a digest just before pinning and pin the digest you booted. The checks are the
  [bump rule](#strata-and-project-maya-the-bump-rule) above; `lineage.source_ref` is the release's commit.
- The volatile parts are the image's own entrypoint (`/opt/strata/entrypoint.sh` and the settings it reads:
  `MODEL`, `FAMILY`, `MODEL_PATH`, `CONTEXT`, `KV`, `VISION`, `GPUS`, `PRESET`, `API_KEY`), the run config setup writes
  (`/data/install/strata-<family>.json`, where `launcher/strata_config.py` puts the launch settings) and how setup
  keeps an option it doesn't choose itself (`user_args`). Re-read the release notes for those.
- The composes forward a short list of Strata's own engine switches (`STRATA_GDN_TC`, `STRATA_GDN_CONVL2`;
  docs/engines/STRATA.md, "Engine switches"). When a release turns on a new default that reaches our slugs and names
  the variable that turns it off, add that variable to all five composes and to `ENGINE_EXACT` in
  `test-compose-env-declared-unused.sh`, once `strings` on the image's
  `/opt/strata/components/engine/<version>/strata` shows the engine reads it. A listed name the engine no longer
  reads comes out the same way.
- The data volume (`strata-data-v2`) carries one pack per model family and the shared MTP layer between versions,
  in a folder every topology shares; a release that changes the pack format re-prepares a family's pack at its first
  start in each topology.

### strata-maya

- Pinned by digest to `ghcr.io/noonghunna/strata-maya-club3090`, which the maintainer builds on the reference
  rig from each Project Maya release we take, with our own Dockerfile (`build.sh --tag <vX.Y.Z> --push` from a
  clean master checkout; recipe in `models/glm-5.3-flash/strata-maya/image/`). A published image is not a pin: the
  checks are the [bump rule](#strata-and-project-maya-the-bump-rule) above (`verify-full`'s MTP check reads Maya's
  drafts from the engine log through `strata_metrics.py --container`), and `lineage.source_ref` is the image's
  `club3090.upstream.revision` label.
- The volatile parts are Maya's own interfaces, which the launcher leans on: the `serve/server.py` and
  `tools/iq_pack.py` flags (the image build fails when one disappears), the pack layout (`sources.json`, the
  pack next to the shards), `maya.py`'s `MODELS` table (the launcher's SHA-256 check reads it; a quant renamed
  there stops the first start with the reason) and the `glm spec:` log line our draft count parses. Maya
  releases several times a day, so read the commits between the two pins for any of those.
- The data volume keeps one pack per quant between versions. A release that changes the pack format needs
  `REPACK=1` once (the launcher rebuilds a pack only when a shard changed).

### vllm-hyperqwen

- Pinned by digest to syv-ai's official `ghcr.io/syv-ai/hyperqwen`, which their CI builds on every push to
  main (`:sha-<7>`; there are no versioned releases). A new image is not a pin: boot
  `vllm-hyperqwen/qwen38-27b-single-ultrafast` on the new digest, run `verify-full`, then
  `scripts/engine-pin-bump.sh vllm-hyperqwen ghcr.io/syv-ai/hyperqwen@sha256:<digest>`, and update
  `lineage.source_ref` (the commit behind `:sha-<7>`) and `base_version` (`vllm.__version__` in the image).
- The volatile parts are HyperQwen's own interfaces, which the compose leans on: its launcher's env names
  (`SPEC`, `CTX`, `DFLASH_TOKENS`, `GPU_UTIL`, `EXTRA_ARGS`, `PORT`, `BIND_HOST`), the image's
  `docker/entrypoint.sh`, the model-directory layout prepare expects (`Qwen3.8-27B-W4A16-AutoRound`, its
  `-fast` sibling, `Qwen3.8-27B-DFlash2-W4A16`) with its `HF_REPO`, and the efforts its template accepts
  (`low|medium|xhigh`). Read the commits between the two pins for any of those.
- Prepare runs at every start and is idempotent: a new image that adds a preparation step applies it in
  place, in the model directory, at the first start after the bump.

### qengine

- Pinned by digest to the author's own `ghcr.io/bpx03/qengine:<release>-vision` (the engine plus PyTorch for the vision
  encoder, which the slugs' `VISION` default needs; `:<release>` is the same build without PyTorch).
  Each image's `org.opencontainers.image.revision` label is its source commit, which `lineage.source_ref` records.
  A bump is `scripts/engine-pin-bump.sh qengine ghcr.io/bpx03/qengine@sha256:<digest>` and `source_ref`, then a boot
  of `qengine/qwen38-27b-dual` on the new digest with `verify-full` (the slugs since 2026-10-28; boot
  `qengine/qwen38-flash-next-dual` too when the release changes the Flash-Next engine).
- The volatile parts are QEngine's own interfaces: the image's entrypoint and its `QE_*` variables (`QE_GPUS`,
  `QE_MODEL`, `QE_MAX_SEQS`, `QE_RESERVE_GB`, `QE_EXTRA_ARGS`) and Flash-Next's `FLASH_*` ones, `server/app.py`'s flags
  (`--think`, `--mtp-k`) and `QE_REASONING_EFFORT`, the model directories it expects (`Qwen3.8-27B-INT4`, RedHatAI's layout, plus
  the `token_rank.bin` and `model_mtp_ft.safetensors` release assets; `Qwen3.8-Flash-Next-QE-INT4`, the author's HF
  folder, which a new release may re-convert), and `/v1/models` `owned_by: "qengine"`, which engine-kind.sh reads.
- The composes set the default effort with `QE_REASONING_EFFORT` (from 1.3.1) and refuse to start on an image whose
  `server/app.py` doesn't read it: a new image that drops or renames it stops the boot there, on purpose. Read its
  `--reasoning-effort` help before adapting the compose; the template's own default is `xhigh`.
  ⚠️ PolyForm Noncommercial license.

### Luce DFlash / xtransformers / future engines

- When adding a new engine, add an "Engine-specific notes" subsection here
  documenting its pinning convention + which files in its image are most
  volatile across versions.
