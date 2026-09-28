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

### Luce DFlash / xtransformers / future engines

- When adding a new engine, add an "Engine-specific notes" subsection here
  documenting its pinning convention + which files in its image are most
  volatile across versions.
