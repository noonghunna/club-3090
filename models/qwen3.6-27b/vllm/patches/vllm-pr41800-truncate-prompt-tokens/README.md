# vLLM PR #41800 overlay — `truncate_prompt_tokens` kwarg on `get_max_tokens`

## What this fixes

Agentic clients (opencode, codex-cli, and similar IDE/agent runtimes) send `truncate_prompt_tokens` on chat-completion requests. Pre-[vLLM PR #41800](https://github.com/vllm-project/vllm/pull/41800), `vllm.entrypoints.utils.get_max_tokens()` doesn't accept that kwarg — and the kwarg propagates from the request handler down into the function call — so requests fail with:

```
HTTP 400: {"error":{"message":"get_max_tokens() got an unexpected keyword argument 'truncate_prompt_tokens'",...}}
```

The fix is upstream PR #41800 (merged 2026-05-06 at commit `d5b31c95`). It adds the kwarg to the function signature and a small body block that clamps `input_length` to `min(input_length, truncate_prompt_tokens or max_model_len)` before the existing length check.

## When this overlay is needed

This overlay is needed on engines pinned to vLLM SHAs that **predate `d5b31c95`**:

| Engine | Pinned SHA | Pre-fix? |
|---|---|---|
| `vllm-nightly-mtp` | `01d4d1ad` (2026-05-04) | ✅ needs overlay |
| `vllm-nightly-dflash` | `e47c98ef` (~2026-05-05) | ✅ needs overlay (20 commits behind d5b31c95) |
| `vllm-nightly-full` | `e47c98ef` | ✅ needs overlay |
| `vllm-nightly-clean` | `bf610c2f` (2026-05-15) | ❌ already includes fix |

If a compose routes through `vllm-nightly-clean`, the overlay is unnecessary — the function signature already accepts the kwarg upstream.

## How the overlay works

`install.sh` is a Python anchor-based in-place patcher. It does two surgical edits to the in-container `/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/utils.py`:

1. **Signature**: adds `truncate_prompt_tokens: int | None = None,` to `get_max_tokens`'s signature, anchored to the existing `override_max_tokens: int | None = None,` line.
2. **Body**: inserts a 6-line truncation-aware `input_length` adjustment block before the existing `if max_model_len < input_length:` check, anchored to that line.

Each insertion carries a sentinel comment (`# PATCH: truncate_prompt_tokens kwarg (club3090/pr41800)`) so re-running the install on an already-patched file is a no-op. Post-patch the file is AST-validated before write.

Why anchor-based and not full-file replacement: the PR diff is +14 / -0 across a 200-line file — replacing the full file would shadow other upstream changes in `utils.py`. Anchor-based insertion is drift-resistant to unrelated upstream movement.

## Composes that wire this overlay in (as of v0.7.3 ship)

* `models/qwen3.6-27b/vllm/compose/dual/docker-compose.yml` (gpu-mode `27b`)
* `models/qwen3.6-27b/vllm/compose/dual/turbo.yml` (gpu-mode `27b-turbo`)
* `models/qwen3.6-27b/vllm/compose/dual/dflash.yml` (gpu-mode `27b-dflash`)
* `models/qwen3.6-27b/vllm/compose/dual/dflash-noviz.yml` (gpu-mode `27b-dflash-noviz`, the compose from issue #138)

## How to add this overlay to another affected compose

In any compose that routes through `vllm-nightly-mtp` / `vllm-nightly-dflash` / `vllm-nightly-full`, add:

1. **Volume mount** in the `volumes:` block:

   ```yaml
       - ../../patches/vllm-pr41800-truncate-prompt-tokens/install.sh:/etc/club3090/install-pr41800.sh:ro
   ```

2. **Install line** in the `entrypoint:` bash script, before `exec vllm serve`:

   ```bash
   bash /etc/club3090/install-pr41800.sh
   ```

Run `bash install.sh` (the file in this directory) standalone to test against a transient vLLM container before wiring into a compose. See the smoke test in the next section.

## Smoke test

```bash
docker run --rm --entrypoint /bin/bash \
  -v $(pwd)/install.sh:/install.sh:ro \
  vllm/vllm-openai:nightly-01d4d1ad375dc5854779c593eee093bcebb0cada \
  -c '
    python3 -c "from vllm.entrypoints.utils import get_max_tokens; import inspect; print(inspect.signature(get_max_tokens))"
    bash /install.sh
    python3 -c "from vllm.entrypoints.utils import get_max_tokens; import inspect; print(inspect.signature(get_max_tokens))"
  '
```

Expected: signature lacks `truncate_prompt_tokens` BEFORE install, has it AFTER. Verified on `01d4d1ad` (2026-05-15).

## When to drop this overlay

When **both** are true:

1. PR #41800 has merged upstream (it has — 2026-05-06 at `d5b31c95`)
2. The engine's pinned nightly SHA bumps past `d5b31c95`

For the Genesis-anchored engines, the bump happens with Sander's next Genesis release cycle (v7.73.x). For `vllm-nightly-dflash` and `vllm-nightly-full`, the bump happens when their respective overlays (PR #41703 DFlash, PR #42102 INT8 PTH KV) are re-validated against a newer nightly.

Track in `docs/UPSTREAM.md`.

## Source

- vLLM PR #41800: https://github.com/vllm-project/vllm/pull/41800
- Merged commit: `d5b31c95`
- Tracking issue: noonghunna/club-3090#139
- Triggered by: noonghunna/club-3090#138 (SEVENID's opencode boot failure)
- Patch summary: +7 lines in `vllm/entrypoints/utils.py` (the actual fix) + 5 call-site forward-compat additions in other files (we skip those — the signature fix alone unblocks all known TypeError reports)
