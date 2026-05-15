# vLLM PR #40886 overlay — AWQ compressed-tensors MoE key remapping

## What this fixes

`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` ships in `compressed-tensors` pack-quantized format, which stores per-expert weights with `_packed` (int32) and `_scale` (bfloat16) suffixes. The vLLM `gemma4.py::_weight_iterator` (as of bf610c2f, 2026-05-15) only handles the float-checkpoint key pattern and does not remap the `_packed` / `_scale` suffixed variants on MoE expert weights. Without this overlay, model load fails with a `KeyError` on the first packed expert key.

The fix is from [vLLM PR #40886](https://github.com/vllm-project/vllm/pull/40886) by @tajwali (open as of 2026-05-15, last updated 2026-04-25). PR author tested on **RTX 3090 24 GB** (same SKU as ours) with vLLM 0.19.1. The patch is +23 / -0 — pure insertion of 4 conditional branches in `_weight_iterator` that intercept the four `_packed` / `_scale` MoE key patterns and yield per-expert float-shaped keys that the existing FusedMoE loader path expects.

## Why an anchor-based Python patcher (not full-file replacement)

The PR is a small insertion. Full-file replacement risks shadowing unrelated upstream changes in `gemma4.py` that we DO want (PR #41745 Gemma 4 MTP support, etc.). The anchor-based patcher in `install.sh` locates the existing `if "moe.gate_up_proj" in name and weight.dim() == 3:` line inside `_weight_iterator` and inserts the patch immediately above it. Idempotent: a sentinel comment is included in the inserted block, and re-running on an already-patched file is a no-op.

## How to use

### From a compose

Bind-mount `install.sh` into the container at a known path, then invoke it from the entrypoint **before** `vllm serve` runs:

```yaml
services:
  vllm-gemma-4-26b-a4b-awq-tp2:
    image: ${VLLM_IMAGE:-vllm/vllm-openai:nightly-${VLLM_NIGHTLY_SHA}}
    volumes:
      - ../../patches/vllm-pr40886-awq-moe-keys/install.sh:/etc/club3090/install-pr40886.sh:ro
    entrypoint:
      - /bin/bash
      - -c
      - |
        bash /etc/club3090/install-pr40886.sh
        exec vllm serve "$@"
      - --
    command:
      - --model
      - /root/.cache/huggingface/gemma-4-26b-a4b-awq-4bit
      # ... other flags ...
```

Same sidecar pattern as `vllm-pr35936-required-fallback/install.sh` — runs once per container start, leaves the container's RW layer in the patched state.

### Override env vars

If vLLM moves the install path of `gemma4.py`:

```bash
CLUB3090_PR40886_TARGET=/some/other/path/gemma4.py bash install.sh
```

## When to drop this overlay

When **both** of these are true:

1. PR #40886 has merged upstream
2. The engine's pinned nightly SHA is past the merge commit

Track in `docs/UPSTREAM.md`.

## Smoke test

Manual:

```bash
# Spin up a transient container, install the patch, verify the sentinel.
docker run --rm \
  -v $(pwd)/install.sh:/install.sh:ro \
  vllm/vllm-openai:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036 \
  bash -c 'bash /install.sh && grep -c "club3090/pr40886" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py'
```

Expected: prints `1` (sentinel present after install).

## Source PR

- PR head: `tajwali/vllm` @ `652819dad0bf9bbb0436d6660822e7aff30c3ff0` (branch `fix/gemma4-compressed-tensors-moe-key-remapping`)
- Vendored as of 2026-05-15
- Patch summary: 4 `if` branches added to `_weight_iterator` in `vllm/model_executor/models/gemma4.py`
  - `moe.gate_up_proj_packed [E, 2I, H/8]` → split into per-expert `gate_proj.weight_packed` + `up_proj.weight_packed`
  - `moe.gate_up_proj_scale [E, 2I, G]` → same split for scales
  - `moe.down_proj_packed [E, H, I/8]` → yield per-expert `down_proj.weight_packed`
  - `moe.down_proj_scale [E, H, G]` → yield per-expert `down_proj.weight_scale`
