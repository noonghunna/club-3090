# `repack_prefix.py` — safetensors prefix repacker

Rename the `model.language_model.` ↔ `model.` weight-key prefix in a safetensors
checkpoint, stream-copying the tensor bytes **verbatim** (no decode/re-encode).

---

## Do you need this?

**No — not for the checkpoints club-3090 ships.** Our AutoRound INT4 Qwen3.6
loads directly with MTP speculative decoding; there is **no repack step** in
`setup.sh` / `pull.sh` / any compose. If you're just running the shipped stack,
ignore this tool.

This is a standalone utility for one narrow case: **you are rolling your own
AutoRound INT4 quant** of a vision-capable Qwen3.6 (or otherwise have a
checkpoint whose key prefix doesn't match the load path you want).

## The problem it solves

AutoRound INT4 quants of a *vision* Qwen3.6 store the language-model weights
under `model.language_model.*` — the multimodal `Qwen3_5ForConditionalGeneration`
layout. Depending on how you load, you want a different prefix, and a mismatch
means the loader can't find the weights (or pulls in a vision tower you didn't
want):

| You want to… | Layout the loader expects | Direction |
|---|---|---|
| Load as a plain **text-only** `Qwen3_5ForCausalLM` | `model.*` | **`strip`** |
| Fire vLLM's **MTP speculative** path (and skip the vision encoder) | `model.language_model.*` | **`inject`** |

## What it does

- Renames the tensor keys in every shard's safetensors **JSON header** and
  rewrites `model.safetensors.index.json`.
- **Stream-copies the tensor data byte-for-byte** — it never decodes,
  dequantizes, or re-encodes weights, so the result is bit-identical and fast
  even on multi-GB shards.
- Re-prefixes the `quantization_config` keys in `config.json`
  (`block_name_to_quantize`, `extra_config`) so the quant metadata still points
  at the right layers. The match is dot-anchored, so `model.layers` never
  matches *inside* `model.language_model.layers`.
- Preserves the safetensors `__metadata__` block and the 8-byte header
  alignment.
- **Pure Python stdlib** — no `torch`, `safetensors`, or `numpy` required.

## Usage

```bash
# strip: model.language_model.* -> model.*   (plain text-only checkpoint)
python3 tools/repack_prefix.py --src /path/to/quant --dst /path/to/out --direction strip

# inject: model.* -> model.language_model.*  (embed for vLLM MTP spec-dec)
python3 tools/repack_prefix.py --src /path/to/quant --dst /path/to/out --direction inject

# preview the rename + file map without writing anything
python3 tools/repack_prefix.py --src /path/to/quant --dst /path/to/out --direction strip --dry-run
```

`--src` is the checkpoint directory (the one with `config.json` +
`*.safetensors` + `model.safetensors.index.json`); `--dst` is written fresh.

## Benefits

- **Lossless** — header-only rename + verbatim data copy → bit-identical
  weights, zero quality change.
- **Fast** — no tensor decode; I/O-bound streaming copy (32 MiB chunks).
- **Dependency-free** — runs anywhere `python3` does; nothing to install.

## Layout

| File | Role |
|---|---|
| [`repack_prefix.py`](repack_prefix.py) | Standalone CLI. |
| [`../scripts/lib/safetensors_repack.py`](../scripts/lib/safetensors_repack.py) | Importable core (header I/O, renamer, verbatim rewrite, config re-prefix). |
| [`../scripts/lib/repackage_autoround_int4.py`](../scripts/lib/repackage_autoround_int4.py) | Higher-level AutoRound-INT4 repackager built on the core. |
| [`../scripts/tests/test-repack-prefix.sh`](../scripts/tests/test-repack-prefix.sh) | Smoke test (generates a dummy checkpoint, exercises strip + roundtrip + dry-run). |

## Scope

A standalone utility only — it is **not wired into** any serving, pull, or setup
path, and nothing in the shipped stack depends on it. Contributed by
[@BlackBox-Labs](https://github.com/noonghunna/club-3090/pull/700); reproduced
and validated on a maintainer rig (strip + roundtrip smoke test pass, output
byte-exact and loads clean).
