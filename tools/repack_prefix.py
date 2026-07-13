#!/usr/bin/env python3
"""Repack the ``model.language_model.`` <-> ``model.`` prefix in a safetensors checkpoint.

Renames weight keys in safetensors JSON headers and stream-copies the binary
data verbatim (never decode/re-encode).  Re-prefixes ``quantization_config``
keys in ``config.json`` so the checkpoint loads correctly.

Two directions:

  strip  (default)   ``model.language_model.*`` -> ``model.*``
                     Use when a VLM-wrapper artifact should become a plain
                     ``Qwen3_5ForCausalLM`` text-only checkpoint.

  inject              ``model.*`` -> ``model.language_model.*``
                     Use when a text-only checkpoint must be embedded into a
                     ``Qwen3_5ForConditionalGeneration`` multimodal layout so
                     vLLM's MTP speculative path fires.

STDLIB ONLY.  Reads default to utf-8 per repo convention.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# Allow standalone invocation even when scripts/lib/ isn't on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "..", "scripts", "lib")
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)

# fmt: off
from safetensors_repack import (  # noqa: E402 — import after sys.path fixup
    make_renamer,
    read_header,
    reprefix_config,
    rewrite_file,
)
# fmt: on


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repack safetensors checkpoint prefix.",
    )
    ap.add_argument("--src", required=True, help="source checkpoint directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument(
        "--direction",
        choices=["strip", "inject"],
        default="strip",
        help="strip: model.language_model.* -> model.*  |  "
        "inject: model.* -> model.language_model.*",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be done without writing files")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    rename_key = make_renamer(args.direction)

    # --- Load index ---------------------------------------------------------
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        print(f"FATAL: no index found at {idx_path}", file=sys.stderr)
        return 1

    idx = json.load(open(idx_path, encoding="utf-8"))
    wm = idx["weight_map"]

    ref_files = sorted(set(wm.values()))
    model_shards = [f for f in ref_files if f != "model_extra_tensors.safetensors"]
    has_extra = "model_extra_tensors.safetensors" in ref_files
    n_out = len(model_shards) + (1 if has_extra else 0)

    fmap = {}  # old filename -> new contiguously-numbered filename
    for i, f in enumerate(sorted(model_shards), start=1):
        fmap[f] = f"model-{i:05d}-of-{n_out:05d}.safetensors"
    if has_extra:
        fmap["model_extra_tensors.safetensors"] = (
            f"model-{n_out:05d}-of-{n_out:05d}.safetensors"
        )

    renamed_keys = sum(1 for k in wm if rename_key(k) != k)
    print(f"direction: {args.direction}")
    print(f"referenced files: {len(ref_files)}")
    print(f"weight_map keys: {len(wm)}  keys renamed: {renamed_keys}")
    print("file remap:")
    for a, b in fmap.items():
        print(f"  {a}  ->  {b}")

    if args.dry_run:
        print("dry-run: no files written")
        return 0

    os.makedirs(dst, exist_ok=True)

    # --- Rewrite safetensors files ------------------------------------------
    total_size = 0
    for old_f in ref_files:
        s = os.path.join(src, old_f)
        if not os.path.exists(s):
            print(f"WARNING: referenced file not found, skipping: {s}", file=sys.stderr)
            continue
        d = os.path.join(dst, fmap[old_f])
        print(f"rewriting {old_f} -> {fmap[old_f]}")
        sizes = rewrite_file(s, d, rename_key)
        total_size += sum(sizes.values())

    # --- Write new index ----------------------------------------------------
    new_wm = {rename_key(k): fmap[v] for k, v in wm.items()}
    new_idx = {"metadata": {"total_size": total_size}, "weight_map": new_wm}
    json.dump(
        new_idx,
        open(os.path.join(dst, "model.safetensors.index.json"), "w", encoding="utf-8"),
        indent=2,
    )

    # --- Re-prefix config.json ----------------------------------------------
    src_cfg_path = os.path.join(src, "config.json")
    if os.path.exists(src_cfg_path):
        cfg = json.load(open(src_cfg_path, encoding="utf-8"))
        cfg = reprefix_config(cfg, args.direction)
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
        print("re-prefixed config.json")

    # --- Copy auxiliary files (best-effort) ---------------------------------
    for aux in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "quantization_config.json",
    ):
        p = os.path.join(src, aux)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, aux))
            print(f"copied {aux}")

    print(f"\nDONE ({args.direction}). total_size={total_size/1e9:.2f} GB -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
