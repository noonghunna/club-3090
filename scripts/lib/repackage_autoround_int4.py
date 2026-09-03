#!/usr/bin/env python3
"""Repackage an AutoRound INT4 checkpoint that the transformers `_weight_conversions`
revert bug left malformed.

Two output modes (tensor DATA is never decoded/re-encoded in either):

  --mode text-only  (default)
    1. Strips the stray ``model.language_model.`` weight-key prefix down to
       ``model.`` so a text-only ``Qwen3_5ForCausalLM`` loader finds its tensors.
    2. Drops orphan duplicate shards (index-referenced files only).
    3. Emits a clean, contiguously-numbered shard set + flat index.

  --mode multimodal-wrapper  (--ref-config REQUIRED)
    Produces the reference-shaped ``Qwen3_5ForConditionalGeneration`` layout so
    vLLM's MTP speculative path fires (speculative.py fires on top-level
    ``model_type == "qwen3_5"``; the draft sizes itself from
    ``text_config.mtp_num_hidden_layers``). Input is the deduped ``model.*``
    artifact:
      * weight keys: ``model.<X>`` -> ``model.language_model.<X>`` (single
        infix; the loader mapper inserts the internal ``.model.``). ``lm_head.*``
        and ``mtp.*`` are untouched.
      * config.json: rebuilt from --ref-config (canonical nested wrapper with
        text_config + vision_config), overlaying THIS checkpoint's
        ``quantization_config`` with its keys re-prefixed to
        ``model.language_model.layers.*`` so Marlin/AutoRound matches every
        quantized tensor.
    No vision weights are needed -- serve with ``--language-model-only`` (the
    vision tower becomes a meta-device StageMissingLayer).

Each safetensors file is rewritten by renaming keys in its JSON header and
stream-copying the data section verbatim.

STDLIB ONLY. Reads default to utf-8 per repo convention.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys

from safetensors_repack import (
    make_renamer,
    reprefix_config,
    rewrite_file,
)

# mode->direction mapping
_MODE_DIR = {"text-only": "strip", "multimodal-wrapper": "inject"}


def _direction(mode: str) -> str:
    d = _MODE_DIR.get(mode)
    if d is None:
        raise ValueError(f"unknown mode: {mode!r}")
    return d


def build_mm_config(src_cfg: dict, ref_cfg: dict) -> dict:
    """Reference nested wrapper as template; overlay THIS checkpoint's quant config
    with keys re-prefixed to the language_model layout."""
    cfg = copy.deepcopy(ref_cfg)
    cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    cfg["model_type"] = "qwen3_5"

    src_qc = src_cfg.get("quantization_config")
    if src_qc is None:
        raise ValueError("source config.json has no quantization_config")
    qc = copy.deepcopy(src_qc)
    qc = reprefix_config({"quantization_config": qc}, "inject")["quantization_config"]

    cfg["quantization_config"] = qc

    # ensure the MTP draft can size itself
    tc = cfg.get("text_config")
    if not isinstance(tc, dict):
        raise ValueError("--ref-config has no text_config block")
    if tc.get("mtp_num_hidden_layers") is None:
        src_tc = src_cfg.get("text_config", src_cfg)
        tc["mtp_num_hidden_layers"] = src_tc.get("mtp_num_hidden_layers", 1)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["text-only", "multimodal-wrapper"],
                    default="text-only")
    ap.add_argument("--ref-config",
                    help="reference Qwen3_5ForConditionalGeneration config.json "
                         "(required for --mode multimodal-wrapper)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "multimodal-wrapper" and not args.ref_config:
        ap.error("--ref-config is required for --mode multimodal-wrapper")

    direction = _direction(args.mode)
    src, dst = args.src, args.dst
    rename_key = make_renamer(direction)

    idx_path = os.path.join(src, "model.safetensors.index.json")
    idx = json.load(open(idx_path, encoding="utf-8"))
    wm = idx["weight_map"]

    ref_files = sorted(set(wm.values()))
    model_shards = [f for f in ref_files if f != "model_extra_tensors.safetensors"]
    has_extra = "model_extra_tensors.safetensors" in ref_files
    n_out = len(model_shards) + (1 if has_extra else 0)

    fmap = {}
    for i, f in enumerate(sorted(model_shards), start=1):
        fmap[f] = f"model-{i:05d}-of-{n_out:05d}.safetensors"
    if has_extra:
        fmap["model_extra_tensors.safetensors"] = f"model-{n_out:05d}-of-{n_out:05d}.safetensors"

    renamed_keys = sum(1 for k in wm if rename_key(k) != k)
    print(f"mode: {args.mode}")
    print(f"referenced files: {len(ref_files)} (orphans dropped)")
    print(f"weight_map keys: {len(wm)}  keys renamed: {renamed_keys}")
    print("file remap:")
    for a, b in fmap.items():
        print(f"  {a}  ->  {b}")
    if args.dry_run:
        print("dry-run: no files written")
        return 0

    os.makedirs(dst, exist_ok=True)

    total_size = 0
    for old_f in ref_files:
        s = os.path.join(src, old_f)
        d = os.path.join(dst, fmap[old_f])
        print(f"rewriting {old_f} -> {fmap[old_f]}")
        sizes = rewrite_file(s, d, rename_key)
        total_size += sum(sizes.values())

    new_wm = {rename_key(k): fmap[v] for k, v in wm.items()}
    new_idx = {"metadata": {"total_size": total_size}, "weight_map": new_wm}
    json.dump(new_idx, open(os.path.join(dst, "model.safetensors.index.json"), "w", encoding="utf-8"),
              indent=2)

    src_cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))

    if args.mode == "multimodal-wrapper":
        ref_cfg = json.load(open(args.ref_config, encoding="utf-8"))
        cfg = build_mm_config(src_cfg, ref_cfg)
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
    else:
        cfg = reprefix_config(copy.deepcopy(src_cfg), "strip")
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)

        sqc_path = os.path.join(src, "quantization_config.json")
        if os.path.exists(sqc_path):
            sqc = json.load(open(sqc_path, encoding="utf-8"))
            # quantization_config.json has the same block_name_to_quantize shape
            sqc = reprefix_config({"quantization_config": sqc}, "strip")["quantization_config"]
            json.dump(sqc, open(os.path.join(dst, "quantization_config.json"), "w", encoding="utf-8"), indent=2)

    # copy aux files verbatim (tokenizer + fable-specific chat template)
    for aux in ("chat_template.jinja", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                "added_tokens.json", "preprocessor_config.json"):
        p = os.path.join(src, aux)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, aux))
            print(f"copied {aux}")

    print(f"\nDONE ({args.mode}). total_size={total_size/1e9:.2f} GB -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
