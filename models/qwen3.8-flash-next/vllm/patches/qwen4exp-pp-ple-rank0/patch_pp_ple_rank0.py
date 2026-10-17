#!/usr/bin/env python3
"""Overlay `qwen4exp-pp-ple-rank0`: let Qwen3.8-Flash-Next run with pipeline parallelism.

    patch_pp_ple_rank0.py patch    apply (marker-gated; a second run is a verified no-op)
    patch_pp_ple_rank0.py verify   check the installed result

Stock vLLM (0.31.0, 0.32.0rc3) refuses PP>1 for any Qwen4Exp model with an n-gram PLE,
because later pipeline ranks never see the raw input_ids the PLE reads. Flash-Next has a
single PLE layer (config `ple_layer_ids: [2]`, decoder layer 1), which always lands on
rank 0, so the check is stricter than it needs to be. This overlay:

  1. model_executor/models/config.py and models/qwen4_exp/nvidia/model_state.py:
     refuse PP only when a PLE layer falls outside rank 0's layer range.
  2. v1/core/kv_cache_utils.py: when a UniformTypeKVCacheSpecs group lists layers of
     every pipeline rank, allocate only this rank's layers (which can be none).

Ported from Minachist's flash-next-vllm.patch (Apache-2.0,
huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound, vllm-patch/).
Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>

Each anchor must appear exactly once. Missing or not unique -> exit 1 (refuse the boot).
Stdlib only; never imports vllm.
"""
from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[qwen4exp-pp-ple-rank0]"
MARKER = "# [club3090 qwen4exp-pp-ple-rank0]"

_CHECK = (
    "{indent}{marker} PLE layers must sit on pipeline rank 0 (Minachist, Apache-2.0)\n"
    "{indent}if {ids} and {pp} > 1:\n"
    "{indent}    from vllm.distributed.utils import get_pp_indices as _club_pp_indices\n"
    "{indent}    _club_lo, _club_hi = _club_pp_indices(int({layers}), 0, {pp})\n"
    "{indent}    _club_stranded = sorted(\n"
    "{indent}        i - 1 for i in {ids} if not (_club_lo <= i - 1 < _club_hi)\n"
    "{indent}    )\n"
    "{indent}    if _club_stranded:\n"
    "{indent}        raise {exc}(\n"
    "{indent}            \"Qwen4Exp n-gram PLE needs every PLE layer on pipeline rank 0; \"\n"
    "{indent}            f\"PP={{{pp}}} strands decoder layer(s) {{_club_stranded}} outside \"\n"
    "{indent}            f\"[{{_club_lo}}, {{_club_hi}}). Use VLLM_PP_LAYER_PARTITION or PP=1.\"\n"
    "{indent}        )\n"
)

EDITS = [
    (
        Path("model_executor/models/config.py"),
        "        if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:\n"
        "            raise NotImplementedError(\n"
        "                \"Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 \"\n"
        "                \"because non-first pipeline ranks do not receive the raw input_ids \"\n"
        "                \"it needs. Please run with PP=1.\"\n"
        "            )\n",
        _CHECK.format(indent="        ", marker=MARKER, ids="text_config.ple_layer_ids",
                      pp="parallel_config.pipeline_parallel_size",
                      layers="text_config.num_hidden_layers", exc="NotImplementedError"),
    ),
    (
        Path("models/qwen4_exp/nvidia/model_state.py"),
        "        if vllm_config.parallel_config.pipeline_parallel_size > 1:\n"
        "            raise RuntimeError(\n"
        "                \"N-gram PLE embedding currently requires \"\n"
        "                \"pipeline_parallel_size=1 because non-first pipeline ranks do \"\n"
        "                \"not receive the raw input_ids required by PLE. Please run \"\n"
        "                \"with PP=1.\"\n"
        "            )\n",
        _CHECK.format(indent="        ", marker=MARKER, ids="config.ple_layer_ids",
                      pp="vllm_config.parallel_config.pipeline_parallel_size",
                      layers="config.num_hidden_layers", exc="RuntimeError"),
    ),
    (
        Path("v1/core/kv_cache_utils.py"),
        "            for layer_name, spec in group_spec.kv_cache_specs.items():\n"
        "                layers_by_spec[spec].append(layer_name)\n",
        f"            {MARKER} keep only this rank's layers (PP); can be none\n"
        "            _club_owned = set(group.layer_names)\n"
        "            for layer_name, spec in group_spec.kv_cache_specs.items():\n"
        "                if layer_name in _club_owned:\n"
        "                    layers_by_spec[spec].append(layer_name)\n",
    ),
]


def die(msg: str) -> None:
    print(f"{TAG} REFUSE: {msg}", file=sys.stderr)
    print(f"{TAG} Fix: re-anchor this overlay for the installed vLLM version "
          "(models/qwen3.8-flash-next/vllm/patches/qwen4exp-pp-ple-rank0/README.md).", file=sys.stderr)
    sys.exit(1)


def vllm_dir() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        die("the vllm package is not importable in this container")
    return Path(spec.origin).resolve().parent


def run(mode: str) -> None:
    root = vllm_dir()
    applied = 0
    for rel, anchor, replacement in EDITS:
        path = root / rel
        if not path.is_file():
            die(f"{rel} not found under {root}")
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            if replacement not in text:
                die(f"{rel} carries the marker but not the expected edit")
            continue
        if mode == "verify":
            die(f"{rel} is not patched")
        count = text.count(anchor)
        if count != 1:
            die(f"anchor in {rel} found {count} times (expected 1)")
        tmp = path.with_suffix(path.suffix + ".club3090.tmp")
        tmp.write_text(text.replace(anchor, replacement), encoding="utf-8")
        try:
            py_compile.compile(str(tmp), doraise=True)
        except py_compile.PyCompileError as exc:
            tmp.unlink(missing_ok=True)
            die(f"patched {rel} does not compile: {exc}")
        tmp.replace(path)
        applied += 1
    state = "applied" if applied else "already applied"
    print(f"{TAG} {state}: PP allowed when PLE layers sit on rank 0; per-rank KV groups")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "verify"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    run(sys.argv[1])
