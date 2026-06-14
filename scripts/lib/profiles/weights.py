#!/usr/bin/env python3
"""Weight download recipe reader for shell callers.

The source of truth is ``scripts/lib/profiles/models/*.yml``. This module is
intentionally thin: it prints shell-safe ``KEY=VALUE`` lines for setup.sh and
preflight.sh, and exits non-zero if Python/PyYAML is unavailable so preflight
can fall back to a generic missing-model hint.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - exercised by shell fallback tests
    yaml = None  # type: ignore[assignment]


PROFILE_ROOT = Path(__file__).resolve().parent


ALIASES = {
    "qwen3.6-27b:autoround_int4": ("qwen3.6-27b", "autoround-int4"),
    "qwen3.6-27b:gguf_q4km": ("qwen3.6-27b", "unsloth-q4km"),
    "qwen3.6-27b:gguf_iq4ks": ("qwen3.6-27b", "ubergarm-iq4ks"),
    "qwen3.6-27b:carnice_bf16mtp": ("qwen3.6-27b", "carnice-bf16mtp"),
    "qwen3.6-27b:qwopus_bf16mtp": ("qwen3.6-27b", "qwopus-bf16mtp"),
    "qwen3.6-35b-a3b:autoround_int4": ("qwen3.6-35b-a3b", "autoround-int4"),
    "gemma-4-31b:autoround_int4": ("gemma-4-31b", "autoround-int4"),
    "gemma-4-26b-a4b:autoround_int4_mixed": ("gemma-4-26b-a4b", "autoround-int4-mixed"),
    "gemma-4-26b-a4b:awq_compressed_tensors": ("gemma-4-26b-a4b", "awq"),
    "qwen3.6-27b-autoround-int4": ("qwen3.6-27b", "autoround-int4"),
    "qwen3.6-27b-dflash": ("qwen3.6-27b", "dflash"),
    "qwen3.6-27b-prism-eagle3": ("qwen3.6-27b", "prism_eagle3"),
    "qwen3.6-27b-mtp-head": ("qwen3.6-27b", "mtp_head"),
    "qwen3.6-27b-gguf-q4km": ("qwen3.6-27b", "unsloth-q4km"),
    "qwen3.6-27b-mmproj-f16": ("qwen3.6-27b", "gguf_mmproj_f16"),
    "qwen3.6-27b-gguf-iq4ks": ("qwen3.6-27b", "ubergarm-iq4ks"),
    "qwen3.6-35b-a3b-autoround-int4": ("qwen3.6-35b-a3b", "autoround-int4"),
    "gemma-4-31b-autoround-int4": ("gemma-4-31b", "autoround-int4"),
    "gemma-4-31b-it-AWQ-4bit": ("gemma-4-31b", "awq"),
    "gemma-4-31b-it-assistant": ("gemma-4-31b", "assistant"),
    "gemma-4-31b-it-dflash": ("gemma-4-31b", "dflash"),
    "gemma-4-26b-a4b-autoround-int4-mixed": ("gemma-4-26b-a4b", "autoround-int4-mixed"),
    "gemma-4-26b-a4b-awq-4bit": ("gemma-4-26b-a4b", "awq"),
    "gemma-4-26b-a4b-it-assistant": ("gemma-4-26b-a4b", "assistant"),
    "carnice-v2-27b-int4-recipe-d-bf16mtp": ("qwen3.6-27b", "carnice-bf16mtp"),
    "qwopus3.6-27b-int4-recipe-d-bf16mtp": ("qwen3.6-27b", "qwopus-bf16mtp"),
}


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _require_yaml() -> None:
    if os.environ.get("CLUB3090_WEIGHTS_READER_DISABLE") == "1":
        _die("weight reader disabled", 2)
    if yaml is None:
        _die("PyYAML unavailable", 2)


def _load_models() -> dict[str, dict[str, Any]]:
    _require_yaml()
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((PROFILE_ROOT / "models").glob("*.yml")):
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        model_id = str(data.get("id") or path.stem)
        out[model_id] = data
    return out


def _label(model: dict[str, Any], variant: str, meta: dict[str, Any]) -> str:
    display = model.get("display_name") or model.get("id")
    kind = str(meta.get("kind") or "weights")
    return f"{display} {kind} ({variant})"


def _recipe(model_id: str, variant: str) -> dict[str, str]:
    models = _load_models()
    model = models.get(model_id)
    if not model:
        _die(f"unknown model: {model_id}")
    weights = model.get("weights") or {}
    meta = weights.get(variant)
    if not isinstance(meta, dict):
        alias = ALIASES.get(f"{model_id}:{variant}") or ALIASES.get(variant)
        if alias and alias[0] == model_id:
            variant = alias[1]
            meta = weights.get(variant)
    if not isinstance(meta, dict):
        _die(f"unknown weight variant: {model_id}:{variant}")

    files = meta.get("files") or []
    if isinstance(files, str):
        files = [files]

    setup_env = str(meta.get("setup_env") or "")
    setup_weights_key = str(meta.get("setup_weights_key") or "")
    if not setup_env and setup_weights_key:
        setup_env = f"WEIGHTS={setup_weights_key}"
    if not setup_env and meta.get("hf_repo"):
        setup_env = f"WEIGHT_KEY={model_id}:{variant}"

    return {
        "WEIGHT_KEY": f"{model_id}:{variant}",
        "WEIGHT_VARIANT": variant,
        "WEIGHT_LABEL": _label(model, variant, meta),
        "WEIGHT_MODEL": model_id,
        "WEIGHT_ENGINE": str(meta.get("engine") or ""),
        "WEIGHT_KIND": str(meta.get("kind") or ""),
        "WEIGHT_REPO": str(meta.get("hf_repo") or ""),
        "WEIGHT_REVISION": str(meta.get("revision") or ""),
        "WEIGHT_SUBDIR": str(meta.get("local_subdir") or meta.get("path") or ""),
        "WEIGHT_FILES": " ".join(str(f) for f in files),
        "WEIGHT_VERIFY_GLOB": str(meta.get("verify_glob") or "*.safetensors"),
        "WEIGHT_SETUP_MODEL": model_id,
        "WEIGHT_SETUP_ENV": setup_env,
        "WEIGHT_MANUAL_NOTE": str(meta.get("manual_note") or ""),
    }


def _resolve_key(key: str) -> tuple[str, str]:
    if ":" in key:
        model_id, variant = key.split(":", 1)
        return model_id, variant
    if key in ALIASES:
        return ALIASES[key]
    models = _load_models()
    matches: list[tuple[str, str]] = []
    for model_id, model in models.items():
        for variant, meta in (model.get("weights") or {}).items():
            if key == meta.get("path") or key == meta.get("local_subdir"):
                matches.append((model_id, variant))
    if len(matches) == 1:
        return matches[0]
    _die(f"unknown weight key: {key}")


def _lookup_path(rel: str) -> tuple[str, str]:
    rel = rel.split(" (", 1)[0].strip()
    rel = rel.removeprefix("./").removeprefix("/")
    rel = rel.removeprefix("models/")
    rel = rel.removeprefix("root/.cache/huggingface/")
    rel = rel.removeprefix("/root/.cache/huggingface/")
    if rel.endswith("/config.json"):
        rel = rel[: -len("/config.json")]
    rel = rel.split(":", 1)[0]

    models = _load_models()
    matches: list[tuple[int, str, str]] = []
    for model_id, model in models.items():
        for variant, meta in (model.get("weights") or {}).items():
            subdir = str(meta.get("local_subdir") or meta.get("path") or "")
            if not subdir:
                continue
            files = meta.get("files") or []
            if isinstance(files, str):
                files = [files]
            exact_files = {f"{subdir}/{name}" for name in files}
            if rel in exact_files:
                matches.append((len(rel) + 1000, model_id, variant))
            elif rel == subdir or rel.startswith(f"{subdir}/"):
                matches.append((len(subdir), model_id, variant))
    if not matches:
        _die(f"no recipe for path: {rel}")
    _, model_id, variant = sorted(matches, reverse=True)[0]
    return model_id, variant


def _print_env(recipe: dict[str, str]) -> None:
    for key in sorted(recipe):
        print(f"{key}={shlex.quote(recipe[key])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_entry = sub.add_parser("entry")
    p_entry.add_argument("key")
    p_entry.add_argument("variant", nargs="?")
    p_lookup = sub.add_parser("lookup")
    p_lookup.add_argument("path")
    args = parser.parse_args(argv)

    if args.cmd == "entry":
        if args.variant:
            model_id, variant = args.key, args.variant
        else:
            model_id, variant = _resolve_key(args.key)
    else:
        model_id, variant = _lookup_path(args.path)

    _print_env(_recipe(model_id, variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
