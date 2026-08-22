#!/usr/bin/env python3
"""One-shot migration: compose_registry.py dict-literals → registry.yaml.

Mechanical by construction — nothing is retyped by hand:

  1. Import the LIVE module and serialize COMPOSE_REGISTRY (+ DEFAULTS /
     ENGINE_PREFERENCE / RECOMMENDED_DEFAULT_MODELS) with the module's own
     stdlib emitter (dump_registry_yaml). Entry rows are the `_entry(**kwargs)`
     argument maps (the _entry()-derived keys `pp` / `gpu_assignment_mode` are
     stripped), the same schema profiles-local/registry.local.json already uses.
  2. PROVE the round-trip BEFORE writing: parse the emitted text back with the
     stdlib reader and rebuild every entry through _entry() — the rebuilt
     catalog must equal the imported original bit-for-bit (all ~100 entries,
     including served_name / gateway / serve_aliases / sampler_profiles).
  3. Write atomically (temp sibling + os.replace), then re-verify from disk.

Idempotent: after the cutover the module itself loads registry.yaml, so
re-running this script re-canonicalizes (and `--check` verifies a hand edit
without writing anything).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def _collect(cr) -> dict:
    """Live module state → registry-DATA dict (the YAML's canonical shape)."""
    entries = {}
    for slug, entry in cr.COMPOSE_REGISTRY.items():
        kwargs = {k: v for k, v in entry.items() if k not in cr._DERIVED_ENTRY_KEYS}
        entries[slug] = kwargs
    return {
        "schema": cr._REGISTRY_YAML_SCHEMA,
        "entries": entries,
        "defaults": cr.nest_defaults(cr.DEFAULTS),
        "engine_preference": {k: list(v) for k, v in cr.ENGINE_PREFERENCE.items()},
        "recommended_default_models": list(cr.RECOMMENDED_DEFAULT_MODELS),
    }


def _verify(cr, data: dict, text: str, label: str) -> list[str]:
    """Round-trip proof: text → parse → _entry() rebuild == imported original."""
    problems: list[str] = []
    parsed = cr.parse_registry_text(text, source=label)
    try:
        entries, defaults = cr._build_core_catalog(parsed, label)
    except cr.RegistryDataError as exc:
        return [f"rebuild failed: {exc}"]
    if entries != cr.COMPOSE_REGISTRY:
        bad = sorted(
            s for s in set(entries) | set(cr.COMPOSE_REGISTRY)
            if entries.get(s) != cr.COMPOSE_REGISTRY.get(s)
        )
        problems.append(f"{len(bad)} entries differ after round-trip: {bad[:10]}")
    if defaults != dict(cr.DEFAULTS):
        bad = sorted(
            k for k in set(defaults) | set(cr.DEFAULTS)
            if defaults.get(k) != cr.DEFAULTS.get(k)
        )
        problems.append(f"DEFAULTS differ after round-trip: {bad}")
    if parsed["engine_preference"] != dict(cr.ENGINE_PREFERENCE):
        problems.append("ENGINE_PREFERENCE differs after round-trip")
    if parsed["recommended_default_models"] != list(cr.RECOMMENDED_DEFAULT_MODELS):
        problems.append("RECOMMENDED_DEFAULT_MODELS differ after round-trip")
    if cr.dump_registry_yaml(parsed) != text:
        problems.append("canonical form is not idempotent (dump(parse(text)) != text)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="repo root (default: the checkout this file lives in)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify only: the on-disk file must equal the canonical form",
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    try:
        from scripts.lib.profiles import compose_registry as cr
    except Exception as exc:  # noqa: BLE001 — report, don't traceback
        print(f"[migrate] FATAL: compose_registry does not import: {exc}")
        return 2

    data = _collect(cr)
    text = cr.dump_registry_yaml(data)
    problems = _verify(cr, data, text, "<pre-write round-trip>")
    target = root / cr._REGISTRY_YAML_REL

    if problems:
        for p in problems:
            print(f"[migrate] ROUND-TRIP FAILURE: {p}")
        return 2
    print(
        f"[migrate] round-trip proven: {len(data['entries'])} entries, "
        f"{len(cr.DEFAULTS)} defaults rows, "
        f"{sum(len(v) for v in data['engine_preference'].values())} ranked engines"
    )

    if args.check:
        try:
            current = target.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[migrate] CHECK FAILED: {target} unreadable: {exc}")
            return 2
        if current != text:
            print(
                f"[migrate] CHECK FAILED: {target} is not the canonical form — "
                "re-run this script WITHOUT --check to re-canonicalize"
            )
            return 1
        print(f"[migrate] CHECK OK: {target} is canonical")
        return 0

    try:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        on_disk = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[migrate] FATAL: could not write {target}: {exc}")
        return 2
    if on_disk != text:
        print(f"[migrate] FATAL: {target} does not read back byte-identical")
        return 2
    print(f"[migrate] wrote {target} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
