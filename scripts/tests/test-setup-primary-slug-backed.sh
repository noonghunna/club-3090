#!/usr/bin/env bash
# Gate: the variant `scripts/setup.sh` downloads by default for a model must be
# a variant some registry.yaml entry can actually serve (club-3090#1303).
#
# THE DEFECT
# ----------
# `profiles/models/gemma-4-26b-a4b.yml` carried:
#
#     default_weight_variant: awq          # what the slug side resolves
#     setup:
#       primary: autoround-int4-mixed      # what setup.sh downloads
#
# Both registered slugs (vllm/gemma-26ba4b-single, vllm/gemma-26ba4b-dual) pin
# `weights_variant: awq`. `autoround-int4-mixed` was referenced by no registry
# entry at all — only by composes under `models/gemma-4-26b-a4b/vllm/compose/
# _archive/`. So the obvious user path (setup, then launch the model's own slug)
# downloaded 16 GB nothing could serve, skipped the 17 GB both slugs need, and
# only failed at switch.sh time — 16 GB in.
#
# WHY THE GATE IS NOT "primary == default_weight_variant"
# ------------------------------------------------------
# `setup.primary` is a DELIBERATE override: weights.py resolves
# `setup.get("primary") or default_weight_variant`. Two models override it
# CORRECTLY today — qwen3.8-27b (fp8; many slugs) and deepseek-v4-flash-0731
# (unsloth-iq2-xxs; llamacpp/deepseek-flash-dual-iq2). Asserting equality would
# red both. The real invariant is weaker and true of all three: whatever setup
# fetches, SOME shipped slug must be able to serve it.
#
# STATUS-AGNOSTIC BY DESIGN
# -------------------------
# A deprecated entry still pins its variant, and this repo keeps deprecated
# entries on purpose (deprecated != deleted — the launch-compat and pull-gate
# tests assert on their hardware fields). Three models' primaries are referenced
# only by deprecated slugs today; that is a different question from this one.
# What this gate rejects is a variant NO registry entry names at all — i.e. one
# whose composes survive only under _archive/.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
import json
import pathlib
import subprocess
import sys

import yaml

root = pathlib.Path(sys.argv[1])


def scan(models, entries):
    """models: catalog rows ({id, default_key}). entries: registry.yaml entries.

    Returns a list of human-readable problems."""
    served = {}
    for slug, e in entries.items():
        if not isinstance(e, dict):
            continue
        served.setdefault((e.get("model"), e.get("weights_variant")), []).append(slug)
    problems = []
    for m in models:
        key = m.get("default_key") or ""
        model_id, _, variant = key.partition(":")
        if not model_id or not variant:
            problems.append(f"{m.get('id')}: setup resolves no weights key at all")
            continue
        if not served.get((model_id, variant)):
            other = sorted({v for (mid, v) in served if mid == model_id})
            problems.append(
                f"{model_id}: setup.sh would download '{variant}', which no "
                f"registry.yaml entry serves"
                + (f" (shipped slugs use: {', '.join(other)})" if other else
                   " (this model has no registry entries at all)")
            )
    return problems


# --- positive control FIRST -------------------------------------------------
# A scan that silently finds nothing is indistinguishable from a clean tree, so
# prove it still sees the exact shape #1303 shipped, and that it does NOT flag
# the legitimate override shape.
BAD = scan(
    [{"id": "m", "default_key": "m:archived-variant"}],
    {"eng/m-single": {"model": "m", "weights_variant": "awq"}},
)
GOOD = scan(
    [{"id": "m", "default_key": "m:fp8"}],
    {
        "eng/m-single": {"model": "m", "weights_variant": "fp8"},
        "eng/m-dual": {"model": "m", "weights_variant": "awq"},
    },
)
DEPRECATED_OK = scan(
    [{"id": "m", "default_key": "m:fp8"}],
    {"eng/m-single": {"model": "m", "weights_variant": "fp8", "status": "deprecated"}},
)
if len(BAD) != 1:
    sys.exit(f"FAIL: positive control — planted #1303 shape not flagged: {BAD}")
if GOOD:
    sys.exit(f"FAIL: positive control — a legitimate setup.primary override was flagged: {GOOD}")
if DEPRECATED_OK:
    sys.exit(f"FAIL: positive control — a deprecated-but-present slug must still count: {DEPRECATED_OK}")
print("  ✓ scanner flags the #1303 shape, spares a legitimate override and a deprecated-only slug")

# --- the real scan ----------------------------------------------------------
catalog = json.loads(
    subprocess.run(
        [sys.executable, str(root / "scripts/lib/profiles/weights.py"), "catalog", "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
)["models"]
entries = yaml.safe_load((root / "scripts/lib/profiles/registry.yaml").read_text(encoding="utf-8"))["entries"]
# ⚠️ BOTH LAYERS, or this gate is guaranteed to fail on any LOCAL-layer model.
# `weights.py catalog --json` above reads the merged catalog (core +
# profiles-local/models.d), so a local model arrives in `models`; reading only
# registry.yaml for `entries` meant its serving slug — which lives in
# registry.local.json — was invisible, and the gate reported "this model has no
# registry entries at all" about a model that is served perfectly well. Same
# layer asymmetry as the registry-emit `_weights_meta` bug. Core is loaded FIRST
# so a local slug can never shadow a core one on a key collision.
# Optional by design: the local layer is gitignored and absent on most checkouts.
_local = root / "scripts/lib/profiles/profiles-local/registry.local.json"
if not _local.exists():
    _local = root / "scripts/lib/profiles-local/registry.local.json"
if _local.exists():
    try:
        _extra = json.loads(_local.read_text(encoding="utf-8"))
        if isinstance(_extra, dict):
            for _slug, _e in _extra.items():
                entries.setdefault(_slug, _e)
    except Exception as exc:                      # never let a local file break the gate
        print(f"  note: could not read the local registry layer ({exc}) — core only")

problems = scan(catalog, entries)
if problems:
    print("test-setup-primary-slug-backed: FAIL")
    for p in problems:
        print(f"  ⛔ {p}")
    print()
    print("  Fix: repoint `setup: primary:` in scripts/lib/profiles/models/<id>.yml to a")
    print("  variant a shipped slug serves (or delete the key so it falls back to")
    print("  default_weight_variant). A variant whose only composes live under _archive/")
    print("  cannot be the thing setup.sh downloads.")
    sys.exit(1)

print(f"test-setup-primary-slug-backed: ok ({len(catalog)} models — every setup.sh "
      f"default download is served by a registry slug)")
PY
