#!/usr/bin/env bash
# registry-emit.sh --json — structured JSON catalog emit (shape assertions).
#
# Exercises the ADDITIVE direct-invocation path
# (`bash scripts/lib/registry-emit.sh --json <root>`) and asserts the JSON is
# well-formed with the contract's top-level + per-section keys. Also guards the
# hard requirement that SOURCING the file and calling registry_variant_rows
# still produces byte-identical tab output (the JSON path must be purely
# additive).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "ASSERTION FAILED: $1" >&2
  exit 1
}

# --- 1. Direct --json invocation emits valid JSON ---------------------------
json="$(bash scripts/lib/registry-emit.sh --json "$ROOT_DIR" 2>/dev/null)"
[[ -n "$json" ]] || fail "--json produced no stdout"

# Validate JSON + assert SHAPE (top-level keys, per-section keys, types) in one
# python pass. The JSON is passed via the environment (not stdin) so the heredoc
# can serve as the python script. Exits non-zero with a clear message on the
# first mismatch.
REGISTRY_JSON="$json" python3 - <<'PY'
import json
import os
import sys

d = json.loads(os.environ["REGISTRY_JSON"])

def need(cond, msg):
    if not cond:
        print(f"ASSERTION FAILED: {msg}", file=sys.stderr)
        raise SystemExit(1)

# top-level
need(set(d.keys()) == {"variants", "defaults", "profiles"},
     f"top-level keys != variants/defaults/profiles (got {sorted(d.keys())})")
need(isinstance(d["variants"], list) and d["variants"], "variants must be a non-empty list")
need(isinstance(d["defaults"], list) and d["defaults"], "defaults must be a non-empty list")

# variants — the parse_variant_rows fields (+ source + configured_ctx +
# weights_companions/drafter/vision); port is an int.  configured_ctx is the EXACT
# numeric registry max_ctx int behind ctx_label (the cockpit's divergence badge
# compares the probe against it).  weights_companions = the per-slug extra weight
# keys (DFlash draft / mmproj) the cockpit Download fetches alongside the core;
# drafter / vision are the per-slug facets (display + companion derivation).
VARIANT_KEYS = {
    "slug", "switch_engine", "launch_engine", "compose_dir", "file", "port",
    "model", "engine", "kvcalc_key", "container", "compose_path", "status",
    "ctx_label", "configured_ctx", "status_note", "source",
    "weights_companions", "drafter", "vision",
}
v0 = d["variants"][0]
need(set(v0.keys()) == VARIANT_KEYS,
     f"variant keys mismatch (got {sorted(v0.keys())})")
# weights_companions is a list; drafter a str; vision a bool.
need(isinstance(v0["weights_companions"], list),
     f"variant.weights_companions must be list (got {type(v0['weights_companions']).__name__})")
need(isinstance(v0["vision"], bool),
     f"variant.vision must be bool (got {type(v0['vision']).__name__})")
need(isinstance(v0["port"], int), f"variant.port must be int (got {type(v0['port']).__name__})")
need(v0["source"] == "curated", f"variant.source default must be 'curated' (got {v0['source']!r})")
# configured_ctx is an int (or None) — the exact registry max_ctx behind ctx_label.
need(v0["configured_ctx"] is None or isinstance(v0["configured_ctx"], int),
     f"variant.configured_ctx must be int|None (got {type(v0['configured_ctx']).__name__})")

# defaults
DEFAULT_KEYS = {"model", "engine", "topology", "slug", "source"}
need(set(d["defaults"][0].keys()) == DEFAULT_KEYS,
     f"default keys mismatch (got {sorted(d['defaults'][0].keys())})")

# profiles — four catalog blocks, each a non-empty dict.
prof = d["profiles"]
need(set(prof.keys()) == {"engines", "models", "hardware", "drafters"},
     f"profiles keys != engines/models/hardware/drafters (got {sorted(prof.keys())})")
for block in ("engines", "models", "hardware", "drafters"):
    need(isinstance(prof[block], dict) and prof[block],
         f"profiles.{block} must be a non-empty dict")

# per-profile field shapes
eng = next(iter(prof["engines"].values()))
need({"image", "min_sm", "supported_kv_formats", "supported_weight_formats",
      "supported_drafters", "supported_model_families"} <= set(eng.keys()),
     f"engine profile missing fields (got {sorted(eng.keys())})")

mod = next(iter(prof["models"].values()))
need({"family", "valid_tp", "max_ctx", "hf_repo", "weights"} <= set(mod.keys()),
     f"model profile missing fields (got {sorted(mod.keys())})")

hw = next(iter(prof["hardware"].values()))
need({"vram_gb", "sm", "supported_kv_formats"} <= set(hw.keys()),
     f"hardware profile missing fields (got {sorted(hw.keys())})")

dr = next(iter(prof["drafters"].values()))
need({"type", "accept"} <= set(dr.keys()),
     f"drafter profile missing fields (got {sorted(dr.keys())})")

print(
    f"json shape ok: {len(d['variants'])} variants, {len(d['defaults'])} defaults, "
    f"{len(prof['engines'])} engines, {len(prof['models'])} models, "
    f"{len(prof['hardware'])} hardware, {len(prof['drafters'])} drafters"
)
PY

# --- 2. usage error when invoked with no flag ------------------------------
if bash scripts/lib/registry-emit.sh >/dev/null 2>&1; then
  fail "direct invocation with no flag should exit non-zero"
fi

# --- 3. HARD: sourced tab output stays present + additive --------------------
# Sourcing must remain silent (the direct-invocation guard must not fire) and
# registry_variant_rows must still emit VARIANT/DEFAULT rows.
src_noise="$(bash -c 'source scripts/lib/registry-emit.sh; echo READY' 2>&1)"
[[ "$src_noise" == "READY" ]] || fail "sourcing the file is not silent: $src_noise"

tab="$(bash -c 'source scripts/lib/registry-emit.sh && registry_variant_rows "$1"' _ "$ROOT_DIR" 2>/dev/null)"
[[ "$tab" == VARIANT* ]] || fail "sourced registry_variant_rows lost its VARIANT rows"
grep -q '^DEFAULT' <<<"$tab" || fail "sourced registry_variant_rows lost its DEFAULT rows"

echo "test-registry-json: ok"
