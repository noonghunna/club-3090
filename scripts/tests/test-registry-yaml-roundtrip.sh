#!/usr/bin/env bash
# test-registry-yaml-roundtrip.sh — the registry-as-data migration contract:
#
#   1. The module's loaded state (COMPOSE_REGISTRY / DEFAULTS /
#      ENGINE_PREFERENCE / RECOMMENDED_DEFAULT_MODELS) is EXACTLY what
#      registry.yaml round-trips to: every entry rebuilt through _entry()
#      equals the loaded dict bit-for-bit, and the on-disk file is the
#      canonical form (dump(parse(file)) == file bytes) — so a hand edit that
#      breaks the round-trip fails here, loudly.
#   2. Loading is python-STDLIB-ONLY: the core catalog import must succeed
#      with PyYAML poisoned out of the interpreter (#584 community rigs —
#      the launcher table path imports this module).
#   3. A corrupt registry.yaml fails LOUDLY (RegistryDataError at import),
#      never silently shrinks the catalog.
#   4. The recently added entry fields (served_name / gateway / serve_aliases /
#      sampler_profiles with multi-row dicts) survive dump → parse → _entry().
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1 + 4 — parity, canonical form, new-field survival (one interpreter).
python3 - <<'PY' || fail "round-trip parity / canonical form"
import copy
import sys

from scripts.lib.profiles import compose_registry as cr

raw = cr.load_registry_data()  # default path = this module's registry.yaml
entries, defaults = cr._build_core_catalog(raw, "<guard>")
assert entries == cr.COMPOSE_REGISTRY, "entries drift between YAML and module state"
assert defaults == cr.DEFAULTS, "DEFAULTS drift between YAML and module state"
assert raw["engine_preference"] == cr.ENGINE_PREFERENCE, "ENGINE_PREFERENCE drift"
assert raw["recommended_default_models"] == list(cr.RECOMMENDED_DEFAULT_MODELS), (
    "RECOMMENDED_DEFAULT_MODELS drift"
)

# Canonical form: the on-disk bytes are exactly what the shared writer emits.
text = cr._REGISTRY_YAML_HEADER and (
    __import__("pathlib").Path(cr.__file__).resolve().parent.joinpath("registry.yaml")
    .read_text(encoding="utf-8")
)
assert cr.dump_registry_yaml(raw) == text, (
    "registry.yaml is not the canonical form — run "
    "scripts/lib/profiles/migrate_registry_to_yaml.py (no --check) to re-canonicalize"
)

# Every entry carries the _entry()-derived shape (proves the _entry wrap).
for slug, e in cr.COMPOSE_REGISTRY.items():
    assert e["pp"] == 1 and e["gpu_assignment_mode"] == "contiguous", slug
    assert "pp" not in raw["entries"][slug] and "gpu_assignment_mode" not in raw["entries"][slug], slug

# New fields survive the round-trip — synthetic entry, dump → parse → _entry.
kwargs = {
    "model": "guard-dummy-1b", "weights_variant": "dummy-int4", "workload": "fast-chat",
    "engine": "vllm-stable", "drafter": None, "kv_format": "bf16",
    "tp": 1, "max_ctx": 8192, "max_num_seqs": 1, "mem_util": 0.9,
    "compose_path": "models/guard-dummy-1b/vllm/compose/single/dummy-int4/base.yml",
    "default_port": 8999, "kvcalc_key": "guard:dummy",
    "served_name": "guard-dummy-serve",
    "gateway": True,
    "serve_aliases": ("guard-alias-a", "guard-alias-b"),
    "sampler_profiles": {
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                      "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                      "presence_penalty": 0.0, "repetition_penalty": 1.0},
    },
    "status_note": "guard: colon, # hash, — em-dash, ⚠ unicode",
}
data = copy.deepcopy(raw)
data["entries"]["guard/dummy"] = {
    k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()
}
rt = cr.parse_registry_text(cr.dump_registry_yaml(data), source="<guard>")
rebuilt, _ = cr._build_core_catalog(rt, "<guard>")
want = cr._entry(**kwargs)
got = rebuilt["guard/dummy"]
assert got == want, f"synthetic entry drift: {got!r} != {want!r}"
assert got["serve_aliases"] == ["guard-alias-a", "guard-alias-b"]
assert got["sampler_profiles"]["thinking"]["top_p"] == 0.95
print("PASS: registry.yaml round-trips bit-identical (100 entries + policy maps)")
PY

# 2 — stdlib-only import (PyYAML poisoned out of the interpreter).
stubdir="$(mktemp -d)"
trap 'rm -rf "$stubdir"' EXIT
printf 'raise ImportError("PyYAML hidden by test-registry-yaml-roundtrip")\n' \
    > "$stubdir/yaml.py"
PYTHONPATH="$stubdir" python3 -c "import scripts.lib.profiles.compose_registry" \
    || fail "core registry import requires PyYAML — the #584 stdlib-only invariant is broken"
echo "PASS: core registry loads with PyYAML poisoned (stdlib-only)"

# 3 — corrupt file fails loudly at import.
corrupt="$(mktemp -d)"
mkdir -p "$corrupt/scripts/lib/profiles"
cp "$ROOT/scripts/lib/profiles/__init__.py" "$corrupt/scripts/lib/profiles/" 2>/dev/null || true
cp "$ROOT/scripts/lib/__init__.py" "$corrupt/scripts/lib/" 2>/dev/null || true
cp "$ROOT/scripts/__init__.py" "$corrupt/scripts/" 2>/dev/null || true
cp "$ROOT/scripts/lib/profiles/compose_registry.py" "$corrupt/scripts/lib/profiles/"
printf 'schema: 1\nentries: [this is not a mapping]\n' \
    > "$corrupt/scripts/lib/profiles/registry.yaml"
set +e
err="$(cd "$corrupt" && python3 -c "import scripts.lib.profiles.compose_registry" 2>&1)"
rc=$?
set -e
[[ $rc -ne 0 ]] || fail "corrupt registry.yaml imported SILENTLY — loud-failure invariant broken"
grep -q "RegistryDataError\|registry.yaml" <<<"$err" \
    || fail "corrupt registry.yaml error is not actionable (got: ${err:0:200})"
echo "PASS: corrupt registry.yaml fails loudly with RegistryDataError"

echo "test-registry-yaml-roundtrip: ok"
