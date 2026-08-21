#!/usr/bin/env bash
#
# registry-lookup.sh — sourceable, stdlib-only lookups over the compose registry.
#
# Wraps `registry-emit.sh --json` (bash + python3 stdlib, same deps as the emit
# path itself) and exposes per-slug / per-model accessors so consumers stop
# hand-maintaining port, container and compose-dir literals that drift from the
# registry:
#
#   registry_lookup_port              <slug>          → e.g. 8020
#   registry_lookup_container         <slug>          → e.g. vllm-qwen36-27b-minimal
#   registry_lookup_compose_dir       <slug>          → e.g. models/qwen3.6-27b/vllm/compose
#   registry_lookup_compose_path      <slug>          → e.g. models/.../dual/autoround-int4/fp8-mtp.yml
#   registry_lookup_served_name       <slug>          → e.g. qwen3.6-27b
#   registry_lookup_default_slug      <model> [topo]  → curated DEFAULTS walk, e.g. vllm/minimal
#   registry_lookup_default_port      <model> [topo]  → port of that default slug's variant
#   registry_lookup_default_container <model> [topo]  → container of that default slug's variant
#
# Every function prints the value on stdout and returns non-zero (empty output)
# when the registry cannot be consulted or the slug/model is unknown — callers
# are expected to keep a literal fallback for that case, e.g.:
#
#   port="$(registry_lookup_port vllm/dual 2>/dev/null || true)"; port="${port:-8010}"
#
# Caching: the JSON catalog is emitted ONCE per process into a tmp file
# (${TMPDIR:-/tmp}/registry-lookup.<pid>.json); every lookup after the first
# reads the cache. The emit costs ~1s, so source this file only where a
# registry answer is actually needed — and prefer one shared resolve over
# per-line lookups in hot paths. registry_lookup_cleanup removes the cache file
# early; otherwise the tmp file lives until the consuming process exits.

export PYTHONUTF8="${PYTHONUTF8:-1}"
# Repo root the helper resolves the registry against. Derived from this file's
# location (scripts/lib/ → two levels up) unless the consumer pins it, e.g.
# REGISTRY_LOOKUP_ROOT="$CLUB3090_DIR" before the first lookup.
: "${REGISTRY_LOOKUP_ROOT:=}"

_registry_lookup_cache=""

# Ensure the cached catalog exists; prints nothing, returns non-zero on failure.
registry_lookup_cache_path() {
    # Deterministic per-process path ($$ is shared with command-substitution
    # subshells, unlike plain variables set inside them) — so whether the first
    # lookup happens at top level or inside $( ), later lookups in the SAME
    # process reuse one emit instead of re-running it. mv makes the publish
    # atomic against a concurrent consumer with the same PID shape (none today,
    # but cheap to be correct about).
    local tmp="${TMPDIR:-/tmp}/registry-lookup.$$.json"
    if [[ ! -s "${tmp}" ]]; then
        local root="${REGISTRY_LOOKUP_ROOT}"
        if [[ -z "${root}" ]]; then
            root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
        fi
        [[ -f "${root}/scripts/lib/registry-emit.sh" ]] || return 1
        local staged
        staged="$(mktemp "${TMPDIR:-/tmp}/registry-lookup.XXXXXXXX.json")" || return 1
        if ! bash "${root}/scripts/lib/registry-emit.sh" --json "${root}" > "${staged}" 2>/dev/null \
           || ! [[ -s "${staged}" ]]; then
            rm -f "${staged}"
            return 1
        fi
        mv -f "${staged}" "${tmp}" || { rm -f "${staged}"; return 1; }
    fi
    _registry_lookup_cache="${tmp}"
    return 0
}

# Drop the cached catalog (call from the consumer's EXIT trap if it wants the
# tmp file gone before process exit).
registry_lookup_cleanup() {
    if [[ -n "${_registry_lookup_cache}" ]]; then
        rm -f "${_registry_lookup_cache}"
        _registry_lookup_cache=""
    fi
}

# Internal: print FIELD of the first variant row whose slug matches SLUG.
_registry_lookup_variant_field() {  # <slug> <field>
    registry_lookup_cache_path || return 1
    REGISTRY_LOOKUP_CACHE="${_registry_lookup_cache}" \
    REGISTRY_LOOKUP_SLUG="$1" \
    REGISTRY_LOOKUP_FIELD="$2" \
    python3 - <<'PY'
import json, os, sys
with open(os.environ["REGISTRY_LOOKUP_CACHE"], encoding="utf-8") as f:
    catalog = json.load(f)
slug = os.environ["REGISTRY_LOOKUP_SLUG"]
field = os.environ["REGISTRY_LOOKUP_FIELD"]
for row in catalog.get("variants", []):
    if row.get("slug") == slug:
        val = row.get(field)
        if val is None:
            sys.exit(1)
        print(val)
        sys.exit(0)
sys.exit(1)
PY
}

registry_lookup_port()         { _registry_lookup_variant_field "$1" port; }
registry_lookup_container()    { _registry_lookup_variant_field "$1" container; }
registry_lookup_compose_dir()  { _registry_lookup_variant_field "$1" compose_dir; }
registry_lookup_compose_path() { _registry_lookup_variant_field "$1" compose_path; }
registry_lookup_served_name()  { _registry_lookup_variant_field "$1" served_name; }

# Internal: resolve MODEL's curated default slug (first curated DEFAULTS entry,
# in curated order; TOPOLOGY filters when given) and print FIELD from that
# slug's variant row. This is the "curated DEFAULTS walk", minus user pins and
# arch gating — those need the full model_default_target in registry-emit.sh,
# not a display/defaulting helper.
_registry_lookup_default_field() {  # <model> <field> [topology]
    registry_lookup_cache_path || return 1
    REGISTRY_LOOKUP_CACHE="${_registry_lookup_cache}" \
    REGISTRY_LOOKUP_MODEL="$1" \
    REGISTRY_LOOKUP_FIELD="$2" \
    REGISTRY_LOOKUP_TOPOLOGY="${3:-}" \
    python3 - <<'PY'
import json, os, sys
with open(os.environ["REGISTRY_LOOKUP_CACHE"], encoding="utf-8") as f:
    catalog = json.load(f)
model = os.environ["REGISTRY_LOOKUP_MODEL"]
field = os.environ["REGISTRY_LOOKUP_FIELD"]
topology = os.environ["REGISTRY_LOOKUP_TOPOLOGY"]
slug = None
for entry in catalog.get("defaults", []):
    if entry.get("model") != model or entry.get("source") != "curated":
        continue
    if topology and entry.get("topology") != topology:
        continue
    slug = entry.get("slug")
    break
if not slug:
    sys.exit(1)
for row in catalog.get("variants", []):
    if row.get("slug") == slug:
        val = row.get(field)
        if val is None:
            sys.exit(1)
        print(val)
        sys.exit(0)
sys.exit(1)
PY
}

registry_lookup_default_slug()      { _registry_lookup_default_field "$1" slug "${2:-}"; }
registry_lookup_default_port()      { _registry_lookup_default_field "$1" port "${2:-}"; }
registry_lookup_default_container() { _registry_lookup_default_field "$1" container "${2:-}"; }
