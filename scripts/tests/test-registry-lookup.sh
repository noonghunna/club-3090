#!/usr/bin/env bash
# test-registry-lookup.sh — guards the shared registry lookup helper
# (scripts/lib/registry-lookup.sh) AND the two drift bugs it was written to
# fix:
#
#   1. Helper unit checks against the REAL registry: slug → port / container /
#      compose_dir / compose_path / served_name, the curated DEFAULTS walk,
#      unknown-slug failure shape, and per-process caching (one emit, reused).
#   2. bench.sh / verify.sh / verify-full.sh / verify-stress.sh used to default
#      CONTAINER to 'vllm-qwen36-27b' — a container name NO registry variant
#      ships, so docker-inspect/exec consumers silently no-op'd. The default is
#      now the MODEL's curated-default slug container (evaluated from the
#      scripts' own text, so reordering can't silently unfix it), with the old
#      literal kept only as last-resort fallback.
#   3. gpu-mode.sh status probed :8012 (27b-dflash) and :8011 (27b-turbo) — no
#      registry variant serves those ports. Guard: every LLM probe port in
#      gpu-mode.sh must exist in the registry (explicit infra allowlist aside),
#      and the two dead ports must stay gone.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

# shellcheck source=../lib/registry-lookup.sh
source "$ROOT/scripts/lib/registry-lookup.sh"
REGISTRY_LOOKUP_ROOT="$ROOT"

expect() {  # <label> <expected> <actual>
  [[ "$2" == "$3" ]] || note "$1: expected '$2', got '$3'"
}
expect_rc() {  # <label> <expected_rc> <actual_rc>
  [[ "$2" == "$3" ]] || note "$1: expected rc=$2, got rc=$3"
}

# --- 1a. slug lookups against the live registry -----------------------------
expect "port(vllm/minimal)"        "8020"                    "$(registry_lookup_port vllm/minimal)"
expect "container(vllm/minimal)"   "vllm-qwen36-27b-minimal" "$(registry_lookup_container vllm/minimal)"
expect "served_name(vllm/minimal)" "qwen3.6-27b"             "$(registry_lookup_served_name vllm/minimal)"
expect "compose_dir(vllm/dual)"    "models/qwen3.6-27b/vllm/compose" \
  "$(registry_lookup_compose_dir vllm/dual)"
case "$(registry_lookup_compose_path vllm/dual)" in
  models/qwen3.6-27b/vllm/compose/dual/*/*.yml) : ;;
  *) note "compose_path(vllm/dual): unexpected '$(registry_lookup_compose_path vllm/dual)'" ;;
esac

out="$(registry_lookup_port no-such/slug 2>/dev/null || true)"
[[ -z "$out" ]] || note "unknown slug should print nothing (got '$out')"
registry_lookup_port no-such/slug >/dev/null 2>&1 && rc=0 || rc=$?
expect_rc "unknown slug rc" 1 "$rc"

# --- 1b. curated DEFAULTS walk ----------------------------------------------
expect "default_slug(qwen3.6-27b)"   "vllm/minimal"            "$(registry_lookup_default_slug qwen3.6-27b)"
expect "default_port(qwen3.6-27b)"   "8020"                    "$(registry_lookup_default_port qwen3.6-27b)"
expect "default_port(dual filter)"   "8010"                    "$(registry_lookup_default_port qwen3.6-27b dual)"
expect "default_container(qwen3.6)"  "vllm-qwen36-27b-minimal" "$(registry_lookup_default_container qwen3.6-27b)"
expect "default_container(gemma31b)" "vllm-gemma-4-31b-qat-awq-int4" \
  "$(registry_lookup_default_container gemma-4-31b)"

# --- 1c. one emit per process -------------------------------------------------
p1="$(registry_lookup_cache_path && printf %s "$_registry_lookup_cache")"
p2="$(registry_lookup_cache_path && printf %s "$_registry_lookup_cache")"
expect "cache path stable within one process" "$p1" "$p2"
[[ -s "$p1" ]] || note "cache file missing/empty: $p1"
registry_lookup_cleanup

# --- 2. CONTAINER defaults resolve to REGISTRY containers --------------------
# Evaluated from each script's own text: everything from the MODEL default
# assignment through the final CONTAINER fallback line, with CONTAINER unset.
resolve_default_container() {  # <script>
  awk '/^MODEL="\$\{MODEL:-/{on=1} on{print} /^CONTAINER="\$\{CONTAINER:-/{if(on){exit}}' "$1"
}
for script in scripts/bench.sh scripts/verify.sh scripts/verify-full.sh scripts/verify-stress.sh; do
  resolved="$(ROOT_DIR="$ROOT" bash -c '
    unset CONTAINER MODEL
    eval "$1"
    printf %s "${CONTAINER:-<UNSET>}"
  ' _ "$(resolve_default_container "$script")")"
  expect "$script: default CONTAINER resolves to registry container" \
    "vllm-qwen36-27b-minimal" "$resolved"

  explicit="$(ROOT_DIR="$ROOT" bash -c '
    CONTAINER=my-custom-container
    eval "$1"
    printf %s "${CONTAINER:-<UNSET>}"
  ' _ "$(resolve_default_container "$script")")"
  expect "$script: explicit CONTAINER wins" "my-custom-container" "$explicit"

  grep -q 'registry_lookup_default_container "$MODEL"' "$script" \
    || note "$script: no longer derives its CONTAINER default from the registry"
  grep -q 'CONTAINER="${CONTAINER:-vllm-qwen36-27b}"' "$script" \
    || note "$script: lost the last-resort literal fallback"
done

# --- 3. no status probe targets a port absent from the registry --------------
REGJSON="$(mktemp /tmp/reg-look.XXXX.json)"
"$ROOT/scripts/lib/registry-emit.sh" --json "$ROOT" > "$REGJSON"
bad_ports="$(
  REGJSON="$REGJSON" python3 - <<'PY'
import json, os, re, sys
with open(os.environ["REGJSON"], encoding="utf-8") as f:
    catalog = json.load(f)
reg = {str(v.get("port")) for v in catalog.get("variants", []) if v.get("port")}
# Infra probes that are NOT catalog model endpoints (studio director /
# ComfyUI UI / LiteLLM router) — each deliberately hardcoded in gpu-mode.sh.
infra = {"8090", "8188", "4000"}
text = open("scripts/gpu-mode.sh", encoding="utf-8").read()
bad = sorted(set(re.findall(r"localhost:(\d+)/v1/models", text)) - reg - infra)
if bad:
    print(f"probe ports not in registry (and not infra allowlist): {bad}")
    sys.exit(1)
PY
)" || note "$bad_ports"


for dead in 8011 8012; do
  grep -q ":${dead}" scripts/gpu-mode.sh \
    && note "gpu-mode.sh still references dead probe port :${dead} (no registry variant serves it)"
done

rm -f "$REGJSON"

if [[ "$fail" -ne 0 ]]; then
  echo "test-registry-lookup: FAILED" >&2
  exit 1
fi
echo "test-registry-lookup: ok"
