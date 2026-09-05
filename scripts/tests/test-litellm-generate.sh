#!/usr/bin/env bash
# Gate: the litellm LOCAL block is registry-derived, byte-stable, and drift-proof.
#
# Covers scripts/lib/litellm-emit.sh (#1078 + Wave-2 follow-up) end to end:
#   1. generation is IDEMPOTENT — a second run over an up-to-date config is a
#      byte-level no-op;
#   2. --check PASSES on the regenerated config;
#   3. --check FAILS (exit 1 + diff) when a generated route is hand-mutated;
#   4. everything OUTSIDE the BEGIN/END GENERATED markers survives a rewrite —
#      hand-maintained local routes AND the cloud block;
#   5. the #1073 serve_aliases mechanism: extra names become additional
#      model_name routes on the SAME upstream port (synthetic fixture);
#   6. every formerly hand-written local route (deepseek, 35b-a3b, gemma dual
#      + alias, gemma-12b-int8, deckard) is ABSORBED into the generated block
#      — no route lost, correct port per route;
#   7. non-functional statuses annotate their route `# status: <status>`
#      (deepseek non-functional; the qwen3.8-27b dual-max experimental scene)
#      while functional routes stay clean.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EMIT="scripts/lib/litellm-emit.sh"
CFG="services/litellm/config.yaml"
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 1+2+4: real config — idempotent, check-clean, non-generated content kept --

cp "$CFG" /tmp/litellm-generate.baseline."$$"
bash "$EMIT" >/dev/null || fail "generator errored on the checked-in config"
bash "$EMIT" >/dev/null || fail "second generator pass errored"
cmp -s "$CFG" "/tmp/litellm-generate.baseline.$$" \
  || fail "generation is not idempotent — second run changed $CFG"

bash "$EMIT" --check >/dev/null || fail "--check failed on the regenerated config"
# Generated-block helpers: resolve a route's api_base / full model_name line.
gen_route_base() { # NAME → the api_base of that generated route
  awk -v want="$1" '
    /=== BEGIN GENERATED LOCAL BLOCK/ { blk=1; next }
    /=== END GENERATED LOCAL BLOCK/   { blk=0 }
    blk && /- model_name:/ {
      cur=$0
      sub(/^.*- model_name: /, "", cur)
      sub(/[[:space:]]*#.*$/, "", cur)
    }
    blk && cur == want && /api_base:/ { print $2; exit }
  ' "$CFG"
}
gen_route_line() { # NAME → the raw `- model_name:` line of that generated route
  awk -v want="$1" '
    /=== BEGIN GENERATED LOCAL BLOCK/ { blk=1; next }
    /=== END GENERATED LOCAL BLOCK/   { blk=0 }
    blk && /- model_name:/ {
      cur=$0
      sub(/^.*- model_name: /, "", cur)
      sub(/[[:space:]]*#.*$/, "", cur)
    }
    blk && cur == want && !done++ { print; exit }
  ' "$CFG"
}

# --- 6+7: formerly hand-written routes are absorbed; statuses annotated -------

for pair in \
  "deepseek-v4-flash:8030" \
  "qwen3.6-35b-a3b-autoround:8051" \
  "gemma-4-31b:8032" \
  "gemma-4-31b-autoround:8032" \
  "gemma-4-12b-int8:8038" \
  "deckard-40b:8199"; do
  name="${pair%:*}" port="${pair##*:}"
  if ! [ "$(gen_route_base "$name")" = "http://host.docker.internal:${port}/v1" ]; then
    fail "absorbed route ${name} missing/wrong port (wanted :${port}, got '$(gen_route_base "$name")')"
  fi
done

# The annotation must match the slug's CURRENT registry status, not a status
# frozen into this test. deepseek was `incubating` when this was written and is
# `experimental` since #1175; hard-coding the word made a routine status change
# look like a generator bug. Derive it instead, so the next change cannot drift.
_ds_status="$(command grep -oE 'model_name: deepseek-v4-flash  # status: [a-z_]+' \
  <(sed -n '/BEGIN GENERATED/,/END GENERATED/p' "$CFG") | awk '{print $NF}')"
[ -n "$_ds_status" ] \
  || fail "non-functional deepseek route lacks the '# status:' annotation"
command grep -qE "^[[:space:]]+status: ${_ds_status}\$" scripts/lib/profiles/registry.yaml \
  || fail "deepseek route annotated '# status: ${_ds_status}', which is not a status in registry.yaml"
# ⚠️ No `incubating` slug is currently ROUTED through litellm (the one that
# remains is the opt-in LMCache dual, hidden from --list), so that branch of the
# annotation contract is deliberately uncovered here rather than faked.
grep -qF 'model_name: qwen3.8-27b  # status: experimental' \
  <(sed -n '/BEGIN GENERATED/,/END GENERATED/p' "$CFG") \
  || fail "experimental qwen3.8 dual-max route lacks the '# status:' annotation"
if ! [ "$(gen_route_line deckard-40b)" = "  - model_name: deckard-40b" ]; then
  fail "functional (production) route deckard-40b must NOT carry a '# status:' annotation"
fi
if ! [ "$(gen_route_line gemma-4-31b)" = "  - model_name: gemma-4-31b" ]; then
  fail "functional (caveats) route gemma-4-31b must NOT carry a '# status:' annotation"
fi

# --- 4 (stronger): everything OUTSIDE the markers is byte-identical to pre-run
diff <(sed -n '1,/BEGIN GENERATED LOCAL BLOCK/p' "/tmp/litellm-generate.baseline.$$") \
     <(sed -n '1,/BEGIN GENERATED LOCAL BLOCK/p' "$CFG") >/dev/null \
  || fail "content ABOVE the generated markers changed during regeneration"
diff <(sed -n '/END GENERATED LOCAL BLOCK/,$p' "/tmp/litellm-generate.baseline.$$") \
     <(sed -n '/END GENERATED LOCAL BLOCK/,$p' "$CFG") >/dev/null \
  || fail "hand-maintained/cloud content BELOW the markers changed during regeneration"

for needle in \
  "model_name: deepseek-v4-flash" \
  "model_name: agents-a1" \
  "model_name: gemma-4-31b-autoround" \
  "model_name: deckard-40b" \
  "model_name: qwen3.8-max-nothink" \
  'extra_body: {"thinking_budget": 1}' \
  "os.environ/DASHSCOPE_API_KEY"; do
  grep -qF "$needle" "$CFG" || fail "regeneration lost non-generated content: $needle"
done

# --- 3: hand-mutation of a generated route is caught --------------------------

TMP_CFG="$(mktemp /tmp/litellm-mutated.XXXXXX.yaml)"
trap 'rm -f "$TMP_CFG" /tmp/litellm-generate.baseline."$$"' EXIT
sed 's|host.docker.internal:8091/v1|host.docker.internal:8099/v1|' "$CFG" > "$TMP_CFG"
if LITELLM_CONFIG="$TMP_CFG" bash "$EMIT" --check >/tmp/litellm-mutate.out 2>&1; then
  fail "--check passed despite a hand-mutated generated route (8091→8099)"
fi
grep -q "^FAIL:" /tmp/litellm-mutate.out || fail "--check failure output lacks a FAIL banner"
grep -q "registry-generated" /tmp/litellm-mutate.out || fail "--check failure output lacks the diff"

# --- 5: #1073 serve_aliases mechanism (synthetic fixture) ---------------------

FIX="$(mktemp -d /tmp/litellm-fixture.XXXXXX)"
mkdir -p "$FIX/scripts/lib/profiles"
cat > "$FIX/scripts/lib/profiles/__init__.py" <<'PY'
PY
cat > "$FIX/scripts/lib/profiles/compose_registry.py" <<'PY'
_STUB = {
    "model": "synthetic-1",
    "gateway": True,
    "served_name": "synthetic-1",
    "serve_aliases": ["synthetic-1-alt", "synthetic-1-legacy"],  # #1073 mechanism
    "compose_path": "models/synthetic/none.yml",
    "default_port": 8123,
}
# llama.cpp-family stub: no served_name fact — the name must come from parsing
# the compose's `--alias` (list form, quoted); experimental → status annotation.
_STUB_LLAMA = {
    "model": "synthetic-2",
    "gateway": True,
    "served_name": None,
    "compose_path": "models/synthetic2/compose.yml",
    "default_port": 8124,
    "status": "experimental",
}
FUNCTIONAL_STATUSES = frozenset({"production", "caveats"})
def get_registry(root=None):
    return {"local/synthetic": dict(_STUB), "local/synthetic-llama": dict(_STUB_LLAMA)}
def curated_default_target(model, topology, detected_sm=None):
    return {"synthetic-1": "local/synthetic", "synthetic-2": "local/synthetic-llama"}[model]
PY
mkdir -p "$FIX/models/synthetic2"
cat > "$FIX/models/synthetic2/compose.yml" <<'EOF'
services:
  llama:
    command:
      #   SERVED_NAME --alias value (default: decoy)   ← comment prose must NOT match
      - '--host'
      - '0.0.0.0'
      - '--alias'
      - 'synthetic-2'
      - -np
      - '1'
EOF
echo '{"variants": [], "defaults": []}' > "$FIX/facts.json"
cat > "$FIX/config.yaml" <<'EOF'
model_list:
  # === BEGIN GENERATED LOCAL BLOCK — scripts/lib/litellm-emit.sh (#1078); DO NOT hand-edit ===
  # === END GENERATED LOCAL BLOCK ===

  # === Cloud (hand-maintained) ===
  - model_name: cloud-ref
    litellm_params:
      model: openai/cloud-ref
      api_base: https://cloud.example/v1
      api_key: os.environ/CLOUD_KEY
EOF

LITELLM_EMIT_REGISTRY_JSON="$FIX/facts.json" LITELLM_CONFIG="$FIX/config.yaml" \
  bash "$EMIT" "$FIX" >/dev/null \
  || fail "generator errored on the synthetic alias fixture"

for name in synthetic-1 synthetic-1-alt synthetic-1-legacy; do
  grep -qF "model_name: $name" "$FIX/config.yaml" \
    || fail "alias mechanism: $name not emitted as a model_name"
done
grep -qF 'model_name: synthetic-2  # status: experimental' "$FIX/config.yaml" \
  || fail "alias parsing: synthetic-2 not emitted (or lacks the non-functional '# status:' annotation)"
if grep -qF 'model_name: value' "$FIX/config.yaml"; then
  fail "alias parsing: compose header COMMENT prose leaked in as a served name"
fi
if grep -qE '^  - model_name: -np' "$FIX/config.yaml"; then
  fail "alias parsing: list-form scan ran past the --alias value into the next flag"
fi
[ "$(grep -cF 'api_base: http://host.docker.internal:8123/v1' "$FIX/config.yaml")" -eq 3 ] \
  || fail "alias mechanism: aliases must ride the SAME upstream route (:8123)"
LITELLM_EMIT_REGISTRY_JSON="$FIX/facts.json" LITELLM_CONFIG="$FIX/config.yaml" \
  bash "$EMIT" --check "$FIX" >/dev/null || fail "fixture --check failed after generation"
rm -rf "$FIX"
echo "OK: litellm-emit idempotent, --check green, hand-mutation caught, "
echo "    cloud/hand blocks preserved, absorbed hand routes intact, status"
echo "    annotation + #1073 serve_aliases/--alias emission verified."
