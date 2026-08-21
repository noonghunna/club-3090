#!/usr/bin/env bash
# Gate: the litellm LOCAL block is registry-derived, byte-stable, and drift-proof.
#
# Covers scripts/lib/litellm-emit.sh (#1078) end to end:
#   1. generation is IDEMPOTENT — a second run over an up-to-date config is a
#      byte-level no-op;
#   2. --check PASSES on the regenerated config;
#   3. --check FAILS (exit 1 + diff) when a generated route is hand-mutated;
#   4. everything OUTSIDE the BEGIN/END GENERATED markers survives a rewrite —
#      hand-maintained local routes AND the cloud block;
#   5. the #1073 serve_aliases mechanism: extra names become additional
#      model_name routes on the SAME upstream port (synthetic fixture — no core
#      entry sets aliases yet).
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
def get_registry(root=None):
    return {"local/synthetic": dict(_STUB)}
def curated_default_target(model, topology, detected_sm=None):
    return "local/synthetic"
PY
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
[ "$(grep -cF 'api_base: http://host.docker.internal:8123/v1' "$FIX/config.yaml")" -eq 3 ] \
  || fail "alias mechanism: aliases must ride the SAME upstream route (:8123)"
LITELLM_EMIT_REGISTRY_JSON="$FIX/facts.json" LITELLM_CONFIG="$FIX/config.yaml" \
  bash "$EMIT" --check "$FIX" >/dev/null || fail "fixture --check failed after generation"
rm -rf "$FIX"
echo "OK: litellm-emit idempotent, --check green, hand-mutation caught, "
echo "    cloud/hand blocks preserved, #1073 serve_aliases emission verified."
