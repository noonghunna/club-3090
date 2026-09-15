#!/usr/bin/env bash
# Gate: `litellm-emit.sh --local` routes the gateway to the rig's PINNED scene
# without touching tracked files (#1325).
#
# The tracked services/litellm/config.yaml follows the curated gateway scene
# (#1078). A rig serving a different scene of the same model records that in
# its .env pin (CLUB3090_DEFAULT_<MODEL>, set by `switch.sh --set-default`);
# --local writes a GITIGNORED config.local.yaml that honours the pin:
#   1. pin names a known slug of the SAME model → that slug's port, with the
#      non-functional `# status:` annotation (status is NOT a gate here — the
#      gateway routes to what the rig serves);
#   2. pin read from ROOT/.env when not in the shell env (shell env wins);
#   3. unknown slug / other-model pin / no default_port → curated scene + a
#      stderr notice (never a crash);
#   4. no pin → the local block equals the tracked generated block;
#   5. --local never modifies the tracked config, and everything outside the
#      markers (cloud / hand routes) is carried over;
#   6. --local --check passes on a fresh local config and fails on drift;
#   7. the default output path is gitignored, and the LiteLLM compose mounts
#      ${LITELLM_CONFIG:-./config.yaml}.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EMIT="scripts/lib/litellm-emit.sh"
fail() { echo "FAIL: $*" >&2; exit 1; }

FIX="$(mktemp -d /tmp/litellm-local-pin.XXXXXX)"
trap 'rm -rf "$FIX"' EXIT
mkdir -p "$FIX/scripts/lib/profiles"
: > "$FIX/scripts/lib/profiles/__init__.py"
cat > "$FIX/scripts/lib/profiles/compose_registry.py" <<'PY'
FUNCTIONAL_STATUSES = frozenset({"production", "caveats"})
_REG = {
    # curated gateway scene for synthetic-1
    "vllm/syn-max": {"model": "synthetic-1", "gateway": True, "served_name": "synthetic-1",
                     "compose_path": "models/syn/max.yml", "default_port": 8123,
                     "status": "production"},
    # a second scene of the SAME model — not gateway-flagged, experimental
    "vllm/syn-fast": {"model": "synthetic-1", "gateway": False, "served_name": "synthetic-1",
                      "compose_path": "models/syn/fast.yml", "default_port": 8125,
                      "status": "experimental"},
    # same model, but no port → an unroutable pin
    "vllm/syn-noport": {"model": "synthetic-1", "gateway": False, "served_name": "synthetic-1",
                        "compose_path": "models/syn/noport.yml", "default_port": None},
    # a different gateway model
    "vllm/other": {"model": "synthetic-2", "gateway": True, "served_name": "synthetic-2",
                   "compose_path": "models/other/x.yml", "default_port": 8124,
                   "status": "production"},
}
def get_registry(root=None):
    return {k: dict(v) for k, v in _REG.items()}
def curated_default_target(model, topology, detected_sm=None):
    return {"synthetic-1": "vllm/syn-max", "synthetic-2": "vllm/other"}[model]
def model_default_pin_key(model):
    return "CLUB3090_DEFAULT_" + "".join(c if c.isalnum() else "_" for c in model).upper()
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

emit() { # emit ARGS… — run against the fixture with no ambient pins
  env -u CLUB3090_DEFAULT_SYNTHETIC_1 -u CLUB3090_DEFAULT_SYNTHETIC_2 \
    LITELLM_EMIT_REGISTRY_JSON="$FIX/facts.json" LITELLM_CONFIG="$FIX/config.yaml" \
    LITELLM_LOCAL_CONFIG="$FIX/config.local.yaml" "$@"
}
route_base() { # FILE NAME → api_base of that route
  awk -v want="$2" '
    /- model_name:/ { cur=$0; sub(/^.*- model_name: /, "", cur); sub(/[[:space:]]*#.*$/, "", cur) }
    cur == want && /api_base:/ { print $2; exit }' "$1"
}

# Tracked config generated once, then frozen as the baseline --local must not touch.
emit bash "$EMIT" "$FIX" >/dev/null || fail "tracked generation errored on the fixture"
cp "$FIX/config.yaml" "$FIX/tracked.baseline"

# --- 4: no pin → local block == tracked block -----------------------------------
emit bash "$EMIT" --local "$FIX" >/dev/null || fail "--local errored with no pin"
diff <(sed -n '/BEGIN GENERATED/,/END GENERATED/p' "$FIX/config.yaml") \
     <(sed -n '/BEGIN GENERATED/,/END GENERATED/p' "$FIX/config.local.yaml") >/dev/null \
  || fail "no pin: local generated block differs from the tracked one"

# --- 1: same-model pin → pinned port + status annotation ------------------------
emit env CLUB3090_DEFAULT_SYNTHETIC_1=vllm/syn-fast bash "$EMIT" --local "$FIX" >/dev/null \
  || fail "--local errored with a valid pin"
[ "$(route_base "$FIX/config.local.yaml" synthetic-1)" = "http://host.docker.internal:8125/v1" ] \
  || fail "pin not honoured: synthetic-1 should route to :8125, got '$(route_base "$FIX/config.local.yaml" synthetic-1)'"
grep -qF 'model_name: synthetic-1  # status: experimental' "$FIX/config.local.yaml" \
  || fail "pinned experimental scene lacks the '# status: experimental' annotation"
[ "$(route_base "$FIX/config.local.yaml" synthetic-2)" = "http://host.docker.internal:8124/v1" ] \
  || fail "a pin for synthetic-1 must not move synthetic-2"

# --- 5: tracked config untouched; non-generated content carried over -------------
cmp -s "$FIX/config.yaml" "$FIX/tracked.baseline" || fail "--local modified the TRACKED config"
grep -qF 'os.environ/CLOUD_KEY' "$FIX/config.local.yaml" || fail "--local dropped the cloud block"
[ "$(route_base "$FIX/config.yaml" synthetic-1)" = "http://host.docker.internal:8123/v1" ] \
  || fail "tracked config must keep the curated scene (:8123)"

# --- 6: --local --check --------------------------------------------------------
emit env CLUB3090_DEFAULT_SYNTHETIC_1=vllm/syn-fast bash "$EMIT" --local --check "$FIX" >/dev/null \
  || fail "--local --check failed on a freshly generated local config"
if emit bash "$EMIT" --local --check "$FIX" >/dev/null 2>&1; then
  fail "--local --check passed although the pin changed (local config is stale)"
fi

# --- 2: pin read from ROOT/.env; shell env wins -----------------------------------
printf 'CLUB3090_DEFAULT_SYNTHETIC_1="vllm/syn-fast"\n' > "$FIX/.env"
emit bash "$EMIT" --local "$FIX" >/dev/null || fail "--local errored reading the .env pin"
[ "$(route_base "$FIX/config.local.yaml" synthetic-1)" = "http://host.docker.internal:8125/v1" ] \
  || fail ".env pin not honoured"
emit env CLUB3090_DEFAULT_SYNTHETIC_1=vllm/syn-max bash "$EMIT" --local "$FIX" >/dev/null
[ "$(route_base "$FIX/config.local.yaml" synthetic-1)" = "http://host.docker.internal:8123/v1" ] \
  || fail "shell env pin must win over .env"
rm -f "$FIX/.env"

# --- 3: invalid pins fall back with a notice ---------------------------------------
for bad in vllm/does-not-exist vllm/other vllm/syn-noport; do
  emit env CLUB3090_DEFAULT_SYNTHETIC_1="$bad" bash "$EMIT" --local "$FIX" \
    >/dev/null 2>"$FIX/err" || fail "invalid pin '$bad' crashed --local"
  [ "$(route_base "$FIX/config.local.yaml" synthetic-1)" = "http://host.docker.internal:8123/v1" ] \
    || fail "invalid pin '$bad' should fall back to the curated scene (:8123)"
  grep -qF "$bad" "$FIX/err" || fail "invalid pin '$bad' fell back silently (no stderr notice)"
done

# --- 7: real repo wiring -----------------------------------------------------------
git check-ignore -q services/litellm/config.local.yaml \
  || fail "services/litellm/config.local.yaml is not gitignored"
command grep -qF -- '${LITELLM_CONFIG:-./config.yaml}:/app/config.yaml' services/litellm/docker-compose.yml \
  || fail "LiteLLM compose must mount \${LITELLM_CONFIG:-./config.yaml}"

# Real registry, pinned to a non-gateway scene, written to a temp path.
real_out="$FIX/real.local.yaml"
env CLUB3090_DEFAULT_QWEN3_8_27B=vllm/qwen38-27b-dual-fast LITELLM_LOCAL_CONFIG="$real_out" \
  bash "$EMIT" --local >/dev/null || fail "--local errored on the real registry"
fast_port="$(python3 -c 'from scripts.lib.profiles.compose_registry import get_registry; print(get_registry()["vllm/qwen38-27b-dual-fast"]["default_port"])')"
[ "$(route_base "$real_out" qwen3.8-27b)" = "http://host.docker.internal:${fast_port}/v1" ] \
  || fail "real registry: qwen3.8-27b should follow the dual-fast pin to :${fast_port}"
git diff --quiet -- services/litellm/config.yaml || fail "real --local run modified the tracked config"

echo "OK: litellm-emit --local honours the .env scene pin (same model, any status),"
echo "    falls back loudly on invalid pins, never touches the tracked config,"
echo "    --local --check works, and the compose/gitignore wiring is in place."
