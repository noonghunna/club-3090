#!/usr/bin/env bash
# test-default-url-gateway — guards the registry-derived default endpoint in the
# bench/verify/concurrency-probe scripts (no hand-maintained :8020/:8010 URL
# literals) and the openwebui → LiteLLM gateway backend collapse.
#
# Offline: no server, no docker. The only live dependency is
# scripts/lib/registry-lookup.sh over the checked-in compose_registry.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "FAIL: $1" >&2; exit 1; }

SCRIPTS=(
  "$ROOT_DIR/scripts/bench.sh"
  "$ROOT_DIR/scripts/verify.sh"
  "$ROOT_DIR/scripts/verify-full.sh"
  "$ROOT_DIR/scripts/verify-stress.sh"
  "$ROOT_DIR/scripts/concurrency-probe.sh"
  "$ROOT_DIR/scripts/bench-agentic.sh"
  "$ROOT_DIR/scripts/health.sh"
  "$ROOT_DIR/scripts/quality-test.sh"
  "$ROOT_DIR/scripts/rebench-full.sh"
)
COMPOSE="$ROOT_DIR/services/openwebui/docker-compose.yml"
# --- 1. syntax ---------------------------------------------------------------
for f in "${SCRIPTS[@]}"; do
  bash -n "$f" || fail "bash -n: syntax error in ${f#$ROOT_DIR/}"
done
echo "  ✓ script syntax"

# --- 2. every default-URL site derives from the registry DEFAULTS walk -------
for f in "${SCRIPTS[@]}"; do
  grep -F 'URL="${URL:-http://localhost:${_DEFAULT_ENDPOINT_PORT:-' "$f" >/dev/null \
    || fail "${f#$ROOT_DIR/}: default URL no longer derives from registry_lookup_default_port"
done
echo "  ✓ all nine scripts derive their default URL from the registry"

# --- 3. no un-derived localhost:<port> model-URL literal remains -------------
# Non-comment lines must not carry a hardcoded http://localhost:<port> URL,
# except usage-text examples showing an explicit override (URL=… / bash …).
for f in "${SCRIPTS[@]}"; do
  if grep -nE 'http://localhost:[0-9]+' "$f" \
      | grep -vE '^[0-9]+:[[:space:]]*#' \
      | grep -vE '^[0-9]+:[[:space:]]*(bash |URL=http)' \
      | grep -q .; then
    fail "${f#$ROOT_DIR/}: un-derived localhost:<port> literal outside a comment"
  fi
done
echo "  ✓ no localhost:<port> model-URL literal remains outside comments/examples"

# --- 4. the last-resort fallback tracks today's curated default --------------
# "qwen3.6-27b's curated default wins everywhere": the fallback literal baked
# into every script must equal what registry_lookup_default_port resolves NOW,
# so a registry change that moves the curated default fails loudly here instead
# of silently leaving stale fallbacks behind.
# shellcheck source=lib/registry-lookup.sh
source "$ROOT_DIR/scripts/lib/registry-lookup.sh"
REGISTRY_LOOKUP_ROOT="$ROOT_DIR"
live_port="$(registry_lookup_default_port qwen3.6-27b 2>/dev/null || true)"
[[ -n "$live_port" ]] || fail "registry_lookup_default_port qwen3.6-27b resolved empty"
for f in "${SCRIPTS[@]}"; do
  grep -F "_DEFAULT_ENDPOINT_PORT:-${live_port}" "$f" >/dev/null \
    || fail "${f#$ROOT_DIR/}: fallback port != live curated default ($live_port)"
done
echo "  ✓ fallback literal matches the live curated default (:${live_port})"

# --- 5. behavioral smoke: concurrency-probe boots offline with the derive ----
out="$(bash "$ROOT_DIR/scripts/concurrency-probe.sh" --sweep --dry --n 1,2 --ctx 1k,4k 2>&1)"
command grep -q "\[sweep\] live" <<<"$out" || fail "--sweep --dry broke after the registry-source addition"
[[ $(grep -cE '^[[:space:]]+[0-9]+K:' <<<"$out") -ge 1 ]] || fail "--sweep --dry printed no clipped grid"
echo "  ✓ concurrency-probe --sweep --dry still runs offline"

# --- 6. openwebui compose parses and points model backends at the gateway ----
python3 - "$COMPOSE" <<'PY'
import os, re, sys, yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)
assert doc and "services" in doc and "open-webui" in doc["services"], "compose does not parse / missing service"

env = doc["services"]["open-webui"]["environment"]
def get(key):
    for item in env:
        if item.startswith(key + "="):
            val = item.split("=", 1)[1]
            # Resolve ${VAR:-default} the way docker compose would (host env wins).
            m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*)\}", val)
            if m:
                return os.environ.get(m.group(1)) or m.group(2)
            return val
    raise AssertionError(f"{key} missing from compose environment")

urls = [u for u in get("OPENAI_API_BASE_URLS").split(";") if u]
keys = [k for k in get("OPENAI_API_KEYS").split(";") if k]
assert len(urls) == len(keys), f"*KEYS ({len(keys)}) must pair with *URLS ({len(urls)})"

gateway = "http://host.docker.internal:4000/v1"
assert gateway in urls, f"LiteLLM gateway {gateway} missing: {urls}"
assert "http://host.docker.internal:8090/v1" in urls, \
    "unrouted ai-studio director :8090 must stay a direct backend"

# Every backend is either the gateway or an explicitly kept direct one; no
# per-model host port may survive the collapse.
allowed_ports = {"4000", "8090"}
for u in urls:
    port = u.rsplit(":", 1)[1].split("/")[0]
    assert port in allowed_ports, f"per-model backend not collapsed into the gateway: {u}"
print("  ✓ openwebui compose parses; gateway + director backends are well-formed")
PY

# --- 7. the override env actually restores per-port behavior -----------------
out="$(
  OWUI_OPENAI_API_BASE_URLS="http://host.docker.internal:8010/v1;http://host.docker.internal:8030/v1" \
  OWUI_OPENAI_API_KEYS="sk-noauth;sk-noauth" \
  python3 - "$COMPOSE" <<'PY'
import os, re, sys, yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)
env = doc["services"]["open-webui"]["environment"]
def get(key):
    for item in env:
        if item.startswith(key + "="):
            val = item.split("=", 1)[1]
            m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*)\}", val)
            return (os.environ.get(m.group(1)) or m.group(2)) if m else val
    raise AssertionError(key)

urls = [u for u in get("OPENAI_API_BASE_URLS").split(";") if u]
assert urls == ["http://host.docker.internal:8010/v1", "http://host.docker.internal:8030/v1"], urls
print("override")
PY
)"
command grep -q "^override$" <<<"$out" || fail "OWUI_OPENAI_API_BASE_URLS override did not resolve"
echo "  ✓ OWUI_OPENAI_API_* override restores legacy per-port backends"

echo "test-default-url-gateway: ok"
