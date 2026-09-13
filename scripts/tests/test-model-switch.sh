#!/usr/bin/env bash
# test-model-switch — HTTP contract for tools/model-switch/server.py.
#
# Hermetic + offline: no real GPU switch. The service is pointed at a STUB
# switch script (SWITCH_SCRIPT) and a stub docker (DOCKER_BIN=/bin/true, so no
# model appears running), and we assert the HTTP/validation/auth/lock contract.
# Real end-to-end switching is hardware-bound and validated manually.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP="$(mktemp -d)"
STUB_LOG="$TMP/switch.log"
SRV_PID=""
# Only kill the server if we actually started it — never `kill 0` (which would
# signal the whole process group, e.g. a parent test-suite runner).
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# Stub switch.sh: record arg1 + the FORCE env (so we can assert force propagation) +
# the full args, sleep briefly (so the single-flight lock is observable), succeed.
cat > "$TMP/stub-switch.sh" <<EOF
#!/usr/bin/env bash
echo "arg1=\$1 FORCE=\${FORCE:-unset} all=\$*" >> "$STUB_LOG"
sleep 2
exit 0
EOF
chmod +x "$TMP/stub-switch.sh"

# Stub setup.sh (for /pull): record the invocation, exit 0 fast (no real download).
cat > "$TMP/stub-setup.sh" <<EOF
#!/usr/bin/env bash
echo "setup arg1=\$1 WEIGHT_KEY=\${WEIGHT_KEY:-} EXTRA=\${WEIGHT_EXTRA_KEYS:-} HF_HOME=\${HF_HOME:-}" >> "$TMP/setup.log"
exit 0
EOF
chmod +x "$TMP/stub-setup.sh"

# Hermetic MODEL_DIR: materialize fake weights so the do_switch weights pre-check passes and the
# STUB switch runs (else those switches would 409 weights_missing). Presence re-checks disk live,
# so slugs discovered later (e.g. the force-required one) can be materialized after server start.
MDIR="$TMP/models"; mkdir -p "$MDIR"
materialize() {
  MODEL_DIR="$MDIR" python3 - "$@" <<'PY'
import sys, os, importlib.util
spec = importlib.util.spec_from_file_location("ms", "tools/model-switch/server.py")
os.environ["MODEL_SWITCH_WATCHDOG"] = "0"
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for s in sys.argv[1:]:
    for a in m._slug_artifacts(s):
        if a["subdir"]:
            d = os.path.join(m.MODEL_DIR, a["subdir"]); os.makedirs(d, exist_ok=True)
            open(os.path.join(d, (a["verify_glob"].replace("*", "x") or "x")), "w").close()
PY
}
materialize vllm/dual vllm/gemma-31b-dual

PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
TOKEN="test-token-123"
BASE="http://127.0.0.1:$PORT"

# MODEL_SWITCH_GPU_COUNT=2 makes hardware eligibility deterministic; WATCHDOG=0 keeps
# the background healer out of the HTTP contract tests; STATE_DIR is a throwaway.
CLUB3090_API_TOKEN="$TOKEN" MODEL_SWITCH_PORT="$PORT" MODEL_SWITCH_BIND=127.0.0.1 \
  SWITCH_SCRIPT="$TMP/stub-switch.sh" SETUP_SCRIPT="$TMP/stub-setup.sh" \
  DOCKER_BIN=/bin/true CLUB3090_TOPOLOGY=dual MODEL_DIR="$MDIR" \
  MODEL_SWITCH_GPU_COUNT=2 MODEL_SWITCH_WATCHDOG=0 MODEL_SWITCH_STATE_DIR="$TMP/state" \
  VLLM_API_KEY="" \
  python3 tools/model-switch/server.py >"$TMP/server.out" 2>&1 &
SRV_PID=$!

# Wait for liveness.
for _ in $(seq 1 50); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/healthz" 2>/dev/null)" == "200" ]] && break
  sleep 0.1
done

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
auth=(-H "Authorization: Bearer $TOKEN")

assert_code() {
  local want="$1" got="$2" msg="$3"
  if [[ "$got" != "$want" ]]; then
    echo "FAIL: $msg (want HTTP $want, got $got)" >&2
    cat "$TMP/server.out" >&2
    exit 1
  fi
}
assert_contains() {
  if [[ "$1" != *"$2"* ]]; then
    echo "FAIL: expected to contain: $2" >&2; echo "--- got ---" >&2; echo "$1" >&2; exit 1
  fi
}

# 1. /healthz needs no auth.
assert_code 200 "$(code "$BASE/healthz")" "/healthz open"
# 2. auth required + enforced.
assert_code 401 "$(code "$BASE/status")" "/status without token -> 401"
assert_code 401 "$(code -H 'Authorization: Bearer wrong' "$BASE/status")" "/status wrong token -> 401"
assert_code 200 "$(code "${auth[@]}" "$BASE/status")" "/status with token -> 200"
# 2c. auth-first preserved: unknown routes require auth BEFORE revealing a 404.
assert_code 401 "$(code "$BASE/__nope__")" "unknown GET route without token -> 401 (auth-first)"
assert_code 404 "$(code "${auth[@]}" "$BASE/__nope__")" "unknown GET route with token -> 404"
assert_code 401 "$(code -XPOST "$BASE/__nope__")" "unknown POST route without token -> 401"
# 3. /models lists the registry.
assert_contains "$(curl -s "${auth[@]}" "$BASE/models")" '"vllm/dual"'
# 3a1. GET / self-discovery: OPEN (no auth), valid JSON manifest, lists routes, no handler leak.
assert_code 200 "$(code "$BASE/")" "GET / discovery open (no auth) -> 200"
disc="$(curl -s "$BASE/")"
echo "$disc" | python3 -c "
import sys,json
d=json.load(sys.stdin)
paths={(e['method'],e['path']) for e in d['endpoints']}
assert ('POST','/switch') in paths and ('POST','/heal') in paths, 'missing routes'
assert not any('handler' in e for e in d['endpoints']), 'handler leaked'
assert d['auth']['configured'] is True, 'auth.configured should be true (token set)'
assert d['auth']['token_env']==['CLUB3090_API_TOKEN','VLLM_API_KEY'], 'token_env'
print('discovery-manifest: ok')
" || { echo 'FAIL: GET / discovery manifest' >&2; echo "$disc" >&2; exit 1; }
# 3b. /heal with no target and no desired model yet (fresh state) -> 400.
assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/heal" -d '{}')" "/heal no-body, no desired -> 400"

# Discover representative slugs from the live registry so tests aren't brittle to
# specific slug names: a functional one, one that needs force by STATUS, and a
# GPU-ineligible one (multi>2 on our forced gpu_count=2).
allmodels="$(curl -s "${auth[@]}" "$BASE/models?all=1")"
pick() { echo "$allmodels" | python3 -c "import sys,json;print(next((r['slug'] for r in json.load(sys.stdin)['available'] if $1),''))"; }
FORCE_SLUG="$(pick "r['requires_force'] and r['gpu_eligible'] is not False")"   # non-functional status
INELIGIBLE_SLUG="$(pick "r['gpu_eligible'] is False")"                          # needs >2 GPUs
# Materialize the force-required slug's weights so 6b reaches the stub switch (not weights_missing).
[[ -n "$FORCE_SLUG" ]] && materialize "$FORCE_SLUG"
# 4. unknown slug -> 400 (registry validation, no switch attempted).
assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '{"slug":"__bogus__"}')" "bad slug -> 400"
# 4b. valid JSON, wrong shape -> 400 (not a 500 / dropped connection).
assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '{"slug":42}')" "non-string slug -> 400"
assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '[]')" "non-object body -> 400"
# 5. valid slug -> 200, stub invoked with that slug.
assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '{"slug":"vllm/dual"}')" "slug switch -> 200"
assert_contains "$(cat "$STUB_LOG")" "vllm/dual"
# 6. model id -> resolves to its curated default slug (gemma-4-31b -> vllm/gemma-31b-dual).
assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '{"model":"gemma-4-31b"}')" "model switch -> 200"
assert_contains "$(cat "$STUB_LOG")" "vllm/gemma-31b-dual"
# functional slug switched without force -> stub saw FORCE=unset.
assert_contains "$(grep 'arg1=vllm/dual ' "$STUB_LOG" | head -1)" "FORCE=unset"

# 6b. force consent (real FORCE env plumbing through switch.sh).
if [[ -n "$FORCE_SLUG" ]]; then
  assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d "{\"slug\":\"$FORCE_SLUG\"}")" "force-required slug w/o force -> 400"
  assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d "{\"slug\":\"$FORCE_SLUG\",\"force\":true}")" "force-required slug w/ force -> 200"
  assert_contains "$(grep "arg1=$FORCE_SLUG " "$STUB_LOG" | tail -1)" "FORCE=1"
fi
# 6c. hardware filter: a GPU-ineligible slug is hidden by default, shown under ?all=1.
if [[ -n "$INELIGIBLE_SLUG" ]]; then
  assert_contains "$allmodels" "\"$INELIGIBLE_SLUG\""
  if curl -s "${auth[@]}" "$BASE/models" | grep -q "\"$INELIGIBLE_SLUG\""; then
    echo "FAIL: GPU-ineligible slug $INELIGIBLE_SLUG should be hidden from default /models" >&2; exit 1
  fi
  # ...and switching to it without force is refused (400).
  assert_code 400 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d "{\"slug\":\"$INELIGIBLE_SLUG\"}")" "ineligible slug w/o force -> 400"
fi
# 6d. /heal with an explicit slug -> 200; /down -> 200 + stub called with --down.
assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/heal" -d '{"slug":"vllm/dual"}')" "/heal slug -> 200"
assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/down")" "/down -> 200"
assert_contains "$(cat "$STUB_LOG")" "arg1=--down"

# 6e. weights_missing: a functional slug whose weights aren't materialized -> /switch 409.
MISSING="$(echo "$allmodels" | python3 -c "import sys,json;print(next((r['slug'] for r in json.load(sys.stdin)['available'] if not r['requires_force'] and r['slug'] not in ('vllm/dual','vllm/gemma-31b-dual')),''))")"
if [[ -n "$MISSING" ]]; then
  assert_contains "$(curl -s "${auth[@]}" -XPOST "$BASE/switch" -d "{\"slug\":\"$MISSING\"}")" '"weights_missing"'
  assert_code 409 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d "{\"slug\":\"$MISSING\"}")" "missing weights -> /switch 409"
  # /pull the missing slug: 202 (started) or 507 (disk) — both valid; if 202, stub setup runs.
  pc="$(code "${auth[@]}" -XPOST "$BASE/pull" -d "{\"slug\":\"$MISSING\"}")"
  [[ "$pc" == "202" || "$pc" == "507" ]] || { echo "FAIL: /pull missing -> $pc (want 202 or 507)" >&2; exit 1; }
  if [[ "$pc" == "202" ]]; then
    for _ in $(seq 1 20); do
      ds="$(curl -s "${auth[@]}" "$BASE/status" | python3 -c "import sys,json;print(json.load(sys.stdin)['download']['state'])")"
      [[ "$ds" != "downloading" ]] && break; sleep 0.3
    done
    assert_contains "$(cat "$TMP/setup.log")" "WEIGHT_KEY="
  fi
fi
# 6f. /status exposes model_dir + download; /pull an already-present slug -> 200 ready.
assert_contains "$(curl -s "${auth[@]}" "$BASE/status")" "\"model_dir\": \"$MDIR\""
assert_code 200 "$(code "${auth[@]}" -XPOST "$BASE/pull" -d '{"slug":"vllm/dual"}')" "/pull present -> 200 ready"
# 7. single-flight: a 2nd switch while the (sleeping) stub holds the lock -> 409.
curl -s -o /dev/null "${auth[@]}" -XPOST "$BASE/switch" -d '{"slug":"vllm/dual"}' &
BG_PID=$!
sleep 0.7
assert_code 409 "$(code "${auth[@]}" -XPOST "$BASE/switch" -d '{"slug":"vllm/dual"}')" "concurrent switch -> 409"
wait "$BG_PID" 2>/dev/null || true   # only the in-flight switch, not the server

# 8. security guard: refuse to start unauthenticated on a non-loopback bind.
PORT2="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
set +e
MODEL_SWITCH_BIND=0.0.0.0 MODEL_SWITCH_PORT="$PORT2" CLUB3090_API_TOKEN="" VLLM_API_KEY="" \
  SWITCH_SCRIPT="$TMP/stub-switch.sh" DOCKER_BIN=/bin/true \
  timeout 5 python3 tools/model-switch/server.py >/dev/null 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]; then
  echo "FAIL: unauthenticated non-loopback bind should be refused (rc=$rc)" >&2; exit 1
fi

# 9. Deterministic state-machine unit tests (watchdog, rollback, crash-loop budget,
#    persistence, force consent) — driven directly against server.py's functions.
echo "--- state-machine unit tests ---"
python3 "$ROOT_DIR/scripts/tests/test-model-switch-unit.py" || exit 1

echo "test-model-switch: ok"
