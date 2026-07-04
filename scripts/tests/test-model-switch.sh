#!/usr/bin/env bash
# test-model-switch — HTTP contract for tools/model-switch/server.py.
#
# Hermetic + offline: no real GPU switch. The service is pointed at a STUB
# switch script (SWITCH_SCRIPT) and a stub docker (DOCKER_BIN=/bin/true, so no
# model appears running), and we assert the HTTP/validation/auth/lock contract.
# Real end-to-end switching is hardware-bound and validated manually.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP="$(mktemp -d)"
STUB_LOG="$TMP/switch.log"
SRV_PID=""
# Only kill the server if we actually started it — never `kill 0` (which would
# signal the whole process group, e.g. a parent test-suite runner).
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# Stub switch.sh: record the slug, sleep briefly (so the single-flight lock is
# observable), succeed.
cat > "$TMP/stub-switch.sh" <<EOF
#!/usr/bin/env bash
echo "\$@" >> "$STUB_LOG"
sleep 2
exit 0
EOF
chmod +x "$TMP/stub-switch.sh"

PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
TOKEN="test-token-123"
BASE="http://127.0.0.1:$PORT"

CLUB3090_API_TOKEN="$TOKEN" MODEL_SWITCH_PORT="$PORT" MODEL_SWITCH_BIND=127.0.0.1 \
  SWITCH_SCRIPT="$TMP/stub-switch.sh" DOCKER_BIN=/bin/true CLUB3090_TOPOLOGY=dual \
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
# 3. /models lists the registry.
assert_contains "$(curl -s "${auth[@]}" "$BASE/models")" '"vllm/dual"'
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

echo "test-model-switch: ok"
