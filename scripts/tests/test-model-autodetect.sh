#!/usr/bin/env bash
# test-model-autodetect — MODEL resolution must never invent a name (#1330).
#
# THE BUG THIS LOCKS OUT. `preflight_autodetect_model` no-op'd SILENTLY on an
# unreachable endpoint and callers fell through to `MODEL="${MODEL:-qwen3.6-27b}"`.
# Against a server that was merely still LOADING and serves something else, every
# request 404'd — and the output was indistinguishable from the config under test
# being broken. verify-full reported rc=8, 8/8 failed, on a compose that passed
# 10/10 once the engine had finished loading.
#
# ⚠️ The interesting assertions here are the ones that must STILL WORK. A fix that
# refuses too eagerly breaks the BYO case (llama.cpp ignores the request's model
# field entirely, #371), so every refusal case is paired with a keep case.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=scripts/preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

FAIL=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null' EXIT

# A stand-in /v1/models. Real enough to exercise the parse, cheap enough to run
# four times — and, unlike a mocked curl, it proves the actual HTTP path.
start_server() { # start_server <json-body>
  python3 - "$1" > "$TMP/port" 2>/dev/null <<'PY' &
import sys, json, http.server, socketserver, threading
body = sys.argv[1].encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
with socketserver.TCPServer(("127.0.0.1", 0), H) as s:
    print(s.server_address[1], flush=True)
    s.serve_forever()
PY
  SRV_PID=$!
  local i
  for i in $(seq 1 50); do [[ -s "$TMP/port" ]] && break; sleep 0.1; done
  SRV_PORT="$(cat "$TMP/port" 2>/dev/null)"
  [[ -n "$SRV_PORT" ]] || { bad "mock server never reported a port"; return 1; }
  for i in $(seq 1 50); do curl -sf -m 1 "http://127.0.0.1:${SRV_PORT}/v1/models" >/dev/null 2>&1 && return 0; sleep 0.1; done
  bad "mock server never answered"; return 1
}
stop_server() { [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null; SRV_PID=""; : > "$TMP/port"; }

# --- 1. the happy path still resolves ----------------------------------------
# POSITIVE CONTROL for everything below: if this stops working, the refusals are
# meaningless because nothing ever resolves.
if start_server '{"data":[{"id":"some-served-name"}]}'; then
  ( unset MODEL PREFLIGHT_ENDPOINT_AUTODETECTED
    URL="http://127.0.0.1:${SRV_PORT}"
    preflight_autodetect_model >/dev/null 2>&1
    preflight_resolve_model_or_fail "qwen3.6-27b" >/dev/null 2>&1
    [[ "$MODEL" == "some-served-name" ]] ) \
    && ok "reachable endpoint resolves the SERVED name" \
    || bad "reachable endpoint did not resolve the served name"

  # an explicit MODEL always wins — llama-swap / multi-model endpoints return the
  # first registered model, which is often the wrong one
  ( MODEL="pinned-by-user"; unset PREFLIGHT_ENDPOINT_AUTODETECTED
    URL="http://127.0.0.1:${SRV_PORT}"
    preflight_autodetect_model >/dev/null 2>&1
    preflight_resolve_model_or_fail "qwen3.6-27b" >/dev/null 2>&1
    [[ "$MODEL" == "pinned-by-user" ]] ) \
    && ok "an explicit MODEL= is never overridden" \
    || bad "explicit MODEL= was clobbered"
  stop_server
fi

# --- 2. unreachable endpoint REFUSES rather than guessing ---------------------
( unset MODEL PREFLIGHT_ENDPOINT_AUTODETECTED
  URL="http://127.0.0.1:9"; PREFLIGHT_MODEL_WAIT_S=0
  preflight_autodetect_model >/dev/null 2>&1
  preflight_resolve_model_or_fail "qwen3.6-27b" >/dev/null 2>&1 ) \
  && bad "unreachable endpoint fell back to the literal — this is the #1330 bug" \
  || ok "unreachable endpoint REFUSES (no invented model name)"

# the refusal has to say WHY, or it just moves the confusion
out="$( unset MODEL PREFLIGHT_ENDPOINT_AUTODETECTED
        URL="http://127.0.0.1:9"; PREFLIGHT_MODEL_WAIT_S=0
        preflight_autodetect_model 2>&1 >/dev/null
        preflight_resolve_model_or_fail "qwen3.6-27b" 2>&1 >/dev/null )"
grep -q "UNREACHABLE" <<<"$out" && grep -q "NOT a failure of whatever you are testing" <<<"$out" \
  && ok "the refusal names the cause and says it is not the subject's fault" \
  || bad "refusal text lost its diagnosis: ${out:0:160}"

# --- 3. reachable, but reports NO model ---------------------------------------
# The split that keeps this from over-refusing: we only know better when WE chose
# the endpoint.
if start_server '{"data":[]}'; then
  ( unset MODEL PREFLIGHT_ENDPOINT_AUTODETECTED
    URL="http://127.0.0.1:${SRV_PORT}"; PREFLIGHT_MODEL_WAIT_S=0
    preflight_autodetect_model >/dev/null 2>&1
    preflight_resolve_model_or_fail "qwen3.6-27b" >/dev/null 2>&1
    [[ "$MODEL" == "qwen3.6-27b" ]] ) \
    && ok "user-supplied URL with no model listed KEEPS the literal (BYO / llama.cpp, #371)" \
    || bad "user-supplied URL refused — that breaks the llama.cpp BYO case"

  ( unset MODEL
    PREFLIGHT_ENDPOINT_AUTODETECTED=1
    URL="http://127.0.0.1:${SRV_PORT}"; PREFLIGHT_MODEL_WAIT_S=0
    preflight_autodetect_model >/dev/null 2>&1
    preflight_resolve_model_or_fail "qwen3.6-27b" >/dev/null 2>&1 ) \
    && bad "autodetected container reporting no model still took the literal" \
    || ok "autodetected container reporting no model REFUSES (we know better)"
  stop_server
fi

# --- 4. no silent paths left --------------------------------------------------
# The original defect was not the fallback; it was that the component which
# failed printed NOTHING. Every no-op must now be audible.
for case in "no-url:" "unreachable:http://127.0.0.1:9"; do
  label="${case%%:*}"; u="${case#*:}"
  out="$( unset MODEL PREFLIGHT_ENDPOINT_AUTODETECTED
          URL="$u"; PREFLIGHT_MODEL_WAIT_S=0
          preflight_autodetect_model 2>&1 >/dev/null )"
  [[ -n "${out// /}" ]] && ok "autodetect is audible when it gives up (${label})" \
                        || bad "autodetect said NOTHING on the ${label} path — the #1330 silence is back"
done

# --- 5. every caller uses the guarded resolver --------------------------------
# A new caller copying the old one-liner reintroduces the bug in its own file.
for c in verify-full.sh verify-stress.sh verify.sh bench.sh; do
  # ⚠️ Anchored at COLUMN 0 on purpose. The guard itself keeps an indented
  # `MODEL="${MODEL:-…}"` in its else-branch, for the case where preflight.sh was
  # never sourced. What must not come back is the UNGUARDED top-level form.
  if command grep -qE '^MODEL="\$\{MODEL:-qwen3\.6-27b\}"' "scripts/$c"; then
    bad "scripts/${c} has an UNGUARDED top-level MODEL=\${MODEL:-…} fallback — the #1330 shape"
  fi
  command grep -q "preflight_resolve_model_or_fail" "scripts/$c" \
    && ok "scripts/${c} resolves through the guard" \
    || bad "scripts/${c} does not call preflight_resolve_model_or_fail"
done

# --- 6. the over-correction that actually shipped ------------------------------
# ⚠️ The first version of this fix refused under BENCH_MOCK=1 too, which killed
# every mocked bench run — and SILENTLY, because test-bench-capture drives
# `BENCH_MOCK=1 bash bench.sh` with stderr to /dev/null under `set -e`. The test
# died one line after its last ✓ with no error anywhere. Guard the exemption, or
# the next person re-tightens it and rediscovers that the hard way.
out="$( cd "$ROOT_DIR" && PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none BENCH_MOCK=1 \
        PREFLIGHT_MODEL_WAIT_S=0 timeout 60 bash scripts/bench.sh 2>/dev/null )"
rc=$?
if (( rc == 0 )) && [[ -n "$out" ]]; then
  ok "BENCH_MOCK=1 runs to completion (no endpoint expected, so no refusal)"
else
  bad "BENCH_MOCK=1 bench.sh exited ${rc} with $( [[ -z "$out" ]] && echo "NO output" || echo "output") — the refusal is over-tightened"
fi

(( FAIL )) && { echo "test-model-autodetect: FAIL" >&2; exit 1; }
echo "test-model-autodetect: ok"
