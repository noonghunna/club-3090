#!/usr/bin/env bash
# Guard: served-model resolution prefers the LOADED model and never trusts a
# non-model response — and verify-stress counts tokens when usage is null.
#
# #1360: TabbyAPI (exllamav3) lists every folder in its model directory on
# /v1/models, so taking data[0] returned a folder called `modules`; its
# /v1/model returns the model actually loaded. The same report showed
# verify-stress failing "reasoning-heavy returned only 0 tokens" because TabbyAPI
# returns `usage: null` unless stream_options.include_usage is set, and the old
# `d.get('usage', {})` read crashed on null and fell back to 0.
#
# Stub servers (one per case) pin:
#   1. TabbyAPI shape: /v1/model card wins over /v1/models[0];
#   2. vLLM shape: /v1/model 404 → /v1/models[0], unchanged behaviour;
#   3. a server answering /v1/model with HTML → falls back (never read as an id);
#   4. /v1/model JSON that is not a model card → falls back;
#   5. a URL passed with a trailing /v1/models still resolves;
#   6. preflight_autodetect_model end-to-end sets MODEL from the card;
#   7. verify-stress's completion_tokens_of: usage wins; null usage → labelled
#      estimate; null usage and no text → 0 (so a genuine early stop still fails).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
rc=0
ok()  { printf '  ok   %s\n' "$1"; }
bad() { printf '  FAIL %s\n' "$1"; rc=1; }

TMP="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done; rm -rf "$TMP"; }
trap cleanup EXIT

cat > "$TMP/stub.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
mode = sys.argv[1]
MODELS = {"object": "list", "data": [{"id": "modules"}, {"id": "qwen-loaded"}]}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/v1/models":
            return self.send(200, MODELS)
        if self.path == "/v1/model":
            if mode == "tabby":
                return self.send(200, {"id": "qwen-loaded", "object": "model", "owned_by": "tabbyAPI"})
            if mode == "html":
                return self.send(200, "<html><body>web ui</body></html>", "text/html")
            if mode == "notcard":
                return self.send(200, {"id": "not-a-model", "status": "ok"})
            return self.send(404, {"error": "not found"})
        self.send(404, {"error": "not found"})
srv = HTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
PY

start_stub() {  # start_stub <mode> → prints port
  local fifo="$TMP/port.$1"
  python3 "$TMP/stub.py" "$1" > "$fifo" 2>/dev/null &
  PIDS+=("$!")
  local i
  for i in $(seq 1 50); do [[ -s "$fifo" ]] && break; sleep 0.1; done
  head -1 "$fifo"
}

# shellcheck source=../lib/served-model.sh
source scripts/lib/served-model.sh

check() {  # check <label> <mode> <expected> [url-suffix]
  local port got
  port="$(start_stub "$2")"
  got="$(club_served_model_id "http://127.0.0.1:${port}${4:-}")"
  [[ "$got" == "$3" ]] && ok "$1 → '$got'" || bad "$1 → got '$got', expected '$3'"
}
check "TabbyAPI: /v1/model card wins over /v1/models[0]" tabby qwen-loaded
check "vLLM-shaped: /v1/model 404 falls back to /v1/models[0]" vllm modules
check "HTML on /v1/model is never read as an id" html modules
check "non-card JSON on /v1/model is never read as an id" notcard modules
check "URL with trailing /v1/models still resolves" tabby qwen-loaded /v1/models

# 6. end-to-end through preflight_autodetect_model
port="$(start_stub tabby)"
got="$(bash -c '
  cd "$1"; source scripts/preflight.sh >/dev/null 2>&1
  unset MODEL; URL="http://127.0.0.1:$2"; PREFLIGHT_MODEL_WAIT_S=0
  preflight_autodetect_model 2>/dev/null; printf "%s" "${MODEL:-}"' _ "$ROOT_DIR" "$port")"
[[ "$got" == "qwen-loaded" ]] && ok "preflight_autodetect_model sets MODEL from the card" \
  || bad "preflight_autodetect_model set MODEL='$got', expected 'qwen-loaded'"

# 7. verify-stress token counting
eval "$(sed -n '/^completion_tokens_of()/,/^}/p' scripts/verify-stress.sh)"
if ! declare -F completion_tokens_of >/dev/null; then
  bad "completion_tokens_of not found in verify-stress.sh"
else
  t1="$(printf '%s' '{"usage":{"completion_tokens":812},"choices":[{"message":{"content":"x"}}]}' | completion_tokens_of)"
  [[ "$t1" == "812 usage" ]] && ok "usage present → engine count ($t1)" || bad "usage present → '$t1'"
  long="$(printf 'a%.0s' $(seq 1 2400))"
  t2="$(printf '{"usage":null,"choices":[{"message":{"content":"%s","reasoning_content":"%s"}}]}' "$long" "$long" | completion_tokens_of)"
  [[ "$t2" == "1200 estimated" ]] && ok "usage null + text → labelled estimate ($t2)" || bad "usage null + text → '$t2'"
  t3="$(printf '%s' '{"usage":null,"choices":[{"message":{"content":""}}]}' | completion_tokens_of)"
  [[ "${t3%% *}" == "0" ]] && ok "usage null + no text → 0, so an early stop still fails ($t3)" || bad "usage null + no text → '$t3'"
fi

if [[ $rc -eq 0 ]]; then echo "PASS: served model resolves to the loaded model; null usage is counted"; else echo "FAIL: served-model / usage handling (see above)"; fi
exit $rc
