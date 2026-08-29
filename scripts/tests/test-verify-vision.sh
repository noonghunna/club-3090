#!/usr/bin/env bash
# Guard for verify-full's [10/10] vision check.
#
# Exists because a gate that only ever passes is worthless: the vision check was
# added to catch a projector that is configured but not serving, so the cases
# that MUST be exercised are the failing ones, not the happy path.
# Drives the real script against a mock endpoint — no GPU, no model.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; PORT="${PORT:-8791}"; MOCK_PID=""
FAILED=0
cleanup(){ [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

cat > "$TMP/mock.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
MODE = os.environ["MOCK_MODE"]
BODIES = {
    "ok":      "- Red circle\n- Blue square\n- Green triangle\n- Number: 47",
    "partial": "- Red circle\n- Blue square",
    "synonym": "red ellipse, blue rectangle, green triangle, 47",
}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/v1/models"): return self._send(200, {"data": [{"id": "mock-model"}]})
        self._send(404, {"error": "nope"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8", "replace")
        has_image = "image_url" in raw
        if has_image and MODE == "novision":
            return self._send(400, {"error": {"message": "model does not support image input"}})
        if has_image:
            return self._send(200, {"choices": [{"message": {"content": BODIES[MODE]},
                                    "finish_reason": "stop"}], "usage": {"completion_tokens": 20}})
        return self._send(200, {"choices": [{"message": {"content": "Paris"},
                                "finish_reason": "stop"}], "usage": {"completion_tokens": 5}})
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

start_mock(){
  [[ -n "$MOCK_PID" ]] && { kill "$MOCK_PID" 2>/dev/null; wait "$MOCK_PID" 2>/dev/null; }
  MOCK_MODE="$1" python3 "$TMP/mock.py" "$PORT" >/dev/null 2>&1 &
  MOCK_PID=$!
  for _ in $(seq 1 50); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" == "200" ]] && return 0
    sleep 0.1; done
  echo "  mock failed to start"; return 1
}

expect(){ # $1=mode $2=regex that MUST appear on the vision line $3=label
  start_mock "$1" || { FAILED=$((FAILED+1)); return; }
  local out
  out="$(URL="http://127.0.0.1:$PORT" MODEL=mock-model CONTAINER=none \
        timeout 180 bash "$ROOT/scripts/verify-full.sh" 2>&1 | grep -aA1 '\[10/10\]')"
  if printf '%s' "$out" | grep -qaE "$2"; then
    echo "  ✅ $3"
  else
    echo "  ❌ $3 — expected /$2/, got:"; printf '%s\n' "$out" | sed 's/^/       /'
    FAILED=$((FAILED+1))
  fi
}

echo "test-verify-vision"
# structural: the check is wired and the asset ships
[[ -f "$ROOT/scripts/assets/vision-test.png" ]] && echo "  ✅ ground-truth asset present" \
  || { echo "  ❌ scripts/assets/vision-test.png missing"; FAILED=$((FAILED+1)); }
grep -qa 'run_check "vision" check_vision' "$ROOT/scripts/verify-full.sh" && echo "  ✅ check wired into driver" \
  || { echo "  ❌ run_check \"vision\" not wired"; FAILED=$((FAILED+1)); }

# behavioural: the FAILING cases must actually fail
expect ok       'vision 4/4'                    "4/4 ground truth -> PASS"
expect synonym  'vision 4/4'                    "synonyms (ellipse/rectangle) -> PASS"
expect partial  'vision 2/4.*missed'            "partial 2/4 -> FAIL (corruption signature)"
expect novision 'not multimodal|0/4|text-only'  "non-multimodal endpoint -> SKIP not fail"

echo ""
if [[ "$FAILED" == "0" ]]; then echo "test-verify-vision: PASS"; exit 0
else echo "test-verify-vision: FAIL ($FAILED)"; exit 1; fi
