#!/usr/bin/env bash
# Offline test for scripts/stream-toolcall-probe.py. Stands up a mock SSE server
# that streams either a CLEAN tool-call or the #145 DROP signature, and asserts
# the probe classifies + exit-codes each correctly. No GPU / real model needed.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT=8771
SRV_PID=""
tmplog="$(mktemp)"
cleanup() {
  [[ -n "$SRV_PID" ]] && kill "$SRV_PID" 2>/dev/null || true
  rm -f "$tmplog"
}
trap cleanup EXIT

# Mock OpenAI streaming endpoint. MOCK_MODE=clean → proper delta.tool_calls +
# finish_reason:tool_calls. MOCK_MODE=drop → tool_call XML in delta.content +
# finish_reason:stop (the club-3090#145 streaming-drop signature).
start_server() {
  MOCK_MODE="$1" python3 - "$PORT" <<'PY' &
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
MODE = os.environ.get("MOCK_MODE", "clean")
def sse(obj): return f"data: {json.dumps(obj)}\n\n".encode()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        # /v1/models health
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps({"data":[{"id":"mock"}]}).encode())
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if MODE == "clean":
            frames = [
                {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \"/etc/hosts\"}"}}]},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]},
            ]
        else:  # drop — XML leaks into content, no tool_calls, finish stop
            frames = [
                {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{"content":"<tool_call>\n{\"name\": \"read_file\", "},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{"content":"\"arguments\": {\"path\": \"/etc/hosts\"}}\n</tool_call>"},"finish_reason":None}]},
                {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]},
            ]
        for f in frames:
            self.wfile.write(sse(f)); self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
  SRV_PID=$!
  for _ in $(seq 1 50); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 0.1
  done
  echo "mock server didn't come up" >&2; exit 1
}

run_probe() {
  # tiny sweep: required only, 1 repeat — the mock answers all prompts identically
  python3 scripts/stream-toolcall-probe.py --url "http://127.0.0.1:${PORT}" \
    --model mock --thinking off --tool-choice required --repeat 1 --timeout 10 "$@"
}

# --- CLEAN: every request should PASS, exit 0 ---
start_server clean
set +e; run_probe >"$tmplog" 2>&1; rc=$?; set -e
kill "$SRV_PID" 2>/dev/null || true; SRV_PID=""
if [[ $rc -ne 0 ]]; then echo "FAIL: clean mode exited $rc (expected 0)"; cat "$tmplog"; exit 1; fi
grep -q "DROP 0" "$tmplog" || { echo "FAIL: clean mode reported a DROP"; cat "$tmplog"; exit 1; }
echo "clean mode: probe PASSed, 0 drops, exit 0 ✓"

# --- DROP: every request should be flagged DROP, exit 1 ---
start_server drop
set +e; run_probe >"$tmplog" 2>&1; rc=$?; set -e
kill "$SRV_PID" 2>/dev/null || true; SRV_PID=""
if [[ $rc -ne 1 ]]; then echo "FAIL: drop mode exited $rc (expected 1)"; cat "$tmplog"; exit 1; fi
grep -q "STREAMING TOOL-CALL DROPS" "$tmplog" || { echo "FAIL: drop mode didn't flag the #145 signature"; cat "$tmplog"; exit 1; }
echo "drop mode: probe flagged DROP, exit 1 ✓"

echo "test-stream-toolcall-probe: ok"
