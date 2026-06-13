#!/usr/bin/env bash
# #255: bench-agentic.sh must NOT abort the context ramp when a turn fails to
# emit a parseable tool call — it should synthesize one, inject the fixture
# tool_result, count the miss, and reach the configured TURNS. This test stands
# up a mock SSE endpoint and asserts both the miss path (ramp continues) and the
# success path (normal tool-call flow still works), fully offline (no GPU).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

tmp="$(mktemp -d)"
port_file="$tmp/port"
cleanup() { [[ -n "${server_pid:-}" ]] && kill "$server_pid" 2>/dev/null || true; rm -rf "$tmp"; }
trap cleanup EXIT

# --- mock OpenAI-compatible SSE endpoint -----------------------------------
# MOCK_EMIT_TOOLCALL=0 → stream content only (no tool_calls) = the parse-miss.
# MOCK_EMIT_TOOLCALL=1 → stream a proper tool_call = the success path.
cat > "$tmp/mock.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

EMIT_TC = os.environ.get("MOCK_EMIT_TOOLCALL", "0") == "1"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            body = json.dumps({"data": [{"id": "mock"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def ev(d): self.wfile.write(f"data: {json.dumps(d)}\n".encode()); self.wfile.flush()
        ev({"choices": [{"delta": {"content": "Looking into it."}}]})
        if EMIT_TC:
            ev({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_mock_0", "function": {"name": "Read", "arguments": "{}"}}]}}]})
        ev({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 100, "completion_tokens": 5}})
        self.wfile.write(b"data: [DONE]\n"); self.wfile.flush()

srv = HTTPServer(("127.0.0.1", 0), H)
with open(sys.argv[1], "w") as f: f.write(str(srv.server_address[1]))
srv.serve_forever()
PY

run_bench() {  # $1 = MOCK_EMIT_TOOLCALL
  rm -f "$port_file"
  MOCK_EMIT_TOOLCALL="$1" python3 "$tmp/mock.py" "$port_file" & server_pid=$!
  for _ in $(seq 1 50); do [[ -s "$port_file" ]] && break; sleep 0.1; done
  local port; port="$(cat "$port_file")"
  PREFLIGHT_NO_AUTODETECT=1 URL="http://127.0.0.1:${port}" MODEL=mock \
    SESSIONS=1 TURNS=3 QUIET=1 bash scripts/bench-agentic.sh 2>&1
  kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true
}

assert_contains() { [[ "$1" == *"$2"* ]] || { echo "ASSERT FAIL: missing '$2'"; echo "$1"; exit 1; }; }
assert_absent()   { [[ "$1" != *"$2"* ]] || { echo "ASSERT FAIL: unexpected '$2'"; echo "$1"; exit 1; }; }

echo "── miss path: no parseable tool call → ramp must continue to turn 3 ──"
out="$(run_bench 0)"
assert_absent  "$out" "FAIL"                       # ramp did NOT abort
assert_contains "$out" "tool-call misses: 3/3"     # all 3 turns missed, counted
# turn 3 reached (the summary table prints the turn-3 row)
echo "$out" | grep -qE "^\s*3\s" || { echo "ASSERT FAIL: turn 3 not reached"; echo "$out"; exit 1; }
echo "  ✓ ramp reached configured depth despite 100% tool-call misses"

echo "── success path: proper tool call → normal flow, zero misses ──"
out="$(run_bench 1)"
assert_absent  "$out" "FAIL"
assert_absent  "$out" "tool-call misses"           # no misses line when all succeed
echo "$out" | grep -qE "^\s*3\s" || { echo "ASSERT FAIL: turn 3 not reached (success path)"; echo "$out"; exit 1; }
echo "  ✓ success path intact, no false misses"

echo "test-bench-agentic-ramp: ok"
