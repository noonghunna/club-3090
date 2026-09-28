#!/usr/bin/env bash
# Guard: scripts/compaction-probe.sh — the ONE thing we ship that measures what
# happens AFTER a context compaction at depth (#1052's reported shape).
#
# Pins the properties that make it worth trusting, all offline against a mock:
#   1. the 0.90 default is NOT 0.95 — verify-stress measured this model class
#      fillable to 91% of n_ctx with 669 MB free, under its own 1024 MB margin,
#      and compaction peaks ABOVE steady state because the new sequence prefills
#      while the old blocks are still held. A 0.95 default measures the ceiling.
#   2. asking for >0.92 WARNS rather than silently measuring an OOM;
#   3. a missing /metrics endpoint says so rather than silently falling back to
#      client timing — a silent fallback there hands back exactly the class of
#      number that #1096 turned out to be;
#   4. it refuses cleanly when nothing is serving.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null' EXIT

# 1. the default is 0.90, stated in the wrapper and honoured by the core
command grep -q 'FILL_FRAC="${FILL_FRAC:-0.90}"' scripts/compaction-probe.sh \
  && echo "  ok   wrapper defaults FILL_FRAC to 0.90" \
  || { echo "  FAIL wrapper's FILL_FRAC default is not 0.90"; rc=1; }
command grep -q 'default=0.90' scripts/lib/compaction_probe.py \
  && echo "  ok   core defaults fill-frac to 0.90" \
  || { echo "  FAIL core's fill-frac default is not 0.90"; rc=1; }
if command grep -qE 'FILL_FRAC:-0\.9[5-9]|default=0\.9[5-9]' scripts/compaction-probe.sh scripts/lib/compaction_probe.py; then
  echo "  FAIL a 0.95+ default crept in — that measures the ceiling, not compaction"; rc=1
fi

# 2. mock endpoint: /v1/models + /metrics + a chat completion
cat > "$TMP/mock.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
WITH_METRICS = os.environ.get("MOCK_METRICS", "1") == "1"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(json.dumps({"data": [{"id": "mock", "max_model_len": 4096}]}))
        elif self.path.startswith("/metrics"):
            if not WITH_METRICS: self.send_response(404); self.end_headers(); return
            self._send('vllm:request_generation_tokens_sum{m="mock"} 100.0\n'
                       'vllm:request_decode_time_seconds_sum{m="mock"} 1.0\n'
                       'vllm:num_preemptions_total{m="mock"} 0.0\n', "text/plain")
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send(json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "usage": {"prompt_tokens": 9999, "completion_tokens": 8}}))
srv = HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(srv.server_address[1]))
srv.serve_forever()
PY

start_mock() {  # $1 = MOCK_METRICS
  MOCK_METRICS="$1" python3 "$TMP/mock.py" "$TMP/port" & MOCK_PID=$!
  for _ in $(seq 1 50); do [[ -s "$TMP/port" ]] && break; sleep 0.1; done
  PORT="$(cat "$TMP/port")"
}

# 3. >0.92 must WARN
start_mock 1
out="$(URL="http://127.0.0.1:${PORT}" MODEL=mock FILL_FRAC=0.95 REPS=1 \
       timeout 60 bash scripts/compaction-probe.sh 2>&1 || true)"
kill "$MOCK_PID" 2>/dev/null; MOCK_PID=""
command grep -qi "above the ~0.92" <<<"$out" \
  && echo "  ok   fill_frac>0.92 warns that it measures the ceiling" \
  || { echo "  FAIL no warning at fill_frac=0.95"; rc=1; }

# 4. NEGATIVE CONTROL — the default must NOT warn, or check 3 proves nothing
rm -f "$TMP/port"; start_mock 1
out="$(URL="http://127.0.0.1:${PORT}" MODEL=mock REPS=1 \
       timeout 60 bash scripts/compaction-probe.sh 2>&1 || true)"
kill "$MOCK_PID" 2>/dev/null; MOCK_PID=""
if command grep -qi "above the ~0.92" <<<"$out"; then
  echo "  FAIL the 0.90 default warned — the check fires unconditionally"; rc=1
else
  echo "  ok   the default does not warn (negative control)"
fi

# 5. missing /metrics must SAY so, not silently use client timing
rm -f "$TMP/port"; start_mock 0
out="$(URL="http://127.0.0.1:${PORT}" MODEL=mock REPS=1 \
       timeout 60 bash scripts/compaction-probe.sh 2>&1 || true)"
kill "$MOCK_PID" 2>/dev/null; MOCK_PID=""
command grep -qi "metrics unavailable" <<<"$out" \
  && echo "  ok   a missing /metrics is announced, not silently swallowed" \
  || { echo "  FAIL silent fallback to client timing"; rc=1; }

# 6. nothing serving -> clean refusal, not a traceback
out="$(URL="http://127.0.0.1:9" timeout 30 bash scripts/compaction-probe.sh 2>&1 || true)"
command grep -q "no model at" <<<"$out" && ! command grep -q "Traceback" <<<"$out" \
  && echo "  ok   refuses cleanly when nothing is serving" \
  || { echo "  FAIL did not refuse cleanly with no endpoint"; rc=1; }

[[ "$rc" == "0" ]] && echo "PASS: compaction-probe defaults, warnings and refusals hold" \
                   || echo "FAIL: compaction-probe regression"
exit "$rc"
