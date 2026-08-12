#!/usr/bin/env bash
set -euo pipefail

# test-thinking-control.sh — the reasoning-switch detection contract.
#
# WHICH request field turns a model's reasoning off is not universal, and an
# unrecognised one is silently ignored — no error, no warning. The whole script
# layer had the Qwen key (`chat_template_kwargs.enable_thinking`) baked in, so
# on a model that uses a different one the switch did nothing, the model reasoned
# at full effort, and every short-budget check spent its entire allowance on the
# reasoning preamble before emitting a content token. verify-full's [3/9] and
# [5/9] then failed reporting "Model may be loading badly or wrong chat
# template" — sending you to debug a template that was in fact correct.
# Found on Inkling-Small 2026-08-12, whose template takes an effort DIAL
# (`reasoning_effort`: none/minimal/low/medium/high/xhigh, default 0.9).
#
# Why this test exists rather than a live model run: any single served model
# exercises exactly ONE branch of the detector. Inkling covers the template-scan
# reasoning_effort path and nothing else — in particular the behavioural-probe
# path, which is what EVERY vLLM/SGLang model takes (they ship no /props), would
# never execute. This drives all of them against a mock endpoint, deterministic
# and GPU-free.
#
# Asserts, per scenario: the detected control, the chat_template_kwargs object,
# and the top-level OpenAI `reasoning_effort` fragment.
#
#   template scan (llama.cpp /props)
#     1. template mentions enable_thinking      → enable_thinking
#     2. template mentions reasoning_effort     → reasoning_effort + std param
#     3. template mentions BOTH                 → enable_thinking  ← no-regression
#     4. template mentions neither              → none, empty kwargs
#   behavioural probe (vLLM / SGLang — no /props)
#     5. enable_thinking yields content         → enable_thinking
#     6. only reasoning_effort yields content   → reasoning_effort + std param
#     7. neither yields content                 → none
#   safety
#     8. endpoint unreachable                   → enable_thinking (today's shape)
#     9. env override wins AND suppresses the top-level param
#
# Scenario 3 is the load-bearing one: it is what guarantees this change can only
# ADD coverage. Every model passing today keeps its exact request shape.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUTF8="${PYTHONUTF8:-1}"

PASS=0
fail() { echo "ASSERTION FAILED: $*" >&2; exit 1; }
ok()   { PASS=$((PASS + 1)); printf '  ok  %s\n' "$*"; }

command -v python3 >/dev/null 2>&1 || { echo "python3 unavailable — skipping"; exit 0; }
command -v curl    >/dev/null 2>&1 || { echo "curl unavailable — skipping";    exit 0; }

TMP="$(mktemp -d)"
MOCK_PID=""
cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

# --- mock endpoint --------------------------------------------------------
# MODE selects the served behaviour:
#   props:<text>   → /props returns that chat_template; completions always 200
#   noprops:<key>  → /props 404s; completions return content ONLY when the
#                    request's chat_template_kwargs carries <key> ("-" = never)
cat > "$TMP/mock.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = os.environ["MOCK_MODE"]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/props":
            if MODE.startswith("props:"):
                return self._send(200, {"chat_template": MODE.split(":", 1)[1]})
            return self._send(404, {"error": "not found"})
        if self.path.endswith("/v1/models"):
            return self._send(200, {"data": [{"id": "mock-model"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        content = "OK"
        if MODE.startswith("noprops:"):
            want = MODE.split(":", 1)[1]
            kw = body.get("chat_template_kwargs") or {}
            # Empty content = the model reasoned the budget away, which is
            # exactly what an ignored switch looks like from the client side.
            content = "OK" if (want != "-" and want in kw) else ""
        self._send(200, {"choices": [{"message": {"content": content}}]})


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

free_port() { python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()"; }

start_mock() {
  [[ -n "$MOCK_PID" ]] && { kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; }
  PORT="$(free_port)"
  MOCK_MODE="$1" python3 "$TMP/mock.py" "$PORT" >/dev/null 2>&1 &
  MOCK_PID=$!
  for _ in $(seq 1 50); do
    curl -sf -m 1 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 0.1
  done
  fail "mock endpoint did not come up on port ${PORT}"
}

# Run detection in a clean subshell so no scenario leaks state into the next.
# Echoes: <control>|<kwargs>|<top-level fragment>
detect() {
  local url="$1"
  env -u THINK_CONTROL -u THINK_OFF_KW -u THINK_ON_KW -u THINK_OFF_STD -u THINK_ON_STD \
      -u VERIFY_THINK_OFF -u VERIFY_THINK_ON "${@:2}" \
      bash -c '
        set -euo pipefail
        source scripts/preflight.sh >/dev/null 2>&1 || true
        declare -F preflight_detect_thinking_control >/dev/null \
          || { echo "MISSING||"; exit 0; }
        URL="'"$url"'" MODEL="mock-model"
        preflight_detect_thinking_control "'"$url"'" "mock-model" 2>/dev/null
        printf "%s|%s|%s\n" "${THINK_CONTROL:-}" "${THINK_OFF_KW:-}" "${THINK_OFF_STD:-}"
      '
}

expect() {
  local label="$1" url="$2" want="$3"; shift 3
  local got
  got="$(detect "$url" "$@" | tail -1)"
  [[ "$got" == "$want" ]] || fail "${label}: expected '${want}', got '${got}'"
  ok "$label"
}

KW_QWEN='{"enable_thinking": false}'
KW_EFFORT='{"reasoning_effort": "none"}'
STD_EFFORT='"reasoning_effort": "none", '

echo "--- template scan (llama.cpp /props)"
start_mock 'props:{%- if enable_thinking %}<think>{%- endif %}'
expect "1. template declares enable_thinking"  "http://127.0.0.1:${PORT}" "enable_thinking|${KW_QWEN}|"

start_mock 'props:Thinking effort level: {{ reasoning_effort }}'
expect "2. template declares reasoning_effort" "http://127.0.0.1:${PORT}" "reasoning_effort|${KW_EFFORT}|${STD_EFFORT}"

start_mock 'props:{{ reasoning_effort }} and {%- if enable_thinking %}'
expect "3. BOTH → keeps today's shape"         "http://127.0.0.1:${PORT}" "enable_thinking|${KW_QWEN}|"

start_mock 'props:{{ messages }} plain template, no switch at all'
expect "4. template declares neither"          "http://127.0.0.1:${PORT}" "none|{}|"

echo "--- behavioural probe (vLLM / SGLang — no /props; opt-in via THINK_PROBE=1)"
start_mock 'noprops:enable_thinking'
expect "5. probe finds enable_thinking"        "http://127.0.0.1:${PORT}" "enable_thinking|${KW_QWEN}|" THINK_PROBE=1

start_mock 'noprops:reasoning_effort'
expect "6. probe finds reasoning_effort"       "http://127.0.0.1:${PORT}" "reasoning_effort|${KW_EFFORT}|${STD_EFFORT}" THINK_PROBE=1

start_mock 'noprops:-'
expect "7. probe finds no working switch"      "http://127.0.0.1:${PORT}" "none|{}|" THINK_PROBE=1

start_mock 'noprops:reasoning_effort'
expect "7b. probe NOT run without opt-in → safe default" "http://127.0.0.1:${PORT}" "enable_thinking|${KW_QWEN}|"

echo "--- safety"
DEAD_PORT="$(free_port)"
expect "8. unreachable endpoint → safe default" "http://127.0.0.1:${DEAD_PORT}" "enable_thinking|${KW_QWEN}|"

start_mock 'props:Thinking effort level: {{ reasoning_effort }}'
expect "9. env override wins + suppresses std param" "http://127.0.0.1:${PORT}" \
  'reasoning_effort (off overridden)|{"enable_thinking": false}|' \
  VERIFY_THINK_OFF='{"enable_thinking": false}'

echo ""
echo "test-thinking-control: ${PASS} assertions passed"
