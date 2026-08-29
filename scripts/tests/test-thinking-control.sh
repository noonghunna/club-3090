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
#   props:<text>        → /props returns that chat_template; completions always 200
#   props-ignore:<text> → /props returns that chat_template, but completions ALWAYS
#                         return empty content: the template NAMES a switch the model
#                         does not honour (GLM-5.3-Flash forced thinking, 2026-08-29)
#   props-lowonly:<text> → /props returns that chat_template; completions return
#                         content ONLY when chat_template_kwargs carries
#                         reasoning_effort="low". Models the GLM-5.3 family, where
#                         `none` is NOT a valid level and is silently ignored while
#                         `low` works — the case that made a probe testing only
#                         `none` declare the model switch-less.
#   props-xhigh:<text>  → /props returns that chat_template; content appears at
#                         reasoning_effort="low" (so the OFF ladder resolves), but
#                         REASONING appears ONLY at "xhigh" — `high` yields none.
#                         Models Qwen3.8-27B, whose template FORCES xhigh and remaps
#                         high -> xhigh. A probe hardcoding `high` reads this model
#                         as "reasoning suspiciously short".
#   noprops:<key>       → /props 404s; completions return content ONLY when the
#                         request's chat_template_kwargs carries <key> ("-" = never)
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
            if MODE.startswith("props:") or MODE.startswith("props-ignore:") or MODE.startswith("props-lowonly:") or MODE.startswith("props-xhigh:"):
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
        if MODE.startswith("props-xhigh:"):
            kw = body.get("chat_template_kwargs") or {}
            eff = kw.get("reasoning_effort")
            # OFF ladder resolves at `low`; the ON ladder must climb past `high`.
            content = "OK" if eff in ("minimal", "low", "medium") else ""
            # >=50 chars: the ON ladder applies verify-full [7]'s own bar, so a
            # short fixture string would be rejected as "suspiciously short" and the
            # ladder would climb PAST xhigh. The fixture must clear the gate it tests.
            reasoning = ("step one, then step two, then step three, and finally a "
                         "conclusion that is comfortably over fifty characters"
                         ) if eff == "xhigh" else ""
            return self._send(200, {"choices": [{"message": {
                "content": content, "reasoning_content": reasoning}}]})
        if MODE.startswith("props-lowonly:"):
            kw = body.get("chat_template_kwargs") or {}
            content = "OK" if kw.get("reasoning_effort") == "low" else ""
        if MODE.startswith("props-ignore:"):
            # Switch named in the template, ignored by the model: the client sees
            # the reasoning preamble eat the whole budget, so content is empty.
            content = ""
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

# The ON ladder is invisible to detect()/expect(), which only emit the OFF triple.
# Scenario 16 was silently asserting nothing about the ON side until this existed.
detect_on() {
  local url="$1"
  env -u THINK_CONTROL -u THINK_OFF_KW -u THINK_ON_KW -u THINK_OFF_STD -u THINK_ON_STD \
      -u VERIFY_THINK_OFF -u VERIFY_THINK_ON "${@:2}" \
      bash -c '
        set -euo pipefail
        source scripts/preflight.sh >/dev/null 2>&1 || true
        declare -F preflight_detect_thinking_control >/dev/null \
          || { echo "MISSING"; exit 0; }
        URL="'"$url"'" MODEL="mock-model"
        preflight_detect_thinking_control "'"$url"'" "mock-model" 2>/dev/null
        printf "%s\n" "${THINK_ON_KW:-}"
      '
}

expect_on() {
  local label="$1" url="$2" want="$3"; shift 3
  local got
  got="$(detect_on "$url" "$@" | tail -1)"
  [[ "$got" == "$want" ]] || fail "${label}: expected ON '${want}', got '${got}'"
  ok "$label"
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

echo "--- scan branch: switch NAMED but not HONOURED (GLM-5.3-Flash, 2026-08-29)"
# The template names a key; the model ignores it and reasons the budget away.
# Without verification the scan trusts the name, TOK_SCALE stays 1, and a
# forced-thinking model gets a 30-token budget -> verify-full 4/9 with hints
# blaming the model. Downgrading to `none` is what widens the budget.
start_mock 'props-ignore:Thinking effort level: {{ reasoning_effort }}'
expect "10. named reasoning_effort IGNORED -> none" "http://127.0.0.1:${PORT}" "none|{}|" THINK_PROBE=1

start_mock 'props-ignore:{%- if enable_thinking %}<think>{%- endif %}'
expect "11. named enable_thinking IGNORED -> none"  "http://127.0.0.1:${PORT}" "none|{}|" THINK_PROBE=1

# NEGATIVE CONTROL: identical mock, no opt-in. Measurement scripts (bench, soak,
# quality) must fire ZERO extra requests and keep the scan's answer, so this MUST
# still report reasoning_effort. If this ever flips to `none`, the verification
# leaked out of its opt-in and is putting uncontrolled load on measured runs.
start_mock 'props-ignore:Thinking effort level: {{ reasoning_effort }}'
expect "12. IGNORED but no opt-in -> scan answer kept" "http://127.0.0.1:${PORT}" "reasoning_effort|${KW_EFFORT}|${STD_EFFORT}"

# ⭐ The GLM-5.3 case: template NAMES reasoning_effort, `none` is silently ignored,
# `low` works. The probe must try `low` before giving up — otherwise it downgrades to
# `none`, THINK_OFF_KW becomes {}, no switch is sent on ANY check, and a
# forced-thinking model burns its whole budget reasoning (verify-full [8]).
start_mock 'props-lowonly:Thinking effort level: {{ reasoning_effort }}'
expect "14. effort dial honours ONLY low -> keeps reasoning_effort, off=low" "http://127.0.0.1:${PORT}" \
  'reasoning_effort|{"reasoning_effort": "low"}|"reasoning_effort": "low", ' THINK_PROBE=1

# NEGATIVE CONTROL for 14: same mock, no opt-in. Without the probe the scan answer
# stands and the off-value stays the DEFAULT `none` — proving the `low` path is
# reached only via the behavioural probe, never guessed.
start_mock 'props-lowonly:Thinking effort level: {{ reasoning_effort }}'
expect "15. low-only mock, no opt-in -> default off-value none" "http://127.0.0.1:${PORT}" \
  "reasoning_effort|${KW_EFFORT}|${STD_EFFORT}"

# ⭐ The Qwen3.8-27B case: OFF resolves at `low`, but REASONING only appears at
# `xhigh` — its template forces xhigh and remaps high -> xhigh. A probe hardcoding
# `high` on the ON side reads this as "reasoning suspiciously short" (verify [7]) and
# blames the model. The ON ladder must climb high -> xhigh.
start_mock 'props-xhigh:Thinking effort level: {{ reasoning_effort }}'
# OFF resolves at `minimal` — the ladder takes the FIRST working level, and minimal
# is correctly ordered ahead of low (lower effort is closer to off).
expect "16a. xhigh model: OFF ladder stops at the first working level" "http://127.0.0.1:${PORT}" \
  'reasoning_effort|{"reasoning_effort": "minimal"}|"reasoning_effort": "minimal", ' THINK_PROBE=1
start_mock 'props-xhigh:Thinking effort level: {{ reasoning_effort }}'
expect_on "16b. ⭐ ON ladder CLIMBS past high to xhigh" "http://127.0.0.1:${PORT}" \
  '{"reasoning_effort": "xhigh"}' THINK_PROBE=1
# Negative control: a model where `high` DOES reason must keep `high`, never climb.
start_mock 'props:Thinking effort level: {{ reasoning_effort }}'
expect_on "16c. ON ladder does NOT climb when high works" "http://127.0.0.1:${PORT}" \
  '{"reasoning_effort": "high"}' THINK_PROBE=1

# A working switch must be untouched by the new step (scenario 3's promise).
start_mock 'props:Thinking effort level: {{ reasoning_effort }}'
expect "13. HONOURED switch unchanged under opt-in" "http://127.0.0.1:${PORT}" "reasoning_effort|${KW_EFFORT}|${STD_EFFORT}" THINK_PROBE=1

echo "--- safety"
DEAD_PORT="$(free_port)"
expect "8. unreachable endpoint → safe default" "http://127.0.0.1:${DEAD_PORT}" "enable_thinking|${KW_QWEN}|"

start_mock 'props:Thinking effort level: {{ reasoning_effort }}'
expect "9. env override wins + suppresses std param" "http://127.0.0.1:${PORT}" \
  'reasoning_effort (off overridden)|{"enable_thinking": false}|' \
  VERIFY_THINK_OFF='{"enable_thinking": false}'

echo ""
echo "test-thinking-control: ${PASS} assertions passed"
