#!/usr/bin/env bash
# #255: bench-agentic.sh must NOT abort the context ramp when a turn fails to
# emit a parseable tool call — it should synthesize one, inject the fixture
# tool_result, count the miss, and reach the configured TURNS. This test stands
# up a mock SSE endpoint and asserts both the miss path (ramp continues) and the
# success path (normal tool-call flow still works), fully offline (no GPU).
#
# #809 residual: bench-agentic carried the same `max(wall - ttft, 1e-6)` epsilon
# divide bench.sh shed in #854 — eleven turns reading 409,726 → 1,808,323 TPS.
# The canvas leg below reproduces that shape against the old code and asserts the
# fix; the additivity leg asserts an autoregressive ramp is untouched.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

tmp="$(mktemp -d)"
port_file="$tmp/port"
cleanup() { [[ -n "${server_pid:-}" ]] && kill "$server_pid" 2>/dev/null || true; rm -rf "$tmp"; }
trap cleanup EXIT

# --- mock OpenAI-compatible SSE endpoint -----------------------------------
# MOCK_EMIT_TOOLCALL=0 → stream content only (no tool_calls) = the parse-miss.
# MOCK_EMIT_TOOLCALL=1 → stream a proper tool_call = the success path.
# MOCK_MODE=ar        → chunks separated in time, so the decode window is a real
#                       fraction of wall (an autoregressive stream).
# MOCK_MODE=canvas    → EVERYTHING in one chunk, then [DONE]: TTFT == wall and
#                       the decode window is zero-width. This is the dLLM shape
#                       that made the old `max(wall - ttft, 1e-6)` print eleven
#                       turns at 409,726 → 1,808,323 TPS (#809).
cat > "$tmp/mock.py" <<'PY'
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

EMIT_TC = os.environ.get("MOCK_EMIT_TOOLCALL", "0") == "1"
MODE = os.environ.get("MOCK_MODE", "ar")

TC = [{"index": 0, "id": "call_mock_0",
       "function": {"name": "Read", "arguments": "{}"}}]
USAGE = {"prompt_tokens": 100, "completion_tokens": 5}

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
        if MODE == "canvas":
            # Denoise the canvas, THEN emit it. The prefill+denoise time has to
            # dominate or the shape is not the reported one: on a real dLLM TTFT
            # is seconds and the post-first-chunk read is microseconds, so the
            # window is ~0% of wall. Without this sleep the mock's sub-millisecond
            # wall makes the same microseconds look like 13% of it, and the guard
            # correctly declines to fire.
            time.sleep(0.2)
            # One canvas, one chunk: content + tool call + usage arrive together,
            # so the client's first-token timestamp IS its last-token timestamp.
            delta = {"content": "Looking into it."}
            if EMIT_TC:
                delta["tool_calls"] = TC
            ev({"choices": [{"delta": delta}], "usage": USAGE})
        elif MODE == "reasoning":
            # THINKING MODEL SHAPE. The server streams `reasoning` deltas for a
            # long time, THEN a short content tail. If TTFT is taken at the first
            # CONTENT delta, the whole reasoning phase is charged to prefill and
            # the decode window collapses below the degenerate threshold — the
            # turn reports "single-block emission, wall ~2 tok/s", which reads as
            # a decode COLLAPSE that never happened (2026-09-08; it mimics #1096
            # closely enough to be mistaken for it).
            for _ in range(6):
                ev({"choices": [{"delta": {"reasoning": "thinking... "}}]})
                time.sleep(0.05)
            ev({"choices": [{"delta": {"content": "Done."}}]})
            if EMIT_TC:
                ev({"choices": [{"delta": {"tool_calls": TC}}]})
            ev({"choices": [{"delta": {}}], "usage": USAGE})
        else:
            ev({"choices": [{"delta": {"content": "Looking into it."}}]})
            # A real decode window. Without this the mock's own burst-write makes
            # every turn look zero-width and the AR leg could not be tested at all.
            time.sleep(0.2)
            if EMIT_TC:
                ev({"choices": [{"delta": {"tool_calls": TC}}]})
            ev({"choices": [{"delta": {}}], "usage": USAGE})
        self.wfile.write(b"data: [DONE]\n"); self.wfile.flush()

srv = HTTPServer(("127.0.0.1", 0), H)
with open(sys.argv[1], "w") as f: f.write(str(srv.server_address[1]))
srv.serve_forever()
PY

run_bench() {  # $1 = MOCK_EMIT_TOOLCALL  $2 = MOCK_MODE (default ar)  [extra env ...]
  rm -f "$port_file"
  local tc="$1" mode="${2:-ar}"; shift 2 2>/dev/null || shift $#
  MOCK_EMIT_TOOLCALL="$tc" MOCK_MODE="$mode" python3 "$tmp/mock.py" "$port_file" & server_pid=$!
  for _ in $(seq 1 50); do [[ -s "$port_file" ]] && break; sleep 0.1; done
  local port; port="$(cat "$port_file")"
  env PREFLIGHT_NO_AUTODETECT=1 URL="http://127.0.0.1:${port}" MODEL=mock \
    SESSIONS=1 TURNS=3 QUIET=1 "$@" bash scripts/bench-agentic.sh 2>&1
  kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true
}

assert_contains() { [[ "$1" == *"$2"* ]] || { echo "ASSERT FAIL: missing '$2'"; echo "$1"; exit 1; }; }
assert_absent()   { [[ "$1" != *"$2"* ]] || { echo "ASSERT FAIL: unexpected '$2'"; echo "$1"; exit 1; }; }

echo "── miss path: no parseable tool call → ramp must continue to turn 3 ──"
out="$(run_bench 0 ar)"
assert_absent  "$out" "FAIL"                       # ramp did NOT abort
assert_contains "$out" "tool-call misses: 3/3"     # all 3 turns missed, counted
# turn 3 reached (the summary table prints the turn-3 row)
echo "$out" | grep -qE "^\s*3\s" || { echo "ASSERT FAIL: turn 3 not reached"; echo "$out"; exit 1; }
echo "  ✓ ramp reached configured depth despite 100% tool-call misses"

echo "── success path: proper tool call → normal flow, zero misses ──"
out="$(run_bench 1 ar)"
ar_out="$out"
assert_absent  "$out" "FAIL"
assert_absent  "$out" "tool-call misses"           # no misses line when all succeed
echo "$out" | grep -qE "^\s*3\s" || { echo "ASSERT FAIL: turn 3 not reached (success path)"; echo "$out"; exit 1; }
echo "  ✓ success path intact, no false misses"

# ===========================================================================
# #809 residual: bench-agentic carried bench.sh's `max(wall - ttft, 1e-6)`
# ===========================================================================
# bench.sh's copy was fixed in #854; this one was tracked as a sibling instance.
# The reported artifact: eleven turns reading 409,726 → 1,808,323 TPS, with only
# turn 12 plausible at 164.9. The realism bound below is the assertion that
# matters — NO decode figure anywhere in the output may exceed it.
REALISM_BOUND=100000
assert_no_absurd_decode() {   # $1=output $2=context
  # NB: the payload goes through a FILE, not a second stdin redirection —
  # `python3 - <<'PY' <<<"$1"` silently feeds the here-string to `-` as the
  # PROGRAM (last redirection wins) and the check never runs.
  printf '%s\n' "$1" > "$tmp/absurd.txt"
  python3 - "$2" "$REALISM_BOUND" "$tmp/absurd.txt" <<'PY' || { echo "$1"; exit 1; }
import re, sys
ctx, bound, path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
bad = []
for ln in open(path, encoding="utf-8"):
    # The decode column only. Prompt-token and VRAM figures are legitimately
    # large and are not throughput claims.
    m = re.match(r"\s*\d+\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s", ln)
    if m and float(m.group(1)) > bound:
        bad.append(ln.rstrip())
if bad:
    sys.exit(f"{ctx}: decode figure above the realism bound ({bound:.0f}):\n" + "\n".join(bad))
PY
}

echo "── canvas granularity: zero-width decode window must print n/a, never 8e5 ──"
out="$(run_bench 1 canvas QUIET=0)"
assert_no_absurd_decode "$out" "canvas ramp"
assert_contains "$out" "n/a"                       # the per-turn column
assert_contains "$out" "zero-width decode window"  # ...and WHY it is n/a
# ⛔ The label was "single-block emission" and the trigger was a
# window/wall RATIO. Both retired 2026-09-08: the ratio encoded answer
# length + context size, so a healthy short answer after a long prefill
# tripped it. The trigger is now "one chunk ⇒ nothing to time".
assert_contains "$out" "decode-window  unmeasurable"
assert_contains "$out" "⚠ CANVAS GRANULARITY"
assert_contains "$out" "HEADLINE number is wall TPS"
# The ramp itself must be unaffected — TTFT is measured, not derived, and the
# curve is what this producer exists for.
echo "$out" | grep -qE "^\s*3\s" || { echo "ASSERT FAIL: canvas ramp did not reach turn 3"; echo "$out"; exit 1; }
assert_contains "$out" "TTFT growth by accumulated context"
echo "  ✓ canvas: n/a per turn with a reason, exclusion stated, ramp + TTFT curve intact"

echo "── additivity: an AR run carries none of the canvas machinery ──"
for unwanted in "CANVAS GRANULARITY" "zero-width decode window" "decode-window  unmeasurable"; do
  assert_absent "$ar_out" "$unwanted"
done
assert_no_absurd_decode "$ar_out" "AR ramp"
echo "  ✓ AR output unchanged (the guard is a ratio — it cannot fire on a real decode window)"

echo "── DECODE_GRANULARITY=token suppresses the classification ──"
out="$(run_bench 1 canvas QUIET=0 DECODE_GRANULARITY=token)"
assert_absent   "$out" "⚠ CANVAS GRANULARITY"
assert_contains "$out" "n/a"                       # the per-turn guard is NOT overridable
assert_no_absurd_decode "$out" "canvas ramp, granularity forced to token"
echo "  ✓ the label is declarable; the zero-width guard is not overridable"

echo "── thinking model: reasoning deltas count as first token ──"
# A reasoning-parser model streams `reasoning` deltas long before the first
# `content` delta. Taking TTFT at the content delta charges the ENTIRE reasoning
# phase to prefill, and the turn reports as a zero-width decode window — i.e. a
# COLLAPSE that did not happen. Measured 2026-09-08 on a live thinking model at
# ~21K ctx: first reasoning delta 28.64s, first content delta 34.69s, window
# 1.25s = 3.5% of wall = flagged. 278 reasoning deltas had streamed normally.
out="$(run_bench 1 reasoning QUIET=0)"
assert_absent "$out" "zero-width decode window"
assert_absent "$out" "decode-window  unmeasurable"
# and it must still produce a real per-turn decode number, not n/a
if grep -qE "^\s*[0-9]+\s+[0-9,]+\s+[0-9]+\s+n/a" <<<"$out"; then
  echo "ASSERT FAIL: a thinking-model turn still reported n/a decode"; echo "$out"; exit 1
fi
assert_no_absurd_decode "$out" "reasoning ramp"
echo "  ✓ reasoning deltas start the decode window — a thinking model is measurable"

echo "test-bench-agentic-ramp: ok"
