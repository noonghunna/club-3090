#!/usr/bin/env bash
#
# Guard for scripts/deep-context-probe.sh (club-3090#1259).
#
# The probe exists because every other instrument stops at ~35K accumulated
# context. Its value is entirely in being RUNNABLE when someone needs it at 140K,
# so this checks the things that would make it fail at that moment: a missing
# MODEL that 404s and reads as a dead server, a bad MODE, a dead endpoint, and
# the output contracts a reader depends on.
#
# The contracts are checked against a FAKE ENGINE (stdlib http.server, below)
# that reproduces the shapes measured on SGLang v0.5.19 / read from vLLM v0.29.0
# source on 2026-09-13 — in particular the false-clean this probe was caught
# producing: `prompt_tokens_details` absent (a server flag is off) being read as
# cached=0 while the engine was reusing 4,160 tokens. No GPU server required.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${ROOT}/scripts/deep-context-probe.sh"
P="${ROOT}/scripts/lib/deep_context_probe.py"
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

[[ -x "$S" ]] || bad "wrapper executable" "executable scripts/deep-context-probe.sh" "missing or not +x"
[[ -f "$P" ]] || bad "implementation present" "scripts/lib/deep_context_probe.py" "missing"
bash -n "$S" 2>/dev/null || bad "wrapper syntax" "clean bash -n" "syntax error"
python3 -c "import ast,sys; ast.parse(open('$P').read())" 2>/dev/null || bad "probe syntax" "parseable python" "syntax error"

# --- refusal: MODEL is required. A wrong/absent model 404s, which reads as a
# dead server — the exact confusion this must not create.
out="$(URL=http://127.0.0.1:9 MODEL= bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 2 ]] || bad "missing MODEL exits 2" "2" "$rc"
command grep -qi '404' <<<"$out" || bad "missing-MODEL message explains the 404 risk" "a mention of 404" "absent"
ok "refuses without MODEL, and says why it matters"

# --- refusal: bad MODE
out="$(URL=http://127.0.0.1:9 MODEL=x MODE=sideways bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 2 ]] || bad "bad MODE exits 2" "2" "$rc"
ok "refuses an unknown MODE"

# --- refusal: dead endpoint, and it must point at the ESTATE_PORT trap (#1293)
out="$(URL=http://127.0.0.1:9 MODEL=x bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 1 ]] || bad "dead endpoint exits 1" "1" "$rc"
command grep -qi 'docker port' <<<"$out" || bad "dead-endpoint hint names the port trap" "a 'docker port' hint" "absent"
ok "refuses a dead endpoint and names the #1293 port trap"

# --- static contract: an unmeasurable decode window must not render as 0.0
command grep -q "'n/a'" "$P" || bad "unmeasurable decode renders n/a" "an 'n/a' branch (#1267)" "absent"
command grep -q 'def ok(win, tok)' "$P" || bad "decode window/token floor present" "a window+token guard" "absent"
ok "unmeasurable decode reports n/a, not a 0.0 that reads as silent-empty (#1267)"

# --- static contract: TTFT must be taken on the first chunk carrying choices,
# not the first chunk with CONTENT — otherwise a reasoning model's reasoning
# phase is charged to prefill (the #1096 retraction).
command grep -q 'if ttft is None: ttft = time.time() - t0' "$P" \
  || bad "TTFT anchored on first choices chunk" "ttft set when ch is truthy" "not found"
ok "TTFT anchored on the first chunk carrying choices (#1096 trap avoided)"

# ===========================================================================
# Fake engine. REPORT mode selects how prefix reuse is (not) exposed:
#   usage    prompt_tokens_details = {"cached_tokens": N} only when N > 0 (SGLang
#            --enable-cache-report shape; vLLM --enable-prompt-tokens-details is
#            the same but sends 0 explicitly)
#   metrics  field absent; /metrics publishes the SGLang prefix-hit counter
#   none     field absent; /metrics is 404 (every shipped sglang compose today)
# Every content chunk carries cumulative usage.completion_tokens (continuous
# usage stats) and one chunk carries MANY tokens (1 -> 9 -> 16, as measured under
# DFlash), so a chunk-counting decode rate reads ~5x low.
# ===========================================================================
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/fake_engine.py" <<'PYEOF'
import json, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, REPORT = int(sys.argv[1]), sys.argv[2]
# Two REPORT values change the STREAM SHAPE rather than the cached-token source;
# both reproduce a decode n/a seen on a real 141K run. They report cached tokens
# exactly like "usage" so only the decode path is under test.
SHAPE = REPORT if REPORT in ("usage_once", "one_token", "one_chunk", "truncated") else "normal"
if SHAPE != "normal": REPORT = "usage"
SEEN = []; COUNTER = {"cache": 0}; LOCK = threading.Lock(); KV_TOTAL = 10000
def toks(s): return len(s) // 4 + 10
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, json.dumps({"data": [{"id": "fake"}]}), "application/json")
        if self.path == "/metrics" and REPORT == "metrics":
            with LOCK:
                res = min(KV_TOTAL, sum(toks(p) for p in SEEN)); n = min(33, len(SEEN))
                body = (f'sglang:realtime_tokens_total{{mode="prefill_cache",}} {COUNTER["cache"]}\n'
                        f'sglang:realtime_tokens_total{{mode="prefill_compute",}} 0\n'
                        f'sglang:realtime_tokens_created{{mode="prefill_cache",}} 1.7e9\n'
                        f'sglang:kv_used_tokens{{tp_rank="0"}} 0\nsglang:kv_evictable_tokens{{tp_rank="0"}} {res}\n'
                        f'sglang:kv_available_tokens{{tp_rank="0"}} {KV_TOTAL - res}\n'
                        f'sglang:mamba_used_tokens{{tp_rank="0"}} 0\nsglang:mamba_evictable_tokens{{tp_rank="0"}} {n}\n'
                        f'sglang:mamba_available_tokens{{tp_rank="0"}} {33 - n}\n')
            return self._send(200, body, "text/plain")
        self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
        prompt = "".join(m.get("content") or "" for m in req["messages"]); ptok = toks(prompt)
        with LOCK:
            cached = min(ptok, max([toks(p) for p in SEEN if prompt.startswith(p)] or [0]))
            COUNTER["cache"] += cached
            if prompt not in SEEN: SEEN.append(prompt)
        # nocache: reuse is observable and reports ZERO, and the re-query gets no
        # speed-up either — cached==0 with a cold-cost TTFT. That is the branch
        # that returned "CLEAN eviction" regardless of pressure (#1299).
        time.sleep(min(1.0, (ptok if REPORT == "nocache" else ptok - cached) * 0.0001))
        maxtok = int(req.get("max_tokens", 16))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def chunk(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
        chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], "usage": None})
        steps = [(1, "OK")] if maxtok <= 8 else [(1, "one"), (9, ", two, three, four, five"), (16, ", six, seven, eight,")]
        if SHAPE == "one_token" and maxtok > 8: steps = [(1, "one")]   # model stops early: no window at all
        # one_chunk: the whole reply arrives in ONE chunk with no per-chunk usage
        # (a real speculative-decoding shape - one DFlash chunk carried 8 tokens).
        # There is no content WINDOW at all, so only the wall basis can measure it.
        if SHAPE == "one_chunk" and maxtok > 8: steps = [(16, "one, two, three, four, five, six")]
        for cum, piece in steps:
            chunk({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                   # usage_once / one_chunk: content streams but per-chunk usage
                   # never arrives, so there is no usage DELTA to difference.
                   "usage": (None if SHAPE in ("usage_once", "one_chunk") else
                             {"prompt_tokens": ptok, "completion_tokens": cum, "total_tokens": ptok + cum})})
            time.sleep(0.15)
        ctok = steps[-1][0]
        # truncated: the reply hit the cap. The probe must SAY so — the fragment
        # lands in the history and biases every later turn (measured: 48 -> 4 tok).
        chunk({"choices": [{"index": 0, "delta": {},
                            "finish_reason": ("length" if SHAPE == "truncated" else "stop")}],
               "usage": None})
        chunk({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
               "prompt_tokens_details": ({"cached_tokens": 0} if REPORT == "nocache" else
                                         {"cached_tokens": cached} if (REPORT == "usage" and cached > 0) else None)}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
PYEOF

# run_fake REPORT MODE [env...] -> output in $out. Kills the server by PID (never pkill -f).
run_fake() {
  local report="$1" mode="$2"; shift 2
  local port; port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  python3 "$TMP/fake_engine.py" "$port" "$report" >"$TMP/fake.$report.$mode.log" 2>&1 &
  local pid=$!
  local i; for i in $(seq 1 50); do curl -sf -m 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break; sleep 0.1; done
  out="$(env URL="http://127.0.0.1:$port" MODEL=fake MODE="$mode" "$@" bash "$S" 2>&1)"; rc=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
# 3rd column of the depth/breadth table rows (turn/session lines start with a number)
col_cached() { awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9,]+$/ {print $3}' <<<"$1"; }
col_decode() { awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9,]+$/ {print $5}' <<<"$1"; }

# --- contract 3 (the false-clean): field ABSENT and no counter -> cached is n/a,
# NEVER 0, and the output names the server flags that would light it up.
run_fake none depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(none) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
vals="$(col_cached "$out")"
[[ -n "$vals" ]] || bad "depth table rendered (none)" "turn rows" "none: $out"
if command grep -qvx 'n/a' <<<"$vals"; then bad "absent field renders n/a" "every cached cell n/a" "$(tr '\n' ' ' <<<"$vals")"; fi
command grep -q 'reuse HAPPENED' <<<"$out" || bad "self-test detects unreported reuse" "'reuse HAPPENED ... NOT REPORT'" "absent: $out"
command grep -q -- '--enable-cache-report' <<<"$out" || bad "names the SGLang flag" "--enable-cache-report" "absent"
command grep -q -- '--enable-prompt-tokens-details' <<<"$out" || bad "names the vLLM flag" "--enable-prompt-tokens-details" "absent"
ok "absent prompt_tokens_details renders cached=n/a (never 0) and names both engine flags"

# --- contract 4: field present -> numeric, and it tracks the growing prefix
run_fake usage depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(usage) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'cached column source: usage' <<<"$out" || bad "self-test picks the usage field" "source: usage" "absent: $out"
last="$(col_cached "$out" | tail -1 | tr -d ,)"
[[ "$last" =~ ^[0-9]+$ && "$last" -gt 0 ]] || bad "cached is numeric and >0 once the prefix repeats" ">0" "'$last'"
first="$(col_cached "$out" | head -1)"
[[ "$first" == "0" ]] || bad "first turn (nothing cached, field omitted) reads 0 once the field is proven live" "0" "'$first'"
ok "present prompt_tokens_details renders numbers; omitted-when-zero reads 0 only after the self-test proved the field live"

# --- contract 5: field absent but the engine's prefix-hit counter exists -> the
# counter delta is used and the column is labelled with that source
run_fake metrics depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(metrics) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'cached column source: metrics' <<<"$out" || bad "self-test falls back to the /metrics counter" "source: metrics" "absent: $out"
last="$(col_cached "$out" | tail -1 | tr -d ,)"
[[ "$last" =~ ^[0-9]+$ && "$last" -gt 0 ]] || bad "counter-delta cached is numeric and >0" ">0" "'$last'"
command grep -qE '^ +[0-9]+ +[0-9,]+ +[0-9,n/a]+ +[0-9.]+ +[~0-9.n/a]+ +[0-9a-z/%.]+ +[0-9]+% +[0-9]+/33' <<<"$out" \
  || bad "pool residency columns come from /metrics (kv %, mamba slots/total)" "'NN% n/33' columns" "absent: $out"
ok "absent field + /metrics counter -> counter delta, labelled 'metrics'; pool residency from /metrics"

# --- contract 6: decode_tps counts TOKENS (usage.completion_tokens), not chunks.
# The fake streams 16 tokens in 3 chunks ~0.15s apart: tokens/window ~ 50,
# chunks/window ~ 7. Anything under 20 means chunks were counted.
d="$(col_decode "$out" | head -1)"
python3 -c "import sys; v=float(sys.argv[1].lstrip('~')); sys.exit(0 if v > 20 else 1)" "$d" 2>/dev/null \
  || bad "decode_tps derives from completion_tokens" ">20 tok/s for 16 tokens in 3 chunks" "'$d'"
[[ "$d" != ~* ]] || bad "per-chunk usage honoured (no '~' approx marker)" "exact rate" "'$d'"
ok "decode_tps counts tokens from continuous usage, not chunks (spec-dec safe)"

# --- contract 6b (the gap the 141K run hit): an engine that streams content but
# sends usage ONCE has no usage DELTA to difference. The old code stopped there
# and printed a bare n/a for every turn past 104K — the one number the report was
# commissioned to produce. It must now fall through to the content-chunk window,
# still count TOKENS from usage, mark the row '~', and name the basis it used.
run_fake usage_once depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(usage_once) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
d="$(col_decode "$out" | head -1)"
[[ "$d" != "n/a" ]] || bad "usage-once stream still yields a decode rate" "a number" "n/a: $out"
[[ "$d" == ~* ]] || bad "a fallback basis is marked approximate" "'~NN.N'" "'$d'"
python3 -c "import sys; v=float(sys.argv[1].lstrip('~')); sys.exit(0 if v > 20 else 1)" "$d" 2>/dev/null \
  || bad "fallback still counts TOKENS not chunks" ">20 tok/s for 16 tokens in 3 chunks" "'$d'"
command grep -q 'decode basis: usage-window' <<<"$out" || bad "the fallback basis is named on the row" "'decode basis: usage-window'" "absent: $out"
ok "usage sent ONCE still measures decode (content-chunk window), marked '~' and basis named"

# --- contract 6c: when decode is genuinely unmeasurable (one token, no window),
# n/a is correct — but it must carry the stream shape that caused it, so a reader
# can tell "the model stopped early" from "the probe went blind". An n/a with no
# reason is the #1267 ambiguity one level up.
run_fake one_token depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(one_token) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
d="$(col_decode "$out" | head -1)"
[[ "$d" == "n/a" ]] || bad "a single-token reply is not timeable" "n/a" "'$d'"
command grep -q 'decode n/a: 1 completion tok' <<<"$out" || bad "n/a names the token count" "'decode n/a: 1 completion tok'" "absent: $out"
command grep -q 'finish_reason=stop' <<<"$out" || bad "n/a names the finish_reason" "finish_reason=stop" "absent: $out"
ok "unmeasurable decode reports n/a WITH the stream shape that caused it"

# --- contract 6f: a reply cut off at the cap must be called out AT THE TURN IT
# HAPPENS. This is the probe's own worst defect, and it is silent: the fragment
# goes into the history as what the assistant said, the model imitates it, and
# replies decay (measured on a real 141K run: 48 -> 33 -> 21 -> 11 -> 7 -> 4,
# stuck at 4, decode n/a from 104K). Two same-boot controls proved the engine was
# never involved — single-turn held 48/48 to 190,670 tokens, and multi-turn with a
# fixed well-formed history showed no trend to 144,521.
run_fake truncated depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(truncated) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'TRUNCATED at the 48-token cap' <<<"$out" \
  || bad "a capped reply is flagged at its own turn" "'TRUNCATED at the 48-token cap'" "absent: $out"
command grep -q 'biases every later turn' <<<"$out" \
  || bad "the flag says WHY truncation matters" "the history-bias consequence" "absent: $out"
ok "a reply truncated at the cap is flagged as history poisoning, at the turn it happens"

# --- contract 6g: the depth ask must COMPLETE inside the cap. Asking for sixty
# numbers needs ~90 tokens and truncated every early turn, which is what started
# the decay. A static check because the failure only shows up 100K deep.
command grep -q 'count from one to twenty in words' "$P" \
  || bad "depth ask fits the cap" "an ask that completes inside DEPTH_MAX_TOK" "still asking for more than fits"
ok "the depth ask completes inside the token cap (no truncation by construction)"

# --- contract 6h: the depth history must be a FIXED assistant turn, never the
# model's own reply. Feeding replies back makes the conversation a feedback loop
# -- any downward drift is read as the house style and reinforced. Measured
# twice on a live server: 48 -> 4 tokens with an over-cap ask, and still 40 -> 2
# by 98K after that was fixed, both taking the decode column to n/a across the
# exact range this probe exists to measure. Controls on the same boot: single
# turn held 48/48 at 190,670 tokens, fixed-history multi-turn showed no trend to
# 144,521. Only the fed-back arm collapsed.
python3 - "$P" <<'PYEOF' || bad "depth history is a fixed turn" "append CANNED_REPLY, not m['text']" "the model's own reply is fed back"
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
body = src[src.index("def run_depth"):]
body = body[:body.index("\ndef ")] if "\ndef " in body else body
appends = re.findall(r'msgs\.append\(\{"role": "assistant".*', body)
sys.exit(0 if appends and all("CANNED_REPLY" in a for a in appends) else 1)
PYEOF
ok "the depth history is a fixed assistant turn, so replies cannot feed back on themselves"

# --- contract 6i: a drafter-acceptance gauge reading exactly 0 is UNPOPULATED,
# not collapsed, and must not raise the alarm. SGLang leaves spec_accept_rate at
# zero until a speculation step completes: on the first live run turns 1-2 read
# 0.00 and the probe cried "ACCEPTANCE LOW" at a drafter that turn 3 showed was
# healthy. That is the absence-vs-zero trap the whole probe exists to avoid, so
# it gets a gate rather than a comment. Static, because reproducing an
# unpopulated engine gauge against a fake is not worth the fidelity it buys.
python3 - "$P" <<'PYEOF' || bad "zero acceptance gauge treated as unpopulated" "an explicit v == 0 -> None guard" "0 would raise a false alarm"
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
body = src[src.index("def spec_view"):]
body = body[:body.index("\ndef ")] if "\ndef " in body else body
# the guard must return "no reading" for 0, BEFORE any threshold comparison
guard = re.search(r"if v is None or v == 0:\s*\n\s*return None, False", body)
thresh = body.index("v < 0.20") if "v < 0.20" in body else -1
sys.exit(0 if (guard and thresh > guard.end()) else 1)
PYEOF
ok "a zero acceptance gauge reads as unpopulated, not as a collapsed drafter"

# --- contract 6j: both engines must report acceptance on the SAME scale, or the
# cross-engine A/B is meaningless. vLLM gives accepted/drafted; SGLang must use
# spec_accept_rate (a fraction), NOT spec_accept_length (mean tokens per step),
# which would invite comparing 3.55 against 100%.
command grep -q 'sglang:spec_accept_rate' "$P" \
  || bad "SGLang acceptance uses the comparable fraction" "sglang:spec_accept_rate" "absent"
command grep -q 'sglang:spec_accept_length' "$P" \
  && bad "SGLang acceptance must not use accept_length" "spec_accept_rate only" "accept_length is still referenced"
ok "acceptance is accepted/drafted on both engines, so the A/B is on one scale"

# --- contract 6d: the LAST basis is reachable. One chunk carrying the whole
# reply is a real speculative-decoding shape (one DFlash chunk carried 8 tokens),
# and it leaves no content window at all — only the wall basis can measure it.
run_fake one_chunk depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(one_chunk) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
d="$(col_decode "$out" | head -1)"
[[ "$d" != "n/a" ]] || bad "a single-chunk reply still measures" "a number" "n/a: $out"
command grep -q 'decode basis: wall' <<<"$out" || bad "the wall basis is named on the row" "'decode basis: wall'" "absent: $out"
ok "a whole reply in one chunk still measures, via the wall basis, and says so"

# --- contract 6e: the four bases stay ordered by fidelity. This is a STATIC
# check on purpose. The floor-fallthrough it protects cannot be exercised
# end-to-end — an engine's final usage chunk lands after the content and widens
# the usage window past the floor by itself, so a fake that tried to force the
# fallthrough would pass against the OLD code too and prove nothing.
python3 - "$P" <<'PYEOF' || bad "decode bases ordered by fidelity" "usage-delta, usage-window, chunk-count, wall in order" "reordered or missing"
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
want = ["usage-delta", "usage-window", "chunk-count", "wall"]
got = [m for m in re.findall(r'basis = "([a-z-]+)"', src)]
sys.exit(0 if got == want else 1)
PYEOF
ok "the four decode bases are chained in descending fidelity (chunk-count below both token-counted bases)"

# --- contract 7: a breadth run UNDER the nominal pool must not claim there is no
# eviction pressure, and must not let any verdict name eviction as the cause.
#
# Nominal pool is not the reusable radix budget: measured on SGLang, a
# 30,895-token prefix that reused perfectly (30,848 tok, 0.23s vs a 24.62s cold
# cost) was gone after five unrelated ~28K sessions — ~66% of
# max_total_num_tokens. The old wording said "no KV eviction pressure ... the
# verdicts below are NOT an eviction test" and so told a reader to discard a
# result that was right, while verdict() separately printed "CLEAN eviction" on
# the very same rows. Both halves are #1299.
run_fake usage breadth SESSIONS=2 SESSION_CTX=600 KV_POOL=100000
[[ $rc -eq 0 ]] || bad "breadth run exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -qi 'no KV eviction pressure' <<<"$out" \
  && bad "sub-pool run must not claim there is no eviction pressure" "no such claim" "the old wording is back"
command grep -q 'does NOT mean there is no' <<<"$out" \
  || bad "sub-pool run states that nominal pool does not rule out eviction" "the ~66% caveat" "absent: $out"
command grep -qc 'HEALTHY reuse' <<<"$out" || bad "breadth re-query verdicts rendered" "HEALTHY reuse rows" "absent: $out"
ok "a sub-pool breadth run does not claim 'no eviction pressure', and still classifies re-queries"

# --- contract 8: concurrent mode runs, interleaves, compacts, and reports what
# the compaction did to the OTHER session. This is the case depth and breadth
# both miss, and the phase labels are the contract — each claim it makes has to
# be attributable to a phase, or a reader cannot tell a compaction effect from
# ordinary growth.
run_fake usage concurrent SESSIONS=2 TARGET_CTX=1500 TURN_TOKENS=500
[[ $rc -eq 0 ]] || bad "concurrent run exits 0" "0" "$rc: $(tail -3 <<<"$out")"
for phase in grow baseline "COMPACT A" after-compact "A regrow" after-regrow; do
  command grep -q "$phase" <<<"$out" || bad "concurrent reports the '$phase' phase" "$phase rows" "absent: $out"
done
# both sessions must actually appear, or "concurrent" is a single-session run
command grep -qE '^ +grow +A ' <<<"$out" || bad "session A grows" "an A row" "absent"
command grep -qE '^ +grow +B ' <<<"$out" || bad "session B grows" "a B row" "absent"
# and the verdict on the neighbour must be stated either way, never left implied
command grep -qE 'kept its prefix across|LOST REUSE after|reuse unobservable' <<<"$out" \
  || bad "states what the compaction did to the other session" "an explicit survived/lost/unobservable line" "absent: $out"
ok "concurrent mode interleaves two sessions, compacts one, and states the effect on the other"

# --- contract 8b: the neighbour verdict must not claim survival when reuse is
# UNOBSERVABLE. Same absence-vs-zero discipline as the rest of the probe: on a
# server that does not report cached tokens, "kept its prefix" would be a guess.
run_fake none concurrent SESSIONS=2 TARGET_CTX=1500 TURN_TOKENS=500
[[ $rc -eq 0 ]] || bad "concurrent run (no cache reporting) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'reuse unobservable' <<<"$out" \
  || bad "unobservable reuse is reported as such" "'reuse unobservable'" "absent: $out"
command grep -q 'kept its prefix across' <<<"$out" \
  && bad "must not claim survival without evidence" "no survival claim" "claimed it anyway"
ok "with reuse unobservable, the neighbour verdict says so instead of claiming survival"

# --- contract 7b (the negative control #1299 asks for): with reuse absent and a
# cold-cost TTFT, a run that CANNOT establish pressure must report the
# observation without naming a cause. The fake reports cached=0 via a pool it
# cannot exceed, so "eviction" must not appear anywhere in a verdict cell.
run_fake nocache breadth SESSIONS=2 SESSION_CTX=600 KV_POOL=100000
[[ $rc -eq 0 ]] || bad "breadth run (cached=0, no speed-up) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
verdict_cells="$(awk -F'   ' '/^ +[0-9]+ +[0-9,]+/ {print $NF}' <<<"$out")"
command grep -qi 'eviction' <<<"$verdict_cells" \
  && bad "an unpressured run must not name eviction" "no 'eviction' in any verdict" "$(tr '\n' '|' <<<"$verdict_cells")"
ok "with pressure unestablished, verdicts report the observation and name no mechanism"

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-deep-context-probe" >&2; exit 1; fi
echo "PASS: test-deep-context-probe (club-3090#1259)"
