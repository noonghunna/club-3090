#!/usr/bin/env bash
# test-served-slots — guards for scripts/lib/served-slots.sh, the ONE served-slot-count
# detector (#1577), and for how a run that could not read it is reported.
#
# rebench-full.sh used to carry a private copy of the detector with no SGLang sources, so
# an SGLang endpoint reached by --url fell back to the N=1 control; and for a vLLM
# endpoint (which reports no slot count over HTTP) the fallback was visible only as a
# line in the step log. This checks: every source answers, the order is the probe's,
# non-integers don't count, both scripts use the shared detector, and an N=1-only run
# is flagged by the summary note and in REPORT.md.
#
# Hermetic: the endpoints are local python stub servers and `docker` is a PATH stub —
# nothing boots, nothing touches the estate.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT_DIR/scripts/lib/served-slots.sh"
REPORT_PY="$ROOT_DIR/scripts/rebench-report.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

WORK="$(mktemp -d)"
PIDS=()
cleanup() {
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill "$p" 2>/dev/null || true; done
  rm -rf "$WORK"
}
trap cleanup EXIT

bash -n "$LIB" || fail "bash -n: syntax error in served-slots.sh"
python3 -m py_compile "$REPORT_PY" || fail "py_compile rebench-report.py"
echo "  ✓ syntax"

# --- stub endpoints ------------------------------------------------------------
# One tiny server per engine shape; each answers only the paths that engine has.
cat > "$WORK/stub.py" <<'PY'
import http.server, json, sys
routes = json.loads(sys.argv[1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = routes.get(self.path.split("?")[0])
        if body is None:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
PY
# Started in THIS shell, never inside $(...): a stub launched in a command-substitution
# subshell records its PID there (lost to the EXIT trap) and outlives the test holding
# whatever stdout/stderr it inherited — a piped run then never sees EOF.
start_stub() {  # $1 = variable to set to the base URL, $2 = routes json
  local out="$WORK/port.$1" port=""
  python3 "$WORK/stub.py" "$2" > "$out" 2>/dev/null </dev/null &
  PIDS+=("$!")
  for _ in $(seq 1 50); do port="$(head -1 "$out" 2>/dev/null || true)"; [[ -n "$port" ]] && break; sleep 0.1; done
  [[ -n "$port" ]] || fail "stub server did not start"
  printf -v "$1" 'http://127.0.0.1:%s' "$port"
}
MODELS='{"object":"list","data":[{"id":"m"}]}'
start_stub SGL  "{\"/get_server_info\":{\"max_running_requests\":4},\"/v1/models\":$MODELS}"
start_stub LCPP "{\"/props\":{\"total_slots\":3},\"/v1/models\":$MODELS}"
start_stub VLLM "{\"/v1/models\":$MODELS}"
start_stub BAD  "{\"/get_server_info\":{\"max_running_requests\":true},\"/props\":{\"total_slots\":0}}"
# Cleanup self-test: the EXIT trap can only stop stubs whose PIDs it holds.
[[ "${#PIDS[@]}" -eq 4 ]] || fail "expected 4 recorded stub PIDs, got ${#PIDS[@]} — the EXIT trap would leak servers"
for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || fail "recorded stub PID $p is not running"; done

# Positive control on the stubs: if they don't answer, every "empty" result below is vacuous.
curl -s -m 3 "$SGL/get_server_info" | command grep -q '"max_running_requests": 4' \
  || fail "stub self-test: SGLang stub does not answer /get_server_info"
echo "  ✓ stub endpoints up (self-test)"

# --- hermetic docker: `docker inspect <name>` answers a canned command line ----
SHIMDIR="$WORK/shim"; mkdir -p "$SHIMDIR"
cat > "$SHIMDIR/docker" <<'SHIM'
#!/usr/bin/env bash
# docker inspect <name> --format ... -> the command line stored for <name>; anything else: benign.
if [[ "$1" == "inspect" ]]; then
  f="${STUB_CMDS_DIR}/$2"
  [[ -f "$f" ]] && { cat "$f"; exit 0; }
  exit 1
fi
exit 0
SHIM
chmod +x "$SHIMDIR/docker"
export STUB_CMDS_DIR="$WORK/cmds"; mkdir -p "$STUB_CMDS_DIR"
echo "vllm serve /model --max-num-seqs 8 --port 8000"          > "$STUB_CMDS_DIR/c-vllm"
echo "vllm serve /model --max-num-seqs=6"                      > "$STUB_CMDS_DIR/c-vllm-eq"
echo "llama-server -m /m.gguf -np 2 -c 32768"                  > "$STUB_CMDS_DIR/c-lcpp"
echo "python -m sglang.launch_server --max-running-requests 5" > "$STUB_CMDS_DIR/c-sgl"
echo "vllm serve /model --port 8000"                           > "$STUB_CMDS_DIR/c-noflag"
export PATH="$SHIMDIR:$PATH"

# shellcheck source=../lib/served-slots.sh
source "$LIB"
expect() {  # label, expected "<N>\t<source>" ('' for none), url, [container]
  local got; got="$(served_slots "$3" "${4:-}")"
  [[ "$got" == "$2" ]] || fail "$1: expected '$(printf %q "$2")', got '$(printf %q "$got")'"
  echo "  ✓ $1"
}
T=$'\t'
expect "SGLang over --url → /get_server_info (the source rebench used to lack)" "4${T}server max_running_requests" "$SGL"
expect "llama.cpp over --url → /props total_slots"                             "3${T}server /props total_slots"   "$LCPP"
expect "vLLM over --url → nothing (it reports no slot count over HTTP)"         ""                                 "$VLLM"
expect "non-integers don't count (bool true, 0)"                                ""                                 "$BAD"
expect "container --max-num-seqs wins over the endpoint"                        "8${T}container max-num-seqs"      "$SGL" c-vllm
expect "container --max-num-seqs=N (= form)"                                    "6${T}container max-num-seqs"      "$VLLM" c-vllm-eq
expect "container -np"                                                          "2${T}container -np"               "$VLLM" c-lcpp
expect "container --max-running-requests"                                       "5${T}container max-running-requests" "$VLLM" c-sgl
expect "container without a slot flag falls through to the endpoint"            "4${T}server max_running_requests" "$SGL" c-noflag
expect "CONTAINER=none is ignored"                                              "3${T}server /props total_slots"   "$LCPP" none

# --- the marker + summary note -------------------------------------------------
M="$WORK/concurrency-slots.txt"
concurrency_slots_marker "$M" undetected undetected "1" 0
note="$(concurrency_slots_note "$M")"
[[ "$note" == *"N=1 only"* && "$note" == *"CONCURRENCY_RUNGS"* ]] || fail "undetected run: summary note missing ('$note')"
echo "  ✓ undetected + no override → summary note names N=1 only + the fix"
concurrency_slots_marker "$M" undetected undetected "1 2 4 8" 1
[[ -z "$(concurrency_slots_note "$M")" ]] || fail "CONCURRENCY_RUNGS override must silence the note"
concurrency_slots_marker "$M" 4 "server max_running_requests" "1 2 4" 0
[[ -z "$(concurrency_slots_note "$M")" ]] || fail "a detected slot count must not produce the note"
[[ -z "$(concurrency_slots_note "$WORK/absent.txt")" ]] || fail "no marker (e.g. --skip concurrency) must not produce the note"
echo "  ✓ override / detected / no-marker → no note"

# --- REPORT.md -----------------------------------------------------------------
mk_tag() {  # $1 dir, then marker args
  mkdir -p "$1"; concurrency_slots_marker "$1/concurrency-slots.txt" "${@:2}"
}
mk_tag "$WORK/rb-undetected" undetected undetected "1" 0
mk_tag "$WORK/rb-detected" 4 "server max_running_requests" "1 2 4" 0
mk_tag "$WORK/rb-override" undetected undetected "1 2 4 8" 1
for d in rb-undetected rb-detected rb-override; do
  python3 "$REPORT_PY" "$WORK/$d" --no-discuss >"$WORK/$d.log" 2>&1 || { cat "$WORK/$d.log" >&2; fail "rebench-report.py failed on $d"; }
done
r="$(cat "$WORK/rb-undetected/REPORT.md")"
[[ "$(command grep -c 'N=1 only' <<<"$r")" -ge 2 ]] || fail "REPORT.md: an N=1-only run must be flagged in the TL;DR AND the Concurrency section"
command grep -q "CONCURRENCY_RUNGS='1 2 4'" <<<"$r" || fail "REPORT.md: the warning must carry the fix"
r="$(cat "$WORK/rb-detected/REPORT.md")"
command grep -q 'N=1 only' <<<"$r" && fail "REPORT.md: a detected slot count must not be flagged N=1 only"
command grep -qF 'Concurrency rungs run: **N=1/2/4** (served slots 4, from server max_running_requests)' <<<"$r" \
  || fail "REPORT.md: a detected run must state the rungs and where the slot count came from"
r="$(cat "$WORK/rb-override/REPORT.md")"
command grep -q 'N=1 only' <<<"$r" && fail "REPORT.md: CONCURRENCY_RUNGS must not be flagged N=1 only"
command grep -qF 'N=1/2/4/8** (set by `CONCURRENCY_RUNGS`)' <<<"$r" || fail "REPORT.md: override run must say the rungs were set by hand"
echo "  ✓ REPORT.md: N=1-only flagged (TL;DR + section); detected / override stated, not flagged"

# --- one detector: no private copies come back -----------------------------------
for s in scripts/rebench-full.sh scripts/concurrency-probe.sh; do
  command grep -qE 'source .*scripts/lib/served-slots\.sh' "$ROOT_DIR/$s" || fail "$s must source scripts/lib/served-slots.sh"
  if command grep -nE "grep -oE '[^']*(max-num-seqs|max-running-requests)|get\\(\"(total_slots|max_running_requests)\"" "$ROOT_DIR/$s"; then
    fail "$s parses a slot count itself — use served_slots (scripts/lib/served-slots.sh) so the detectors can't drift again (#1577)"
  fi
done
echo "  ✓ rebench-full.sh + concurrency-probe.sh use the one detector (no private copies)"

echo "PASS test-served-slots"
