#!/usr/bin/env bash
# test-concurrency-probe-verdict — guards the two verdict rules in
# scripts/lib/concurrency_probe.py that produced the false `FAIL — fit` in #1211.
#
# ⚠️ It imports and calls the REAL functions, and statically asserts the REAL
# wiring. A guard that re-implements the rule inline and tests the copy proves
# nothing: PR #1212's version of this test passed unchanged with the admission
# logic gutted to `run_peak = None`, because it never touched run_probe. Every
# assertion here fails if the corresponding production line is reverted — that
# was verified by reverting each one.
#
# The two rules:
#   vram_ok(leak, growth)         — SYMMETRIC band. The old `0 <= leak` failed a
#                                   -1 MB delta, which is what #1211 actually hit.
#   admission_ok(run_peaks, n, wait_peaks)
#                                 — judged on the PER-ROUND PEAK, never the racy
#                                   last sample (#1212's insight), and a peak below
#                                   n fails only when something WAITED (#1782): a
#                                   stream that finished early is not one refused.
# And the peak's source: engine_stats() reads Running / Waiting only from lines
# logged during the round (#1782), so a round with no stats line is no data.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
LIB="$ROOT_DIR/scripts/lib/concurrency_probe.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

bash -n "$PROBE" || fail "bash -n: syntax error"
python3 -m py_compile "$LIB" || fail "py_compile concurrency_probe.py"
echo "  ✓ syntax"

# --- the REAL functions, imported ------------------------------------------------
PYTHONPATH="$ROOT_DIR/scripts/lib" python3 - "$LIB" <<'PY' || fail "verdict rules"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cp", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["cp"] = m
spec.loader.exec_module(m)

bad = []
def ck(cond, msg):
    if not cond: bad.append(msg)

# --- vram_ok: the #1211 case -----------------------------------------------------
# The reporter's own N=2 trace: cold 44302 -> warm 44303 -> final 44302.
leak_1211 = 44302 - 44303
ck(leak_1211 == -1, "fixture drift: the #1211 delta is -1 MB")
ck(m.vram_ok(leak_1211, 200) is True,
   "#1211: a -1 MB post-warm delta must PASS (it failed under `0 <= leak`)")
ck(m.vram_ok(0, 200) is True,     "a 0 MB delta must pass")
ck(m.vram_ok(199, 200) is True,   "growth inside the band must pass")
ck(m.vram_ok(201, 200) is False,  "a real leak above the band must FAIL")
# Bounded below: a container collapsing mid-probe must still fail, not read clean.
ck(m.vram_ok(-20924, 200) is False,
   "VRAM falling by the size of the model must FAIL (collapsed container)")
ck(m.vram_ok(-201, 200) is False, "below the band must FAIL")

# --- admission_ok: judged on the peak -------------------------------------------
ck(m.admission_ok([1, 1, 1, 1, 1], 1) is True,  "N=1 peak=1 is admitted")
ck(m.admission_ok([2, 2], 2) is True,           "N=2 peak=2 is admitted")
ck(m.admission_ok([0, 2, 0], 2) is True,        "a dip after the peak is completion")
ck(m.admission_ok([1, 1], 2, [1, 0]) is False,  "peak<N with a request WAITING fails (real queueing)")
ck(m.admission_ok([2, 2], 4, [2, 2]) is False,  "max_num_seqs=2 at N=4: peak 2, 2 waiting fails")
ck(m.admission_ok([1, 1], 2, [0, 0]) is True,   "#1782: peak<N with nothing waiting is a finished stream, not a refusal")
ck(m.admission_ok([1, 1], 2) is True,           "peak<N with no Waiting data proves nothing")
ck(m.admission_ok([1, 1], 2, [None, None]) is True, "peak<N with Waiting unknown proves nothing")
ck(m.admission_ok([], 2) is True,               "no engine data must not fail admission")
ck(m.admission_ok([None, None], 2) is True,     "all-None must not fail admission")

# --- the /metrics gauges (#1782) ---------------------------------------------------
page = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_running{engine="1",model_name="m"} 1.0
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 9.0
vllm:num_requests_running_total 7
"""
ck(m.metrics_gauges(page) == (3, 3),
   f"gauges sum over engines, and _by_reason / _total are not the gauge: {m.metrics_gauges(page)}")
ck(m.metrics_gauges("sglang:num_running_reqs 4\n") == (4, None), "SGLang's own gauge names read too")
ck(m.metrics_gauges('sglang:num_running_reqs{model_name="m"} 2.0\nsglang:num_queue_reqs{model_name="m"} 3.0\n') == (2, 3),
   "SGLang running / queued")
ck(m.metrics_gauges("qengine_active_sequences 4\nqengine_queued_requests 1\n") == (4, 1), "QEngine's gauges read too")
ck(m.metrics_gauges("sglang:num_requests_total 9\nfoo 1\n") == (None, None), "a page without the gauges reads None, None")
ck(m.metrics_gauges("vllm:num_requests_running 1\n") == (1, None), "an unlabelled sample still parses")

if bad:
    for b in bad: print("  ⛔", b)
    sys.exit(1)
print("  ✓ vram_ok symmetric band (#1211) · admission_ok on the peak (#1212), keyed on Waiting (#1782) · /metrics gauges")
PY

# --- engine_stats reads the gauges from the ROUND's window (#1782) ----------------
# A stub `docker` answers `logs --since … --until …` with the round's lines and
# `logs --tail 400` with the recent tail. The #1782 shape: an MTP request done in
# 1.5 s, inside vLLM's 10 s stats interval, so the round logged no stats line and
# the tail holds only idle `Running: 0` lines.
STUB="$(mktemp -d)"
trap 'rm -rf "$STUB"' EXIT
cat > "$STUB/docker" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$STUB_DIR/args"
case " $* " in
  *" --since "*) cat "$STUB_DIR/window" ;;
  *) cat "$STUB_DIR/tail" ;;
esac
SH
chmod +x "$STUB/docker"
cat > "$STUB/tail" <<'LOG'
INFO 10-22 15:00:01 [loggers.py:255] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 75.0%
INFO 10-22 15:00:11 [loggers.py:255] Engine 000: Avg prompt throughput: 812.4 tokens/s, Avg generation throughput: 25.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 75.0%
LOG
printf 'INFO 10-22 15:00:05 [logger.py:41] Received request chatcmpl-1\n' > "$STUB/window"
STUB_DIR="$STUB" PATH="$STUB:$PATH" PYTHONPATH="$ROOT_DIR/scripts/lib" python3 - "$LIB" "$STUB" <<'PY' || fail "engine_stats window"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cp", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["cp"] = m
spec.loader.exec_module(m)
stub = sys.argv[2]
bad = []
def ck(cond, msg):
    if not cond: bad.append(msg)

# negative control: the tail alone is the bug — it reports a peak of 0 for a round whose request ran
run, wait, run_max, wait_max, hit = m.engine_stats("c")
ck(run_max == 0, f"fixture drift: the tail alone must read the round's peak as 0 (run_max={run_max})")

open(f"{stub}/args", "w").close()
run, wait, run_max, wait_max, hit = m.engine_stats("c", since=1792695559.25, until=1792695560.75)
calls = open(f"{stub}/args").read().splitlines()
ck(any("--since 1792695559.250000 --until 1792695560.750000 c" in c for c in calls),
   f"the round's window must reach docker logs as --since/--until: {calls}")
ck((run, wait, run_max, wait_max) == (None, None, None, None),
   f"#1782: no stats line inside the round is no data, not 0 ({run}, {wait}, {run_max}, {wait_max})")
ck(m.admission_ok([run_max], 1) is True, "#1782: a round with no stats line must not fail admission")
ck(hit == 75.0, f"the hit rate still comes from the tail (got {hit})")

with open(f"{stub}/window", "w") as f:
    f.write("INFO 10-22 15:00:21 [loggers.py:255] Engine 000: Running: 3 reqs, Waiting: 1 reqs, Prefix cache hit rate: 70.0%\n"
            "INFO 10-22 15:00:31 [loggers.py:255] Engine 000: Running: 1 reqs, Waiting: 0 reqs, Prefix cache hit rate: 71.0%\n")
run, wait, run_max, wait_max, hit = m.engine_stats("c", since=1.0, until=2.0)
ck((run, wait, run_max, wait_max) == (1, 0, 3, 1),
   f"in-window lines: last 1/0, peak 3/1 (got {run}, {wait}, {run_max}, {wait_max})")
ck(m.admission_ok([run_max], 3, [wait_max]) is True and m.admission_ok([run_max], 4, [wait_max]) is False,
   "an in-window peak, with its Waiting, still judges admission")

if bad:
    for b in bad: print("  ⛔", b)
    sys.exit(1)
print("  ✓ engine_stats: Running/Waiting from the round's window, hit rate from the tail (#1782)")
PY

# --- GaugePoller: polls /metrics through a round and keeps the peaks (#1782) -----
PYTHONPATH="$ROOT_DIR/scripts/lib" python3 - "$LIB" <<'PY' || fail "GaugePoller"
import http.server, importlib.util, sys, threading, time
spec = importlib.util.spec_from_file_location("cp", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["cp"] = m
spec.loader.exec_module(m)
bad = []
def ck(cond, msg):
    if not cond: bad.append(msg)

pages = {"seq": [(0, 1), (2, 0), (2, 0), (1, 0), (0, 0)], "i": 0, "body": None}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if pages["body"] is not None:
            text = pages["body"]
        else:
            run, wait = pages["seq"][min(pages["i"], len(pages["seq"]) - 1)]
            pages["i"] += 1
            text = (f"vllm:num_requests_running{{engine=\"0\"}} {run}.0\n"
                    f"vllm:num_requests_waiting{{engine=\"0\"}} {wait}.0\n")
        self.send_response(200); self.end_headers(); self.wfile.write(text.encode())
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"

with m.GaugePoller(url, interval=0.02) as g:
    time.sleep(0.3)
ck(g.available and g.samples >= 5, f"the poller must sample through the round (samples={g.samples})")
ck((g.run_max, g.wait_max) == (2, 1), f"peaks running 2 / waiting 1 (got {g.run_max}, {g.wait_max})")
ck((g.run_last, g.wait_last) == (0, 0), f"last sample running 0 / waiting 0 (got {g.run_last}, {g.wait_last})")

pages["body"] = "sglang:num_requests_total 4\n"   # a metrics page with no running / waiting gauge
with m.GaugePoller(url, interval=0.02) as g:
    time.sleep(0.1)
ck(not g.available and g.run_max is None, "no gauges on the page: unavailable, so the log lines are used")
with m.GaugePoller("http://127.0.0.1:9", interval=0.02) as g:
    time.sleep(0.1)
ck(not g.available, "an unreachable /metrics is unavailable, not a peak of 0")
srv.shutdown()
if bad:
    for b in bad: print("  ⛔", b)
    sys.exit(1)
print("  ✓ GaugePoller: peaks from /metrics through the round; no gauges or no server = unavailable")
PY

# --- the WIRING (a correct rule called with the wrong list is still the bug) -----
command grep -q 'clean_fit = (bad == 0) and vram_ok(leak, GROWTH)' "$LIB" \
  || fail "clean_fit must use vram_ok() — a literal '0 <= leak' reintroduces #1211"
command grep -qE 'admitted_ok = admission_ok\(\s*runmax_by_round, N, waitmax_by_round\)' "$LIB" \
  || fail "admission_ok must be called with runmax_by_round (the PEAK list) and waitmax_by_round; \
run_by_round is each round's LAST sample and reintroduces the race one level up"
command grep -qE 'with GaugePoller\(URL, container=CONTAINER\) as gauges, cf\.ThreadPoolExecutor' "$LIB" \
  || fail "each round must poll /metrics while its requests run (#1782), with the container for QEngine's metrics port"
command grep -qE 'run_max, wait_max = gauges\.run_max, gauges\.wait_max' "$LIB" \
  || fail "the polled peaks must replace the log lines' sample when /metrics has the gauges"
command grep -qE 'run_peak = max\(\[r for r in runmax_by_round' "$LIB" \
  || fail "run_peak must come from the peak list"
command grep -q 'runmax_by_round.append' "$LIB" \
  || fail "the per-round peak must actually be collected"
command grep -qE 'engine_stats\(CONTAINER, since=t0, until=t_end\)' "$LIB" \
  || fail "each round must read the engine's gauges from its own window (#1782): \
engine_stats(CONTAINER, since=t0, until=t_end)"
# Admission must NOT be folded into the fit verdict: that puts a sampling race
# into a verdict that a clean run has already earned.
command grep -qE 'clean_fit = .*admitted_ok' "$LIB" \
  && fail "admission must NOT be a term in clean_fit — a run where every request \
completed must not fail on when the metrics sample landed"
echo "  ✓ wiring: peak list, vram_ok, and admission kept out of clean_fit"

# --- the sweep early-stop keys off the peak --------------------------------------
command grep -qE 'running_max.*-lt.*"\$n"' "$PROBE" \
  || fail "sweep early-stop must compare the PEAK running against n"
command grep -qE '"\$waiting_max" -gt 0' "$PROBE" \
  || fail "sweep early-stop must also require something waiting (#1782)"
command grep -q 'running_max=' "$PROBE" \
  || fail "sweep must parse running_max from the RESULT line"
echo "  ✓ sweep early-stop keys off the peak running"

echo "test-concurrency-probe-verdict: ok"
