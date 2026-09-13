#!/usr/bin/env bash
# test-soak-decode-source — decode-figure provenance + engine counters.
#
# Covers the pair #1267 (an unmeasurable turn and a genuine silent-empty turn
# rendered IDENTICALLY) and #1268 (the 100 ms client-timing floor discards real
# decode on fast hardware). Both are failures you cannot see in the output,
# which is what these assertions are shaped around: they check that two
# different states RENDER differently, not merely that a run completes.
#
# Pre-fix, both states printed a bare `decode_tps=0.0` and the reader had to
# redo `wall - ttft` by hand to tell "fine, decoded in 49 ms" from "HTTP 200
# with zero tokens — the failure soak exists to catch". Three real turns from a
# PASSING 25-turn run on the reference rig:
#     wall=656ms    ttft=557ms    -> decode window  99 ms
#     wall=650ms    ttft=550ms    -> decode window 100 ms
#     wall=10606ms  ttft=10531ms  -> decode window  75 ms
#
# Contract asserted here:
#   1. unmeasurable renders as `decode_tps=n/a` + its window width, and is
#      textually DISTINGUISHABLE from a silent-empty turn's `decode_tps=0.0`
#   2. the run summary states the basis split, so p50's denominator is explicit,
#      and the silent-empty discriminator is unchanged (it keys on
#      completion_tokens, never on decode_tps)
#   3. an engine counter, when reachable, produces the figure INSTEAD of the
#      SSE-timed window — on a turn whose window is under the floor, which
#      pre-fix reported as 0.0. Three sources: /metrics TPOT histogram,
#      SGLang's "gen throughput (token/s)" log line, llama.cpp's "eval time"
#   4. llama.cpp's PROMPT eval line (prefill) is never read as the decode rate
#   5. the scrape is bounded to the turn (`--since`) and reads STDERR
#   6. SOAK_ENGINE_COUNTER=off restores pure client-side timing + the floor —
#      the one-word reversal of the #1268 design decision
#
# Negative control (how this test was proven to bite): run it against the
# pre-fix tree — `git stash` / `git checkout <base> -- scripts/soak-*` — and
# leg 1 fails on the first assertion, because there `decode_tps=0.0` is all
# there is. Every engine leg fails there too: no counter is ever read.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=fixtures/soak-harness/soak-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/soak-harness/soak-env.sh"

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || { echo "FAIL: missing '$2' in:" >&2; echo "$1" >&2; exit 1; }; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || { echo "FAIL: must NOT contain '$2':" >&2; echo "$1" >&2; exit 1; }; }

# turn_line <output> <turn-number> — the per-turn log line for that turn.
turn_line() { printf '%s\n' "$1" | command grep -F "turn ${2}/" | head -1; }
# decode_render <line> — just the decode figure + its label, with the volatile
# VRAM reading normalised away. This is the string a reader uses to tell one
# kind of turn from another, so it is the string the test compares.
decode_render() {
  printf '%s\n' "$1" | sed -E 's/^.*decode_tps=/decode_tps=/; s/vram=[0-9]+MiB/vram=<V>MiB/'
}

soak_env_init
trap soak_env_cleanup EXIT

PLAN_DIR="${SOAK_ENV_DIR}/plans"
LOG_DIR="${SOAK_ENV_DIR}/engine-logs"
mkdir -p "$PLAN_DIR" "$LOG_DIR"

# ── 1. #1267: unmeasurable vs silent-empty must not render the same ──────────
# Turns 1-3 decode in ~80 ms (under the floor, real output). Turn 4 is HTTP 200
# with zero completion tokens after >1 s of "thinking" — the genuine failure.
cat > "${PLAN_DIR}/narrow-then-empty" <<'PLAN'
narrow:31900
narrow:31900
narrow:31900
empty:31900
PLAN

export SOAK_SESSIONS=1
export SOAK_TURNS=4
soak_stub_start "${PLAN_DIR}/narrow-then-empty"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-1267"
soak_stub_stop
out_1267="$SOAK_OUT"
summary_1267="$(cat "${SOAK_ENV_DIR}/run-1267/summary.md")"

narrow_line="$(turn_line "$out_1267" 3)"
empty_line="$(turn_line "$out_1267" 4)"
[[ -n "$narrow_line" && -n "$empty_line" ]] || fail "could not find both turn lines in:
$out_1267"

# THE assertion this pair of issues is about.
if [[ "$(decode_render "$narrow_line")" == "$(decode_render "$empty_line")" ]]; then
  fail "unmeasurable and silent-empty turns render identically — the #1267 defect:
  unmeasurable: $narrow_line
  silent-empty: $empty_line"
fi

assert_contains "$narrow_line" "decode_tps=n/a"
assert_contains "$narrow_line" "no decode figure"
assert_contains "$narrow_line" "ms decode window is under the 100 ms client-timing floor"
assert_contains "$empty_line" "decode_tps=0.0"
assert_contains "$empty_line" "SILENT-EMPTY: 0 completion tokens"
# An unmeasurable turn must not present a number that can be read as a rate.
assert_not_contains "$narrow_line" "decode_tps=0.0"

# The summary must now state the denominator its p50 was computed over (#1267),
# and must still classify the silent-empty turn exactly as before.
assert_contains "$summary_1267" "Decode-window basis:"
assert_contains "$summary_1267" "3 unmeasurable (decode window under the 100 ms client-timing floor, no engine counter)"
assert_contains "$summary_1267" "1 silent-empty"
assert_contains "$summary_1267" "computed over the 0 turn(s) carrying a decode figure"
assert_contains "$summary_1267" "Silent-empty turns (HTTP 200 + 0 completion tokens): 1 / 4"
assert_contains "$out_1267" "silent_empty         1 / 4"
# ...and the CSV must carry both states distinctly.
csv_1267="$(tr -d '\r' < "${SOAK_ENV_DIR}/run-1267/turn-log.csv" | cut -d, -f10 | tail -n +2 | paste -sd, -)"
[[ "$csv_1267" == "unmeasurable,unmeasurable,unmeasurable,empty" ]] \
  || fail "expected the CSV basis column to separate the two states, got: '$csv_1267'"

# ── 2. #1268: /metrics TPOT histogram beats the client-side floor ────────────
# Same narrow (sub-floor) turns as leg 1, but now the engine answers. Pre-fix
# these turns are exactly the ones that report 0.0.
cat > "${PLAN_DIR}/narrow" <<'PLAN'
narrow:31900
PLAN

export SOAK_TURNS=3
soak_stub_start "${PLAN_DIR}/narrow" STUB_TPOT_TPS=123.456
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-prom"
soak_stub_stop
out_prom="$SOAK_OUT"
summary_prom="$(cat "${SOAK_ENV_DIR}/run-prom/summary.md")"

assert_contains "$out_prom" "decode-rate source: ENGINE counter"
assert_contains "$out_prom" "vllm:time_per_output_token_seconds"
assert_contains "$out_prom" "decode_tps=123.456"
assert_contains "$out_prom" "(engine-reported: /metrics time_per_output_token_seconds)"
assert_not_contains "$out_prom" "decode_tps=n/a"
assert_not_contains "$out_prom" "decode_tps=0.0"
# The decoy histogram on the same page must not be what got read.
assert_not_contains "$out_prom" "e2e_request_latency"
assert_contains "$summary_prom" "3 engine-reported (prom-tpot)"
assert_contains "$summary_prom" "| p50 decode TPS | 123.46 |"
csv_prom="$(tr -d '\r' < "${SOAK_ENV_DIR}/run-prom/turn-log.csv" | tail -1)"
assert_contains "$csv_prom" ",engine,prom-tpot"

# ── 3. #1268: SGLang's decode-batch log line ────────────────────────────────
# No /metrics here, so the run falls through to the container log — which the
# fake docker prints on STDERR only, the way every engine in this repo logs.
cat > "${LOG_DIR}/sglang.log" <<'LOG'
[2026-09-13 09:00:00 TP0] Prefill batch. #new-seq: 1, #new-token: 1024, #cached-token: 0, token usage: 0.01, #running-req: 0, #queue-req: 0
[2026-09-13 09:00:01 TP0] Decode batch. #running-req: 1, #token: 1040, token usage: 0.02, gen throughput (token/s): 88.75, #queue-req: 0
LOG

soak_stub_docker "${LOG_DIR}/sglang.log"
export SOAK_TURNS=2
soak_stub_start "${PLAN_DIR}/narrow"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-sglang" CONTAINER=sglang-stub
soak_stub_stop
out_sglang="$SOAK_OUT"

assert_contains "$out_sglang" "decode-rate source: ENGINE counter — decode-rate lines in 'docker logs sglang-stub'"
assert_contains "$out_sglang" "decode_tps=88.75"
assert_contains "$out_sglang" "(engine-reported: SGLang 'gen throughput (token/s)' log line)"
assert_not_contains "$out_sglang" "decode_tps=n/a"
csv_sglang="$(tr -d '\r' < "${SOAK_ENV_DIR}/run-sglang/turn-log.csv" | tail -1)"
assert_contains "$csv_sglang" ",engine,sglang-log"
# The scrape must be bounded to the turn being measured, or a stale line from an
# earlier turn can be reported as this turn's rate.
docker_argv="$(cat "${SOAK_ENV_DIR}/docker-argv.log")"
assert_contains "$docker_argv" "logs --since "
assert_contains "$docker_argv" "--tail"

# ── 4. #1268: llama.cpp eval time — and NOT prompt eval time ────────────────
# Both lines carry "tokens per second"; the prefill one is ~6.6x faster here, so
# reading the wrong line would produce a plausible, wrong, unfalsifiable number.
cat > "${LOG_DIR}/llamacpp.log" <<'LOG'
slot release: id  0 | task 12 | stop processing: n_past = 1234, truncated = 0
prompt eval time =    1804.51 ms /  1024 tokens (    1.76 ms per token,   567.34 tokens per second)
       eval time =    2334.63 ms /   199 runs   (   11.73 ms per token,    85.24 tokens per second)
      total time =    4139.14 ms /  1223 tokens
LOG

soak_stub_docker "${LOG_DIR}/llamacpp.log"
soak_stub_start "${PLAN_DIR}/narrow"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-llamacpp" CONTAINER=llama-cpp-stub
soak_stub_stop
out_llamacpp="$SOAK_OUT"

assert_contains "$out_llamacpp" "decode_tps=85.24"
assert_contains "$out_llamacpp" "(engine-reported: llama.cpp 'eval time' log line)"
assert_not_contains "$out_llamacpp" "567.34"

# ── 5. the reversal switch: SOAK_ENGINE_COUNTER=off ─────────────────────────
# The engine counter is reachable (the stub serves /metrics) and must be
# ignored: client-side timing and the floor, exactly as before #1268 — still
# labelled, because #1267's contract is independent of where the figure came
# from. `ok` decodes over ~150 ms (measurable), `narrow` over ~80 ms (not).
cat > "${PLAN_DIR}/ok-then-narrow" <<'PLAN'
ok:31900
narrow:31900
PLAN

rm -f "${SOAK_ENV_BIN}/docker"   # back to host mode for this leg
export SOAK_TURNS=2
soak_stub_start "${PLAN_DIR}/ok-then-narrow" STUB_TPOT_TPS=123.456
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-off" SOAK_ENGINE_COUNTER=off
soak_stub_stop
out_off="$SOAK_OUT"
summary_off="$(cat "${SOAK_ENV_DIR}/run-off/summary.md")"

assert_contains "$out_off" "decode-rate source: CLIENT-side SSE timing — SOAK_ENGINE_COUNTER=off"
assert_not_contains "$out_off" "123.456"
assert_not_contains "$out_off" "engine-reported"
assert_contains "$(turn_line "$out_off" 1)" "(client-timed:"
assert_contains "$(turn_line "$out_off" 1)" "ms decode window)"
assert_contains "$(turn_line "$out_off" 2)" "decode_tps=n/a"
assert_contains "$summary_off" "1 client-timed / 1 unmeasurable"
assert_contains "$summary_off" "computed over the 1 turn(s) carrying a decode figure"

# An invalid value must be refused, not silently coerced into a mode.
set +e
bad_out="$(SOAK_ENGINE_COUNTER=maybe bash "${ROOT_DIR}/scripts/soak-test.sh" --quick 2>&1)"
bad_rc=$?
set -e
[[ "$bad_rc" -eq 2 ]] || fail "SOAK_ENGINE_COUNTER=maybe should exit 2, got ${bad_rc}: $bad_out"
assert_contains "$bad_out" "must be 'auto' or 'off'"

echo "PASS: test-soak-decode-source"
