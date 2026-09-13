#!/usr/bin/env bash
# test-soak-decode-basis — canvas-granularity decode reporting (#809, soak half).
#
# decode TPS = completion_tokens / (wall - ttft) assumes autoregressive
# streaming. Canvas-granularity (block-diffusion) models denoise a whole canvas
# in parallel and emit ~one chunk per canvas, so a short response arrives whole,
# ttft == wall, and the window is zero-width. soak printed decode_tps=0.0 on 20
# of 25 turns in #809 — a wrong number replaced by a misleading one, with the
# zeros then dropped from the summary's percentiles so the run silently
# described only the multi-canvas subset.
#
# Contract asserted here:
#   1. canvas turns report a wall-derived figure, LABELLED, never a bare 0.0
#   2. the summary states the measured/derived split and keeps the two series
#      apart (wall TPS includes prefill; it is not a decode rate)
#   3. genuine silent-empty (completion_tokens == 0) still reports 0.0 and still
#      counts toward the silent-empty verdict — the discriminator must not regress
#   4. a fast AUTOREGRESSIVE rig with a narrow-but-real window (#849: 82 ms on
#      dual NVFP4 5090s) is NOT reclassified as canvas
#   5. SOAK_DECODE_GRANULARITY=autoregressive restores the pre-fix behaviour
#
# ⚠️ Two expectations here were SUPERSEDED by #1267 / #1268, deliberately:
#   - a narrow-but-real window no longer prints `decode_tps=0.0`. That zero was
#     byte-identical to a genuine silent-empty turn, so it now renders as
#     `decode_tps=n/a` plus the window width. The byte-identity-with-master leg
#     therefore covers the `ok` shape only; `narrow` asserts the new contract.
#   - a run with unmeasurable turns now DOES emit a "Decode-window basis:" line.
#     It used to appear only when a canvas turn existed, which left an
#     autoregressive run's p50 describing an unstated subset of its turns.
# What has NOT changed, and is still asserted below: canvas turns stay
# wall-derived and labelled, the two series stay apart, and the silent-empty
# discriminator still keys on completion_tokens.
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

soak_env_init
trap soak_env_cleanup EXIT

PLAN_DIR="${SOAK_ENV_DIR}/plans"
mkdir -p "$PLAN_DIR"
export SOAK_SESSIONS=2
export SOAK_TURNS=5

# ── 1. canvas granularity: derived + labelled, never a bare 0.0 ──────────────
cat > "${PLAN_DIR}/canvas" <<'PLAN'
canvas:31900
PLAN

soak_stub_start "${PLAN_DIR}/canvas"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-canvas"
soak_stub_stop
canvas_out="$SOAK_OUT"
canvas_summary="$(cat "${SOAK_ENV_DIR}/run-canvas/summary.md")"

assert_contains "$canvas_out" "canvas-granularity generation detected"
assert_contains "$canvas_out" "(wall-derived, canvas)"
assert_not_contains "$canvas_out" "decode_tps=0.0 "
assert_contains "$canvas_summary" "Decode-window basis:"
assert_contains "$canvas_summary" "wall-derived (canvas) turns | 10 / 10"
assert_contains "$canvas_summary" "p50 wall-derived TPS (canvas)"
assert_contains "$canvas_summary" "INCLUDES prefill and is not a decode rate"
assert_contains "$canvas_out" "p50_wall_tps_canvas"
# The derived series must NOT be folded into the decode percentiles.
assert_contains "$canvas_summary" "| p50 decode TPS | 0.00 |"
# ...and the CSV must carry the basis so a consumer can tell them apart.
assert_contains "$(head -1 "${SOAK_ENV_DIR}/run-canvas/turn-log.csv")" "decode_basis"
assert_contains "$(tail -1 "${SOAK_ENV_DIR}/run-canvas/turn-log.csv")" ",wall"
# A canvas run with no errors is still a PASS.
[[ "$SOAK_RC" -eq 0 ]] || fail "clean canvas run should exit 0, got $SOAK_RC"

# ── 2. escape hatch: forcing autoregressive disables the canvas derivation ───
# Pre-#1267 this asserted a bare `decode_tps=0.0`. The canvas apparatus is still
# off — that is what this leg exists to prove — but the turns it produces are
# `unmeasurable`, and an unmeasurable turn must no longer wear a silent-empty
# turn's rendering.
soak_stub_start "${PLAN_DIR}/canvas"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-canvas-forced" SOAK_DECODE_GRANULARITY=autoregressive
soak_stub_stop
forced_summary="$(cat "${SOAK_ENV_DIR}/run-canvas-forced/summary.md")"
assert_contains "$SOAK_OUT" "decode_tps=n/a"
assert_contains "$SOAK_OUT" "no decode figure"
assert_not_contains "$SOAK_OUT" "decode_tps=0.0"
assert_not_contains "$SOAK_OUT" "(wall-derived, canvas)"
assert_contains "$(tr -d '\r' < "${SOAK_ENV_DIR}/run-canvas-forced/turn-log.csv" | tail -1)" ",unmeasurable"
# The basis line is now emitted for an unmeasurable run too (#1267) — but it
# must describe them as unmeasurable, never as canvas.
assert_contains "$forced_summary" "Decode-window basis:"
assert_contains "$forced_summary" "unmeasurable"
assert_not_contains "$forced_summary" "wall-derived"

# ── 3. silent-empty must not regress ────────────────────────────────────────
# HTTP 200, >=1s, completion_tokens == 0. Not canvas — nothing was produced —
# so it stays 0.0 AND still counts toward the silent-empty verdict.
cat > "${PLAN_DIR}/silent-empty" <<'PLAN'
empty:31900
PLAN

soak_stub_start "${PLAN_DIR}/silent-empty"
SOAK_SESSIONS=1 SOAK_TURNS=2 soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-empty"
soak_stub_stop
empty_summary="$(cat "${SOAK_ENV_DIR}/run-empty/summary.md")"
assert_contains "$SOAK_OUT" "decode_tps=0.0 "
assert_not_contains "$SOAK_OUT" "(wall-derived, canvas)"
assert_contains "$empty_summary" "Silent-empty turns (HTTP 200 + 0 completion tokens): 2 / 2"
assert_contains "$empty_summary" "returned HTTP 200 with empty completion"
assert_contains "$(tail -1 "${SOAK_ENV_DIR}/run-empty/turn-log.csv")" ",empty"
assert_not_contains "$empty_summary" "Decode-window basis:"
[[ "$SOAK_RC" -ne 0 ]] || fail "100% silent-empty must not exit 0"

# ── 4. stickiness: a canvas turn latches, so later narrow turns derive too ───
cat > "${PLAN_DIR}/mixed" <<'PLAN'
canvas:31900
narrow:31900
narrow:31900
PLAN

soak_stub_start "${PLAN_DIR}/mixed"
SOAK_SESSIONS=1 SOAK_TURNS=3 soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-mixed"
soak_stub_stop
assert_not_contains "$SOAK_OUT" "decode_tps=0.0 "
# NB: csv.writer emits CRLF row terminators, so strip the CR before comparing.
mixed_basis="$(tr -d '\r' < "${SOAK_ENV_DIR}/run-mixed/turn-log.csv" | cut -d, -f10 | tail -n +2 | paste -sd, -)"
[[ "$mixed_basis" == "wall,wall,wall" ]] || fail "expected the canvas classification to latch, got: '$mixed_basis'"

# ── 5. autoregressive regression guard — byte-identical to origin/master ────
# The leg that matters most. Two plans: a normal stream (measurable window) and
# the #849 fast-burst shape (82 ms window — sub-threshold but NOT zero). Both
# must behave exactly as they do on master, per-turn stream included.
BASE_TREE="${SOAK_ENV_DIR}/base-tree"
mkdir -p "${BASE_TREE}/scripts"
have_base=1
for f in soak-test.sh soak-helper.py; do
  if ! git show "origin/master:scripts/${f}" > "${BASE_TREE}/scripts/${f}" 2>/dev/null; then
    echo "SKIP: origin/master not available — cannot run the byte-identity leg" >&2
    have_base=0
    break
  fi
done

normalise() {
  # The per-turn provenance label and the decode-rate-source banner are new in
  # #1267/#1268 and have no counterpart on master. They are dropped here so this
  # leg keeps asserting what it was written to assert: that no FIGURE drifted.
  # Their presence is asserted in test-soak-decode-source.sh.
  sed -E \
    -e '/decode-rate source:/d' \
    -e 's/ \((client-timed|engine-reported)[^)]*\)$//' \
    -e 's#run-ar-(new|base)-[a-z]+#<RUNDIR>#g' \
    -e 's#http://127\.0\.0\.1:[0-9]+#<ENDPOINT>#g' \
    -e 's/[0-9]+\.[0-9]+/<NUM>/g' \
    -e 's/wall=[0-9]+ms/wall=<NUM>ms/g' \
    -e 's/ttft=[0-9]+ms/ttft=<NUM>ms/g' \
    -e 's/\| [0-9]+ ms \|/| <NUM> ms |/g' \
    -e 's/(p[0-9]+_ttft_ms +)[0-9]+/\1<NUM>/g' \
    -e 's/[0-9]+ turn\(s\) exceeded/<NUM> turn(s) exceeded/g'
}

# assert_same <label> <baseline-text> <new-text> — equal once measured
# durations and the run directory are normalised away.
assert_same() {
  local label="$1" d
  d="$(diff <(printf '%s\n' "$2" | normalise) <(printf '%s\n' "$3" | normalise) || true)"
  [[ -z "$d" ]] || {
    echo "FAIL: autoregressive ${label} drifted from origin/master:" >&2
    echo "$d" >&2
    exit 1
  }
}

if [[ "$have_base" == "1" ]]; then
  chmod +x "${BASE_TREE}/scripts/soak-test.sh"
  for shape in ok narrow; do
    printf '%s:31900\n' "$shape" > "${PLAN_DIR}/ar-${shape}"

    soak_stub_start "${PLAN_DIR}/ar-${shape}"
    soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-ar-new-${shape}"
    soak_stub_stop
    new_out="$SOAK_OUT"
    new_summary="$(cat "${SOAK_ENV_DIR}/run-ar-new-${shape}/summary.md")"

    soak_stub_start "${PLAN_DIR}/ar-${shape}"
    soak_run "$BASE_TREE" "${SOAK_ENV_DIR}/run-ar-base-${shape}"
    soak_stub_stop
    base_out="$SOAK_OUT"
    base_summary="$(cat "${SOAK_ENV_DIR}/run-ar-base-${shape}/summary.md")"

    # No canvas apparatus may appear on an autoregressive run.
    assert_not_contains "$new_out" "canvas"
    assert_not_contains "$new_summary" "canvas"
    if [[ "$shape" == "narrow" ]]; then
      # Prove this leg is actually exercising the #849 fast-burst path rather
      # than quietly landing in the measurable branch: the window is under the
      # 100 ms floor, so it must record the 'unmeasurable' basis — NOT be
      # reclassified as canvas. Its RENDERING is #1267's, not master's, so the
      # byte-identity comparison below deliberately does not cover this shape.
      assert_contains "$new_out" "decode_tps=n/a"
      assert_not_contains "$new_out" "decode_tps=0.0"
      assert_contains "$(tr -d '\r' < "${SOAK_ENV_DIR}/run-ar-new-${shape}/turn-log.csv" | tail -1)" ",unmeasurable"
      # NB: no base-vs-new comparison here on purpose, in either direction. An
      # assertion that master still PRINTS the defect would pass today and fail
      # the moment this lands on master — a test that dies of its own success.
    else
      assert_contains "$(tr -d '\r' < "${SOAK_ENV_DIR}/run-ar-new-${shape}/turn-log.csv" | tail -1)" ",decode"
      assert_same "${shape} summary" "$base_summary" "$new_summary"
      assert_same "${shape} stdout"  "$base_out"     "$new_out"
    fi
  done
fi

echo "PASS: test-soak-decode-basis"
