#!/usr/bin/env bash
# test-soak-vram-baseline — the warm-VRAM-baseline anchoring contract (#829).
#
# soak-test.sh used to capture its baseline unconditionally at the END of
# session 1. When the engine died mid-session-1 that sampled a corpse, and every
# later reading then read as multi-GB growth: #827 printed
#     - VRAM grew 32053 MiB > 200 MiB threshold.
#     - VRAM session-to-session oscillation reached 63866 MiB.
# on a run whose only real fault was an unrelated engine crash. That is the
# loudest FAIL in the report and it points triage at a memory leak that does not
# exist.
#
# Contract asserted here:
#   1. an errored session NEVER anchors the baseline
#   2. a later CLEAN session re-anchors it, and pre-anchor rows are excluded
#      from growth + oscillation
#   3. if NO session runs clean, VRAM growth/oscillation are UNMEASURABLE —
#      never a number — and the verdict is not PASS
#   4. an all-clean run is byte-identical to the pre-fix behaviour
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

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

# ── 1. #827 repro: engine dies inside session 1 and never comes back ─────────
# Sessions 1-3 x 5 turns. Turns 1-3 stream normally with the GPU reporting a
# healthy 63876 MiB; turn 4 kills the engine and drops VRAM to the 31823 MiB
# corpse reading; everything after that errors. Pre-fix, the baseline was the
# corpse and the peak was the healthy sample: 63876 - 31823 = 32053.
cat > "${PLAN_DIR}/died-in-session-1" <<'PLAN'
ok:63876
ok:63876
ok:63876
dead:31823
http:500:31823
http:500:31823
PLAN

soak_stub_start "${PLAN_DIR}/died-in-session-1"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-dead"
soak_stub_stop

summary_dead="$(cat "${SOAK_ENV_DIR}/run-dead/summary.md")"

# The phantom arithmetic must be gone from BOTH surfaces.
assert_not_contains "$SOAK_OUT" "VRAM grew 32053"
assert_not_contains "$summary_dead" "VRAM grew 32053"
assert_not_contains "$SOAK_OUT" "oscillation reached 63866"
assert_not_contains "$summary_dead" "oscillation reached 63866"
# ...and no growth FAIL of ANY magnitude may be manufactured from a corpse.
assert_not_contains "$summary_dead" "MiB > 200 MiB threshold"

assert_contains "$SOAK_OUT" "NOT anchoring the warm VRAM baseline here"
assert_contains "$SOAK_OUT" "no error-free session completed"
assert_contains "$SOAK_OUT" "boot_vram_mib        UNMEASURABLE"
assert_contains "$SOAK_OUT" "max_growth_mib       INCONCLUSIVE"
assert_contains "$summary_dead" "Boot VRAM baseline: **UNMEASURABLE**"
assert_contains "$summary_dead" "Max growth observed: **INCONCLUSIVE**"
assert_contains "$summary_dead" "VRAM oscillation | INCONCLUSIVE"
assert_contains "$summary_dead" "UNMEASURABLE — no session completed without errors"
# The genuine fault still surfaces as a failure.
assert_contains "$summary_dead" "returned non-200 status or stream error"
[[ "$SOAK_RC" -ne 0 ]] || fail "a run with no measurable baseline must not exit 0 (got $SOAK_RC)"

# ── 2. session 1 errors, sessions 2+ are clean → re-anchor on session 2 ──────
# Session 1 (turns 1-5) errors with the GPU at a transient 63876 MiB; sessions
# 2 and 3 stream cleanly at a steady 31900 MiB. Pre-fix this produced a ~32000
# MiB phantom growth. Post-fix the baseline is session 2's 31900 and the
# session-1 rows are excluded, so growth is ~0.
cat > "${PLAN_DIR}/reanchor" <<'PLAN'
http:500:63876
http:500:63876
http:500:63876
http:500:63876
http:500:63876
ok:31900
PLAN

soak_stub_start "${PLAN_DIR}/reanchor"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-reanchor"
soak_stub_stop

summary_re="$(cat "${SOAK_ENV_DIR}/run-reanchor/summary.md")"
assert_contains "$SOAK_OUT" "warm baseline after session 2: 31900 MiB"
assert_contains "$summary_re" "Boot VRAM baseline: 31900 MiB"
assert_contains "$summary_re" "Warm baseline anchored at END of session 2"
assert_contains "$summary_re" "Max growth observed: 0 MiB"
assert_not_contains "$summary_re" "MiB > 200 MiB threshold"
# The 63876 MiB session-1 samples must not leak into the peak or the oscillation.
assert_not_contains "$summary_re" "Max VRAM observed: 63876 MiB"
assert_contains "$summary_re" "VRAM oscillation | 0 MiB"

# ── 3. all-clean control: byte-identical to pre-fix (origin/master) ──────────
# The regression guard that matters most — the fix must be invisible on a
# healthy run. Both trees are driven by the same plan and the same fake GPU, so
# the only differences allowed are the run directory and wall-clock timings.
cat > "${PLAN_DIR}/all-clean" <<'PLAN'
ok:31900
PLAN


# A base run that could not start leaves no summary.md. Without this the gate
# dies on `cat:` and the reader has no idea the BASE TREE was the problem.
require_base_summary() {
  local d="$1"
  if [[ ! -f "$d/summary.md" ]]; then
    echo "FAIL: the BASELINE soak run produced no summary.md in $d." >&2
    echo "      The base tree could not run — almost always a dependency that" >&2
    echo "      origin/master's soak-test.sh sources but this gate does not extract." >&2
    exit 1
  fi
  cat "$d/summary.md"
}
BASE_TREE="${SOAK_ENV_DIR}/base-tree"
mkdir -p "${BASE_TREE}/scripts/lib"

# ⚠️ THIS LIST MUST COVER EVERY FILE THE BASE soak-test.sh RESOLVES RELATIVE TO
# ITSELF (BASH_SOURCE), not just the entrypoints. soak-test.sh sources
# lib/club-containers.sh; while that was missing, the base tree was unrunnable,
# the base run produced no summary.md, and this gate died on a bare
# `cat: .../summary.md: No such file or directory` that named neither the cause
# nor the fix. The have_base guard below did not catch it because it only tests
# whether `git show` succeeded — which it did, for the two files it knew about.
for f in soak-test.sh soak-helper.py lib/club-containers.sh; do
  if ! git show "origin/master:scripts/${f}" > "${BASE_TREE}/scripts/${f}" 2>/dev/null; then
    echo "SKIP: origin/master:scripts/${f} not available — cannot run the byte-identity leg" >&2
    BASE_TREE=""
    break
  fi
done

# Coverage check: re-derive the BASE script's own relative dependencies and
# assert we extracted them. This is what makes the next added `source` fail
# LOUDLY here instead of silently three steps later at a missing summary.md.
if [[ -n "$BASE_TREE" ]]; then
  while IFS= read -r dep; do
    [[ -z "$dep" || -f "${BASE_TREE}/scripts/${dep}" ]] && continue
    # ⚠️ FAIL, not SKIP. "origin/master unavailable" is an ENVIRONMENT condition and
    # skipping is honest. An incomplete extraction list is OUR BUG, fixable in-repo —
    # skipping it would silently retire the byte-identity leg while the gate stayed
    # green, which is the exact failure this whole guard exists to prevent.
    echo "FAIL: base tree INCOMPLETE — origin/master's soak-test.sh sources scripts/${dep}," >&2
    echo "      which this gate does not extract. Add it to the extraction list above." >&2
    exit 1
  done < <(command grep -oE 'pwd\)/[A-Za-z0-9_./-]+' "${BASE_TREE}/scripts/soak-test.sh" 2>/dev/null | sed 's|pwd)/||')
fi

soak_stub_start "${PLAN_DIR}/all-clean"
soak_run "$ROOT_DIR" "${SOAK_ENV_DIR}/run-clean-new"
soak_stub_stop
new_out="$SOAK_OUT"
new_rc="$SOAK_RC"
new_summary="$(cat "${SOAK_ENV_DIR}/run-clean-new/summary.md")"

[[ "$new_rc" -eq 0 ]] || fail "all-clean run should PASS, got rc=$new_rc: $new_out"
assert_contains "$new_out" "warm baseline after session 1: 31900 MiB"
assert_contains "$new_summary" "Verdict: **PASS**"
# No #829 apparatus may appear on a healthy run.
assert_not_contains "$new_summary" "UNMEASURABLE"
assert_not_contains "$new_summary" "Warm baseline anchored at END"
assert_not_contains "$new_out" "NOT anchoring"

if [[ -n "$BASE_TREE" ]]; then
  chmod +x "${BASE_TREE}/scripts/soak-test.sh"
  soak_stub_start "${PLAN_DIR}/all-clean"
  soak_run "$BASE_TREE" "${SOAK_ENV_DIR}/run-clean-base"
  soak_stub_stop
  base_summary="$(require_base_summary "${SOAK_ENV_DIR}/run-clean-base")"

  # Normalise the two variable inputs: the run directory and every measured
  # duration (wall/ttft/tps/percentiles vary run to run by construction).
  normalise() {
    sed -E \
      -e 's#run-clean-(new|base)#<RUNDIR>#g' \
      -e 's/[0-9]+\.[0-9]+/<NUM>/g' \
      -e 's/wall=[0-9]+ms/wall=<NUM>ms/g' \
      -e 's/ttft=[0-9]+ms/ttft=<NUM>ms/g' \
      -e 's/\| [0-9]+ ms \|/| <NUM> ms |/g' \
      -e 's/[0-9]+ turn\(s\) exceeded/<NUM> turn(s) exceeded/g'
  }
  diff_out="$(diff <(printf '%s\n' "$base_summary" | normalise) \
                   <(printf '%s\n' "$new_summary" | normalise) || true)"
  [[ -z "$diff_out" ]] || {
    echo "FAIL: all-clean summary drifted from origin/master:" >&2
    echo "$diff_out" >&2
    exit 1
  }
fi

echo "PASS: test-soak-vram-baseline"
