#!/usr/bin/env bash
#
# Guard: no EXECUTED `grep` in scripts/ may be the bare builtin name.
#
# Why (club-3090#1280): `grep` is shimmed to ugrep in interactive shells on some
# rigs, which breaks exact output matching. A gate for this already existed —
# test-bench-capture.sh asserts it for `lib/capture.sh` — but it covered ONE
# file, so 119 executed bare greps sat unguarded across 27 others. The risk
# concentrates in sourced files: `preflight.sh` alone held 24 and is sourced by
# 20 scripts, and a sourced file runs inside the caller's shell, which is
# exactly where the shim reaches.
#
# ⚠️ Must FAIL against the pre-fix tree, or it is asserting the wrong thing.
#
# ⚠️ Scope: EXECUTED greps only. A `grep` inside an echo/printf string is advice
# we print for a user to type; rewriting it would be wrong, and 11 such lines
# exist deliberately. Hence the scanner is quote-aware rather than a regex — a
# naive `grep -E '\| *grep '` flags those 11 and a maintainer learns to ignore
# the gate. It also treats `$( ... )` INSIDE double quotes as code again, which
# a naive scanner misses; that case accounted for 17 of the 119.
#
# scripts/tests/ is out of scope: it asserts ON grep output.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779 — this gate shells out to python3
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCAN="${ROOT}/scripts/tests/lib-scan-bare-grep.py"

[[ -f "$SCAN" ]] || { echo "FAIL: scanner missing: scripts/tests/lib-scan-bare-grep.py" >&2; exit 1; }

# --- positive control FIRST: a scan that silently finds nothing is
# indistinguishable from a clean tree, so prove it can still see a violation.
probe="$(mktemp -d)"; trap 'rm -rf "$probe"' EXIT
mkdir -p "$probe/scripts/lib"
printf '%s\n' 'x="$(foo | grep -c bar)"'            > "$probe/scripts/planted.sh"
printf '%s\n' 'echo "advice: docker ps | grep thing"' > "$probe/scripts/lib/msgonly.sh"
planted="$(python3 "$SCAN" "$probe" | command grep -c '[^[:space:]]' || true)"
if [[ "$planted" != "1" ]]; then
  echo "FAIL: positive control — scanner should find exactly 1 planted violation (and ignore echo advice), found $planted" >&2
  exit 1
fi
echo "  ✓ scanner detects a planted violation and ignores printed advice"

# --- the actual assertion
out="$(python3 "$SCAN" "$ROOT" | command grep '[^[:space:]]' || true)"
if [[ -n "$out" ]]; then
  echo "FAIL: executed bare \`grep\` found — use \`command grep\`:" >&2
  echo "$out" >&2
  echo "" >&2
  echo "  (A hit here IS in code: the scanner already excludes echo/printf text.)" >&2
  echo "FAIL: test-no-bare-grep" >&2
  exit 1
fi
echo "PASS: test-no-bare-grep (club-3090#1280)"
