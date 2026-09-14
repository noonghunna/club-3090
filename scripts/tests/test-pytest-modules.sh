#!/usr/bin/env bash
#
# Guard: the pytest modules under scripts/tests/ actually RUN.
#
# Why (club-3090#1316): the documented gate is `for t in scripts/tests/*.sh`, so
# the four `test_*.py` modules beside those gates were invoked by NOTHING — not
# this suite, not CI (`.github/workflows/release.yml` only regenerates the
# CHANGELOG on tag push). #1205 hard-cut the `local/` namespace and carefully
# updated every `.sh` test it touched; the `.py` ones it could not know to
# update, because no gate would ever have told it. Three of the four modules
# went red and stayed red for a week under a green 172/0 sweep.
#
# A test nothing invokes cannot fail, and therefore protects nothing — it is
# indistinguishable from a deleted test except that it still LOOKS like
# coverage. This wrapper is the one entry point that makes them fail out loud.
#
# ⚠️ Naming contract this relies on:
#     test-<name>.sh   a bash gate     (run directly by the suite loop)
#     test_<name>.py   a pytest module (run HERE, never standalone)
#     anything else    a helper, NOT a test
# `lib-scan-bare-grep.py` is the third kind — a scanner library consumed by
# test-no-bare-grep.sh. Globbing `*.py` would try to execute it and produce a
# spurious failure, which is how #1316 was first mis-scored. Glob `test_*.py`.
#
# ⚠️ These modules MUST be run via `python3 -m pytest` from the repo root, not
# as `python3 scripts/tests/test_promote.py` — run standalone they die with
# `ModuleNotFoundError: No module named 'scripts'`, a harness-shaped error that
# invites shrugging past a real failure.
#
# ⚠️ A missing pytest is a HARD FAIL here, deliberately — not a skip. Skipping
# would reproduce the exact defect this guard exists to close: a gate that is
# green because it did nothing.

set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

fail=0
note() { printf '  %s\n' "$*"; }

# ── The modules, by the naming contract above ───────────────────────────────
shopt -s nullglob
modules=(scripts/tests/test_*.py)
shopt -u nullglob

if [ ${#modules[@]} -eq 0 ]; then
    note "✗ no scripts/tests/test_*.py modules found"
    note "  This guard exists BECAUSE these are easy to lose track of. If they"
    note "  were intentionally removed, remove this guard in the same commit."
    exit 1
fi

if ! python3 -c 'import pytest' 2>/dev/null; then
    note "✗ pytest is not importable, so ${#modules[@]} test modules cannot run"
    note "  Fix: pip install pytest"
    note "  (Deliberately a FAILURE, not a skip — a gate that silently runs"
    note "   nothing is the bug this guard was written to prevent.)"
    exit 1
fi

note "running ${#modules[@]} pytest modules:"
for m in "${modules[@]}"; do note "    $m"; done

# Full output to a log; summarise from the file. Never pipe a long run through
# tail/head at run time — the pipeline's exit status becomes tail's.
log="$(mktemp)"
python3 -m pytest "${modules[@]}" -q --color=no >"$log" 2>&1
rc=$?

summary="$(command grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$log" | tail -1)"
note "pytest: ${summary:-<no summary line>}"

if [ $rc -ne 0 ]; then
    fail=1
    note "✗ pytest exited $rc — failing tests:"
    command grep -E '^(FAILED|ERROR)' "$log" | sed 's/^/      /'
    note "  Full output: $log"
    note "  Reproduce: python3 -m pytest ${modules[*]} -q"
else
    note "✓ all pytest modules pass"
    rm -f "$log"
fi

exit $fail
