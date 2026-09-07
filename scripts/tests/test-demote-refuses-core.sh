#!/usr/bin/env bash
# ⛔ SAFETY GUARD: `demote.py` must never touch the curated catalog (#1202 P3).
#
# Written BEFORE the local/ hard-cut, deliberately. Until now the only thing
# standing between `--slug vllm/dual` and the curated catalog was a string test
# --  `if not slug.startswith("local/")` -- evaluated before anything is read.
# demote.py's own docstring says so: "CORE IS NEVER TOUCHED ... a non-local/ slug
# is refused before anything is read." Removing the namespace deletes that guard,
# so this test pins the PROPERTY (a curated slug is refused, core is untouched)
# rather than the mechanism, and must pass both before and after the change.
set -uo pipefail
# Non-UTF-8 locales break python3 reads/writes on this rig (#599/#584).
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
rc=0

# A real curated slug, straight from the registry — not a literal that could rot.
CORE_SLUG="$(python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
print(next(iter(COMPOSE_REGISTRY)))" 2>/dev/null)"
[[ -n "$CORE_SLUG" ]] || { echo "  FAIL could not read a core slug from the registry"; exit 1; }

# Fingerprint the curated catalog + the local layer before we poke it.
before="$(find scripts/lib/profiles scripts/lib/profiles-local -type f 2>/dev/null \
          | sort | xargs -r md5sum | md5sum)"

for mode in "--dry-run" "-y"; do
  out="$(python3 scripts/lib/profiles/demote.py --slug "$CORE_SLUG" $mode 2>&1)"; code=$?
  if [[ "$code" -eq 0 ]]; then
    echo "  FAIL demote.py $mode ACCEPTED a curated slug ($CORE_SLUG) — exit 0"
    rc=1
  else
    echo "  ok   demote.py $mode refused $CORE_SLUG (exit $code)"
  fi
  # The refusal must be intelligible, not a stack trace or a bare "not found".
  case "$out" in
    *Traceback*) echo "  FAIL refusal was a traceback, not a message"; rc=1 ;;
  esac
done

after="$(find scripts/lib/profiles scripts/lib/profiles-local -type f 2>/dev/null \
         | sort | xargs -r md5sum | md5sum)"
if [[ "$before" == "$after" ]]; then
  echo "  ok   catalog byte-identical after both attempts"
else
  echo "  FAIL demote.py MODIFIED the catalog while refusing a curated slug"; rc=1
fi

# Refuse a vacuous pass: the tool must actually run (a missing file also 'refuses').
if ! python3 scripts/lib/profiles/demote.py --help >/dev/null 2>&1; then
  echo "  FAIL demote.py --help does not run; the refusals above prove nothing"; rc=1
else
  echo "  ok   demote.py is runnable (refusals are real, not import errors)"
fi

[[ "$rc" == "0" ]] && echo "PASS: core is unreachable from demote.py" || echo "FAIL: core-safety regression"
exit "$rc"
