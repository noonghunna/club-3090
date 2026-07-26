#!/usr/bin/env bash
set -euo pipefail

# test-locale-utf8.sh — the non-UTF-8-locale guarantee (#779).
#
# Repo sources are full of unicode (— × → ⚠). On a rig whose locale is neither
# UTF-8 nor C, Python decodes file reads, stdout AND sys.argv with the locale
# codec, which crashed large parts of the script layer: on a real
# en_US.ISO-8859-1 locale the suite was 46 pass / 38 fail.
#
# Python already auto-enables UTF-8 mode (PEP 540) for the C/POSIX locale, so
# `LC_ALL=C` alone is NOT the at-risk case and cannot demonstrate any of this.
# The at-risk case is a genuine single-byte locale (de_DE.iso88591 and friends),
# where UTF-8 mode stays off. Every script that runs python3 therefore exports
# PYTHONUTF8, which covers reads, writes, stdout and argv in one move — the argv
# class especially, where no `encoding=` pin can help (surrogateescape, #777).
#
# `${PYTHONUTF8:-1}` and not `=1`: a user who deliberately sets PYTHONUTF8=0
# keeps control. We supply the default, we don't override an explicit choice.
#
# Asserts:
#   (a) every python3-invoking script under scripts/ exports PYTHONUTF8 — this
#       is the invariant that DECAYS, since a new script silently reintroduces
#       the bug and nothing else would notice;
#   (b) the export precedes the first python3 use in each file (an export below
#       the call site is inert);
#   (c) it defaults rather than forces, so an explicit PYTHONUTF8=0 still wins;
#   (d) functionally, with a REAL single-byte locale when one can be built:
#       representative scripts run clean and emit unicode intact. Skipped with a
#       printed note when `localedef` is unavailable — never silently passed.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "ASSERTION FAILED: $*" >&2
  exit 1
}

# --- (a)+(b)+(c) the static invariant -------------------------------------
missing=()
late=()
forced=()
checked=0

# A python3 mention inside a COMMENT is not an invocation — including the very
# comment this convention ships with, which names python3 in its own text. Match
# only executable lines, or every file self-reports as broken.
first_python3_line() {
  # `|| true` throughout: `set -o pipefail` is on, and a grep that legitimately
  # matches nothing exits 1, which would abort this test with NO message at all.
  grep -nE '(^|[^a-z-])python3' "$1" 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1 || true
}

while IFS= read -r f; do
  # This gate is exempt from its own rule: its probes deliberately set and unset
  # PYTHONUTF8 to prove both directions, so a blanket export would make them
  # vacuous. (It also self-matches on its own function name and grep pattern.)
  [[ "${f##*/}" == "test-locale-utf8.sh" ]] && continue

  # Skip files that never actually invoke python3 — nothing to protect.
  first_py="$(first_python3_line "$f")"
  [[ -n "$first_py" ]] || continue
  checked=$((checked + 1))

  if ! grep -q 'export PYTHONUTF8' "$f"; then
    missing+=("$f")
    continue
  fi

  export_line="$(grep -n 'export PYTHONUTF8' "$f" | head -1 | cut -d: -f1 || true)"
  if (( export_line > first_py )); then
    late+=("$f (export at ${export_line}, first python3 at ${first_py})")
  fi

  # Must DEFAULT, not force: `${PYTHONUTF8:-1}` keeps an explicit 0 working.
  if grep -qE '^export PYTHONUTF8=1\s*$' "$f"; then
    forced+=("$f")
  fi
done < <(find scripts -maxdepth 2 -name '*.sh' -type f | sort)

if (( ${#missing[@]} > 0 )); then
  echo "ASSERTION FAILED: script(s) run python3 without exporting PYTHONUTF8 (#779)." >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "  Fix: add near the top, above the first python3 call:" >&2
  echo '    export PYTHONUTF8="${PYTHONUTF8:-1}"' >&2
  exit 1
fi

if (( ${#late[@]} > 0 )); then
  echo "ASSERTION FAILED: PYTHONUTF8 exported BELOW the first python3 call (inert)." >&2
  printf '  %s\n' "${late[@]}" >&2
  exit 1
fi

if (( ${#forced[@]} > 0 )); then
  echo "ASSERTION FAILED: PYTHONUTF8 forced to 1 instead of defaulted." >&2
  printf '  %s\n' "${forced[@]}" >&2
  echo '  Use `export PYTHONUTF8="${PYTHONUTF8:-1}"` so an explicit 0 still wins.' >&2
  exit 1
fi

(( checked >= 80 )) || fail "expected to check 80+ python3-invoking scripts, checked ${checked} (discovery broke?)"

# --- (d) functional leg on a REAL single-byte locale -----------------------
# Synthesized rather than assumed: modern minimal images ship C.UTF-8 and often
# no single-byte locale at all, so there is usually nothing installed to test
# against — which is precisely why this class went unnoticed for so long.
if ! command -v localedef >/dev/null 2>&1; then
  echo "test-locale-utf8: ok (static invariant over ${checked} scripts; functional leg SKIPPED — localedef unavailable)"
  exit 0
fi

TMP_LOC="$(mktemp -d)"
trap 'rm -rf "$TMP_LOC"' EXIT

if ! localedef -f ISO-8859-1 -i en_US "${TMP_LOC}/en_US.ISO-8859-1" >/dev/null 2>&1; then
  echo "test-locale-utf8: ok (static invariant over ${checked} scripts; functional leg SKIPPED — localedef could not build ISO-8859-1)"
  exit 0
fi

hostile=(env "LOCPATH=${TMP_LOC}" LC_ALL=en_US.ISO-8859-1 LANG=en_US.ISO-8859-1)

# Sanity-check the hostile env is actually hostile — otherwise this leg proves
# nothing. PYTHONUTF8 is explicitly cleared here so we observe the raw locale.
# `-u` must precede the assignments: `env FOO=1 -u BAR cmd` treats `-u` as the
# command name, not an option.
raw_mode="$(env -u PYTHONUTF8 "LOCPATH=${TMP_LOC}" LC_ALL=en_US.ISO-8859-1 LANG=en_US.ISO-8859-1 \
  python3 -c 'import sys; print(sys.flags.utf8_mode)')"
[[ "$raw_mode" == "0" ]] || fail "synthesized locale did not disable UTF-8 mode (utf8_mode=${raw_mode}); the functional leg would be vacuous"

# A script's exported default must reach the python3 it spawns.
probe="$("${hostile[@]}" bash -c '
  export PYTHONUTF8="${PYTHONUTF8:-1}"
  python3 -c "import sys; print(sys.flags.utf8_mode, sys.stdout.encoding)"
')"
[[ "$probe" == "1 utf-8" ]] || fail "the exported default did not reach the child python (got: ${probe})"

# An explicit 0 must still win — we default, we do not override.
probe0="$("${hostile[@]}" PYTHONUTF8=0 bash -c '
  export PYTHONUTF8="${PYTHONUTF8:-1}"
  python3 -c "import sys; print(sys.flags.utf8_mode)"
')"
[[ "$probe0" == "0" ]] || fail "an explicit PYTHONUTF8=0 was overridden (got: ${probe0}); the default must not force"

# End to end: a real script that reads unicode-bearing repo sources and prints
# unicode must produce BYTE-IDENTICAL output under both locales. Stronger than
# probing for one character — it catches silent mojibake as well as crashes,
# which matters because a single-byte locale DECODES any byte happily and
# corrupts quietly rather than raising (only the encode side blows up).
if ! ref="$(bash scripts/switch.sh --list 2>&1)"; then
  fail "switch.sh --list failed under the NORMAL locale; the comparison would be meaningless"
fi
if ! out="$("${hostile[@]}" bash scripts/switch.sh --list 2>&1)"; then
  echo "ASSERTION FAILED: switch.sh --list failed under a real ISO-8859-1 locale" >&2
  echo "$out" | tail -20 >&2
  exit 1
fi
# Guard against a vacuous comparison: if the output were pure ASCII, matching
# bytes would prove nothing about encoding at all.
if ! printf '%s' "$ref" | grep -qP '[^\x00-\x7F]' 2>/dev/null; then
  fail "switch.sh --list emitted no non-ASCII characters; this leg cannot detect an encoding fault"
fi
if [[ "$ref" != "$out" ]]; then
  echo "ASSERTION FAILED: switch.sh --list output differs between locales (mojibake)." >&2
  diff <(printf '%s' "$ref") <(printf '%s' "$out") | head -10 >&2
  exit 1
fi

echo "test-locale-utf8: ok (static invariant over ${checked} scripts; functional leg on a real ISO-8859-1 locale)"
