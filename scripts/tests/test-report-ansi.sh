#!/usr/bin/env bash
# test-report-ansi — report.sh's output carries no terminal escape sequences.
#
# The report is Markdown for a file or an issue, but the tools it embeds (verify-*, soak,
# bench, `nvidia-smi topo`) colour their output even when piped. Pasted reports carried
# 37-39 raw sequences each, which read as "[32m✓[0m" / "[4mGPU0…[0m" (#1427). report.sh
# now strips them in redact(), which every embedded block passes through.
#
#   1. strip_ansi + redact, the REAL functions out of report.sh, on a line with colour,
#      bold, cursor, OSC-title and charset sequences, ✓ / ×, and a literal "[32m" with no
#      ESC: every sequence goes, the rest stays byte-for-byte, redacted and --no-redact.
#   2. the same under en_US.UTF-8 when the rig has it. There GNU sed reads [@-~] in
#      collation order, which leaves "m" out, so a strip without LC_ALL=C matched no colour
#      code at all — the first version of this fix did exactly that.
#   3. every block report.sh pipes into details() goes through redact() (or strip_ansi), so
#      a new embed can't bring the codes back.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="$ROOT/scripts/report.sh"
fails=0
fail() { echo "  ✗ $*" >&2; fails=$((fails + 1)); }
ok() { echo "  ✓ $*"; }

fns="$(awk '/^strip_ansi\(\) \{$/,/^}$/' "$REPORT"; awk '/^redact\(\) \{$/,/^}$/' "$REPORT")"
[[ "$fns" == *"strip_ansi()"* && "$fns" == *"redact()"* ]] || { echo "  ✗ strip_ansi/redact not found in report.sh" >&2; exit 1; }

ESC=$'\033'
input="${ESC}[32m✓${ESC}[0m pass · ${ESC}[1;33mwarn${ESC}[0m · ${ESC}[4mGPU0${ESC}[0m · ${ESC}]0;title"$'\a'"x · ${ESC}(Bcs${ESC}[m · ${ESC}[2Kerase · 2× 3090 · keep [32m literal · hf_abcdefghijklmnopqrstuvwxyz0123456789"
plain="✓ pass · warn · GPU0 · x · cs · erase · 2× 3090 · keep [32m literal · hf_abcdefghijklmnopqrstuvwxyz0123456789"

run() {  # run <locale> <REDACT> -> the function's output for $input
  env -i PATH="$PATH" LC_ALL="$1" bash -c "REDACT=$2; USER_NAME=nobody-xyz; HOST_SHORT=nohost-xyz; $fns"$'\n''printf "%s\n" "$1" | redact' _ "$input"
}

locales=(C C.UTF-8)
if locale -a 2>/dev/null | command grep -qiE '^en_US\.utf-?8$'; then locales+=(en_US.UTF-8)
else echo "  ⊘ en_US.UTF-8 not installed here — the collation case is not exercised on this machine"; fi

for loc in "${locales[@]}"; do
  out="$(run "$loc" 0)"
  if [[ "$out" == *"$ESC"* ]]; then fail "[$loc, --no-redact] escape sequences left: $(printf '%s' "$out" | cat -v)"
  elif [[ "$out" != "$plain" ]]; then fail "[$loc, --no-redact] text changed beyond the sequences: $(printf '%s' "$out" | cat -v)"
  else ok "[$loc, --no-redact] every sequence stripped; ✓ × and the literal [32m kept"; fi
  out="$(run "$loc" 1)"
  if [[ "$out" == *"$ESC"* ]]; then fail "[$loc, redacted] escape sequences left: $(printf '%s' "$out" | cat -v)"
  elif [[ "$out" != "${plain/hf_abcdefghijklmnopqrstuvwxyz0123456789/hf_<REDACTED>}" ]]; then fail "[$loc, redacted] unexpected output: $(printf '%s' "$out" | cat -v)"
  else ok "[$loc, redacted] sequences stripped and the token still redacted"; fi
done

# 3. every embed reaches details() through redact() / strip_ansi (continuation lines joined).
bad="$(sed -e ':a' -e '/\\$/N; s/\\\n//; ta' "$REPORT" | command grep -nE '\|[[:space:]]*details[[:space:]]' | command grep -vE 'redact|strip_ansi' || true)"
if [[ -n "$bad" ]]; then fail "a block reaches details() without redact()/strip_ansi: $bad"
else ok "every block piped into details() goes through redact() or strip_ansi"; fi

if [[ "$fails" -gt 0 ]]; then echo "test-report-ansi: $fails failure(s)" >&2; exit 1; fi
echo "test-report-ansi: ok"
