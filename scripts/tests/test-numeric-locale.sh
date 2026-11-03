#!/usr/bin/env bash
# test-numeric-locale — numbers parse and print with a dot whatever the user's locale (#1926, #159).
#
# Why this exists (2026-11-03): on a decimal-comma locale (es_ES, de_DE, fr_FR …) mawk, which is
# Debian's and Ubuntu's default awk, parses "1.65" as 1 and prints 8.00 as "8,00", and bash's printf
# rejects "9.0". #1926's bench (es_ES) read QEngine's 10.02 GB of host-backed experts as "8,00 GB"
# (3.81 + 1.65 + 4.56 summed as 3 + 1 + 4) and printed mem_total_gib=188,5. gawk, our rig's awk,
# ignores the locale's decimal point unless --use-lc-numeric, which is why nothing here caught it.
# launch.sh had forced LC_NUMERIC=C since #159; nothing else did.
#
#   (a) every scripts/*.sh and scripts/lib/*.sh that calls awk or a printf with a %.Nf format carries
#       the numeric-locale line before its first such use: the invariant that decays as scripts are added;
#   (b) that line also moves an LC_ALL, which would override LC_NUMERIC, into the other categories;
#   (c) functional, under a real es_ES.UTF-8 (built with localedef) set through LANG, LC_NUMERIC and
#       LC_ALL in turn, with mawk as awk: capture.sh's QEngine offload reader sums 10.02 GB, and bash
#       printf "%.2f" 9.0 prints 9.00. Skipped, with a notice, where localedef or mawk is missing.
# ⚠️ Against the pre-fix tree this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail() { echo "FAIL: test-numeric-locale — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# --- (a) the line, before the first numeric use, in every file that does numeric work --------------
python3 - <<'PY' || fail "a script or lib does numeric work without the numeric-locale line before it (see above)"
import glob, re, sys
LINE = ('[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do '
        'export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C')
USE = re.compile(r'(^|[^A-Za-z_])awk([^A-Za-z_]|$)|printf.*%[0-9]*\.[0-9]+f')
bad, n = [], 0
for p in sorted(glob.glob("scripts/*.sh") + glob.glob("scripts/lib/*.sh")):
    lines = open(p, encoding="utf-8").read().split("\n")
    uses = [i for i, l in enumerate(lines) if USE.search(l) and not l.lstrip().startswith("#") and l.strip() != LINE]
    if not uses:
        continue
    n += 1
    at = next((i for i, l in enumerate(lines) if l.strip() == LINE), None)
    if at is None:
        bad.append(f"{p}: no numeric-locale line (first numeric use at line {uses[0] + 1})")
    elif at > uses[0]:
        bad.append(f"{p}: the line (line {at + 1}) comes after the first numeric use (line {uses[0] + 1})")
for b in bad:
    print("  ⛔", b)
if bad:
    sys.exit(1)
print(f"  ✓ {n} scripts and libs do numeric work, each pins the numeric locale first")
PY

# --- (b) LC_ALL moves to the other categories; LC_NUMERIC ends as C ----------------------------------
got="$(env LC_ALL=C.UTF-8 LC_CTYPE=ignored bash -c 'source scripts/lib/capture.sh
  printf "%s|%s|%s|%s" "${LC_ALL-unset}" "$LC_NUMERIC" "$LC_CTYPE" "$LANG"')"
[[ "$got" == "unset|C|C.UTF-8|C.UTF-8" ]] \
  || fail "LC_ALL must give its value to the other categories and leave LC_NUMERIC=C: got '$got'"
pass "an LC_ALL is moved into the other categories (LC_CTYPE keeps its value), LC_NUMERIC ends as C"

# --- (c) functional, under a real decimal-comma locale, with mawk as awk ----------------------------
if ! command -v localedef >/dev/null 2>&1 || ! command -v mawk >/dev/null 2>&1; then
  echo "  ⚠ skipped the functional leg: needs localedef and mawk (gawk ignores the locale's decimal point)"
  echo "test-numeric-locale: ok"
  exit 0
fi
L="$tmp/locale"; mkdir -p "$L" "$tmp/bin"
localedef -f UTF-8 -i es_ES "$L/es_ES.UTF-8" >/dev/null 2>&1 || true
[[ -d "$L/es_ES.UTF-8" ]] || { echo "  ⚠ skipped the functional leg: localedef could not build es_ES.UTF-8"; echo "test-numeric-locale: ok"; exit 0; }
ln -s "$(command -v mawk)" "$tmp/bin/awk"
# The positive control: mawk under this locale really does read a dot as the end of the number.
ctl="$(LOCPATH="$L" LC_ALL=es_ES.UTF-8 "$tmp/bin/awk" 'BEGIN { printf "%.2f", "1.65" + 0 }')"
[[ "$ctl" == "1,00" ]] || fail "positive control: mawk under es_ES.UTF-8 should read \"1.65\" as 1 and print 1,00, got '$ctl' — the leg below would prove nothing"
printf '%s\n' "==== qengine server start: QEngine 1.3.4.3, pid 1, build 5083e04 (/app), log /x.log" \
  "[flash] GPU1 expert budget: 17.62 GiB free -> 35138 of 38400 routed groups on the device, 3262 host-backed (1.65 GB)" \
  "[flash] GPU2 expert budget: 16.88 GiB free -> 33242 of 42240 routed groups on the device, 8998 host-backed (4.56 GB)" \
  "[flash] GPU0 expert budget: 17.55 GiB free -> 34728 of 42240 routed groups on the device, 7512 host-backed (3.81 GB)" \
  > "$tmp/qe-3cards.log"
for how in LANG LC_NUMERIC LC_ALL; do
  got="$(env -u LC_ALL -u LC_NUMERIC PATH="$tmp/bin:$PATH" LOCPATH="$L" LANG=C.UTF-8 "$how=es_ES.UTF-8" \
    bash -c 'source scripts/lib/capture.sh; CAP_LOG="$1"; CONTAINER=; cap_offload_detected ""; printf "|%.2f" 9.0' _ "$tmp/qe-3cards.log" 2>&1)"
  [[ "$got" == $'boot log (QEngine: 10.02 GB of experts host-backed over 3 cards)\n|9.00' ]] \
    || fail "with $how=es_ES.UTF-8 and mawk: got '$got' (#1926 read '8,00 GB')"
done
pass "under es_ES.UTF-8 via LANG, LC_NUMERIC and LC_ALL, with mawk: 10.02 GB (not 8,00) and printf 9.00"

echo "test-numeric-locale: ok"
