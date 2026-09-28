#!/usr/bin/env bash
#
# Guard: every published port mapping must route through the full two-level
# override chain `${ESTATE_PORT:-${PORT:-<default>}}`.
#
# WHY (club-3090#1293): `ESTATE_PORT` is injected by
# scripts/lib/profiles/estate_cli.py and scripts/lib/generate_compose.py when the
# estate RELOCATES a slug — running two concurrently, or dodging a busy port.
# `PORT` is the plain user knob. ESTATE_PORT must win.
#
# 12 of 158 composes omitted the ESTATE_PORT level and bound `${PORT:-<default>}`
# directly. The estate CLI injected ESTATE_PORT, the compose ignored it, and the
# container silently bound the DEFAULT port. Nothing errored — the relocation
# just did not happen.
#
# The failure mode does not read as "wrong port". Hit live: ESTATE_PORT=8199 was
# set, the compose bound 8101, and a probe against 8199 returned `Connection
# refused` against a perfectly healthy container. First read was "the server
# crashed"; it took `docker port` to see the truth. A silently-relocated-nowhere
# endpoint looks exactly like a dead one.
#
# A further 2 composes had ESTATE_PORT but no `PORT` fallback, so they were
# missing the plain knob every other compose gives the user.
#
# ⚠️ Must FAIL against the pre-fix tree (12 + 2 offenders), or it is asserting
# the wrong thing.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
# #779: this gate shells out to python3, which decodes reads/argv with the
# LOCALE codec unless UTF-8 mode is on — and these composes are full of unicode.
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
scan="$tmp/scan.py"

# ONE scanner, used for both the control and the real run. A control that
# exercises different code proves nothing about the code that matters.
cat > "$scan" <<'PYEOF'
import glob, os, re, sys

root = sys.argv[1]
# The canonical host-side expression, as emitted by generate_compose.py:
#   "${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-${PORT:-NNNN}}:<container port>"
CANON = re.compile(r'\$\{ESTATE_PORT:-\$\{PORT:-[^}]*\}\}')

bad, seen = [], 0
for f in sorted(glob.glob(os.path.join(root, "models/**/*.yml"), recursive=True)):
    raw = open(f, encoding="utf-8", errors="replace").read()
    rel = os.path.relpath(f, root)
    in_ports = False
    for line in raw.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            continue                      # prose, not a mapping
        if re.match(r'^ports:\s*$', s):
            in_ports = True
            continue
        if not in_ports:
            continue
        if s.startswith("- "):
            seen += 1
            if not CANON.search(s):
                bad.append("%s|%s" % (rel, s))
            continue
        in_ports = False
print("SEEN %d" % seen)
for b in bad:
    print("BAD " + b)
PYEOF

# --- positive control FIRST -------------------------------------------------
# A scan that silently matches nothing is indistinguishable from a clean tree.
mkdir -p "$tmp/probe/models/planted/eng/compose/dual/q"
printf '%s\n' \
  '    ports:' \
  '      - "${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-${PORT:-8000}}:8000"' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/ok.yml"
printf '%s\n' \
  '    ports:' \
  '      - "${BIND_HOST:-0.0.0.0}:${PORT:-8000}:8000"' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/portonly.yml"
printf '%s\n' \
  '    ports:' \
  '      - "${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-8000}:8000"' \
  > "$tmp/probe/models/planted/eng/compose/dual/q/estateonly.yml"

ctl="$(python3 "$scan" "$tmp/probe")"
[[ "$(printf '%s\n' "$ctl" | command grep -c '^BAD ')" == "2" ]] || {
  echo "FAIL: positive control — scanner should flag BOTH the PORT-only and the ESTATE_PORT-only mapping" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
[[ "$ctl" == *"SEEN 3"* ]] || {
  echo "FAIL: positive control — scanner should have seen 3 planted mappings" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
echo "  ✓ scanner flags a PORT-only and an ESTATE_PORT-only mapping, passes the canonical chain"

# --- the actual assertion ---------------------------------------------------
out="$(python3 "$scan" "$ROOT")"
seen="$(printf '%s\n' "$out" | command sed -n 's/^SEEN //p')"
bad="$(printf '%s\n' "$out" | command grep '^BAD ' || true)"

if [[ -z "$seen" || "$seen" -lt 1 ]]; then
  echo "FAIL: scanned no published port mappings — the search is wrong, not the tree" >&2
  exit 1
fi

fails=0
if [[ -n "$bad" ]]; then
  echo "FAIL: port mapping(s) not on the canonical \${ESTATE_PORT:-\${PORT:-<default>}} chain:" >&2
  printf '%s\n' "$bad" | command sed 's/^BAD /        /; s/|/\n            /' >&2
  echo "" >&2
  echo "  Without the ESTATE_PORT level the estate CLI's relocation is silently ignored" >&2
  echo "  and the container binds the default instead (club-3090#1293)." >&2
  fails=$((fails+1))
fi

# The generator is the source of this contract — if it stops emitting the chain,
# every newly generated compose regresses silently.
command grep -qF 'ESTATE_PORT:-${{PORT:-' "$ROOT/scripts/lib/generate_compose.py" \
  || { echo "FAIL: generate_compose.py no longer emits the \${ESTATE_PORT:-\${PORT:-…}} chain" >&2
       fails=$((fails+1)); }

[[ "$fails" -eq 0 ]] || exit 1
echo "PASS: test-compose-estate-port — $seen/$seen published port mapping(s) honour ESTATE_PORT then PORT (#1293)"
