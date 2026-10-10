#!/usr/bin/env bash
#
# Guard: never pipe `nvidia-smi` or `docker logs` into an early-exiting
# `grep -q` in scripts/.
#
# Why (club-3090#1574): grep -q exits on its first match. The writer then takes
# a SIGPIPE on its next write, and under `pipefail` the pipeline returns 141, so
# a FOUND match reads as a miss. bench.sh, report.sh, launch.sh and verify-full.sh
# all run with pipefail. A 4× 3090 rig with its NVLink bridge in use was reported
# as "driver P2P grant: REFUSED" in 7 of 8 bench reports, and launch.sh's
# host-side NVLink probe could miss the bridge the same way. It is a race, not a
# certainty, which is why it survived: `set -o pipefail; nvidia-smi topo -m |
# grep -q GPU0` returned 141 on 300 of 300 runs on the reference rig, but on a
# small output it sometimes wins.
#
# Fix shape: capture, then match — `out="$(nvidia-smi …)" || true;
# command grep -q PAT <<<"$out"` — or count, which reads to the end:
# `[[ "$(docker logs c | command grep -c PAT)" -gt 0 ]]`.
#
# Scope: these two producers, because they are the ones measured to write in
# pieces. `docker ps`, `docker info`, `benchlocal-cli --help` and echo/printf
# pipes came back 0 on every run (one write, smaller than the pipe buffer).
# `grep -m1` on the same producers loses only the exit status, not the value it
# prints, so it is not flagged. scripts/tests/ is out of scope.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Lines that are not comments and pipe one of the producers into grep -q.
PATTERN='^[^#]*(nvidia-smi|docker[[:space:]]+logs)[^#]*\|[[:space:]]*(command[[:space:]]+)?grep[[:space:]]+(-[a-zA-Z]*q|--quiet|--silent)'
scan() {  # scan <root> → offending lines as path:line:text
  command grep -rnE "$PATTERN" "$1/scripts" --include='*.sh' 2>/dev/null \
    | command grep -v "^$1/scripts/tests/" || true
}

# --- positive control FIRST: a scan that finds nothing is indistinguishable from
# a clean tree, so prove it still sees both producers and both spellings.
probe="$(mktemp -d)"; trap 'rm -rf "$probe"' EXIT
mkdir -p "$probe/scripts/lib" "$probe/scripts/tests"
cat > "$probe/scripts/planted.sh" <<'EOS'
if nvidia-smi topo -m 2>/dev/null | command grep -qP '\bNV[0-9]+\b'; then :; fi
docker logs "$c" 2>&1 | grep -qiE "mmproj" && x=1
EOS
planted="$(scan "$probe" | command grep -c 'planted.sh' || true)"
if [[ "$planted" != "2" ]]; then
  echo "FAIL: positive control — expected 2 planted violations, found $planted" >&2
  exit 1
fi
echo "  ✓ scanner detects planted nvidia-smi and docker-logs pipes into grep -q"

# --- negative control: the fixed shapes, comments and tests must not register.
cat > "$probe/scripts/lib/fixed.sh" <<'EOS'
# Never `nvidia-smi … | grep -q` (comment explaining the rule)
topo="$(nvidia-smi topo -m 2>/dev/null)" || true
if command grep -qP '\bNV[0-9]+\b' <<<"$topo"; then :; fi
if [[ "$(docker logs "$c" 2>&1 | command grep -ciE "mmproj")" -gt 0 ]]; then :; fi
v="$(nvidia-smi 2>/dev/null | command grep -m1 -oE 'CUDA Version: [0-9.]+' || true)"
EOS
printf '%s\n' 'nvidia-smi topo -m | grep -q NV' > "$probe/scripts/tests/asserts-on-output.sh"
false_hits="$(scan "$probe" | command grep -cv 'planted.sh' || true)"
if [[ "$false_hits" -ne 0 ]]; then
  echo "FAIL: negative control — scanner fired on fixed shapes / comments / tests:" >&2
  scan "$probe" | command grep -v 'planted.sh' >&2
  exit 1
fi
echo "  ✓ captured-then-matched, grep -c, grep -m1, comments and scripts/tests/ do not register"

# --- the real tree --------------------------------------------------------------
hits="$(scan "$ROOT")"
if [[ -n "$hits" ]]; then
  echo "FAIL: nvidia-smi / docker logs piped into grep -q — under pipefail a found match reads as a miss (#1574):" >&2
  printf '%s\n' "${hits#"$ROOT/"}" | sed "s|$ROOT/||" >&2
  echo "Fix: capture first — out=\"\$(nvidia-smi …)\" || true; command grep -q PAT <<<\"\$out\" — or count with grep -c." >&2
  exit 1
fi
echo "test-no-grep-q-sigpipe: ok"
