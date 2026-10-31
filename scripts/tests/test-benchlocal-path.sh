#!/usr/bin/env bash
# test-benchlocal-path — benchlocal-cli is found in ~/.local/bin when this shell's PATH lacks it, and a rebench
# that asks for the 8-pack checks for it before running anything.
#
# Why this exists: #1888's 512K rebench ran every other step, then failed its 8-pack at "benchlocal-cli not
# found on $PATH", with benchlocal-cli installed and working in the user's own shell. pip --user, uv tool and
# pipx put it in ~/.local/bin, which a non-login shell often lacks.
#
#   1. club_benchlocal_on_path: finds it in ~/.local/bin (or $UV_TOOL_BIN_DIR first), puts that on PATH;
#   2. quality-test.sh runs with it only in ~/.local/bin, and still refuses (exit 127) when it is nowhere;
#   3. rebench-full.sh --with-8pack-thinking stops at the start when it is nowhere, and proceeds when it's in
#      ~/.local/bin.
# Runs with a fake HOME and PATH=/usr/bin:/bin, so the rig's own benchlocal-cli can't leak in.
# ⚠️ Against the pre-fix tree this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail() { echo "FAIL: test-benchlocal-path — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
MINPATH="/usr/bin:/bin"
if [[ -n "$(PATH=$MINPATH command -v benchlocal-cli || true)" ]]; then
  echo "SKIP: benchlocal-cli is installed system-wide, so 'nowhere' can't be staged"; exit 0
fi

mkbl() {  # <dir> — a stub benchlocal-cli that answers `list`
  mkdir -p "$1"
  printf '#!/usr/bin/env bash\n[[ "$1" == list ]] && echo toolcall-15\nexit 0\n' > "$1/benchlocal-cli"
  chmod +x "$1/benchlocal-cli"
}
home_with="$tmp/home-with"; mkbl "$home_with/.local/bin"
home_without="$tmp/home-without"; mkdir -p "$home_without"
uvbin="$tmp/uvbin"; mkbl "$uvbin"

# --- 1. the helper -----------------------------------------------------------------
h() { env -i HOME="$1" PATH="$MINPATH" ${2:+UV_TOOL_BIN_DIR="$2"} bash -c \
  'source scripts/lib/benchlocal-path.sh; if club_benchlocal_on_path; then echo "rc=0 bl=$(command -v benchlocal-cli)"; else echo rc=1; fi' 2>&1; }
out="$(h "$home_with")"
[[ "$out" == *"rc=0 bl=$home_with/.local/bin/benchlocal-cli"* && "$out" == *"[benchlocal] using $home_with/.local/bin/benchlocal-cli"* ]] \
  || fail "~/.local/bin not found or not announced: $out"
out="$(h "$home_with" "$uvbin")"
[[ "$out" == *"rc=0 bl=$uvbin/benchlocal-cli"* ]] || fail "UV_TOOL_BIN_DIR must win over ~/.local/bin: $out"
[[ "$(h "$home_without")" == "rc=1" ]] || fail "nowhere: expected rc=1, got '$(h "$home_without")'"
pass "club_benchlocal_on_path: ~/.local/bin found and announced, \$UV_TOOL_BIN_DIR first, nothing found = 1"

# --- 2. quality-test.sh ------------------------------------------------------------
qt() { env -i HOME="$1" PATH="$MINPATH" CLUB3090_CONFIG_DIR="$CLUB3090_CONFIG_DIR" PYTHONUTF8=1 \
  PREFLIGHT_NO_AUTODETECT=1 URL=http://127.0.0.1:9 timeout 120 bash scripts/quality-test.sh --list-packs 2>&1; }
set +e
out="$(qt "$home_with")"; rc=$?
[[ $rc -eq 0 && "$out" == *toolcall-15* && "$out" == *"[benchlocal] using"* ]] \
  || { echo "$out" | tail -8 >&2; fail "quality-test.sh with benchlocal-cli only in ~/.local/bin (rc=$rc)"; }
out="$(qt "$home_without")"; rc=$?
set -e
[[ $rc -eq 127 && "$out" == *"Installed already?"* ]] || { echo "$out" | tail -8 >&2; fail "nowhere: expected exit 127 with the PATH hint (rc=$rc)"; }
pass "quality-test.sh: runs with benchlocal-cli only in ~/.local/bin; still refuses, with the PATH hint, when it's nowhere"

# --- 3. rebench-full.sh --with-8pack-thinking ---------------------------------------
rb() { env -i HOME="$1" PATH="$MINPATH" CLUB3090_CONFIG_DIR="$CLUB3090_CONFIG_DIR" PYTHONUTF8=1 \
  timeout 120 bash scripts/rebench-full.sh --url http://127.0.0.1:9 --with-8pack-thinking=off 2>&1; }
set +e
out="$(rb "$home_without")"; rc=$?
[[ $rc -eq 2 && "$out" == *"--with-8pack-thinking needs benchlocal-cli"* && "$out" != *verify-full* ]] \
  || { echo "$out" | tail -8 >&2; fail "rebench without benchlocal-cli must stop at the start (rc=$rc)"; }
out="$(rb "$home_with")"; rc=$?
set -e
[[ "$out" == *"[benchlocal] using"* && "$out" != *"needs benchlocal-cli"* ]] \
  || { echo "$out" | tail -8 >&2; fail "rebench with benchlocal-cli in ~/.local/bin must get past the check"; }
pass "rebench-full.sh: stops before any step when the 8-pack can't run; gets past the check with ~/.local/bin"

echo "test-benchlocal-path: ok"
