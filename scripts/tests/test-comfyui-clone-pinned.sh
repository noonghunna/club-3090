#!/usr/bin/env bash
# test-comfyui-clone-pinned — services/comfyui/entrypoint.sh's clone_pinned must land a
# node on its pinned commit from whichever mirror still has it (club-3090#1394).
#
# WHY THIS TEST EXISTS
# --------------------
# The HiDream-O1 node's author deleted their GitHub account. The entrypoint cloned it
# with `git clone --depth 1 <dead url>` under `set -e`, so every FRESH ComfyUI install
# stopped at that line — and the maintainer rig never saw it, because its clone already
# existed. clone_pinned replaces that for gone-upstream nodes: pin a SHA, try mirrors in
# order, warn (not exit) if none has it.
#
# Offline: the "mirrors" are local bare repos (file:// URLs) plus a URL that does not
# exist. Every case runs the REAL function text extracted from the entrypoint, under the
# entrypoint's own `set -euo pipefail`.
#
# Contract:
#   1. fresh install: skips a dead mirror and one that lacks the SHA, lands on the SHA.
#   2. existing clone already at the SHA: NO network (every mirror dead) — a local patch
#      is reset away (so the entrypoint's patch step re-applies cleanly), HEAD stays.
#   3. existing SHALLOW clone at an OLDER commit, dead origin: moves to the SHA.
#   4. no mirror has it: returns 0 with a WARN — the rest of ComfyUI still boots.
#   5. a pre-existing non-git directory at the destination is never deleted.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRY="${ROOT}/services/comfyui/entrypoint.sh"
fails=0
fail() { echo "✗ $*" >&2; fails=$((fails+1)); }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0 HOME="$tmp/home"
mkdir -p "$HOME"
git config --global user.email t@example.invalid; git config --global user.name t
git config --global init.defaultBranch main
git config --global protocol.file.allow always

# The real function, extracted verbatim.
FN="$tmp/fn.sh"
awk '/^clone_pinned\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "$ENTRY" > "$FN"
command grep -q '^clone_pinned() {' "$FN" || { echo "✗ clone_pinned() not found in $ENTRY" >&2; exit 1; }
run() { bash -c "set -euo pipefail; source '$FN'; clone_pinned \"\$@\"; echo RC=\$?" _ "$@" 2>&1; }

# Mirror A carries c1 -> c2 (the pin). Mirror B is an unrelated repo. DEAD does not exist.
work="$tmp/work"; git init -q "$work"
echo one > "$work/f"; git -C "$work" add f; git -C "$work" commit -qm c1; C1="$(git -C "$work" rev-parse HEAD)"
echo two > "$work/f"; git -C "$work" commit -qam c2;                   PIN="$(git -C "$work" rev-parse HEAD)"
git clone -q --bare "$work" "$tmp/A.git"
other="$tmp/other"; git init -q "$other"; echo x > "$other/g"; git -C "$other" add g; git -C "$other" commit -qm x
git clone -q --bare "$other" "$tmp/B.git"
A="file://$tmp/A.git"; B="file://$tmp/B.git"; DEAD="file://$tmp/does-not-exist.git"

echo "--- 1. fresh install: skip dead + lacking mirrors, land on the pin ---"
out="$(run "$tmp/n1" "$PIN" "$DEAD" "$B" "$A")"
[[ "$(git -C "$tmp/n1" rev-parse HEAD 2>/dev/null)" == "$PIN" ]] || fail "1: not on the pin — $out"
[[ "$out" == *"mirror unusable: $DEAD"* ]] || fail "1: dead mirror not reported — $out"
[[ "$out" == *"lacks ${PIN:0:8}: $B"* ]]     || fail "1: lacking mirror not reported — $out"
[[ "$out" == *"from $A"* && "$out" == *"RC=0"* ]] || fail "1: did not report the serving mirror — $out"

echo "--- 2. existing clone at the pin: no network, local patch reset ---"
git clone -q "$A" "$tmp/n2"; git -C "$tmp/n2" remote set-url origin "$DEAD"
echo patched > "$tmp/n2/f"
out="$(run "$tmp/n2" "$PIN" "$DEAD")"                       # every mirror dead: must not matter
[[ "$(git -C "$tmp/n2" rev-parse HEAD)" == "$PIN" ]] || fail "2: HEAD moved — $out"
[[ "$(cat "$tmp/n2/f")" == "two" ]] || fail "2: local modification survived the reset"
[[ "$out" == *"pinned @ ${PIN:0:8}"* && "$out" != *"WARN"* ]] || fail "2: fast path not taken — $out"

echo "--- 3. existing shallow clone at an OLDER commit, dead origin: moves to the pin ---"
git clone -q --bare "$work" "$tmp/old.git"; git -C "$tmp/old.git" update-ref refs/heads/main "$C1"
git clone -q --depth 1 "file://$tmp/old.git" "$tmp/n3"; git -C "$tmp/n3" remote set-url origin "$DEAD"
[[ "$(git -C "$tmp/n3" rev-parse HEAD)" == "$C1" ]] || fail "3: fixture not at the older commit"
out="$(run "$tmp/n3" "$PIN" "$DEAD" "$A")"
[[ "$(git -C "$tmp/n3" rev-parse HEAD)" == "$PIN" ]] || fail "3: not moved to the pin — $out"

echo "--- 4. no mirror has it: WARN and continue, never exit ---"
out="$(run "$tmp/n4" "$PIN" "$DEAD" "$B")"
[[ "$out" == *"WARN: no mirror carries ${PIN:0:8}"* ]] || fail "4: no WARN — $out"
[[ "$out" == *"RC=0"* ]] || fail "4: aborted the entrypoint instead of continuing — $out"

echo "--- 5. a pre-existing non-git directory is never deleted ---"
mkdir -p "$tmp/n5"; echo keep > "$tmp/n5/user-file"
out="$(run "$tmp/n5" "$PIN" "$DEAD" "$A")"
[[ -f "$tmp/n5/user-file" ]] || fail "5: user's directory was deleted — $out"
[[ "$out" == *"WARN"* && "$out" == *"RC=0"* ]] || fail "5: expected a WARN and continue — $out"

if [[ "$fails" -gt 0 ]]; then
  echo "test-comfyui-clone-pinned: $fails failure(s)" >&2
  exit 1
fi
echo "test-comfyui-clone-pinned: ok"
