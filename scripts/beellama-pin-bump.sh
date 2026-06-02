#!/usr/bin/env bash
# beellama-pin-bump — bump the centralized beellama image pin to the latest published
# tag from Anbeeld's GHCR (engines/beellama-local.yml install.spec, injected as
# BEELLAMA_IMAGE across all beellama composes).
#
# Run MANUALLY, on demand. It does NOT push: it discovers the newest commit-pinned tag,
# rewrites the one install.spec line, prints the diff, and reminds you to validate +
# open a PR. Per the engine-pin policy, pins bump via PR with re-validation, never
# silently (mirrors docs/NIGHTLY_BUMP_RUNBOOK.md for the vLLM nightly pin).
#
# Usage:
#   scripts/beellama-pin-bump.sh                 # track server-cuda-v0.3.0-*, bump + show diff
#   scripts/beellama-pin-bump.sh --check         # dry-run: show latest vs current, no edit
#   scripts/beellama-pin-bump.sh --channel server-cuda   # track a different tag prefix (e.g. a stable channel)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="$HERE/lib/profiles/engines/beellama-local.yml"
PKG="users/Anbeeld/packages/container/beellama.cpp/versions"
REPO_IMG="ghcr.io/anbeeld/beellama.cpp"
CHANNEL="server-cuda-v0.3.0"   # commit-pinned CUDA channel to track (NOT the moving -dev alias)
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --channel) CHANNEL="$2"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac; shift
done

command -v gh >/dev/null || { echo "ERROR: gh CLI required (GHCR tag discovery)" >&2; exit 3; }
[ -f "$PROFILE" ] || { echo "ERROR: engine profile not found: $PROFILE" >&2; exit 3; }

# Newest commit-pinned tag on the channel (GHCR versions are newest-first; skip the moving -dev alias).
latest=$(gh api "$PKG" --paginate \
  --jq ".[].metadata.container.tags[]? | select(startswith(\"${CHANNEL}-\")) | select(test(\"-dev$\")|not)" \
  2>/dev/null | head -1)
[ -n "$latest" ] && [ "$latest" != "null" ] || { echo "ERROR: no commit-pinned '${CHANNEL}-*' tag found on $REPO_IMG" >&2; exit 4; }
new_spec="$REPO_IMG:$latest"

cur_spec=$(grep -oE "spec: \S+" "$PROFILE" | head -1 | awk '{print $2}')
echo "[pin-bump] channel : $CHANNEL"
echo "[pin-bump] current : $cur_spec"
echo "[pin-bump] latest  : $new_spec"
if [ "$cur_spec" = "$new_spec" ]; then echo "[pin-bump] already up to date — nothing to do."; exit 0; fi
if [ "$CHECK" -eq 1 ]; then echo "[pin-bump] --check: would bump current → latest (no edit made)."; exit 0; fi

# Rewrite the single install.spec line (everything else in the profile untouched).
sed -i -E "s#^(  spec: ).*#\1${new_spec}#" "$PROFILE"
echo "[pin-bump] bumped install.spec → $new_spec"
echo
echo "NEXT (engine-pin policy — validate, then PR; do NOT push silently):"
echo "  1. git switch -c chore/beellama-pin-\$(echo $latest | grep -oE '[a-f0-9]{6,}$' || echo bump)"
echo "  2. validate on a 3090: switch.sh --force beellama/gemma-dflash-dual + verify-full (MODEL/URL set)"
echo "  3. for t in scripts/tests/*.sh; do bash \"\$t\"; done   # gate stays green"
echo "  4. commit + open a PR (a human merges)."
