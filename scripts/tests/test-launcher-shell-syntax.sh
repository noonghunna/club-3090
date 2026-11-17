#!/usr/bin/env bash
# test-launcher-shell-syntax — every model launcher script parses under sh AND bash.
#
# The composes run the launchers with the image's /bin/sh (dash on every image we
# ship), and dash accepted an apostrophe inside "${1:?…}" ("is this script's
# argument") that bash outside POSIX mode cannot parse. All three Project Maya
# launchers carried it: `bash -n` failed on each, and `bash <launcher>` would not
# run them at all. This guard parses every tracked launcher with both shells.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0
checked=0

# --- positive control: the checker must catch the construct that slipped through ---
probe="$(mktemp)"
trap 'rm -f "$probe"' EXIT
printf '%s\n' 'TAG="${1:?the quant is this script'"'"'s argument}"' > "$probe"
if bash -n "$probe" 2>/dev/null; then
  echo "✗ positive control: bash -n accepted an apostrophe inside \"\${1:?…}\" — this check can't catch the bug it exists for" >&2
  exit 1
fi

# --- every tracked launcher, both shells ------------------------------------------
while IFS= read -r f; do
  checked=$((checked+1))
  for sh_ in sh bash; do
    if ! err="$("$sh_" -n "$f" 2>&1)"; then
      echo "✗ $sh_ -n $f: ${err%%$'\n'*}" >&2
      fails=$((fails+1))
    fi
  done
done < <(git ls-files 'models/*/launcher/*.sh' 'models/*/*/launcher/*.sh')

[[ $checked -eq 0 ]] && { echo "✗ found no launcher scripts — the search is wrong, not the tree" >&2; exit 1; }
if [[ $fails -gt 0 ]]; then
  echo "$fails launcher parse failure(s)" >&2
  exit 1
fi
echo "✓ $checked launcher script(s) parse under sh and bash (positive control: bash -n rejects the old construct)"
