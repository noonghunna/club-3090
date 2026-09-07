#!/usr/bin/env bash
# Guard: `switch.sh --local` shows only the models YOU registered, and every
# local row is MARKED in the normal listing (#1202 P5).
#
# Local rows used to be indistinguishable: nothing rendered a provenance marker.
# That mattered little while they lived under a `local/` namespace, and matters a
# lot now that P3 gave them the same `<engine>/<name>` shape as curated rows.
#
# The bug this pins first: listing is TWO passes, one that counts and one that
# renders. Filtering only the first produced a header reading "variants: 1" above
# every curated row — a filter that reported success while doing nothing.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0

# 1. provenance side channel: silent on a pristine checkout, and does not widen
#    the tab row (which is index-coupled across five readers).
out="$(bash -c 'source scripts/lib/registry-emit.sh; registry_local_slugs "$PWD"' 2>/dev/null)"
if [[ -n "$out" && "$(find scripts/lib/profiles-local -type f 2>/dev/null | command grep -cv -e README -e gitignore)" == "0" ]]; then
  echo "  FAIL registry_local_slugs reported rows with no local layer present"; rc=1
else
  echo "  ok   provenance channel silent with no local layer"
fi

# 2. the flag exists and is wired to BOTH passes
if command grep -q -- '--local) LIST_REQUESTED=1; LIST_LOCAL=1' scripts/switch.sh; then
  echo "  ok   --local flag parsed"
else
  echo "  FAIL --local flag missing"; rc=1
fi
n="$(command grep -c 'LIST_LOCAL:-0' scripts/switch.sh)"
if [[ "$n" -ge 2 ]]; then
  echo "  ok   --local applied in both the count and render passes ($n sites)"
else
  echo "  FAIL --local applied in only $n pass — the header would filter while rows do not"
  rc=1
fi

# 3. the marker is rendered, and only for local rows
if command grep -q 'ann " · local"' scripts/switch.sh; then
  echo "  ok   local rows carry a provenance marker"
else
  echo "  FAIL local rows render no marker"; rc=1
fi

# 4. shadowed rows are surfaced, not silently dropped — the "listed but marked"
#    promise from #1202 P2. A shadowed row is absent from the merged registry by
#    design (core wins), so a footer is the only place it can appear.
if command grep -q 'shadowed local slug' scripts/switch.sh; then
  echo "  ok   shadowed local slugs surfaced in a footer"
else
  echo "  FAIL shadowed local slugs would vanish silently"; rc=1
fi

# 5. NEGATIVE CONTROL: a pristine listing must be unchanged — no marker, no
#    footer, and the same variant count as without the feature.
listing="$(LIST_ALL=1 bash scripts/switch.sh --list 2>/dev/null)"
if [[ "$listing" == *"· local"* ]]; then
  echo "  FAIL a pristine checkout rendered a local marker"; rc=1
elif [[ "$listing" == *"shadowed local slug"* ]]; then
  echo "  FAIL a pristine checkout rendered the shadowed footer"; rc=1
elif [[ "$listing" != *"Available variants"* ]]; then
  echo "  FAIL --list produced no listing at all; the checks above prove nothing"; rc=1
else
  echo "  ok   pristine listing unchanged (no marker, no footer)"
fi

[[ "$rc" == "0" ]] && echo "PASS: --local filters and local rows are marked" \
                   || echo "FAIL: switch --local regression"
exit "$rc"
