#!/usr/bin/env bash
#
# Guard for preflight.sh's in-container model-path extractor (club-3090#1306).
#
# The extractor pulls /root/.cache/huggingface/<subdir> tokens out of compose
# files to check the weights are on disk. Its character class must exclude every
# character that cannot appear in a directory name, or it captures punctuation
# from the surrounding syntax and manufactures a path that can never exist.
#
# That is not hypothetical: the class excluded " ' whitespace and , but NOT
# backslash, and the gemma composes write the drafter path inside escaped JSON
# (...it-assistant\",\"num_speculative_tokens...). The capture became
# `gemma-4-26b-a4b-it-assistant\` and preflight refused to boot EIGHT composes
# across all three gemma models, reporting a file that was present as missing.
#
# This asserts on the real composes rather than a fixture: a fixture would have
# to reproduce the escaping to be meaningful, and then it tests the fixture.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

# The extractor, lifted verbatim from preflight.sh so the two cannot drift apart
# silently. If preflight.sh changes this regex, line 1 of this test must change
# with it — which is the point.
EXTRACT() { command grep -hv '^[[:space:]]*#' "$@" 2>/dev/null \
  | command grep -oE '/root/\.cache/huggingface/[^"'\''\\[:space:],]+' || true; }

# --- 1. the live regex in preflight.sh must exclude backslash ---------------
if command grep -c 'cache/huggingface/\[\^' "${ROOT}/scripts/preflight.sh" >/dev/null 2>&1 \
   && command grep 'cache/huggingface/\[\^' "${ROOT}/scripts/preflight.sh" | command grep -q '\\\\'; then
  ok "preflight.sh's extractor excludes backslash"
else
  bad "preflight extractor excludes backslash" "a backslash inside the character class" "not excluded"
fi

# --- 2. no captured subdir may contain a character illegal in a dir name ----
mapfile -t composes < <(find "${ROOT}/models" -name '*.yml' -not -path '*/_archive/*' 2>/dev/null | sort)
[[ ${#composes[@]} -gt 0 ]] || bad "composes found" "at least one compose" "none"
dirty=""
while IFS= read -r tok; do
  sub="${tok#/root/.cache/huggingface/}"
  case "$sub" in
    # `}` is NOT dirty: ${VAR:-default} is resolved downstream by
    # _preflight_compose_path_default. Only characters that can never survive
    # into a directory name are a defect here.
    *\\*|*\"*|*\'*) dirty+="  ${tok}"$'\n' ;;
  esac
done < <(EXTRACT "${composes[@]}" | sort -u)
if [[ -n "$dirty" ]]; then
  bad "captured model paths are usable" "no stray syntax characters" $'\n'"$dirty"
else
  ok "every captured model path is a usable directory name (${#composes[@]} composes scanned)"
fi

# --- 3. negative control: the OLD class must fail this very corpus ----------
# Without it, assertion 2 could pass because nothing matches at all.
OLD_EXTRACT() { command grep -hv '^[[:space:]]*#' "$@" 2>/dev/null \
  | command grep -oE '/root/\.cache/huggingface/[^"'\''[:space:],]+' || true; }
old_dirty=$(OLD_EXTRACT "${composes[@]}" | command grep -c '\\' || true)
if [[ "$old_dirty" -lt 1 ]]; then
  bad "negative control fires" "the pre-fix class to capture >=1 backslash path" "it captured none — this test proves nothing"
else
  ok "negative control: the pre-fix class captures ${old_dirty} unusable path(s) on this corpus"
fi

# --- 4. and the extractor must still find real paths (not just match nothing)
found=$(EXTRACT "${composes[@]}" | sort -u | wc -l)
[[ "$found" -ge 10 ]] || bad "extractor still finds paths" ">=10 distinct model paths" "$found"
ok "extractor resolves ${found} distinct model paths"

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-preflight-model-path-extract" >&2; exit 1; fi
echo "PASS: test-preflight-model-path-extract (club-3090#1306)"
