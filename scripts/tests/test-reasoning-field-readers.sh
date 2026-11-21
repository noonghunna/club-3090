#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# test-reasoning-field-readers.sh — every reader of a reply's reasoning must accept
# BOTH field names.
#
# Servers name the reasoning part of a reply differently: llama.cpp and SGLang send
# `reasoning_content`; vLLM >= 0.34 sends `reasoning` (it keeps `reasoning_content`
# only as a deprecated INPUT name). A reader that checks only `reasoning_content`
# goes quietly wrong on vLLM, with no error:
#   - bench.sh started TTFT at the first ANSWER token, charging the whole reasoning
#     phase to prefill (Inkling-2-Mini on vLLM, 2026-11-21: 2-8 s TTFT on a
#     27-token prompt, decode_TPS 170-2,029; wall TPS was fine);
#   - preflight's reasoning-length probes read 0 at effort `high`, so the ON
#     ladder climbed past a level that was reasoning;
#   - verify-stress.sh and deep_context_probe.py timed their windows the same way.
#
# The rule: within two lines of every `.get("reasoning_content")` /
# `.get('reasoning_content')` under scripts/ (tests excluded), the same reader also
# asks for `.get("reasoning")` / `.get('reasoning')`. Writes (dict keys) are not
# reads and don't match. A reader that genuinely needs one name only can say so
# with a trailing `# reasoning-field: one-name-ok (<why>)` on the line.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bad=0
hits=0
while IFS=: read -r file line _; do
  [[ -n "$file" ]] || continue
  hits=$((hits + 1))
  window="$(sed -n "$((line > 2 ? line - 2 : 1)),$((line + 2))p" "$file")"
  if command grep -qE "get\(['\"]reasoning['\"]\)" <<<"$window"; then
    continue
  fi
  if sed -n "${line}p" "$file" | command grep -q 'reasoning-field: one-name-ok'; then
    continue
  fi
  echo "  ✗ ${file}:${line} reads reasoning_content but not reasoning:"
  echo "      $(sed -n "${line}p" "$file" | sed -E 's/^[[:space:]]+//' | cut -c1-140)"
  bad=$((bad + 1))
done < <(command grep -rnE "get\(['\"]reasoning_content['\"]\)" scripts \
           --include='*.sh' --include='*.py' --exclude-dir=tests 2>/dev/null || true)

if (( hits == 0 )); then
  echo "FAIL: found no reasoning_content readers at all — the scan pattern no longer matches the tree"
  exit 1
fi
if (( bad > 0 )); then
  echo "FAIL: ${bad} of ${hits} reasoning readers accept only \`reasoning_content\`"
  echo "  Fix: also read \`reasoning\` (vLLM >= 0.34), e.g."
  echo "       delta.get(\"content\") or delta.get(\"reasoning_content\") or delta.get(\"reasoning\")"
  exit 1
fi
echo "PASS: all ${hits} reasoning readers under scripts/ accept both reasoning_content and reasoning"
