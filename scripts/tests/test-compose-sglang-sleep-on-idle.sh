#!/usr/bin/env bash
# test-compose-sglang-sleep-on-idle — every compose that launches an SGLang server
# passes --sleep-on-idle.
#
# WHY THIS TEST EXISTS
# --------------------
# Stock SGLang busy-polls its scheduler event loop on every TP rank while nothing
# is being served: ~1 CPU core per GPU, all day (measured on
# sgl/qwen38-27b-dual-fast, v0.5.20: 192 % of a core idle vs 3 % for vLLM). The
# upstream flag that stops it defaults OFF, so a new SGLang compose copied from an
# upstream recipe silently brings the busy-poll back (#1443).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail=0; n=0
while IFS= read -r f; do
  n=$((n + 1))
  # the flag must be on the launch line's continuation, not just in a comment
  if ! command grep -qE '^[[:space:]]+--sleep-on-idle([[:space:]]|\\|$)' "$f"; then
    echo "✗ $f launches SGLang without --sleep-on-idle" >&2; fail=1
  fi
done < <(command grep -rlE 'sglang\.launch_server|sglang serve' models --include='*.yml' | sort)
[[ $n -gt 0 ]] || { echo "✗ found no SGLang composes — the scan itself is broken" >&2; exit 1; }
[[ $fail -eq 0 ]] && echo "test-compose-sglang-sleep-on-idle: ok ($n SGLang composes, all pass --sleep-on-idle)"
exit $fail
