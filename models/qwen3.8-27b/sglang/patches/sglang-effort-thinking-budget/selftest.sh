#!/usr/bin/env bash
# selftest.sh [image] — offline drift guard for patch sglang-effort-thinking-budget.
#
# The patches.yml drift_guard.check. Run it on every SGLang pin bump BEFORE booting the new
# image: it needs Docker but no GPU and starts nothing that serves. In ONE throwaway container
# (`docker run --rm --network none`, default lmsysorg/sglang:v0.5.21) it runs:
#   * scripts/tests/test-effort-budget.sh with the image's own python (the module's unit tests);
#   * selftest_container.py — the stock-behaviour control, the negative controls (anchor
#     removed / duplicated, half-patched tree, malformed map, module not mounted), install.sh
#     twice (second = byte-for-byte no-op), the AST call-site checks, and request scenarios
#     through the REAL patched SGLang functions (see that file's docstring for what is stubbed).
# A failure here on a new pin means re-anchor (README.md "Re-anchoring"), not "skip the check".
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

IMAGE="${1:-lmsysorg/sglang:v0.5.21}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
[ -f "$ROOT/scripts/lib/effort_budget.py" ] || { echo "selftest: repo root not found from $HERE" >&2; exit 2; }
command -v docker >/dev/null || { echo "selftest: docker is required" >&2; exit 2; }

echo "[sglang-effort-thinking-budget] selftest on $IMAGE (no GPU, no network)"
set +e
docker run --rm --network none --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUTF8=1 \
  -v "$ROOT:/repo:ro" \
  -v "$ROOT/scripts/lib/effort_budget.py:/etc/club3090/effort_budget.py:ro" \
  -v "$HERE:/etc/club3090/effort-thinking-budget:ro" \
  "$IMAGE" -c '
    echo "--- 0. scripts/tests/test-effort-budget.sh on the image python ($(python3 -V)) ---"
    bash /repo/scripts/tests/test-effort-budget.sh || exit 1
    python3 /etc/club3090/effort-thinking-budget/selftest_container.py
  '
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "[sglang-effort-thinking-budget] selftest PASS on $IMAGE"
else
  echo "[sglang-effort-thinking-budget] selftest FAIL on $IMAGE (rc=$rc)" >&2
fi
exit "$rc"
