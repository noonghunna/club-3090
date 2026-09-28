#!/usr/bin/env bash
# test-envelope-matrix — #1363. Resolves EVERY registry slug against EVERY
# hardware profile and asserts the launcher contract holds.
#
# WHY THIS EXISTS. #1361 cannot be fixed by widening the launchers' slug-prefix
# gate, because three injectors fire from registry fields and would go live on
# 73 slugs at once. A simulation found MOE_RESERVE_MB reaching 110 (slug,card)
# pairs with NO allowlist arm in either launcher -- and an unallowlisted key is
# not a silent no-op, it is `exit 2`: an unlaunchable slug.
#
# Four properties:
#   A TOTAL      resolve_variant_pin must not raise for a registered slug
#   B ALLOWLIST  emitted keys accepted by BOTH launchers' case arms
#   C CONSUMED   emitted keys appear as ${KEY:- in that slug's own compose
#   D POST-FLIP  what WOULD reach the launcher once #1365 removes the raise,
#                plus a STATIC check of keys emittable by construction
#
# ⚠️ B passes today only because 73 slugs raise BEFORE they can emit -- the raise
# masks the landmine. D is the one that sees it, which is why D is asserted now
# rather than after the gate flips and a user finds it.
#
# Allowlists are PARSED from both launchers, never duplicated here: a key added
# to one and not the other still hard-fails, and a copy would drift.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/tests/fixtures/envelope-matrix.py || {
  echo "test-envelope-matrix: FAILED" >&2
  exit 1
}
echo "test-envelope-matrix: ok"
