#!/usr/bin/env bash
#
# rerun-failed-packs.sh — re-test ONLY the packs that had failures in a prior
# quality run, and report which failures reproduced vs flipped (flakes).
#
# Why: benchlocal-cli's finest run granularity is --pack (no per-scenario run
# filter — see noonghunna/benchlocal-cli#82), and a --full re-run costs 1-2 h.
# After any full run, the variance question is almost always "are these
# failures real?" — answering it only needs the packs that failed.
#
# What it does:
#   1. Parses a saved RunResult JSON: failed scenarios -> the affected packs.
#   2. Re-runs ONLY those packs via quality-test.sh (keeping its hermes-env +
#      timeout guards), passing --previous-result so benchlocal emits its own
#      per-scenario delta, matching the ORIGINAL run's thinking mode from the
#      JSON (thinking_enabled) — no mode flag to get wrong.
#   3. Prints a consolidated verdict per original failure:
#      REPRODUCED (real) / FIXED (flake or environment) + any NEW regressions
#      in the re-run packs.
#
# Usage:
#   bash scripts/rerun-failed-packs.sh <result.json> [--repeat N] [extra quality-test.sh args...]
#
#   URL= / MODEL= env override the endpoint (default: autodetect via
#   quality-test.sh, same as any other run). --repeat N re-runs each scenario
#   N times (benchlocal aggregates at >=50% pass) — use N>=3 for flakiness
#   stats on the failing packs.
#
#   RERUN_DRY=1  print the plan (packs + mode + commands) without running.
#
# Output JSONs land next to the original as <original>.rerun.<pack>.json.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULT_JSON="${1:-}"
if [[ -z "$RESULT_JSON" || "$RESULT_JSON" == "-h" || "$RESULT_JSON" == "--help" ]]; then
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi
if [[ ! -f "$RESULT_JSON" ]]; then
  echo "✗ result JSON not found: $RESULT_JSON" >&2
  echo "  Fix: pass a saved RunResult (e.g. results/quality/quality-<ts>.json)" >&2
  exit 2
fi
shift

# --- 1. plan: failed packs + thinking mode from the original run -------------
PLAN="$(python3 - "$RESULT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
failed_packs = []
failures = []
for p in d.get("packs", []):
    pid = p.get("pack_id", "?")
    bad = [s for s in p.get("scenarios", []) if s.get("passed") is not True]
    if bad:
        failed_packs.append(pid)
        for s in bad:
            failures.append(f"{pid}/{s.get('id','?')}:{s.get('failure_mode','fail')}")
mode = "--enable-thinking" if d.get("thinking_enabled") else "--no-thinking"
print(mode)
print(" ".join(failed_packs))
print(" ".join(failures))
PY
)"
MODE_FLAG="$(sed -n '1p' <<<"$PLAN")"
FAILED_PACKS="$(sed -n '2p' <<<"$PLAN")"
ORIG_FAILURES="$(sed -n '3p' <<<"$PLAN")"

if [[ -z "$FAILED_PACKS" ]]; then
  echo "✓ no failed scenarios in $RESULT_JSON — nothing to re-run."
  exit 0
fi

echo "[rerun] original run: $RESULT_JSON  (mode: ${MODE_FLAG#--})"
echo "[rerun] failed packs: $FAILED_PACKS"
echo "[rerun] original failures: $(wc -w <<<"$ORIG_FAILURES" | tr -d ' ')"

if [[ "${RERUN_DRY:-0}" == "1" ]]; then
  for pack in $FAILED_PACKS; do
    echo "[dry] quality-test.sh --pack $pack $MODE_FLAG --previous-result $RESULT_JSON --save-json ${RESULT_JSON}.rerun.${pack}.json $*"
  done
  exit 0
fi

# --- 2. re-run each failed pack via the wrapper -------------------------------
for pack in $FAILED_PACKS; do
  echo
  echo "════ re-running pack: $pack ════"
  bash "${SCRIPT_DIR}/quality-test.sh" --pack "$pack" "$MODE_FLAG" \
    --previous-result "$RESULT_JSON" \
    --save-json "${RESULT_JSON}.rerun.${pack}.json" \
    "$@"
done

# --- 3. consolidated verdict ---------------------------------------------------
python3 - "$RESULT_JSON" $FAILED_PACKS <<'PY'
import json, sys
orig_path, packs = sys.argv[1], sys.argv[2:]
orig = json.load(open(orig_path, encoding="utf-8"))
orig_state = {}
for p in orig.get("packs", []):
    for s in p.get("scenarios", []):
        orig_state[(p["pack_id"], s["id"])] = s.get("passed") is True

repro, fixed, new_reg, missing = [], [], [], []
for pack in packs:
    try:
        rr = json.load(open(f"{orig_path}.rerun.{pack}.json", encoding="utf-8"))
    except FileNotFoundError:
        missing.append(pack)
        continue
    for p in rr.get("packs", []):
        for s in p.get("scenarios", []):
            key = (p["pack_id"], s["id"])
            now_pass = s.get("passed") is True
            was_pass = orig_state.get(key)
            tag = f"{key[0]}/{key[1]}"
            if was_pass is False and not now_pass:
                repro.append(f"{tag} ({s.get('failure_mode','fail')})")
            elif was_pass is False and now_pass:
                fixed.append(tag)
            elif was_pass is True and not now_pass:
                new_reg.append(f"{tag} ({s.get('failure_mode','fail')})")

print("\n════ rerun verdict ════")
print(f"REPRODUCED ({len(repro)}) — likely real:")
for t in repro: print(f"  ✗ {t}")
print(f"FIXED on re-run ({len(fixed)}) — flake / environment:")
for t in fixed: print(f"  ↺ {t}")
if new_reg:
    print(f"NEW regressions ({len(new_reg)}) — variance cuts both ways:")
    for t in new_reg: print(f"  ⚠ {t}")
if missing:
    print(f"(no rerun JSON for: {', '.join(missing)})")
print("\nNote: single re-run separates 'stable' from 'flaky', not 'model' from")
print("'harness'. For flakiness RATES, add --repeat 3 and read benchlocal's")
print(">=50% aggregation in the per-pack delta output.")
PY
