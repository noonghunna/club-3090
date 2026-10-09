#!/usr/bin/env bash
# vllm-effort-budget installer (patch B) — runs in the compose entrypoint before `vllm serve`,
# AFTER the effort_budget.py shell-env line has exported CLUB3090_REASONING_EFFORT_BUDGETS.
#
#   1. copies /etc/club3090/effort_budget.py (scripts/lib/effort_budget.py) and the vLLM glue
#      next to the vllm package, on every boot (a restarted container keeps site-packages);
#   2. validates the map in the environment — a bad map refuses the boot here instead of
#      failing every request later;
#   3. inserts ONE call before `request.to_sampling_params(` in chat serving (marker-gated:
#      a second run is a verified no-op). Anchor missing or not unique -> exit 1, see README.
#
# Prints exactly one `[vllm-effort-budget] applied…` / `already applied…` line on success.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$here/patch_effort_budget.py" copy || exit 1
python3 -c '
import sys
import club3090_effort_budget as m
try:
    m.budgets_from_env()
except m.BudgetConfigError as exc:
    sys.exit(f"[vllm-effort-budget] REFUSE: {exc} (set by effort_budget.py shell-env from the THINKING_BUDGET_* knobs)")
import club3090_effort_budget_vllm  # noqa: F401 — the glue parses the same map at import
' || exit 1
python3 "$here/patch_effort_budget.py" patch || exit 1
