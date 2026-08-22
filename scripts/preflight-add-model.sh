#!/usr/bin/env bash
#
# Scoped preflight for a model/slug being added to the catalog.
#
#   bash scripts/preflight-add-model.sh <slug>     # e.g. vllm/minimal
#   bash scripts/preflight-add-model.sh --help
#
# Runs the fast, high-signal subset of the add-model gates in one pass:
#   - scripts/diagnose-profile.sh <slug>   (registry -> cross-ref -> fits() ->
#                                           kv-calc projection -> calibration
#                                           freshness -> overlay matching)
#   - the 9 catalog guard tests that catch the recurring ship-blockers:
#       test-profiles-compat               (Step 4b: engine/hardware/scenario)
#       test-compose-registry-disk         (registry <-> disk parity + counts)
#       test-compose-status-drift          (compose header Status <-> registry)
#       test-compose-mounts-resolve        (the ../ mount depth)
#       test-model-weights-registry        (files:/verify_glob traps, #910/#911)
#       test-switch-registry-parity        (switch.sh derives from registry)
#       test-launch-registry-parity        (launch.sh derives from registry)
#       test-default-resolver              (<model>/default walk stays honest)
#       test-docs-slugs-resolve            (docs never name a dead slug)
#   - tools/kv-calc.py --calibration       (verdict accuracy vs measured rows)
#
# This is a TRIAGE gate, not a substitute for the full suite: before commit,
# the FULL `for t in scripts/tests/*.sh; do bash "$t"; done` sweep remains
# authoritative (see docs/ADDING_MODELS.md Step 7).

set -uo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs —
# same invariant as every other script here (AGENTS.md "File encoding";
# guarded by test-locale-utf8.sh).
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

GATES=(
  test-profiles-compat
  test-compose-registry-disk
  test-compose-status-drift
  test-compose-mounts-resolve
  test-model-weights-registry
  test-switch-registry-parity
  test-launch-registry-parity
  test-default-resolver
  test-docs-slugs-resolve
)

usage() {
  echo "Usage: $0 <slug>"
  echo ""
  echo "Scoped preflight for adding a model to the catalog: diagnose-profile"
  echo "plus the 9 catalog guard tests plus kv-calc --calibration, with a"
  echo "pass/fail summary. Exit 0 only when everything is green."
  echo ""
  echo "Example: $0 vllm/minimal"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SLUG="${1:-}"
if [[ -z "${SLUG}" || $# -gt 1 ]]; then
  usage >&2
  [[ -n "${SLUG}" ]] && echo "" >&2 && echo "ERROR: exactly one slug argument is required; got: $*" >&2
  exit 64
fi

if [[ ! -t 1 || -n "${NO_COLOR:-}" ]]; then
  C_GOOD="" C_BAD="" C_HEAD="" C_DIM="" C_RESET=""
else
  C_GOOD=$'\033[32m' C_BAD=$'\033[31m' C_HEAD=$'\033[1m' C_DIM=$'\033[2m' C_RESET=$'\033[0m'
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

RESULTS=()
FAILURES=0

record() { # <name> <rc> <logfile>
  local name="$1" rc="$2" log="$3"
  if [[ "$rc" == "0" ]]; then
    RESULTS+=("${C_GOOD}✓${C_RESET} ${name}")
  else
    RESULTS+=("${C_BAD}✗${C_RESET} ${name} ${C_DIM}(failed — last 20 lines printed above; re-run it directly for the full log)${C_RESET}")
    FAILURES=$((FAILURES + 1))
  fi
}

run_gate() { # <label> <cmd...>
  local label="$1"; shift
  local log="${TMP_DIR}/$(echo "${label}" | tr ' /:' '___').log"
  echo "${C_HEAD}[preflight] ${label}${C_RESET}"
  if ( "$@" ) >"${log}" 2>&1; then
    record "${label}" 0 "${log}"
  else
    local rc=$?
    tail -20 "${log}" | sed 's/^/    /'
    record "${label}" "${rc}" "${log}"
  fi
  echo ""
}

echo "${C_HEAD}[preflight-add-model] slug: ${SLUG}${C_RESET}"
echo ""

run_gate "diagnose-profile ${SLUG}" bash "${ROOT_DIR}/scripts/diagnose-profile.sh" "${SLUG}"

for gate in "${GATES[@]}"; do
  if [[ -f "${ROOT_DIR}/scripts/tests/${gate}.sh" ]]; then
    run_gate "${gate}" bash "${ROOT_DIR}/scripts/tests/${gate}.sh"
  else
    echo "ERROR: gate script not found: scripts/tests/${gate}.sh" >&2
    exit 2
  fi
done

run_gate "kv-calc --calibration" python3 "${ROOT_DIR}/tools/kv-calc.py" --calibration

echo "${C_HEAD}================ preflight summary ================${C_RESET}"
for line in "${RESULTS[@]}"; do
  echo "  ${line}"
done
echo "${C_HEAD}===================================================${C_RESET}"
echo ""

if (( FAILURES > 0 )); then
  echo "${C_BAD}${FAILURES} gate(s) FAILED.${C_RESET} The last 20 lines of each failure were printed above — re-run the failing gate directly for the full log."
else
  echo "${C_GOOD}All preflight gates green for ${SLUG}.${C_RESET}"
fi
echo ""
echo "Note: this scoped pass is triage only. The FULL scripts/tests/*.sh suite"
echo "remains authoritative before commit — run it once at the end and baseline"
echo "any failure against the last release tag (docs/ADDING_MODELS.md Step 7)."

exit $(( FAILURES > 0 ? 1 : 0 ))
