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

# Catalog rules (docs/ADDING_ENGINES.md, docs/ADDING_MODELS.md "Which case is
# this?"): required for a curated slug, advisory for a local one, because the
# local layer may run things the catalog would never ship. Findings print either
# way; only a curated slug's required failures count against the preflight.
RULES_ORIGIN="$(python3 - "${ROOT_DIR}" "${SLUG}" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/lib/profiles")
from compose_registry import get_registry
print((get_registry(sys.argv[1]).get(sys.argv[2]) or {}).get("origin") or "core")
PY
)"
RULES_LOG="${TMP_DIR}/catalog_rules.log"
if [[ "${RULES_ORIGIN}" == "local" ]]; then
  echo "${C_HEAD}[preflight] catalog rules ${SLUG} (advisory: local slug)${C_RESET}"
  python3 "${ROOT_DIR}/scripts/lib/profiles/catalog_rules.py" --slug "${SLUG}" >"${RULES_LOG}" 2>&1 || true
  sed 's/^/    /' "${RULES_LOG}"
  RESULTS+=("${C_DIM}·${C_RESET} catalog rules ${SLUG} ${C_DIM}(advisory for a local slug; required before export_pr.py)${C_RESET}")
else
  echo "${C_HEAD}[preflight] catalog rules ${SLUG}${C_RESET}"
  python3 "${ROOT_DIR}/scripts/lib/profiles/catalog_rules.py" --slug "${SLUG}" --strict >"${RULES_LOG}" 2>&1
  rules_rc=$?
  sed 's/^/    /' "${RULES_LOG}"
  record "catalog rules ${SLUG}" "${rules_rc}" "${RULES_LOG}"
fi
echo ""

# Effort ladder source (#1951): the model profile's `effort:` block is the one
# source the gateway render and the omp overrides will read; the compose keeps
# only the REASONING_EFFORT default. Informational here — the enforcing gate is
# the effort-block finding in catalog rules above (curated) and
# scripts/tests/test-effort-profiles.sh (whole catalog).
EFFORT_ERR="$(mktemp 2>/dev/null)" || EFFORT_ERR="/dev/null"   # no capture file: the rc still surfaces
EFFORT_LINE="$(python3 - "${ROOT_DIR}" "${SLUG}" <<'PY' 2>"${EFFORT_ERR}"
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/lib/profiles")
from compat import load_profiles
from compose_registry import get_registry
entry = get_registry(sys.argv[1]).get(sys.argv[2])
if not entry:
    print(f"?: slug {sys.argv[2]!r} is not in the registry — run this after "
          "the slug is catalogued")
    sys.exit(0)
mid = entry.get("model") or "?"
eff = (load_profiles(sys.argv[1] + "/scripts/lib/profiles").models.get(mid) or
       type("x", (), {"effort": None})()).effort
if not eff:
    print(f"{mid}: NO `effort:` block — declare the probed ladder or an honest status")
elif eff.get("levels"):
    flags = " + budget-map" if eff.get("budget_map") else ""
    flags += " + template-kwarg" if eff.get("template_kwarg") else ""
    print(f"{mid}: levels {'/'.join(eff['levels'])}{flags}")
else:
    print(f"{mid}: status {eff.get('status')}")
PY
)"
EFFORT_RC=$?
if [[ ${EFFORT_RC} -ne 0 ]]; then
  # Surface the failure instead of swallowing it: a broken probe must not
  # read as "no effort line to show". An empty stderr still names the rc.
  EFFORT_WHY="$(head -n 1 "${EFFORT_ERR}" 2>/dev/null)"
  if [[ -n "${EFFORT_WHY}" ]]; then
    echo "${C_DIM}[preflight] effort (#1951): check failed (rc=${EFFORT_RC}) — ${EFFORT_WHY}${C_RESET}" >&2
  else
    echo "${C_DIM}[preflight] effort (#1951): check failed (rc=${EFFORT_RC}) — no stderr from the probe${C_RESET}" >&2
  fi
elif [[ -n "${EFFORT_LINE}" ]]; then
  echo "${C_DIM}[preflight] effort (#1951): ${EFFORT_LINE}${C_RESET}"
  echo ""
fi
rm -f "${EFFORT_ERR}"

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
