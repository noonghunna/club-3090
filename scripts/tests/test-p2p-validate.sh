#!/usr/bin/env bash
# test-p2p-validate.sh — guards scripts/p2p-validate.sh + its payload.
#
# WHY THIS TEST EARNS ITS KEEP: every FAILURE branch of the verdict matrix is
# unreachable on a healthy rig. A working machine only ever exercises PASS/PASS,
# so the hang and silent-corruption paths — the entire reason the tool exists —
# would otherwise ship having never run once.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779 — python3 under a C/POSIX locale mangles non-ASCII

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${ROOT_DIR}/scripts/p2p-validate.sh"
PAYLOAD="${ROOT_DIR}/scripts/lib/p2p_collective_check.py"
fail=0
pass() { echo "  ok   — $1"; }
bad()  { echo "  FAIL — $1" >&2; fail=1; }

echo "== test-p2p-validate =="

[[ -r "$WRAPPER" ]] && pass "wrapper present" || bad "wrapper missing"
[[ -x "$WRAPPER" ]] && pass "wrapper executable" || bad "wrapper not executable"
[[ -r "$PAYLOAD" ]] && pass "payload present" || bad "payload missing"

bash -n "$WRAPPER" 2>/dev/null && pass "wrapper parses" || bad "wrapper syntax error"
python3 -m py_compile "$PAYLOAD" 2>/dev/null && pass "payload compiles" || bad "payload syntax error"

# --- the verdict matrix: arm_a arm_b -> expected exit code ---
# Control-failed MUST outrank everything: never accuse P2P when the baseline is broken.
python3 - "$PAYLOAD" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p2pcheck", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

cases = [
    ("PASS",  "PASS",  m.EXIT_HEALTHY,      "healthy"),
    ("HANG",  "PASS",  m.EXIT_HANG,         "hang with a good control"),
    ("WRONG", "PASS",  m.EXIT_WRONG,        "silent corruption"),
    ("ERROR", "PASS",  m.EXIT_INCONCLUSIVE, "arm A errored"),
    # control broken -> inconclusive regardless of arm A
    ("PASS",  "HANG",  m.EXIT_INCONCLUSIVE, "control hung"),
    ("HANG",  "HANG",  m.EXIT_INCONCLUSIVE, "both hung"),
    ("WRONG", "ERROR", m.EXIT_INCONCLUSIVE, "control errored"),
]
bad = 0
for a, b, want, name in cases:
    msg, rc = m.verdict(a, b)
    if rc != want:
        print(f"  FAIL — verdict({a},{b}) = {rc}, want {want} ({name})"); bad = 1
    elif not msg.strip():
        print(f"  FAIL — verdict({a},{b}) returned an empty message"); bad = 1
    else:
        print(f"  ok   — verdict({a},{b}) -> {rc} ({name})")

# the two severe verdicts must name the workaround, or the user is stranded
for a in ("HANG", "WRONG"):
    msg, _ = m.verdict(a, "PASS")
    if "NCCL_P2P_DISABLE" not in msg:
        print(f"  FAIL — {a} verdict does not tell the user the workaround"); bad = 1
    else:
        print(f"  ok   — {a} verdict names NCCL_P2P_DISABLE")

# distinct exit codes: callers must be able to distinguish hang from corruption
codes = {m.EXIT_HEALTHY, m.EXIT_HANG, m.EXIT_WRONG, m.EXIT_INCONCLUSIVE, m.EXIT_UNUSABLE}
if len(codes) != 5:
    print("  FAIL — exit codes are not distinct"); bad = 1
else:
    print("  ok   — exit codes distinct")
sys.exit(bad)
PY
[[ $? -eq 0 ]] || fail=1

# --- payload must never mutate the rig ---
for danger in 'setpci' 'nvidia-smi -[a-z]* [0-9]' 'modprobe' 'rm -rf' 'docker run'; do
  if command grep -qE "$danger" "$PAYLOAD"; then
    bad "payload contains a state-changing call: $danger"
  fi
done
pass "payload is read-only (no state-changing calls)"

# --- correctness check must be value-based, not completion-based ---
command grep -q 'wrong' "$PAYLOAD" && command grep -q 'expected' "$PAYLOAD" \
  && pass "payload verifies VALUES, not just completion" \
  || bad "payload does not appear to check result values"

# --- arms must use distinct ports (the bug that broke the first version) ---
command grep -q 'BASE_PORT + 1' "$PAYLOAD" \
  && pass "arms use distinct ports (killed arm can hold its port)" \
  || bad "arms may share a port — a killed arm A will EADDRINUSE arm B"

# --- hang must be killed by process GROUP, not just the parent ---
command grep -q 'killpg' "$PAYLOAD" \
  && pass "timeout kills the process group (mp.spawn children outlive proc.kill)" \
  || bad "timeout does not kill the process group — children would survive"

[[ $fail -eq 0 ]] && echo "== PASS ==" || echo "== FAIL =="
exit $fail
