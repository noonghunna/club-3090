#!/usr/bin/env bash
# test-kv-offload-agentic-probe — scripts/kv-offload-agentic-probe.py must compile, refuse
# --disk without --container, and plant/extract needles correctly.
#
# WHY THIS TEST EXISTS
# --------------------
# The agentic probe's PASS hinges on each conversation's needle (a planted "access code")
# surviving the evict -> revisit round-trip. If seed_user_msg stopped embedding the code in
# the first user message, every revisit would read MISSING while the counters still moved —
# a silent "hit" that says nothing about state integrity. The engine-counter parsing is
# identical to kv-offload-probe.py and already covered by test-kv-offload-probe.sh; this
# test covers only the new pure helpers. Offline: no server, no GPU.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

python3 -m py_compile scripts/kv-offload-agentic-probe.py || { echo "✗ kv-offload-agentic-probe.py does not compile" >&2; exit 1; }
out="$(python3 scripts/kv-offload-agentic-probe.py --disk 2>&1)"
[[ "$out" == *"--disk needs --container"* ]] || { echo "✗ --disk without --container must be refused (got: $out)" >&2; exit 1; }

python3 - <<'PY'
import importlib.util, random, re, sys
spec = importlib.util.spec_from_file_location("aprobe", "scripts/kv-offload-agentic-probe.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
fails = []

orig = "Run: ls <repo>/ && echo done"
code, msg = probe.seed_user_msg(random.Random(1096), orig)
if not re.fullmatch(r"[A-Z]+-\d{4}", code):
    fails.append(f"seed_user_msg code {code!r} is not WORDS-#### (4 digits)")
if f"the access code is {code}" not in msg or orig not in msg:
    fails.append(f"seed_user_msg must plant the code AND keep the original first turn (got: {msg[:80]!r})")
c1, _ = probe.seed_user_msg(random.Random(42), orig)
c2, _ = probe.seed_user_msg(random.Random(43), orig)
if c1 == c2:
    fails.append(f"two fresh-RNG conversations got the same code {c1!r} — conversations must differ")

if probe.needle_ok("ALPHA-1234", "The access code is ALPHA-1234.") is not True:
    fails.append("needle_ok must be True when the code is in the answer")
if probe.needle_ok("ALPHA-1234", "I don't remember any code.") is not False:
    fails.append("needle_ok must be False when the code is absent")
if probe.needle_ok("ALPHA-1234", None) is not False:
    fails.append("needle_ok must be False for a None answer")

for f in fails:
    print("✗", f, file=sys.stderr)
sys.exit(1 if fails else 0)
PY
[[ $? -eq 0 ]] || exit 1
echo "test-kv-offload-agentic-probe: ok (compiles, arg guard, needle plant+extract)"
