#!/usr/bin/env bash
# test-p2p-validate.sh — guards scripts/p2p-validate.sh + its payload.
#
# WHY THIS TEST EARNS ITS KEEP: every FAILURE branch of the verdict matrix is
# unreachable on a healthy rig. A working machine only ever exercises PASS/PASS,
# so the hang and silent-corruption paths — the entire reason the tool exists —
# would otherwise ship having never run once.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
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

# --- #1507: the sizing line must time the peer copy with BOTH GPUs awake ---
# A peer copy only raises the SOURCE GPU's P-state; an idle destination stays at P8 / Gen1 and caps
# the copy at ~2-3 GB/s, so a healthy x8/x8 Gen4 rig read 0.49x ("no peer advantage") instead of
# 1.98x. A fake torch records every copy and sync: host traffic must reach BOTH GPUs before the
# first timed peer copy, and every sync must name both devices (a bare synchronize() waits on one).
out="$(python3 - "$PAYLOAD" <<'PY' 2>&1
import importlib.util, sys, time, types
log = []
class T:
    def __init__(self, device): self.device = device
    def numel(self): return 1024
    def element_size(self): return 4
    def copy_(self, other):
        log.append(("copy", other.device, self.device)); time.sleep(0.0005); return self
cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                             synchronize=lambda dev=None: log.append(("sync", dev)))
torch = types.SimpleNamespace(cuda=cuda, float32="f32",
                              empty=lambda n, dtype=None, device="cpu", pin_memory=False: T(str(device)))
sys.modules["torch"] = torch
spec = importlib.util.spec_from_file_location("p2p", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = m.peer_bandwidth(warm_s=0.02, reps=1)
assert r is not None, "peer_bandwidth returned None under the fake torch"
first_peer = next(i for i, e in enumerate(log) if e == ("copy", "cuda:0", "cuda:1"))
woken = {e[2] for e in log[:first_peer] if e[0] == "copy" and e[1] == "cpu"}
assert woken >= {"cuda:0", "cuda:1"}, f"GPUs given host traffic before the first timed peer copy: {sorted(woken)}"
syncs = [e[1] for e in log if e[0] == "sync"]
assert syncs and None not in syncs and set(syncs) == {0, 1}, f"sync calls: {syncs[:6]}"
print("ok")
PY
)"
[[ "$out" == "ok" ]] && pass "peer bandwidth wakes both GPUs first and syncs both devices (#1507)" \
  || bad "peer bandwidth is timed with an idle destination GPU or a one-device sync (#1507): $out"

# --- #1620: the sizing line must measure the pair the user asked for ---------
# `--gpus 0,3` reached the two collective arms (each runs under
# CUDA_VISIBLE_DEVICES=$GPUS) but not the bandwidth probe, which ran in the parent
# on cuda:0 -> cuda:1, so on a 4-card rig it always measured GPU0<->GPU1 and printed
# that number under a 0,3 run. Drive main() with fakes and record the CUDA mask in
# force wherever the bandwidth is actually measured, in-process or in a child.
out="$(python3 - "$PAYLOAD" <<'PY' 2>&1
import contextlib, importlib.util, io, os, sys
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
spec = importlib.util.spec_from_file_location("p2p", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.GPUS = "0,3"
m._sh = lambda cmd: ("0, RTX 3090\n1, RTX 3090\n2, RTX 3090\n3, RTX 3090"
                     if "--query-gpu=index,name" in cmd else "")
m._arm = lambda name, extra_env, port: "PASS"
measured = []
def fake_bw(*a, **k):                                   # measured in THIS process
    measured.append(os.environ.get("CUDA_VISIBLE_DEVICES")); return (12.5, 6.25)
m.peer_bandwidth = fake_bw
class Done:                                             # measured in a child process
    returncode, stderr = 0, ""
    stdout = "BW=12.5,6.25\n"
def fake_run(cmd, **kw):
    env = kw.get("env") or os.environ
    measured.append(env.get("CUDA_VISIBLE_DEVICES")); return Done()
m.subprocess.run = fake_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = m.main()
out = buf.getvalue()
assert rc == m.EXIT_HEALTHY, f"main() returned {rc}"
assert measured, "the bandwidth probe never ran"
assert all(v == "0,3" for v in measured), \
    f"bandwidth measured under CUDA_VISIBLE_DEVICES={measured!r}, not the requested pair 0,3"
line = next((l for l in out.splitlines() if "bandwidth" in l), "")
assert "12.50" in line and "0,3" in line, f"sizing line missing the numbers or the pair: {line!r}"
print("ok")
PY
)"
[[ "$out" == "ok" ]] && pass "bandwidth probe runs on the --gpus pair, not cuda:0<->cuda:1 (#1620)" \
  || bad "bandwidth probe ignores --gpus (#1620): $out"

[[ $fail -eq 0 ]] && echo "== PASS ==" || echo "== FAIL =="
exit $fail
