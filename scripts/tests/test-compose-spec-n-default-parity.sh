#!/usr/bin/env bash
# test-compose-spec-n-default-parity.sh — the drafter entrypoint and the drafter flag
# must default to the SAME depth.
#
# WHY: composes with the drafter toggle decide ON/OFF in the entrypoint from
# `_spec_n="$${SPEC_N:-<default>}"`, while the depth flag itself is interpolated
# host-side, e.g. `--spec-draft-n-max ${SPEC_N:-${MTP_DRAFT_N_MAX:-2}}`. 26 llama.cpp-
# family composes defaulted the entrypoint to `1` while the flag read a second variable
# (MTP_DRAFT_N_MAX / DRAFT_N_MAX / MTP_N_MAX, default 1-5) that was never declared, so
# the entrypoint could not see it:
#   - the banner printed `[spec] drafter ON (SPEC_N=1)` while the server drafted at 2;
#   - MTP_DRAFT_N_MAX=0, documented as "disables", kept the drafter loaded at n-max 0
#     (the entrypoint still read 1 = ON), which is exactly the half-off state the
#     toggle exists to avoid.
# So this guard checks default parity on every compose with the toggle, and exercises
# the DELIVERY PATH (rendered entrypoint + argv) with a NEGATIVE CONTROL.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

fails=0

# --- 1. static: entrypoint default == every host-side SPEC_N default ----------
static_out=$(python3 - <<'PY'
import glob, re, subprocess
files = subprocess.run(['grep', '-rl', '--include=*.yml', '_spec_n="$${SPEC_N:-', 'models'],
                       capture_output=True, text=True, encoding='utf-8').stdout.split()
checked = 0; bad = []
for f in sorted(files):
    if '/_archive/' in f: continue
    s = open(f, encoding='utf-8').read()
    m = re.search(r'_spec_n="\$\$\{SPEC_N:-((?:\$\$\{[A-Z_]+:-\d+\})|\d+)\}"', s)
    if not m:
        bad.append(f'{f}: entrypoint _spec_n default is not a literal or ${{VAR:-N}}'); continue
    checked += 1
    ep = m.group(1).replace('$$', '$')
    for host in re.findall(r'(?<!\$)\$\{SPEC_N:-((?:\$\{[A-Z_]+:-\d+\})|\d+)\}', s):
        if host != ep:
            bad.append(f'{f}: flag defaults SPEC_N to {host} but the entrypoint to {ep}')
    v = re.match(r'\$\{([A-Z_]+):-', ep)
    if v and not re.search(rf'^\s*- {v.group(1)}(=|\s*$)', s, re.M):
        bad.append(f'{f}: entrypoint reads {v.group(1)} but `environment:` does not declare it (docker will not forward it)')
print(f'CHECKED {checked}')
for b in bad: print('BAD', b)
PY
)
checked=$(command grep -oE '^CHECKED [0-9]+' <<<"$static_out" | cut -d' ' -f2)
while IFS= read -r line; do
  [[ "$line" == BAD* ]] || continue
  echo "FAIL: ${line#BAD }" >&2; fails=$((fails+1))
done <<<"$static_out"
[[ "${checked:-0}" -eq 0 ]] && { echo "FAIL: found no composes with the drafter toggle — the search is wrong, not the tree" >&2; exit 1; }

# --- 2. delivery path + NEGATIVE CONTROL -------------------------------------
# Render the compose, then run its real entrypoint with the real argv; only the final
# `exec /app/llama-server` is swapped for a printer.
SAMPLE=models/qwen3.8-27b/llama-cpp/compose/single/unsloth-iq4xs/q4kv-vision.yml
if command -v docker >/dev/null 2>&1 && [[ -f "$SAMPLE" ]]; then
  run() {  # $@ = VAR=value overrides; prints "<banner>|<argv…>"
    env -u SPEC_N -u SPEC -u MTP_DRAFT_N_MAX MODEL_DIR="${MODEL_DIR:-/tmp}" "$@" \
      docker compose -f "$SAMPLE" config --format json 2>/dev/null | python3 -c '
import json, subprocess, sys
svc = next(iter(json.load(sys.stdin)["services"].values()))
ep, cmd = svc["entrypoint"], svc["command"]
script = ep[2].replace("$$", "$").replace("exec /app/llama-server", "exec printf \"%s \"")  # config output keeps $$ escaped
env = {"PATH": "/usr/bin:/bin"}
env.update({k: v for k, v in (svc.get("environment") or {}).items() if v is not None})
p = subprocess.run(["bash", "-c", script] + ep[3:] + cmd, capture_output=True, text=True, env=env)
banner = [l for l in p.stderr.splitlines() if l.startswith("[spec]")]
print((banner[0] if banner else "<no banner>") + "|" + p.stdout)'
  }
  expect() {  # name, output, banner-substring, argv-must-contain, argv-must-not-contain
    local name="$1" out="$2" want_b="$3" want_a="$4" not_a="$5"
    [[ "${out%%|*}" == *"$want_b"* ]] || { echo "FAIL: $name — banner '${out%%|*}', expected '$want_b'" >&2; fails=$((fails+1)); }
    [[ -z "$want_a" || "${out#*|}" == *"$want_a"* ]] || { echo "FAIL: $name — argv lacks '$want_a'" >&2; fails=$((fails+1)); }
    [[ -z "$not_a" || "${out#*|}" != *"$not_a"* ]] || { echo "FAIL: $name — argv still carries '$not_a'" >&2; fails=$((fails+1)); }
  }
  # Negative control: the default must report the flag's own depth (2), not the old 1.
  expect "default"             "$(run)"                   "drafter ON (SPEC_N=2)" "--spec-draft-n-max 2" ""
  expect "MTP_DRAFT_N_MAX=0"   "$(run MTP_DRAFT_N_MAX=0)" "drafter OFF"           ""                     "--spec-type"
  expect "SPEC_N=3"            "$(run SPEC_N=3)"          "drafter ON (SPEC_N=3)" "--spec-draft-n-max 3" ""
else
  echo "NOTE: docker unavailable — static check only; the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails drafter-default parity check(s) failed" >&2
  exit 1
fi
echo "PASS: $checked composes default the drafter entrypoint and flag to the same depth; the rendered iq4xs entrypoint reports 2 by default, strips the drafter at MTP_DRAFT_N_MAX=0, honours SPEC_N=3"
