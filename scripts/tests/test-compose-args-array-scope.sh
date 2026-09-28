#!/usr/bin/env bash
# Every "${X_ARGS[@]}" used on an `exec` line must be ASSIGNED at a block depth no
# deeper than that exec -- i.e. not inside a branch the exec's own path may skip.
#
# WHY: a compose entrypoint that ends in `if ...; then exec A; else exec B; fi` will
# happily accept an array assigned inside ONE branch. The other branch expands an
# unset array to nothing and the feature is SILENTLY DEAD on that path, while
# `docker compose config`, `bash -n` and every other gate stay green -- the script
# is perfectly valid bash.
#
# Caught for real: porting #1170's MAMBA_BLOCK_SIZE knob to the fp8 composes put
# MAMBA_ARGS=() inside the `_NVLINK_ENABLED=1` then-branch. The else-branch is the
# no-NVLink path -- most rigs, including this one -- so the knob would have shipped
# dead exactly where it was most likely to be used.
#
# ⚠️ Indentation is NOT a sufficient proxy: in these composes the branch execs sit
# at the same column as a block nudged inside the branch. Depth is tracked from the
# shell keywords instead.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
# #779: this gate parses compose YAML through python3; without PYTHONUTF8 a
# non-UTF8 locale mangles the entrypoint text and the scan silently misreads.
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "$0")/../.." || exit 1

python3 - <<'PY' || exit 1
import re, sys, yaml, pathlib

def script_of(path):
    try: d = yaml.safe_load(open(path))
    except Exception: return None
    for svc in (d.get("services") or {}).values():
        # ⚠️ entrypoint AND command. 17 services set
        # entrypoint: ['/bin/bash','-c'] and put the real script in `command:`;
        # reading only `entrypoint` skipped them ENTIRELY and this gate scored
        # them as vacuously clean. Found 2026-09-14 when a malformed array
        # expansion shipped to 13 composes with every gate green.
        cand = []
        for key in ("entrypoint", "command"):
            v = svc.get(key)
            if isinstance(v, list):
                # NOT v[-1]: these composes end with a literal "--", and picking
                # it yields a 2-char string that passes every check vacuously.
                cand += [e for e in v if isinstance(e, str) and "exec " in e]
            elif isinstance(v, str) and "exec " in v:
                cand.append(v)
        if cand: return max(cand, key=len)
    return None

OPEN  = re.compile(r'^\s*(if|for|while|until|case)\b')
CLOSE = re.compile(r'^\s*(fi|done|esac)\b')
ALT   = re.compile(r'^\s*(else|elif)\b')
ONELINE = re.compile(r'\b(fi|done|esac)\s*;?\s*$')

def scope_paths(lines):
    """Path of enclosing BRANCH ids per line.  `else`/`elif` starts a NEW id at the
    same depth, so an assignment in the then-branch does not dominate the
    else-branch -- which plain depth counting cannot express."""
    paths, stack, nxt = [], [], [0]
    def fresh():
        nxt[0] += 1
        return nxt[0]
    for ln in lines:
        if CLOSE.match(ln) and stack: stack.pop()
        elif ALT.match(ln) and stack: stack[-1] = fresh()
        paths.append(tuple(stack))
        if OPEN.match(ln) and not ONELINE.search(ln): stack.append(fresh())
    return paths

checked = refs = 0
problems = []

for p in sorted(pathlib.Path("models").rglob("*.yml")):
    if "compose" not in str(p): continue
    s = script_of(p)
    if not s: continue
    checked += 1
    lines = s.splitlines()
    paths = scope_paths(lines)
    for i, ln in enumerate(lines):
        if not re.match(r'^\s*exec\b', ln): continue
        for arr in set(re.findall(r'\$\{?([A-Z_]+_ARGS)\[@\]', ln)):
            refs += 1
            asg = [j for j, l in enumerate(lines) if re.match(rf'^\s*{arr}=\(', l)]
            if not asg:
                problems.append(f"{p}: exec uses ${{{arr}[@]}} but it is never assigned"); continue
            # an assignment dominates the exec iff it comes first AND its branch
            # path is a prefix of the exec's (same scope, or an enclosing one)
            ok = any(j < i and paths[i][:len(paths[j])] == paths[j] for j in asg)
            if not ok:
                problems.append(
                    f"{p}: {arr} is assigned only inside a branch that the exec on line "
                    f"{i+1} does not share -- that path expands an unset array, silently "
                    f"dropping the feature")

if checked == 0 or refs == 0:
    print(f"FAIL: vacuous run (checked={checked}, refs={refs}) -- extraction is broken", file=sys.stderr); sys.exit(1)
for m in problems: print("FAIL: " + m, file=sys.stderr)
if problems: sys.exit(1)
print(f"  checked {checked} entrypoints, {refs} array reference(s) -- each assigned before, and no deeper than, every exec that uses it")
PY
echo "test-compose-args-array-scope.sh OK"
