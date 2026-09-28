#!/usr/bin/env bash
# Every array expansion in a compose entrypoint must be wrapped in UNESCAPED
# double quotes: "${NAME[@]}" — never \"${NAME[@]}\".
#
# WHY (club-3090 regression, 2026-09-14): a scripted rollout wrote the injected
# line as
#       \"$${CACHE[@]}\" \
# instead of the form its own sibling two lines below uses,
#       "$${SPEC[@]}"
# because the quotes were escaped for a Python string literal and landed on disk
# verbatim. Bash then emits a stray quote character and the server dies at
# argument parsing:
#       sglang serve: error: unrecognized arguments: ""
# ALL 13 SGLang composes shipped unbootable, and it reached master.
#
# ⚠️ WHY NOTHING ELSE CATCHES IT:
#   • the YAML parses — `\"` is legal inside a block scalar
#   • `bash -n` passes — `\"` is valid bash, just a literal quote character
#   • `docker compose config` renders happily
#   • test-compose-args-array-scope passes — the array IS referenced and IS
#     assigned at a reachable depth; that gate checks SCOPE, not QUOTING
#   • a presence grep for `CACHE[@]` matches — presence is not correctness
# The break is only visible by BOOTING, which no gate does.
#
# ⚠️ Scope: array expansions only. Escaped quotes are legitimate elsewhere in
# these entrypoints — e.g. the JSON payload in
# `--preferred-sampling-params "{\"temperature\": ...}"` — so a blanket ban on
# `\"` would false-positive on real code.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
# #779: parses compose YAML through python3; without PYTHONUTF8 a non-UTF8
# locale mangles the entrypoint text and the scan silently misreads.
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "$0")/../.." || exit 1

python3 - <<'PY' || exit 1
import pathlib, re, sys, yaml

# an array expansion preceded by a BACKSLASH-quote, or followed by one
BAD = re.compile(r'\\"\s*\$\$?\{\w+\[@\]\}|\$\$?\{\w+\[@\]\}\s*\\"')
# any array expansion at all, to prove the scan reached real content
ANY = re.compile(r'\$\$?\{(\w+)\[@\]\}')

def entrypoints(path):
    try:
        d = yaml.safe_load(open(path, encoding="utf-8"))
    except Exception:
        return []
    out = []
    for svc in (d or {}).get("services", {}).values():
        # ⚠️ BOTH keys, never `entrypoint or command`. These composes set
        # entrypoint: ['/bin/bash','-c'] and put the 17K-line script in
        # `command:` — a short-circuit on the truthy 2-element entrypoint
        # reads 11 characters and silently scans nothing. That is exactly how
        # the 2026-09-14 regression slipped past the sibling scope gate.
        for key in ("entrypoint", "command"):
            v = svc.get(key)
            if isinstance(v, list):
                out += [e for e in v if isinstance(e, str)]
            elif isinstance(v, str):
                out.append(v)
    return out

problems, checked, refs = [], 0, 0
for p in sorted(pathlib.Path("models").rglob("*.yml")):
    if "compose" not in str(p):
        continue
    for script in entrypoints(p):
        checked += 1
        for i, ln in enumerate(script.splitlines(), 1):
            for m in ANY.finditer(ln):
                refs += 1
            if BAD.search(ln):
                problems.append(f"{p}:{i}: escaped quotes around an array expansion\n      {ln.strip()}")

if checked == 0 or refs == 0:
    print(f"REFUSING: scan found nothing (entrypoints={checked} array refs={refs}) — "
          "the YAML shape or quoting convention changed; fix this gate, do not delete it")
    sys.exit(1)

if problems:
    print(f"✗ {len(problems)} malformed array expansion(s):")
    for x in problems:
        print("   " + x)
    print('\n  Expected form:  "${NAME[@]}"   (unescaped double quotes)')
    print('  Bash treats \\" as a literal quote character, so the expansion')
    print("  becomes a stray argument and the engine refuses to start.")
    sys.exit(1)

print(f"✓ array quoting OK ({refs} expansions across {checked} entrypoints)")
PY
