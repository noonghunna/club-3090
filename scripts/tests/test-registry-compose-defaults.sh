#!/usr/bin/env bash
# test-registry-compose-defaults — the registry's max_num_seqs / mem_util must match the compose defaults they describe.
#
# WHY THIS TEST EXISTS
# --------------------
# The registry repeats two numbers that really live in each compose: the concurrency cap
# (vLLM --max-num-seqs, SGLang --max-running-requests) and the memory fraction (vLLM
# --gpu-memory-utilization, SGLang --mem-fraction-static). Nothing kept them in step, and on
# 2026-09-25 twenty entries had drifted: all 18 Qwen3.8 / ThinkingCap SGLang dual-fast + multi4/multi8
# slugs still said max_num_seqs 1 after the composes moved to 2 (2026-09-15), the dual-fast pair said
# mem_util 0.9 after 0.95, and two vLLM slugs kept a mem_util their compose had since derated. Tools
# that read the registry (c3, kv-calc) then describe a config the compose does not boot.
# The compose is what runs, so it is the source of truth here; fix the registry, not the compose.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

python3 - <<'PY'
import re, sys
sys.path.insert(0, ".")
from scripts.lib.profiles import compose_registry as cr

FLAGS = {  # engine dir -> (concurrency flag, memory flag)
    "vllm": ("--max-num-seqs", "--gpu-memory-utilization"),
    "sglang": ("--max-running-requests", "--mem-fraction-static"),
}
VALUE = r'"?\$?\{?(?:[A-Z_]+:-)?([0-9.]+)\}?"?'

def flag_default(txt, flag):
    """The default a compose boots with for `flag`: inline (`flag X` / `flag=X` / `flag "${V:-X}"`) or the
    next YAML list item (`- flag` then `- "${V:-X}"`). None if absent or not a plain/defaulted value."""
    f = re.escape(flag)
    vals = re.findall(rf'{f}[ =]+{VALUE}(?=[\s"\\]|$)', txt, re.M)
    vals += re.findall(rf'-\s+{f}\s*\n\s*-\s+{VALUE}\s*$', txt, re.M)
    vals = sorted(set(vals))
    return vals[0] if len(vals) == 1 else (None if not vals else "AMBIGUOUS:" + ",".join(vals))

reg = cr.get_registry()
bad, checked, unparsed = [], 0, []
for slug, e in sorted(reg.items()):
    e = e if isinstance(e, dict) else vars(e)
    cp = e.get("compose_path") or ""
    parts = cp.split("/")
    eng = next((k for k in FLAGS if k in parts), None)
    if not eng or "_archive" in parts:
        continue
    try:
        txt = open(cp, encoding="utf-8").read()
    except OSError:
        continue
    cflag, mflag = FLAGS[eng]
    for field, flag, cast in (("max_num_seqs", cflag, int), ("mem_util", mflag, float)):
        want = e.get(field)
        got = flag_default(txt, flag)
        if want is None:
            continue
        if got is None or got.startswith("AMBIGUOUS"):
            unparsed.append(f"{slug}:{field}={got}")
            continue
        checked += 1
        if cast(got) != cast(want):
            bad.append(f"{slug}: registry {field}={want} but {cp} boots {flag} {got}")

for b in bad:
    print(f"✗ {b}", file=sys.stderr)
if bad:
    print(f"test-registry-compose-defaults: {len(bad)} drifted field(s) of {checked} checked — fix registry.yaml "
          f"(the compose is what runs), then re-run migrate_registry_to_yaml.py", file=sys.stderr)
    sys.exit(1)
if checked < 100:
    print(f"✗ only {checked} fields compared — the parser has stopped recognising the compose flags", file=sys.stderr)
    sys.exit(1)
print(f"test-registry-compose-defaults: ok ({checked} registry fields match their compose defaults; "
      f"{len(unparsed)} not statically resolvable, skipped)")
PY
