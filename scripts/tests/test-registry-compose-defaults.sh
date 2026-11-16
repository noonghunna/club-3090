#!/usr/bin/env bash
# test-registry-compose-defaults — the registry's max_num_seqs / mem_util / max_ctx must match the compose defaults they describe.
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
#
# max_ctx (#2054) is the third number, checked on every engine: the card pages and c3 show it, and a
# context change made on one side only ships one context while documenting another. Until 2026-11-16
# only the generated slug tables noticed (test-docs-slug-tables), and only by accident. It is read from
# whichever context flag or env default the compose sets; deprecated slugs are skipped (nothing boots
# them, and three retired ik-llama slugs disagree), and a compose whose context isn't a plain or
# defaulted literal is counted as not statically resolvable.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
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
VALUE = r'"?\$*\{?(?:[A-Z_]+:-)?([0-9.]+)\}?"?'  # \$* also reads a runtime-deferred $${V:-X}

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

# max_ctx, on every engine: the context the compose boots with, from its context flag or env default
CTX_FLAGS = ("--max-model-len", "--context-length", "--ctx-size", "-c", "--max-ctx", "--max-seq-len",
             "--max-context")
CTX_ENVS = ("CONTEXT", "MAX_MODEL_LEN", "CTX_SIZE", "CTX", "MAX_CTX", "CONTEXT_LENGTH", "MAX_SEQ_LEN")
INT = r'"?\$*\{?(?:[A-Z_]+:-)?([0-9]+)\}?"?'

def ctx_defaults(txt):
    """Every context default the compose's non-comment lines set, as ints: a flag inline or as the next list
    item, or an env key (`KEY: "${KEY:-N}"`, `- KEY=${KEY:-N}`). One value is a reading; none or several is not."""
    body = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    vals = set()
    for flag in CTX_FLAGS:
        f = re.escape(flag)
        vals.update(re.findall(rf'(?<![\w-]){f}[ =]+{INT}(?=[\s"\\]|$)', body, re.M))
        vals.update(re.findall(rf"-\s+['\"]?{f}['\"]?\s*\n\s*-\s+['\"]?{INT}['\"]?\s*$", body, re.M))
    for key in CTX_ENVS:
        vals.update(re.findall(rf'^\s+{key}:\s*{INT}\s*$', body, re.M))
        vals.update(re.findall(rf'^\s+-\s+{key}=\$\{{{key}:-([0-9]+)\}}', body, re.M))
    return sorted({int(v) for v in vals})

ctx_checked, ctx_unparsed = 0, []
for slug, e in sorted(reg.items()):
    e = e if isinstance(e, dict) else vars(e)
    cp, want = e.get("compose_path") or "", e.get("max_ctx")
    if not want or "_archive" in cp.split("/") or e.get("status") == "deprecated":
        continue
    try:
        txt = open(cp, encoding="utf-8").read()
    except OSError:
        continue
    got = ctx_defaults(txt)
    if len(got) != 1:
        ctx_unparsed.append(f"{slug}:max_ctx={got or None}")
        continue
    ctx_checked += 1
    if got[0] != int(want):
        bad.append(f"{slug}: registry max_ctx={want} but {cp} boots context {got[0]}")

for b in bad:
    print(f"✗ {b}", file=sys.stderr)
if bad:
    print(f"test-registry-compose-defaults: {len(bad)} drifted field(s) of {checked + ctx_checked} checked — fix "
          f"registry.yaml (the compose is what runs), then re-run migrate_registry_to_yaml.py", file=sys.stderr)
    sys.exit(1)
if checked < 100:
    print(f"✗ only {checked} fields compared — the parser has stopped recognising the compose flags", file=sys.stderr)
    sys.exit(1)
if ctx_checked < 120:
    print(f"✗ only {ctx_checked} max_ctx values compared — the parser has stopped recognising the compose context "
          f"settings", file=sys.stderr)
    sys.exit(1)
print(f"test-registry-compose-defaults: ok ({checked} registry fields and {ctx_checked} max_ctx values match their "
      f"compose defaults; {len(unparsed) + len(ctx_unparsed)} not statically resolvable, skipped)")
PY
