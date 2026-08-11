#!/usr/bin/env bash
# test-compose-no-repeated-args.sh — no compose may pass the same CLI flag twice
# inside one `command:` block.
#
# Why this exists: llama.cpp treats repeated flags as LAST-WINS, not additive.
# Passing `-ot A -ot B -ot C` keeps only C and discards A and B, announcing it
# with a single line that is trivially read past:
#
#   DEPRECATED: argument '-ot' specified multiple times, use comma-separated
#   values instead (only last value will be used)
#
# Three shipped DeepSeek composes did exactly that for months (#948). The
# damage was not the dead default — the `blk.99999` sentinels are a deliberate
# no-op — but the LAUNCHER path: OT_G0/OT_G1 are resolved from the detected
# card's VRAM and injected as the first two `-ot` args, then silently thrown
# away because a third `-ot` followed. The per-card residency policy never took
# effect on ANY rig, and every bench taken on those composes actually measured
# all-experts-on-CPU rather than the interleaved placement the file describes.
#
# Nothing in the suite could catch it: the only symptom lived in a boot log,
# and no test reads boot logs. Hence a static guard.
#
# The correct shape for a multi-valued flag is ONE occurrence with
# comma-separated values, which also preserves rule order (earlier rules win):
#   -ot 'ruleA,ruleB,ruleC'
#
# If a flag is ever genuinely repeatable on some engine, add it to
# REPEATABLE_OK below WITH the engine and a link proving the semantics are
# additive — never just to silence this test.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Flags known to be genuinely additive rather than last-wins. Empty today:
# every repeat found in this repo has been a bug.
REPEATABLE_OK=""

python3 - "$REPEATABLE_OK" <<'PY'
import collections, pathlib, re, sys

ok = {f for f in sys.argv[1].split(",") if f}
fails = 0
checked = 0

for path in sorted(pathlib.Path("models").rglob("*.yml")):
    if "/compose/" not in str(path):
        continue
    text = path.read_text(encoding="utf-8")
    start = re.search(r"^\s*command:\s*$", text, re.M)
    if not start:
        continue          # folded/string form or no command block
    checked += 1
    block = text[start.end():]
    # The command list ends at the next same-or-shallower service key.
    end = re.search(r"^\s{0,4}[a-z_]+:\s*$", block, re.M)
    if end:
        block = block[:end.start()]
    # Only lines that are ENTIRELY a flag token; values (paths, regexes,
    # negative numbers) never match because of the trailing anchor.
    flags = re.findall(r"^\s*-\s*'?\"?(-{1,2}[A-Za-z][\w-]*)'?\"?\s*$", block, re.M)
    for flag, n in collections.Counter(flags).items():
        if n > 1 and flag not in ok:
            print(f"FAIL: {path} passes {flag} {n} times", file=sys.stderr)
            print(f"        llama.cpp keeps only the LAST one and discards the rest.", file=sys.stderr)
            print(f"        Use a single {flag} with comma-separated values instead.", file=sys.stderr)
            fails += 1

if fails:
    print(f"{fails} repeated-argument failure(s) across {checked} composes", file=sys.stderr)
    raise SystemExit(1)
print(f"test-compose-no-repeated-args: ok ({checked} composes, no repeated flags)")
PY
