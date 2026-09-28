#!/usr/bin/env bash
#
# Guard: no comment may follow a shell line-continuation inside a compose.
#
# Why (the #1292 regression): a documentation block was inserted here —
#
#     --mem-fraction-static "${MEM_FRACTION:-0.82}" \
#     # ⭐ MAMBA STATE SLOTS vs CONCURRENCY ...
#
# The trailing `\` splices the comment onto the command, the `#` comments out the
# remainder of that logical line, and because the command is an `exec` the shell
# is replaced immediately — so EVERY flag after that point is silently dropped.
#
# On the 5 SGLang dual composes that meant losing --host (so the server bound
# 127.0.0.1 and was unreachable from the host), --enable-metrics,
# --served-model-name, --reasoning-parser, --tool-call-parser, and the entire
# SPEC array — i.e. speculative decoding was OFF on slugs whose whole purpose is
# the DFlash2 drafter. `speculative_algorithm: None` in the boot dump.
#
# ⛔ Nothing caught it. The compose suite was fully green on the broken tree:
# the YAML is valid, the flags are all present in the file, and the container
# starts and answers its own healthcheck. Only the engine's resolved server_args
# disagreed with the compose text.
#
# ⚠️ Must FAIL against the pre-fix tree.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

out="$(python3 - "$ROOT" <<'PYEOF'
import glob, os, sys
root = sys.argv[1]
bad = []
for pat in ("models/**/*.yml", "scripts/lib/profiles-local/**/*.yml"):
    for f in glob.glob(os.path.join(root, pat), recursive=True):
        lines = open(f, encoding="utf-8", errors="replace").read().split("\n")
        for i in range(1, len(lines)):
            prev, cur = lines[i-1].rstrip(), lines[i].lstrip()
            # A '#' on the line spliced onto a continuation kills the rest of the
            # command. Blank continuation lines are fine; only comments bite.
            # ⚠️ Only a CODE line matters. A '\' at the end of a COMMENT line is
            # just text — doc blocks are full of pasted shell that legitimately
            # wraps, and flagging those makes the gate noise a maintainer learns
            # to ignore (46 such lines exist in this tree).
            if prev.lstrip().startswith("#"):
                continue
            if prev.endswith("\\") and cur.startswith("#"):
                bad.append(f"{os.path.relpath(f, root)}:{i+1}: {cur[:80]}")
print("\n".join(bad))
PYEOF
)"

# positive control first: a scan that silently finds nothing is indistinguishable
# from a clean tree.
probe="$(mktemp -d)"; trap 'rm -rf "$probe"' EXIT
mkdir -p "$probe/models/x"
printf '%s\n' '        --flag "a" \' '        # planted comment' '        --lost-flag "b"' > "$probe/models/x/planted.yml"
n="$(python3 - "$probe" <<'PYEOF'
import glob, os, sys
root=sys.argv[1]; n=0
for f in glob.glob(os.path.join(root,"models/**/*.yml"), recursive=True):
    L=open(f,encoding="utf-8").read().split("\n")
    for i in range(1,len(L)):
        if L[i-1].lstrip().startswith("#"): continue
        if L[i-1].rstrip().endswith("\\") and L[i].lstrip().startswith("#"): n+=1
print(n)
PYEOF
)"
[[ "$n" == "1" ]] || { echo "FAIL: positive control — expected 1 planted hit, got $n" >&2; exit 1; }
echo "  ✓ scanner detects a planted continuation-comment"

if [[ -n "$out" ]]; then
  echo "FAIL: comment follows a line continuation — every flag after it is SILENTLY DROPPED:" >&2
  echo "$out" >&2
  echo "  Move the comment ABOVE the command, or end the continuation first." >&2
  exit 1
fi
echo "PASS: test-compose-continuation-comments (#1292 regression guard)"
