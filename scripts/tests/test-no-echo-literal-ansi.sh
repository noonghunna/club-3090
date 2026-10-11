#!/usr/bin/env bash
#
# Guard: never hand an ANSI colour code written as a backslash sequence to a
# plain `echo` in scripts/.
#
# Why: bash's `echo` does not interpret backslash escapes unless given -e, so
# `echo "\033[33m⚠\033[0m …"` prints the characters backslash-0-3-3 instead of
# a colour. verify-stress.sh printed `\033[33m⚠\033[0m VRAM margin thin at
# ceiling` in four user reports (#1574, #1601, #1609, #1618), and its
# engine-restart messages had the same shape; gpu-mode.sh's prune modes did it
# through colour variables (`CYAN='\033[0;36m'` … `echo "${CYAN}…"`).
#
# Fix shape: printf with the codes in the format string (verify-stress.sh's own
# pass/fail helpers), `echo -e` (gpu-mode.sh's convention), or real escape bytes
# from `$'\033[33m'`.
#
# Two shapes are checked: a literal sequence inside echo's first quoted
# argument, and a variable assigned a backslash sequence (in quotes, not $'…')
# that a plain echo in the same file expands. scripts/tests/ is out of scope.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779 — python3 under a C/POSIX locale mangles non-ASCII
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# echo, any flags that do not include -e, then a quoted argument holding \033 / \x1b / \e[ / \E[.
LITERAL='^[^#]*\becho([[:space:]]+-[a-df-zA-Z]+)*[[:space:]]+("[^"]*|'"'"'[^'"'"']*)\\(033|x1b|e\[|E\[)'
scan_literal() {  # scan_literal <root> → path:line:text
  command grep -rnE "$LITERAL" "$1/scripts" --include='*.sh' 2>/dev/null \
    | command grep -v "^$1/scripts/tests/" || true
}
scan_vars() {  # scan_vars <root> → path:line:text
  python3 - "$1" <<'PY'
import os, re, sys
root = sys.argv[1]
assign = re.compile(r"""\b([A-Za-z_][A-Za-z0-9_]*)=(["'])\\(?:033|x1b|e\[|E\[)""")
plain_echo = re.compile(r"""\becho(?:\s+-[a-df-zA-Z]+)*\s+(?!-)""")   # flags, none of them -e
for dirpath, dirs, files in os.walk(os.path.join(root, "scripts")):
    if os.path.relpath(dirpath, root).split(os.sep)[:2] == ["scripts", "tests"]:
        dirs[:] = []; continue
    for f in sorted(files):
        if not f.endswith(".sh"):
            continue
        path = os.path.join(dirpath, f)
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        code = [l.split("#", 1)[0] if l.lstrip().startswith("#") else l for l in lines]
        names = {m.group(1) for l in code for m in assign.finditer(l)}
        if not names:
            continue
        uses = re.compile(r"\$\{?(?:%s)\b" % "|".join(sorted(names)))
        for n, l in enumerate(code, 1):
            for m in plain_echo.finditer(l):
                if uses.search(l, m.end()):
                    print(f"{path}:{n}:{lines[n-1].strip()}"); break
PY
}

# --- positive control FIRST: a scan that finds nothing is indistinguishable from
# a clean tree, so prove both scanners still see every spelling they claim to.
probe="$(mktemp -d)"; trap 'rm -rf "$probe"' EXIT
mkdir -p "$probe/scripts/lib" "$probe/scripts/tests"
cat > "$probe/scripts/planted.sh" <<'EOS'
echo "    \033[33m⚠\033[0m VRAM margin thin"
echo -n '\e[31mred\e[0m'
echo -E "\x1b[2mdim"
C_WARN="\033[1;33m"; C_RST="\033[0m"
CYAN='\033[0;36m'
echo "${CYAN}── Before ──"
  echo "  $C_WARN!$C_RST thin"
EOS
planted=$(( $(scan_literal "$probe" | command grep -c 'planted.sh' || true) \
          + $(scan_vars "$probe" | command grep -c 'planted.sh' || true) ))
if [[ "$planted" != "5" ]]; then
  echo "FAIL: positive control — expected 5 planted violations, found $planted" >&2
  scan_literal "$probe" >&2; scan_vars "$probe" >&2
  exit 1
fi
echo "  ✓ scanners detect literal \\033 / \\e[ / \\x1b in plain echo, with or without -n/-E, and colour variables"

# --- negative control: the fixed shapes, comments and tests must not register.
cat > "$probe/scripts/lib/fixed.sh" <<'EOS'
# Never `echo "\033[33m"` without -e (comment explaining the rule)
RED='\033[0;31m'; NC='\033[0m'
echo -e "${RED}error${NC}"
echo -ne "\033[2K\r"
printf "    \033[33m⚠\033[0m %s\n" "$msg"
printf "  ${RED}✗${NC} %s\n" "$1"
ok=$'\033[32m'; off=$'\033[0m'
echo "${ok}fine${off}"
echo "Restarting container '${CONTAINER}'…"
EOS
printf '%s\n' 'echo "\033[31mliteral in a test fixture"' > "$probe/scripts/tests/asserts-on-output.sh"
false_hits="$( { scan_literal "$probe"; scan_vars "$probe"; } | command grep -cv 'planted.sh' || true)"
if [[ "$false_hits" -ne 0 ]]; then
  echo "FAIL: negative control — scanners fired on fixed shapes / comments / tests:" >&2
  { scan_literal "$probe"; scan_vars "$probe"; } | command grep -v 'planted.sh' >&2
  exit 1
fi
echo "  ✓ echo -e, printf formats, \$'…' bytes, comments and scripts/tests/ do not register"

# --- the real tree --------------------------------------------------------------
hits="$( { scan_literal "$ROOT"; scan_vars "$ROOT"; } )"
if [[ -n "$hits" ]]; then
  echo "FAIL: plain echo given a backslash colour code — it prints \\033[…m as text (#1574):" >&2
  printf '%s\n' "$hits" | sed "s|$ROOT/||" >&2
  echo "Fix: printf \"…\\033[33m…\\033[0m %s\\n\" \"\$arg\", or echo -e, or \$'\\033[33m'." >&2
  exit 1
fi
echo "test-no-echo-literal-ansi: ok"
