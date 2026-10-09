#!/usr/bin/env bash
# test-tests-stub-lifecycle — tree-wide guard: a test that starts a stub server must be able to
# stop it. Two leak shapes were found in the wild, each leaving servers running for days:
#
#   1. A function that records "$!" called inside a command substitution. The PID lands in the
#      substitution's subshell and never reaches the EXIT trap. test-served-model-id.sh started
#      every stub that way and left six behind per run: ~60 orphans on one rig.
#   2. A long-lived server whose only stop is an explicit `kill` after use. Any failure between
#      start and kill under `set -e` exits first. test-concurrency-probe.sh's EXIT trap only
#      removed a temp dir, and two engine stubs were found orphaned days later.
#
# Rule 1: a function whose body records "$!" must not be called as "$(fn" / `fn`.
# Rule 2: a test that backgrounds a server (serve_forever / HTTPServer / http.server in the file
#         plus a background `&`) must set an EXIT trap whose handler kills: the trap string
#         itself, or the function it names, contains `kill`.
# Comment lines are ignored. The guard proves itself on built-in bad fixtures (negative controls).
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/guard.py" <<'PY'
import re, sys

FUNC = re.compile(r'(?:^|;)\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{')
PID_REC = re.compile(r'(\+=\(\s*"?\$!"?\s*\)|=\s*"?\$!"?(\s|;|$))')
SERVER = re.compile(r'serve_forever|HTTPServer|http\.server')
BG = re.compile(r'(^|[^&|>])&\s*($|;|#|[A-Za-z_][A-Za-z0-9_]*=\$!)')
TRAP = re.compile(r'(?:^|;)\s*trap\s+(.+?)\s+(?:EXIT|0)\b')

def strip_comment(line):
    """Drop a bash comment: an unquoted # at the start of a word (quote-aware, so '#' inside
    strings and ${#x} survive)."""
    q, prev = None, " "
    for i, c in enumerate(line):
        if q:
            if c == q and (q == "'" or line[i - 1] != "\\"):
                q = None
        elif c in "'\"":
            q = c
        elif c == "#" and prev in " \t;":
            return line[:i]
        prev = c
    return line

def code_lines(text):
    return [strip_comment(l) for l in text.split("\n")]

def functions(lines):
    out, i = {}, 0
    while i < len(lines):
        m = FUNC.search(lines[i])
        if m:
            name, body = m.group(1), [lines[i]]
            if lines[i].rstrip().endswith("}") and lines[i].count("{") <= lines[i].count("}"):
                out[name] = "\n".join(body); i += 1; continue
            j = i + 1
            while j < len(lines) and lines[j].rstrip() != "}":
                body.append(lines[j]); j += 1
            out[name] = "\n".join(body); i = j + 1; continue
        i += 1
    return out

def check(path):
    errs = []
    lines = code_lines(open(path, encoding="utf-8").read())
    text = "\n".join(lines)
    funcs = functions(lines)
    # Rule 1
    for name, body in funcs.items():
        if PID_REC.search(body):
            for n, l in enumerate(lines, 1):
                if re.search(r'(\$\(|`)\s*' + re.escape(name) + r'(\s|\)|`|$)', l):
                    errs.append(f"{path}:{n}: '{name}' records $! but is called inside a command substitution "
                                f"— the PID never reaches the EXIT trap; have it set a variable instead (printf -v)")
    # Rule 2
    if SERVER.search(text) and any(BG.search(l) for l in lines):
        handlers = [m.group(1).strip().strip("'\"") for m in (TRAP.search(l) for l in lines) if m]
        def kills(h):
            if "kill" in h:
                return True
            return any("kill" in funcs.get(tok, "") for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', h))
        if not any(kills(h) for h in handlers):
            errs.append(f"{path}: backgrounds a server but no EXIT trap kills it — a failure between start "
                        f"and kill leaves it running; register the PID and kill it in the EXIT trap")
    return errs

errs = [e for p in sys.argv[1:] for e in check(p)]
print("\n".join(errs))
sys.exit(1 if errs else 0)
PY

# ---- negative controls: the guard must catch both shapes -------------------------------
cat > "$TMP/bad1.sh" <<'SH'
PIDS=(); trap 'for p in "${PIDS[@]}"; do kill "$p"; done' EXIT
start_stub() {
  python3 -c 'import http.server; http.server.HTTPServer(("127.0.0.1",0), None).serve_forever()' &
  PIDS+=("$!")
  echo 1234
}
port="$(start_stub)"
SH
cat > "$TMP/bad2.sh" <<'SH'
set -euo pipefail
trap 'rm -rf "$TMP"' EXIT
python3 -c 'import http.server; http.server.HTTPServer(("127.0.0.1",0), None).serve_forever()' & SPID=$!
false
kill "$SPID"
SH
cat > "$TMP/good.sh" <<'SH'
PIDS=(); cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
start_stub() {  # sets $2; never call as "$(start_stub ...)"
  python3 -c 'import http.server; http.server.HTTPServer(("127.0.0.1",0), None).serve_forever()' &
  PIDS+=("$!")
  printf -v "$2" '%s' 1234
}
start_stub vllm port
SH
out="$(python3 "$TMP/guard.py" "$TMP/bad1.sh" || true)"
[[ "$out" == *"'start_stub' records \$! but is called inside a command substitution"* ]] \
  || { echo "FAIL: rule 1 did not fire on a \$(start_stub) fixture: $out" >&2; exit 1; }
out="$(python3 "$TMP/guard.py" "$TMP/bad2.sh" || true)"
[[ "$out" == *"backgrounds a server but no EXIT trap kills it"* ]] \
  || { echo "FAIL: rule 2 did not fire on a trap without kill: $out" >&2; exit 1; }
python3 "$TMP/guard.py" "$TMP/good.sh" >/dev/null || { echo "FAIL: the guard flags a correct fixture" >&2; exit 1; }
echo "  ✓ negative controls: both leak shapes caught; the correct shape passes"

# ---- the tree --------------------------------------------------------------------------
# (this file is skipped: its bad fixtures are deliberate)
mapfile -t files < <(command grep -rlE 'serve_forever|HTTPServer|http\.server|\$!' "$ROOT_DIR/scripts/tests" --include='*.sh' \
  | command grep -v '/test-tests-stub-lifecycle\.sh$' | sort)
[[ "${#files[@]}" -gt 10 ]] || { echo "FAIL: found only ${#files[@]} candidate tests — the search is broken" >&2; exit 1; }
if ! out="$(python3 "$TMP/guard.py" "${files[@]}")"; then
  echo "FAIL: stub server lifecycle:" >&2; echo "$out" | sed "s|$ROOT_DIR/||" >&2; exit 1
fi
echo "  ✓ ${#files[@]} tests that background processes: every stub server can be stopped"
echo "test-tests-stub-lifecycle: ok"
