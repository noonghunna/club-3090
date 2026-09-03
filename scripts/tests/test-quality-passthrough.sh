#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-passthrough — guards the #987/#1023 contract:
#
#   1. `--` pass-through forwards everything after it VERBATIM to
#      `benchlocal-cli run`.
#   2. Promoted first-class flags (--retry-runaways, --strict-thinking,
#      --report/--report-out) reach the benchlocal argv with validation.
#   3. DRIFT GUARD: every flag documented for quality-test.sh in CLAUDE.md
#      (AGENTS.md) / docs/QUALITY_TEST.md invocations is either forwarded by
#      the wrapper or appears after `--` (pass-through). This is the check
#      that would have caught CLAUDE.md recommending --strict-thinking while
#      the wrapper rejected it (#1023).
#
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER="scripts/quality-test.sh"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output/args to contain: $needle" >&2
    echo "--- got ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output/args NOT to contain: $needle" >&2
    echo "--- got ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

tmp_bin="$(mktemp -d)"
tmp_log="$(mktemp)"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin"
  rm -f "$tmp_log" "$before_list" "$after_list"
}
trap cleanup EXIT

cat > "${tmp_bin}/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models)
      printf '{"data":[{"id":"mock-model"}]}'
      exit 0
      ;;
  esac
done
exit 0
MOCK_CURL
chmod +x "${tmp_bin}/curl"

cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BENCHLOCAL_MOCK_LOG}"
json_out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-json)
      json_out="${2:-}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  cat > "$json_out" <<'JSON'
{"packs":[{"pack_id":"toolcall-15","status":"ok","passed":1,"total":1,"score":1.0}]}
JSON
fi
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}/benchlocal-cli"

qrun() {
  PATH="${tmp_bin}:$PATH" BENCHLOCAL_MOCK_LOG="$tmp_log" PREFLIGHT_NO_AUTODETECT=1 \
    URL=http://mock MODEL=mock-model bash "$WRAPPER" "$@"
}

# ---------------------------------------------------------------------------
echo "--- 1. '--' pass-through forwards everything verbatim, appended LAST ---"
: > "$tmp_log"
out="$(qrun --quick --no-thinking -- --retry-runaways --strict-thinking --report md 2>&1)"
assert_contains "$out" "[quality-test] pass-through (4 arg(s)): --retry-runaways --strict-thinking --report md"
args="$(cat "$tmp_log")"
assert_contains "$args" "--retry-runaways --strict-thinking --report md"
# pass-through args come LAST (argparse-style: last occurrence wins, so a
# pass-through flag can override a wrapper one)

# empty pass-through (`--` alone) forwards nothing and does not break the run
: > "$tmp_log"
qrun --quick -- >/dev/null 2>&1
args="$(cat "$tmp_log")"
assert_not_contains "$args" "pass-through"

# ---------------------------------------------------------------------------
echo "--- 2. promoted first-class flags reach the benchlocal argv ---"
: > "$tmp_log"
qrun --quick --retry-runaways >/dev/null 2>&1
args="$(cat "$tmp_log")"
assert_contains "$args" "--retry-runaways"
# exactly once (first-class forwarding, no duplicate via another path)
[[ "$(grep -o -- '--retry-runaways' <<<"$args" | wc -l)" == "1" ]] || { echo "--retry-runaways forwarded more than once: $args" >&2; exit 1; }

: > "$tmp_log"
qrun --quick --strict-thinking >/dev/null 2>&1
args="$(cat "$tmp_log")"
assert_contains "$args" "--strict-thinking"

: > "$tmp_log"
out="$(qrun --quick --report md 2>&1)"
assert_contains "$out" "[quality-test] report: Results Card v2 (md)"
args="$(cat "$tmp_log")"
assert_contains "$args" "--report md"

: > "$tmp_log"
out="$(qrun --quick --report md --report-out card.md 2>&1)"
assert_contains "$out" "[quality-test] report-out: card.md"
args="$(cat "$tmp_log")"
assert_contains "$args" "--report md --report-out card.md"

# validation: --report and --report-out require values; --report-out requires --report
for bad in "--report" "--report-out"; do
  if qrun --quick "$bad" >/dev/null 2>&1; then
    echo "expected '$bad' without a value to be rejected (exit 2)" >&2
    exit 1
  fi
done
if qrun --quick --report-out card.md >/dev/null 2>&1; then
  echo "expected --report-out without --report to be rejected (exit 2)" >&2
  exit 1
fi

# unknown flags still rejected (the wrapper did not turn into a no-op sieve)
if qrun --quick --bogus-flag >/dev/null 2>&1; then
  echo "expected --bogus-flag to be rejected (exit 2)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
echo "--- 3. drift guard: documented flags are executable ---"

# Flags the wrapper itself parses out of argv (from its own case arms).
wrapper_flags() {
  awk '
    /^while \[\[ \$# -gt 0 \]\]; do$/ { inloop=1; next }
    inloop && /^done$/ { exit }
    inloop && /^[[:space:]]+[^[:space:]#][^)#]*\)/ {
      line=$0
      sub(/^[[:space:]]+/, "", line); sub(/\).*/, "", line)
      n=split(line, toks, "|")
      for (i=1; i<=n; i++) if (toks[i] ~ /^-/) print toks[i]
    }
  ' "$1" | sort -u
}

# Scan fenced bash/sh blocks: any command invoking scripts/quality-test.sh may
# use ONLY wrapper-known flags BEFORE a bare `--`; anything after `--` is
# pass-through by construction. Prints "file:line FLAG" violations.
doc_flag_violations() {
  local known_file="$1"; shift
  python3 - "$known_file" "$@" <<'PY'
import sys

known_path, doc_paths = sys.argv[1], sys.argv[2:]
with open(known_path) as f:
    known = {ln.strip() for ln in f if ln.strip()}

def check_command(cmd, lineno, path, out):
    if "scripts/quality-test.sh" not in cmd:
        return
    tail = cmd.split("scripts/quality-test.sh", 1)[1]
    for tok in tail.split():
        if tok == "--":
            break  # everything after `--` is verbatim pass-through: reachable
        if tok.startswith("--") and tok not in known:
            out.append(f"{path}:{lineno} documents unexecutable flag {tok}")

violations = []
for path in doc_paths:
    with open(path) as f:
        lines = f.read().splitlines()
    in_fence = False
    fence_lang = ""
    buf, buf_start, pending = "", 0, False
    for lineno, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("```"):
            if pending:
                check_command(buf, buf_start, path, violations)
                buf, pending = "", False
            in_fence = not in_fence
            fence_lang = s[3:].strip().lower() if in_fence else ""
            continue
        if not in_fence or fence_lang not in ("", "bash", "sh", "shell", "console"):
            continue
        if not s:
            continue
        if s.endswith("\\"):
            if not pending:
                buf_start = lineno
            buf += " " + s[:-1].strip()
            pending = True
            continue
        if pending:
            check_command(buf + " " + s, buf_start, path, violations)
            buf, pending = "", False
        else:
            check_command(s, lineno, path, violations)

for v in violations:
    print(v)
sys.exit(1 if violations else 0)
PY
}

# PROSE drift scan — the check that catches the original #1023 bug (CLAUDE.md
# recommended --strict-thinking in a prose sentence; no fenced invocation to
# trip the block scanner above). Every flag-shaped token anywhere in the docs
# must be one of:
#   - a wrapper-known flag (forwarded by quality-test.sh itself),
#   - PASS_THROUGH_OK: documented as reachable via the `--` pass-through,
#   - OTHER_TOOL_OK: belongs to another command documented nearby
#     (rebench-full.sh, rescore, vLLM boot flags, switch.sh, ...),
#   - or on a line that is a DIRECT benchlocal-cli invocation ("benchlocal-cli
#     run" / starting with benchlocal-cli) — that tool's own surface.
# Merely MENTIONING benchlocal-cli does not exempt a line: that is exactly how
# the --strict-thinking recommendation slipped through.
doc_prose_violations() {
  local known_file="$1"; shift
  python3 - "$known_file" "$@" <<'PY'
import re, sys

known_path, doc_paths = sys.argv[1], sys.argv[2:]
with open(known_path) as f:
    known = {ln.strip() for ln in f if ln.strip()}

# benchlocal-cli flags intentionally documented as `--`-pass-through-reachable
PASS_THROUGH_OK = {
    "--model-turn-timeout", "--timeout-ceiling-s",
    "--request-delay", "--max-total-tokens", "--max-transient-retries",
    "--measured-tps", "--reference-tps", "--retry-on-timeout",
    "--temperature", "--top-p", "--top-k", "--min-p", "--repeat-penalty",
    "--enable-sandboxed-packs",          # aider pack via raw benchlocal-cli
    "--exit-on-regression",              # quality-baseline.sh passes it through
}
# flags belonging to OTHER commands the docs mention alongside quality-test.sh
OTHER_TOOL_OK = {
    "--with-8pack-thinking",             # rebench-full.sh
    "--reasoning-parser", "--served-model-name", "--add-host",  # vLLM / docker boot
    "--in-place",                        # benchlocal-cli rescore
    "--dry-run",                         # quality-baseline.sh / report.sh
    "--clear-default", "--set-default", "--profile-like",       # switch.sh
    "--spec-file",                       # promote.py / export_pr.py (#1143)
}
allowed = known | PASS_THROUGH_OK | OTHER_TOOL_OK

flag_re = re.compile(r"--[a-z0-9]+(?:-[a-z0-9]+)+")
link_re = re.compile(r"\]\([^)]*\)")  # strip markdown link targets (anchors)

violations = []
for path in doc_paths:
    for lineno, raw in enumerate(open(path), 1):
        line = link_re.sub("", raw)
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        # a DIRECT benchlocal-cli invocation (`benchlocal-cli run|inspect|rescore|…`)
        # owns its own flag surface; prose that merely mentions benchlocal-cli
        # does NOT.
        if re.search(r"[\s`]benchlocal-cli\s+(run|inspect|rescore|reproduce|list)\b", line):
            continue
        for tok in flag_re.findall(line):
            if tok not in allowed:
                violations.append(f"{path}:{lineno} documents unexecutable flag {tok}")

for v in violations:
    print(v)
sys.exit(1 if violations else 0)
PY
}

wrapper_flags "$WRAPPER" > "${tmp_bin}/known-flags.txt"
if ! grep -q '^--timeout-per-case$' "${tmp_bin}/known-flags.txt" \
  || ! grep -q '^--full$' "${tmp_bin}/known-flags.txt" \
  || ! grep -q '^--help$' "${tmp_bin}/known-flags.txt"; then
  echo "drift-guard self-check: failed to extract wrapper flags from $WRAPPER" >&2
  exit 1
fi
if violations="$(doc_flag_violations "${tmp_bin}/known-flags.txt" AGENTS.md docs/QUALITY_TEST.md 2>&1 \
  && doc_prose_violations "${tmp_bin}/known-flags.txt" AGENTS.md docs/QUALITY_TEST.md 2>&1)"; then
  :
else
  echo "DRIFT GUARD FAILED — documented quality-test.sh flags are not executable:" >&2
  echo "$violations" >&2
  exit 1
fi

# negative control A: a hypothetical undocumented flag in a doc invocation MUST fail
cat > "${tmp_bin}/bad-doc.md" <<'MD'
## Example

```bash
bash scripts/quality-test.sh --full --hypothetical-future-flag --no-thinking
```
MD
if doc_flag_violations "${tmp_bin}/known-flags.txt" "${tmp_bin}/bad-doc.md" >/dev/null 2>&1; then
  echo "drift guard did NOT catch a hypothetical undocumented flag (fenced)" >&2
  exit 1
fi

# negative control A2: the same flag in PROSE mentioning quality-test.sh MUST fail
cat > "${tmp_bin}/bad-prose.md" <<'MD'
For CI, run `bash scripts/quality-test.sh --quick --hypothetical-future-flag`.
MD
if doc_prose_violations "${tmp_bin}/known-flags.txt" "${tmp_bin}/bad-prose.md" >/dev/null 2>&1; then
  echo "drift guard did NOT catch a hypothetical undocumented flag (prose)" >&2
  exit 1
fi

# negative control B: the same flag AFTER `--` is reachable — must pass
cat > "${tmp_bin}/good-doc.md" <<'MD'
```bash
bash scripts/quality-test.sh --full --no-thinking -- --hypothetical-future-flag
```
MD
doc_flag_violations "${tmp_bin}/known-flags.txt" "${tmp_bin}/good-doc.md" >/dev/null

# negative control C: reproduces the ORIGINAL #1023 bug — a doc that recommends
# --strict-thinking in prose while the wrapper does NOT forward it must FAIL.
# (Remove the promoted flags from the known set to simulate the pre-fix wrapper.)
grep -v -e '^--strict-thinking$' -e '^--retry-runaways$' -e '^--report$' -e '^--report-out$' \
  "${tmp_bin}/known-flags.txt" > "${tmp_bin}/pre-fix-flags.txt"
cat > "${tmp_bin}/pre-fix-doc.md" <<'MD'
benchlocal-cli flags both failure modes automatically: per-pack `thinking_validity` in the saved JSON, and `--strict-thinking` for a CI exit code.
MD
if doc_prose_violations "${tmp_bin}/pre-fix-flags.txt" "${tmp_bin}/pre-fix-doc.md" >/dev/null 2>&1; then
  echo "drift guard would NOT have caught the original --strict-thinking drift" >&2
  exit 1
fi

echo "test-quality-passthrough: ok"
