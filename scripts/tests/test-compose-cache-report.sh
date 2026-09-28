#!/usr/bin/env bash
#
# Guard: every SGLang compose must pass `--enable-cache-report`.
#
# WHY (club-3090#1297): SGLang gates the OpenAI `usage.prompt_tokens_details`
# block on that flag (entrypoints/openai/usage_processor.py —
# `enable_cache_report: bool = False`, then `if enable_cache_report:`). Without
# it EVERY response carries `prompt_tokens_details: null`, streaming and
# non-streaming alike, so a client cannot see how much of its prompt the prefix
# cache served. 0 of 15 SGLang composes passed it while 20 vLLM composes passed
# the counterpart `--enable-prompt-tokens-details` (#1246) — a silent asymmetry
# between the two engines we ship, with no error on either side.
#
# What it cost: on sgl/qwen38-27b-dual-superfast the engine logged
# `#cached-token: 4,160` out of a 4,212-token prompt (99% served from cache) and
# TTFT fell 31x on the repeat, while the API reported nothing. A 31x timing drop
# is an inference, not a measurement — and community bench reports taken on an
# SGLang slug were blind to the one axis several open questions turn on.
#
# ⚠️⚠️ ABSENT IS NOT ZERO ON SGLANG. Even with the flag ON, SGLang OMITS
# cached_tokens when the value is genuinely 0; vLLM instead sends
# `cached_tokens: 0` explicitly. So a consumer that reads "field missing" as
# "nothing was reused" cannot distinguish no-reuse from reporting-off — which is
# exactly the misreading #1297 is about. Read a missing field as UNKNOWN.
#
# ⚠️ Comment lines do not count. A flag that only appears in a header table is
# documentation, not an argument — #1292 shipped a whole compose whose flags
# were present in the file and absent from the process.
#
# ⚠️ Must FAIL against the pre-fix tree (0/15), or it is asserting the wrong
# thing. The positive control below proves the scanner can still see a miss.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
# #779: this gate shells out to python3, which decodes reads/argv with the
# LOCALE codec unless UTF-8 mode is on — and these composes are full of unicode.
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
scan="$tmp/scan.py"

# ONE scanner, used for both the control and the real run. A control that
# exercises different code proves nothing about the code that matters.
cat > "$scan" <<'PYEOF'
import glob, os, sys

root = sys.argv[1]
FLAG = "--enable-cache-report"
bad = []
seen = 0
for f in sorted(glob.glob(os.path.join(root, "models/*/sglang/compose/**/*.yml"),
                          recursive=True)):
    raw = open(f, encoding="utf-8", errors="replace").read()
    # Strip comment-only lines FIRST: a flag named in a header table is prose,
    # and counting it would let a compose document a knob it never passes.
    body = "\n".join(l for l in raw.split("\n") if not l.lstrip().startswith("#"))
    if "sglang.launch_server" not in body:
        continue          # not an SGLang server compose
    seen += 1
    if FLAG not in body:
        bad.append(os.path.relpath(f, root))
print("SEEN %d" % seen)
for b in bad:
    print("MISSING " + b)
PYEOF

# --- positive control FIRST -------------------------------------------------
# A scan that silently matches nothing is indistinguishable from a clean tree.
# Plant one compliant compose and one that names the flag ONLY in a comment.
mkdir -p "$tmp/probe/models/planted/sglang/compose/dual/q"
printf '%s\n' \
  '        exec python3 -m sglang.launch_server \' \
  '          --enable-cache-report \' \
  '          --host 0.0.0.0' \
  > "$tmp/probe/models/planted/sglang/compose/dual/q/ok.yml"
printf '%s\n' \
  '        # --enable-cache-report  <- documented but never passed' \
  '        exec python3 -m sglang.launch_server \' \
  '          --host 0.0.0.0' \
  > "$tmp/probe/models/planted/sglang/compose/dual/q/bad.yml"

ctl="$(python3 "$scan" "$tmp/probe")"
[[ "$(printf '%s\n' "$ctl" | command grep -c '^MISSING ')" == "1" ]] || {
  echo "FAIL: positive control — scanner should flag exactly 1 planted compose" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
[[ "$ctl" == *"SEEN 2"* ]] || {
  echo "FAIL: positive control — scanner should have seen 2 planted composes" >&2
  printf '%s\n' "$ctl" >&2; exit 1; }
echo "  ✓ scanner flags a missing flag and ignores one that is only a comment"

# --- the actual assertion ---------------------------------------------------
out="$(python3 "$scan" "$ROOT")"
seen="$(printf '%s\n' "$out" | command sed -n 's/^SEEN //p')"
missing="$(printf '%s\n' "$out" | command grep '^MISSING ' || true)"

# A scan that found no composes at all would "pass" vacuously.
if [[ -z "$seen" || "$seen" -lt 1 ]]; then
  echo "FAIL: scanned no SGLang composes — the search is wrong, not the tree" >&2
  exit 1
fi

if [[ -n "$missing" ]]; then
  echo "FAIL: SGLang compose(s) do not pass --enable-cache-report:" >&2
  printf '%s\n' "$missing" | command sed 's/^MISSING /        /' >&2
  echo "" >&2
  echo "  Without it every response carries prompt_tokens_details: null and the" >&2
  echo "  prefix cache is invisible to clients (club-3090#1297)." >&2
  exit 1
fi
echo "PASS: test-compose-cache-report — $seen/$seen SGLang composes pass --enable-cache-report (#1297)"
