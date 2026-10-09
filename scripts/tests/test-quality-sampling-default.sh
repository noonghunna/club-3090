#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-sampling-default — guards #1579's decision: quality numbers measure the model AS
# SERVED. quality-test.sh and rebench-full.sh default to the server's sampler (the compose's
# model-card sampler); the packs' fixed sampler stays as the opt-in reproducible baseline
# (--pack-sampling / SAMPLING_FROM_SERVER=0).
#
#   1. quality-test.sh: the default sends --sampling-from-server; --pack-sampling and
#      SAMPLING_FROM_SERVER=0 don't. Choices that already fix the sampler are not overridden:
#      explicit --temperature-style flags after `--` (benchlocal refuses them alongside server
#      sampling), --retry-failed (restores its baseline's), --resume (restores the original's —
#      and an explicit sampling flag with --resume is still refused).
#   2. --both-modes: both legs on the server sampler; --pack-sampling carries to both.
#   3. quality-baseline.sh: the regression gate uses --pack-sampling unless you choose.
#   4. rerun-failed-packs.sh: re-runs under the ORIGINAL result's sampler, overrides included.
#   5. rebench-full.sh: both 8-pack legs default SAMPLING_FROM_SERVER to 1.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

tmp="$(mktemp -d)"
before="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort \
    | comm -13 "$before" - | xargs -r rm -f || true
  rm -rf "$tmp"; rm -f "$before"
}
trap cleanup EXIT

mkdir -p "$tmp/bin"
cat > "$tmp/bin/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;; esac; done
exit 0
EOF
# Mock benchlocal-cli: one argv line per real run (help probes answered, not logged).
cat > "$tmp/bin/benchlocal-cli" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [[ "$a" == "--help" ]] && { echo "--run-meta --reasoning-effort --progress"; exit 0; }; done
printf '%s\n' "$*" >> "$MOCK_ARGV"
exit 0
EOF
chmod +x "$tmp/bin"/*

qt() {   # qt [VAR=val …] -- <quality-test args>; argv log in $tmp/argv, output in $out
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  : > "$tmp/argv"
  out="$(env PATH="$tmp/bin:$PATH" MOCK_ARGV="$tmp/argv" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
         "${envs[@]+"${envs[@]}"}" bash scripts/quality-test.sh "$@" 2>&1)" || rc=$?
}
has()  { command grep -q -- "$1" "$tmp/argv"; }

# ---------------------------------------------------------------------------
echo "--- 1. quality-test.sh ---"
qt -- --quick
has "--sampling-from-server" || { printf '%s\n' "$out" | tail -5 >&2; fail "the default should send --sampling-from-server"; }
[[ "$out" == *"sampling: the server's"* ]] || fail "the default should say it uses the server's sampler"
pass "default → the server's sampler"
qt -- --quick --pack-sampling
if has "--sampling-from-server"; then fail "--pack-sampling still sent --sampling-from-server"; fi
[[ "$out" == *"each pack's own fixed sampler"* ]] || fail "--pack-sampling should say so"
pass "--pack-sampling → the packs' fixed sampler"
qt SAMPLING_FROM_SERVER=0 -- --quick
if has "--sampling-from-server"; then fail "SAMPLING_FROM_SERVER=0 still sent --sampling-from-server"; fi
pass "SAMPLING_FROM_SERVER=0 → the packs' fixed sampler"
qt -- --quick -- --temperature 0.7 --top-p 0.8
if has "--sampling-from-server"; then fail "explicit --temperature got --sampling-from-server too (benchlocal refuses the pair)"; fi
has "--temperature 0.7" || fail "the explicit sampler flags should still reach benchlocal"
pass "explicit sampler flags after -- → no server sampling forced on them"
qt -- --quick -- --retry-failed 2
if has "--sampling-from-server"; then fail "--retry-failed got --sampling-from-server (it restores its baseline's)"; fi
pass "--retry-failed → its baseline's sampler"
: > "$tmp/run.partial.jsonl"
rc=0; qt -- --resume "$tmp/run.partial.jsonl"
if has "--sampling-from-server"; then fail "--resume got a sampling flag (it restores the original's)"; fi
pass "--resume → the original run's sampler, no flag added"
rc=0; qt -- --resume "$tmp/run.partial.jsonl" --pack-sampling
[[ "$rc" == "2" && "$out" == *"drop: --pack-sampling"* ]] || fail "--resume with an explicit sampling flag should still be refused (rc=$rc)"
pass "--resume + an explicit sampling flag → refused, as before"

# ---------------------------------------------------------------------------
echo "--- 2. --both-modes ---"
qt -- --quick --both-modes
[[ "$(command grep -c -- '--sampling-from-server' "$tmp/argv")" == "2" ]] || fail "--both-modes: both legs should be on the server sampler"
pass "both legs on the server's sampler"
qt -- --quick --both-modes --pack-sampling
if has "--sampling-from-server"; then fail "--both-modes --pack-sampling: a leg still used the server sampler"; fi
[[ "$(wc -l < "$tmp/argv")" == "2" ]] || fail "--both-modes --pack-sampling should still run two legs"
pass "--pack-sampling carries to both legs"

# ---------------------------------------------------------------------------
echo "--- 3. quality-baseline.sh ---"
mkdir -p "$tmp/baselines"
bl="$(bash scripts/quality-baseline.sh --slug vllm/minimal --capture --dry-run 2>&1)"
[[ "$bl" == *"--pack-sampling"* ]] || fail "quality-baseline should gate on the reproducible sampler: $bl"
bl="$(bash scripts/quality-baseline.sh --slug vllm/minimal --capture --dry-run --sampling-from-server 2>&1)"
[[ "$bl" != *"--pack-sampling"* ]] || fail "quality-baseline: an explicit --sampling-from-server should win"
pass "--pack-sampling by default; your choice wins"

# ---------------------------------------------------------------------------
echo "--- 4. rerun-failed-packs.sh ---"
mk() {   # mk <file> <json-fields>
  printf '{%s,"thinking_enabled":false,"packs":[{"pack_id":"toolcall-15","scenarios":[{"id":"TC-01","passed":false}]}]}\n' "$2" > "$1"
}
mk "$tmp/server.json" '"sampling_source":"server"'
mk "$tmp/pack.json" '"thinking_mode":"force-off"'
mk "$tmp/explicit.json" '"sampling_overrides":{"temperature":0.7,"top_p":0.8,"max_tokens":4096}'
r="$(RERUN_DRY=1 bash scripts/rerun-failed-packs.sh "$tmp/server.json" 2>&1)"
[[ "$r" == *"--sampling-from-server"* ]] || fail "rerun of a server-sampled result should use the server sampler: $r"
r="$(RERUN_DRY=1 bash scripts/rerun-failed-packs.sh "$tmp/pack.json" 2>&1)"
[[ "$r" == *"--pack-sampling"* ]] || fail "rerun of a pack-sampled result should use --pack-sampling: $r"
r="$(RERUN_DRY=1 bash scripts/rerun-failed-packs.sh "$tmp/explicit.json" 2>&1)"
[[ "$r" == *"--pack-sampling"* && "$r" == *"-- --temperature 0.7 --top-p 0.8"* ]] \
  || fail "rerun of an override run should replay its sampler overrides: $r"
[[ "$r" != *"max_tokens"* && "$r" != *"--max-tokens"* ]] || fail "rerun must not replay max_tokens as a sampler"
pass "server / pack / explicit-override originals each re-run under their own sampler"

# ---------------------------------------------------------------------------
echo "--- 5. rebench-full.sh ---"
n="$(command grep -c 'SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-1}"' scripts/rebench-full.sh)"
[[ "$n" == "2" ]] || fail "rebench-full.sh: both 8-pack legs should default SAMPLING_FROM_SERVER to 1 (found $n)"
if command grep -q 'SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-0}"' scripts/rebench-full.sh; then
  fail "rebench-full.sh still defaults a leg to the packs' sampler"
fi
pass "both 8-pack legs default to the server sampler"

echo "test-quality-sampling-default: ok"
