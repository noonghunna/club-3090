#!/usr/bin/env bash
# test-catalog-baseline — the baseline induction tool's contract
# (catalog-baselines slice 2, scripts/catalog-baseline.sh).
#
# Hermetic: a synthetic rebench tag dir + a COPY of baselines.yml; provenance
# passed explicitly (no nvidia-smi / docker dependency). Asserts the gate
# refusals (fail-loud), the extraction, the dry-run no-write guarantee, and
# the write-path upsert (both add and replace).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
TAGD="$TMP/fake-tag"
mkdir -p "$TAGD"
BL="$TMP/baselines.yml"
cp scripts/lib/profiles/baselines.yml "$BL"
REAL_SUM_BEFORE="$(sha256sum scripts/lib/profiles/baselines.yml | cut -d' ' -f1)"

fail() { echo "FAIL: $1" >&2; exit 1; }

# ── fixture tag dir (A1-shaped artifacts) ────────────────────────────────────
cat > "$TAGD/verify-full.log" <<'EOF'
[9/9] MTP acceptance length threshold ...
All checks passed. Stack is ready for full-functionality use.
EOF
# bench.log: 5 measured runs per prompt (2 prompts → 10 run-N lines)
{
  echo "========== NARRATIVE =========="
  echo "=== measured (5) ==="
  for i in 1 2 3 4 5; do echo "  run-$i  wall= 6.6s ttft= 130ms toks=1000 wall_TPS=150.0 decode_TPS=153.9"; done
  echo "========== CODE =========="
  echo "=== measured (5) ==="
  for i in 1 2 3 4 5; do echo "  run-$i  wall= 5.3s ttft= 128ms toks=800 wall_TPS=149.9 decode_TPS=154.0"; done
} > "$TAGD/bench.log"
cat > "$TAGD/_internal.json" <<'EOF'
{"bench": {"narrative": {"decode_tps_mean": 153.9, "ttft_ms_mean": 130.0},
           "code": {"decode_tps_mean": 154.0, "ttft_ms_mean": 128.0}}}
EOF
cat > "$TAGD/quality-full.json" <<'EOF'
{"packs": [{"passed": 55, "total": 75}, {"passed": 50, "total": 75}]}
EOF
cat > "$TAGD/quality-full-thinking.json" <<'EOF'
{"packs": [{"passed": 60, "total": 75}, {"passed": 50, "total": 75}]}
EOF
cat > "$TAGD/verify-stress.log" <<'EOF'
  ✓ ceiling ladder: all 6 rungs passed — fillable to 240635 tok (91% of n_ctx=262144)
All stress / boundary checks passed. KV-cache and prefill paths are sound.
EOF

ARGS=(--tag-dir "$TAGD" --engine-pin "test/pin:v1" --rig "2x3090-pcie"
      --power-cap-w "370,420" --submitted-by tester --baselines-file "$BL")

# ── 1. dry-run happy path: extraction + NO write ─────────────────────────────
sum_before="$(sha256sum "$BL" | cut -d' ' -f1)"
out="$(bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" --dry-run 2>&1)"
grep -q "narr_tps: 153.9" <<<"$out" || fail "narr_tps not extracted: $out"
grep -q "code_tps: 154.0" <<<"$out" || fail "code_tps not extracted"
grep -q 'quality_8pk: "105/150"' <<<"$out" || fail "quality_8pk not extracted"
grep -q 'quality_8pk_think_on: "110/150"' <<<"$out" || fail "think-on not extracted"
grep -q 'tokens: 240635' <<<"$out" || fail "ctx_validated not extracted"
grep -q "DRY RUN" <<<"$out" || fail "dry-run not flagged"
[[ "$(sha256sum "$BL" | cut -d' ' -f1)" == "$sum_before" ]] || fail "dry-run WROTE the file"

# ── 2. write path: add (vllm/dual has no row) then replace ───────────────────
bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" >/dev/null 2>&1
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
row = d["baselines"]["vllm/dual"]
assert row["narr_tps"] == 153.9 and row["code_tps"] == 154.0, row
assert row["quality_8pk"] == "105/150" and row["quality_8pk_think_on"] == "110/150"
assert row["ctx_validated"]["tokens"] == 240635
assert row["engine_pin"] == "test/pin:v1" and row["submitted_by"] == "tester"
PY
# replace: run again with a different pin — one row, updated in place.
out="$(bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]/test\/pin:v1/test/pin:v2}" 2>&1)" || fail "replace run failed"
grep -q "replaced" <<<"$out" || fail "second induction did not replace"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
assert d["baselines"]["vllm/dual"]["engine_pin"] == "test/pin:v2"
PY

# ── 3. refusals (fail-loud) ──────────────────────────────────────────────────
rm "$TAGD/verify-full.log"
if bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" --dry-run >/dev/null 2>&1; then
  fail "missing verify-full must refuse"
fi
echo "All checks passed" > "$TAGD/verify-full.log"
rm "$TAGD/quality-full.json" "$TAGD/quality-full-thinking.json"
if bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" --dry-run >/dev/null 2>&1; then
  fail "missing quality without --tps-only must refuse"
fi
out="$(bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" --tps-only --dry-run 2>&1)" \
  || fail "--tps-only should proceed"
grep -q "WITHOUT a quality run" <<<"$out" || fail "--tps-only must WARN"

# ── 4. unknown slug refuses ──────────────────────────────────────────────────
if bash scripts/catalog-baseline.sh vllm/__nope__ "${ARGS[@]}" --dry-run >/dev/null 2>&1; then
  fail "unknown slug must refuse"
fi

# real file untouched throughout (checksum across the run — git state may
# legitimately be dirty in a working checkout)
[[ "$(sha256sum scripts/lib/profiles/baselines.yml | cut -d' ' -f1)" == "$REAL_SUM_BEFORE" ]] \
  || fail "REAL baselines.yml was modified by the test"

echo "test-catalog-baseline: ok"
