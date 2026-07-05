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
# guarantee the ADD paths below: strip any real vllm/dual row (the real file
# seeds one since wave-2) AND any real vllm/qwen-27b-dual-max entry (the real
# file carries a submission-only entry since slice 3 — section 5's
# submission-only case needs the slug absent)
python3 - "$BL" <<'PY'
import re
import sys

p = sys.argv[1]
old = open(p).read()
for slug in ("vllm/dual", "vllm/qwen-27b-dual-max"):
    old = re.sub(rf"^  {re.escape(slug)}:\n(?:^(?:    .*|\s*)\n)*?(?=^  \S|^#|^\S|\Z)",
                 "", old, count=1, flags=re.M)
open(p, "w").write(old)
PY

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
  echo "=== summary [narrative] (n=5) ==="
  echo "  decode_TPS     mean= 153.90   std=  0.10   CV= 0.1%   min=153.80   max=154.00"
  echo "  TTFT          mean=   130ms  std=    2ms  min=128ms  max=132ms"
  echo "========== CODE =========="
  echo "=== measured (5) ==="
  for i in 1 2 3 4 5; do echo "  run-$i  wall= 5.3s ttft= 128ms toks=800 wall_TPS=149.9 decode_TPS=154.0"; done
  echo "=== summary [code] (n=5) ==="
  echo "  decode_TPS     mean= 154.00   std=  0.10   CV= 0.1%   min=153.90   max=154.10"
  echo "  TTFT          mean=   128ms  std=    2ms  min=126ms  max=130ms"
  # 2c prefill-probe blocks: the 90K TTFT (17s!) must NOT bleed into the
  # canonical short-prompt ttft_ms (per-block parser protection).
  echo "=== summary [prefill-10k] (n=3) ==="
  echo "  prefill tok/s  mean=7976.62   std= 87.19   CV= 1.1%   min=7902.43   max=8072.65"
  echo "  TTFT          mean=  1254ms  std=   56ms  min=1193ms  max=1303ms"
  echo "=== summary [prefill-90k] (n=3) ==="
  echo "  prefill tok/s  mean=5459.14   std= 23.81   CV= 0.4%   min=5442.01   max=5486.33"
  echo "  TTFT          mean= 17096ms  std=   74ms  min=17012ms  max=17150ms"
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
    ✓ rung 1/6: target=95K  actual=94K tok (36%)  recalled 'violet chinchilla 79'  prefill=7402.9 t/s (13s)  VRAM_free=1403MB
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
# 2c: prefill depth points land; the canonical ttft_ms stays the SHORT-prompt
# value (the 90K block's 17s TTFT must not bleed into it).
grep -q 'prefill_tps: { 10k: 7977, 90k: 5459 }' <<<"$out" || fail "prefill_tps not extracted: $out"
grep -q 'ttft_ms: 130' <<<"$out" || fail "canonical ttft_ms polluted by probe blocks"
# anchor calibration: probe 5459 @90K vs ladder 7403 @94K → ratio 0.74 → OK.
grep -q "anchor calibration OK" <<<"$out" || fail "anchor calibration not reported"
grep -q "DRY RUN" <<<"$out" || fail "dry-run not flagged"
[[ "$(sha256sum "$BL" | cut -d' ' -f1)" == "$sum_before" ]] || fail "dry-run WROTE the file"

# ── 2. write path: add (row stripped from the copy above) then replace ───────
out="$(bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" 2>&1)" || fail "add run failed"
grep -q "added" <<<"$out" || fail "first induction did not ADD"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
row = d["baselines"]["vllm/dual"]
assert row["narr_tps"] == 153.9 and row["code_tps"] == 154.0, row
assert row["quality_8pk"] == "105/150" and row["quality_8pk_think_on"] == "110/150"
assert row["ctx_validated"]["tokens"] == 240635
assert row["engine_pin"] == "test/pin:v1" and row["submitted_by"] == "tester"
# added rows must land BEFORE the gap-list footer, not at EOF (marker-drift
# guard — the footer was retitled once and silently broke the insertion anchor)
txt = open(sys.argv[1]).read()
assert txt.index("  vllm/dual:") < txt.index("KNOWN GAPS"), \
    "added row landed after the gap-list footer (insertion anchor drifted)"
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

# ── 5. bundle mode (slice 3): provenance FROM the bundle → submissions map ───
# Synthetic volunteer bundle: the same gate artifacts + rig.txt +
# container-config.json (a 2×5090 rig, one global power line, a pin that is
# NOT this rig's resolved pin — so any local-fallback leak fails loudly).
BSRC="$TMP/bundle-src/results/rebench/synth-fp8w"
mkdir -p "$BSRC"
cp "$TAGD"/verify-full.log "$TAGD"/bench.log "$TAGD"/_internal.json \
   "$TAGD"/verify-stress.log "$BSRC"/
cat > "$BSRC/rig.txt" <<'EOF'
hostname: volunteer-box
GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaa)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-bbb)
power_cap_w: 575.00
EOF
cat > "$BSRC/container-config.json" <<'EOF'
[{"Config": {"Image": "vllm/test-pin:v9"}}]
EOF
tar czf "$TMP/bundle.tgz" -C "$TMP/bundle-src" results

BARGS=(--from-bundle "$TMP/bundle.tgz" --source "https://example.test/disc#42"
       --submitted-by volunteer1 --tps-only --baselines-file "$BL")

# refusals: --source and --submitted-by are the trust boundary — REQUIRED
if bash scripts/catalog-baseline.sh vllm/dual --from-bundle "$TMP/bundle.tgz" \
     --submitted-by v --tps-only --baselines-file "$BL" --dry-run >/dev/null 2>&1; then
  fail "bundle mode without --source must refuse"
fi
if bash scripts/catalog-baseline.sh vllm/dual --from-bundle "$TMP/bundle.tgz" \
     --source x --tps-only --baselines-file "$BL" --dry-run >/dev/null 2>&1; then
  fail "bundle mode without --submitted-by must refuse (no \$USER default)"
fi

# dry-run: every provenance field derived FROM the bundle, none from this rig
sum_before="$(sha256sum "$BL" | cut -d' ' -f1)"
out="$(bash scripts/catalog-baseline.sh vllm/dual "${BARGS[@]}" --dry-run 2>&1)"
grep -q 'rig: "2x5090-pcie"' <<<"$out" || fail "rig not derived from rig.txt: $out"
grep -q 'power_cap_w: \[575, 575\]' <<<"$out" || fail "one global cap line not replicated per card"
grep -q 'engine_pin: "vllm/test-pin:v9"' <<<"$out" || fail "pin not read from container-config.json"
grep -q 'tier: submitted' <<<"$out" || fail "tier: submitted missing"
[[ "$(sha256sum "$BL" | cut -d' ' -f1)" == "$sum_before" ]] || fail "bundle dry-run WROTE the file"

# write: the submissions map lands; the primary row is untouched
out="$(bash scripts/catalog-baseline.sh vllm/dual "${BARGS[@]}" 2>&1)" || fail "bundle add failed"
grep -q "added submissions map" <<<"$out" || fail "first bundle induction did not add the map"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
row = d["baselines"]["vllm/dual"]
assert row["narr_tps"] == 153.9 and row["engine_pin"] == "test/pin:v2", \
    "primary row disturbed by a bundle induction"
s = row["submissions"]["2x5090-pcie"]
assert s["narr_tps"] == 153.9 and s["tier"] == "submitted"
assert s["source"] == "https://example.test/disc#42" and s["submitted_by"] == "volunteer1"
assert s["power_cap_w"] == [575, 575] and s["engine_pin"] == "vllm/test-pin:v9"
assert s["rig"] == "2x5090-pcie"
PY

# same rig_class re-induction REPLACES (one row per rig_class, newest wins)
out="$(bash scripts/catalog-baseline.sh vllm/dual "${BARGS[@]}" 2>&1)" || fail "bundle replace failed"
grep -q "replaced submission" <<<"$out" || fail "same-rig_class re-induction did not replace"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
assert len(d["baselines"]["vllm/dual"]["submissions"]) == 1
PY

# primary re-induction PRESERVES the submissions map (the block regex would
# otherwise swallow it — the slice-3 splice-safety guard)
out="$(bash scripts/catalog-baseline.sh vllm/dual "${ARGS[@]}" --tps-only 2>&1)" \
  || fail "primary re-induction over a row with submissions failed"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
row = d["baselines"]["vllm/dual"]
assert row.get("tier") == "local", "primary induction must stamp tier: local"
assert row["submissions"]["2x5090-pcie"]["tier"] == "submitted", \
    "primary re-induction dropped the submissions map"
PY

# submission-only entry: a slug measured on hardware we don't have
out="$(bash scripts/catalog-baseline.sh vllm/qwen-27b-dual-max "${BARGS[@]}" 2>&1)" \
  || fail "submission-only add failed"
grep -q "submission-only" <<<"$out" || fail "did not create a submission-only entry"
python3 - "$BL" <<'PY'
import sys

import yaml

d = yaml.safe_load(open(sys.argv[1]))
row = d["baselines"]["vllm/qwen-27b-dual-max"]
assert "narr_tps" not in row, "submission-only entry must carry no primary fields"
assert row["submissions"]["2x5090-pcie"]["tier"] == "submitted"
txt = open(sys.argv[1]).read()
assert txt.index("  vllm/qwen-27b-dual-max:") < txt.index("KNOWN GAPS"), \
    "submission-only entry landed after the gap-list footer"
PY

# multi-tag bundle: refuse without --from-tag, select with it
mkdir -p "$TMP/bundle-src/results/rebench/synth-second"
cp "$BSRC"/* "$TMP/bundle-src/results/rebench/synth-second/"
tar czf "$TMP/bundle2.tgz" -C "$TMP/bundle-src" results
if bash scripts/catalog-baseline.sh vllm/dual --from-bundle "$TMP/bundle2.tgz" \
     --source x --submitted-by v --tps-only --baselines-file "$BL" --dry-run >/dev/null 2>&1; then
  fail "multi-tag bundle without --from-tag must refuse"
fi
out="$(bash scripts/catalog-baseline.sh vllm/dual --from-bundle "$TMP/bundle2.tgz" \
     --from-tag synth-second --source x --submitted-by v --tps-only \
     --baselines-file "$BL" --dry-run 2>&1)" || fail "multi-tag --from-tag selection failed"
grep -q 'source_tag: "synth-second"' <<<"$out" || fail "--from-tag did not select the tag in the bundle"

# real file untouched throughout (checksum across the run — git state may
# legitimately be dirty in a working checkout)
[[ "$(sha256sum scripts/lib/profiles/baselines.yml | cut -d' ' -f1)" == "$REAL_SUM_BEFORE" ]] \
  || fail "REAL baselines.yml was modified by the test"

echo "test-catalog-baseline: ok"
