#!/usr/bin/env bash
# Guard: what `catalog.sh register` can actually READ off a real-world compose
# and checkpoint, and what it must REFUSE before writing anything.
#
# Every case here is a failure a real third-party recipe produced on first
# contact (DominikBucko/qwen38-flash-next-2x3090, 2026-09-08). They share one
# shape: the derivation silently returned a wrong-but-plausible value, or died
# on an input the whole ecosystem uses.
#
#   1. NESTED CONFIG. A multimodal checkpoint (…ForConditionalGeneration) states
#      the text dims under `text_config`, so a top-level-only read sees None for
#      every dim and refuses a model it could describe perfectly well.
#   2. A DIRECTORY is what people pass to --weights, because it is what they
#      downloaded. Refusing it with advice they appear to have followed is a
#      dead end.
#   3. SHELL EXPANSIONS. `--max-model-len ${MAX_MODEL_LEN:-262144}` is the
#      commonest idiom in this repo's own composes; the raw token reached an
#      int() and raised an unhandled ValueError instead of refusing.
#   4. ENV-DRIVEN COMPOSES. A compose whose image builds its own command line
#      states the context in `environment:`. 75 composes here do it. Reading
#      only flags silently yielded a 4096 default for a 262K model.
#   5. UNKNOWN WORKLOAD must refuse BEFORE the write. compat validates
#      cross-references AFTER promote writes all three artifacts, and a dangling
#      workload ref does not just break the new row — load_profiles() then fails
#      for EVERY caller, so one typo makes the whole catalog, core included,
#      unloadable until a gitignored JSON is hand-edited.
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
rc=0

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/tools" "$TMP/weights"
cp -r scripts/lib "$TMP/scripts/"
cp -r tools/tui-core "$TMP/tools/"

# ── 1 + 2: nested config, reached through a DIRECTORY ────────────────
cat > "$TMP/weights/config.json" <<'JSON'
{"architectures": ["Qwen4ExpForConditionalGeneration"],
 "model_type": "qwen4_exp",
 "text_config": {"hidden_size": 2560, "num_hidden_layers": 48,
                 "num_attention_heads": 24, "num_key_value_heads": 2,
                 "max_position_embeddings": 262144},
 "vision_config": {"hidden_size": 1152}}
JSON

C="$TMP/envdriven.yml"
cat > "$C" <<'YML'
services:
  theirs:
    image: ghcr.io/someone/their-runtime:locked
    environment:
      PORT: "${PORT:-20299}"
      CUDA_VISIBLE_DEVICES: "0,1"
      MAX_MODEL_LEN: "${MAX_MODEL_LEN:-262144}"
      CPU_OFFLOAD_GB: "${CPU_OFFLOAD_GB:-30}"
      MTP_DEPTH: "${MTP_DEPTH:-3}"
    command: ["/model"]
YML

out="$(scripts/catalog.sh register --compose "$C" --engine their-vllm --engine-type vllm \
        --model nested-probe --weights "$TMP/weights" --workload long-ctx-single \
        --root "$TMP" --dry-run -y 2>&1)"; code=$?
if [[ "$code" -ne 0 ]]; then
  echo "  FAIL a DIRECTORY + nested text_config was refused (exit $code)"; rc=1
  printf '%s\n' "$out" | sed 's/^/       /' | tail -4
else
  echo "  ok   --weights <dir> resolved to config.json"
  command grep -q "hidden=2560 layers=48 heads=24/2" <<<"$out" \
    || { echo "  FAIL nested text_config dims not read: $(command grep -o 'arch .*' <<<"$out")"; rc=1; }
  command grep -q "hidden=2560" <<<"$out" && echo "  ok   nested text_config dims read"
  # 4: the context came from environment:, not a flag, and is NOT the 4096 default
  if command grep -qE "max_ctx +262144" <<<"$out"; then
    echo "  ok   env-driven MAX_MODEL_LEN read (not the 4096 fallback)"
  else
    echo "  FAIL env-driven context not read: $(command grep -o 'max_ctx.*' <<<"$out")"; rc=1
  fi
fi

# ── 3: a shell expansion in a FLAG must not reach int() ──────────────
C2="$TMP/expansion.yml"
cat > "$C2" <<'YML'
services:
  theirs:
    image: ghcr.io/someone/vllm-ish:v1
    ports:
      - "${PORT:-20298}:20298"
    command: >
      vllm serve /model --max-model-len ${MAX_MODEL_LEN:-${CTX:-131072}}
      --tensor-parallel-size ${TP:-2}
YML
out2="$(scripts/catalog.sh register --compose "$C2" --engine their-vllm --engine-type vllm \
         --model expansion-probe --weights "$TMP/weights" --workload long-ctx-single \
         --root "$TMP" --dry-run -y 2>&1)"; code2=$?
if [[ "$code2" -ne 0 ]]; then
  echo "  FAIL a \${VAR:-N} flag value crashed the derivation (exit $code2)"; rc=1
  printf '%s\n' "$out2" | sed 's/^/       /' | tail -3
elif command grep -qE "max_ctx +131072" <<<"$out2" && command grep -qE "tp +2" <<<"$out2"; then
  echo "  ok   \${VAR:-\${VAR2:-N}} resolved to its innermost default"
else
  echo "  FAIL expansion not resolved: $(command grep -oE 'max_ctx.*|tp +[0-9]+' <<<"$out2" | tr '\n' ' ')"; rc=1
fi

# ── 5: unknown workload refuses BEFORE writing ───────────────────────
snap() { find "$TMP/scripts/lib/profiles-local" -type f 2>/dev/null | sort | xargs -r md5sum | md5sum; }
before="$(snap)"
out3="$(scripts/catalog.sh register --compose "$C" --engine wl-engine --engine-type vllm \
         --model wl-probe --weights "$TMP/weights" --workload not-a-real-workload \
         --root "$TMP" -y 2>&1)"; code3=$?
if [[ "$code3" -eq 0 ]]; then
  echo "  FAIL an unknown workload was ACCEPTED — compat will fail for every caller"; rc=1
elif ! command grep -qi "unknown workload" <<<"$out3"; then
  echo "  FAIL refused, but not for the workload: $(tail -1 <<<"$out3")"; rc=1
else
  echo "  ok   unknown workload refused, and it named the known ones"
fi
if [[ "$before" != "$(snap)" ]]; then
  echo "  FAIL the refusal still WROTE to the local layer (the whole point is it must not)"; rc=1
else
  echo "  ok   nothing written on the refusal"
fi

# ── 6: the facts behind the c3 COLUMNS ───────────────────────────────
# spec / offload / provider / size / family / vision were hardcoded to
# None/""/1.0/"dense"/False, so a CPU-offloaded MoE with an MTP drafter
# registered as a plain resident dense model with no speculation. Every column
# blank reads as "this recipe does not do that", not as "nobody looked".
out4="$(scripts/catalog.sh register --compose "$C" --engine col-engine --engine-type vllm \
         --model col-probe --weights "$TMP/weights" --workload long-ctx-single \
         --hf-repo someorg/Some-Model-W4A16 --root "$TMP" --dry-run -y 2>&1)"
check() {  # check <label> <regex>
  if command grep -qE "$2" <<<"$out4"; then echo "  ok   $1"; else
    echo "  FAIL $1 — not in the pre-write table"; rc=1; fi
}
check "MTP depth read from an env-driven compose"  "spec +mtp n=3"
check "provider recorded from --hf-repo"           "provider +someorg/Some-Model-W4A16"
check "family read from the checkpoint, not 'dense'" "family +qwen"
check "vision_config seen"                          "vision +True"
if command grep -qE "size_gb +1\.0( |$)" <<<"$out4"; then
  echo "  FAIL size_gb is still the 1.0 placeholder"; rc=1
else
  echo "  ok   size_gb measured, not the 1.0 placeholder"
fi
# offload: PRESENT but backend unnamed -> must NOT be guessed
if command grep -qE "offload +on, backend unnamed" <<<"$out4"; then
  echo "  ok   offload evidence surfaced without guessing the backend"
else
  echo "  FAIL offload evidence not surfaced: $(command grep -oE 'offload .*' <<<"$out4")"; rc=1
fi
# and a NAMED backend is taken verbatim
C3="$TMP/named-backend.yml"
sed 's|CPU_OFFLOAD_GB: "${CPU_OFFLOAD_GB:-30}"|CPU_OFFLOAD_GB: "30"|' "$C" > "$C3" 2>/dev/null || cp "$C" "$C3"
printf '    command: ["--offload-backend", "uva"]\n' >> "$C3"
out5="$(scripts/catalog.sh register --compose "$C3" --engine col-engine --engine-type vllm \
         --model col-probe2 --weights "$TMP/weights" --workload long-ctx-single \
         --root "$TMP" --dry-run -y 2>&1)"
command grep -qE "offload +uva" <<<"$out5" \
  && echo "  ok   a NAMED offload backend is taken verbatim" \
  || { echo "  FAIL named backend not read: $(command grep -oE 'offload .*' <<<"$out5")"; rc=1; }

# ── 7: the local layer is visible to the WEIGHTS enrichment ──────────
# weights.py feeds c3's provider + GB columns and globbed only the curated
# models/ dir, so a local model's provider was blank whatever its profile said.
if python3 - <<'PYEOF'
import subprocess, sys, json, pathlib
src = pathlib.Path("scripts/lib/profiles/weights.py").read_text(encoding="utf-8")
sys.exit(0 if "profiles-local" in src else 1)
PYEOF
then echo "  ok   weights.py reads the local layer"
else echo "  FAIL weights.py still ignores profiles-local (provider/GB stay blank)"; rc=1; fi

[[ "$rc" == "0" ]] && echo "PASS: catalog.sh derivation is robust to real-world composes and checkpoints" \
                   || echo "FAIL: catalog.sh derivation regression"
exit "$rc"
