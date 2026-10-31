#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output not to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

run_deps() {
  local compose="$1"
  local model_dir="$2"
  (
    export ROOT_DIR MODEL_DIR="$model_dir"
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_compose_deps "$compose"
  ) 2>&1
}

run_deps_unset_model_dir() {
  local compose="$1"
  (
    unset MODEL_DIR
    export ROOT_DIR
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_compose_deps "$compose"
  ) 2>&1
}

expect_missing() {
  local compose="$1"
  local model_dir="$2"
  local output

  if output="$(run_deps "$compose" "$model_dir")"; then
    echo "ASSERTION FAILED: expected missing-model failure for $compose" >&2
    echo "--- output ---" >&2
    echo "$output" >&2
    exit 1
  fi
  printf '%s' "$output"
}

ik_compose="${TMP_DIR}/ik.yml"
cat > "$ik_compose" <<'YAML'
services:
  ik:
    image: ${IK_LLAMA_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp@sha256:5f914f1ccade922417af58c94bd1cbb558052c8852d86678ead3fe693eec0143}
    command: >-
      --model /models/${GGUF_FILE:-qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf}
YAML

out="$(expect_missing "$ik_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf"
assert_contains "$out" "hf download ubergarm/Qwen3.6-27B-GGUF Qwen3.6-27B-MTP-IQ4_KS.gguf"
assert_contains "$out" "WEIGHTS=iq4ks bash scripts/setup.sh qwen3.6-27b"
assert_not_contains "$out" "qwen3.6-27b-autoround-int4"

llama_compose="${TMP_DIR}/llama.yml"
cat > "$llama_compose" <<'YAML'
services:
  llama:
    image: ${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b10236}
    command: >-
      -m /models/${GGUF_FILE:-qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf}
YAML
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-gguf/unsloth-mtp-q4km"
touch "${TMP_DIR}/models/qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf"
out="$(run_deps "$llama_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

vllm_compose="${TMP_DIR}/gemma.yml"
cat > "$vllm_compose" <<'YAML'
services:
  vllm:
    image: vllm/vllm-openai:v0.21.0
    command:
      - --model
      - /root/.cache/huggingface/gemma-4-31b-autoround-int4
      - --speculative-config
      - '{"model":"/root/.cache/huggingface/gemma-4-31b-it-assistant","num_speculative_tokens":4}'
YAML
out="$(expect_missing "$vllm_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "gemma-4-31b-autoround-int4/config.json"
assert_contains "$out" "gemma-4-31b-it-assistant/config.json"
assert_contains "$out" "hf download Intel/gemma-4-31B-it-int4-AutoRound"
assert_contains "$out" "hf download google/gemma-4-31B-it-assistant"
assert_contains "$out" "bash scripts/setup.sh gemma-4-31b"
assert_not_contains "$out" "qwen3.6-27b-autoround-int4"
mkdir -p "${TMP_DIR}/models/gemma-4-31b-autoround-int4" "${TMP_DIR}/models/gemma-4-31b-it-assistant"
touch "${TMP_DIR}/models/gemma-4-31b-autoround-int4/config.json" "${TMP_DIR}/models/gemma-4-31b-it-assistant/config.json"
out="$(run_deps "$vllm_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]


extends_base="${TMP_DIR}/base.yml"
extends_stub="${TMP_DIR}/stub.yml"
cat > "$extends_base" <<'YAML'
services:
  vllm-base:
    image: vllm/vllm-openai:v0.21.0
    command:
      - --model
      - /root/.cache/huggingface/qwen3.6-27b-autoround-int4
YAML
cat > "$extends_stub" <<'YAML'
services:
  vllm-stub:
    extends:
      file: base.yml
      service: vllm-base
YAML
out="$(expect_missing "$extends_stub" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4/config.json"

if out="$(run_deps_unset_model_dir "$extends_stub")"; then
  echo "ASSERTION FAILED: expected missing-model failure with unset MODEL_DIR" >&2
  echo "--- output ---" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "MODEL_DIR not set"
assert_contains "$out" "defaulting to ${ROOT_DIR}/models-cache"

out="$(CLUB3090_WEIGHTS_READER_DISABLE=1 expect_missing "$extends_stub" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4/config.json"
assert_contains "$out" "Check the compose header for its model-specific hf download command."
assert_not_contains "$out" "hf download Lorbus/Qwen3.6-27B-int4-AutoRound"

sglang_compose="${TMP_DIR}/sglang.yml"
cat > "$sglang_compose" <<'YAML'
services:
  sglang:
    image: ghcr.io/sgl-project/sglang:v0.5.12
    volumes:
      - "${MODEL_DIR:?MODEL_DIR is required}/qwen3.6-27b-autoround-int4:/models/target:ro"
      - "${MODEL_DIR}/qwen3.6-27b-prism-eagle3/compressed:/models/drafter:ro"
YAML
out="$(expect_missing "$sglang_compose" "${TMP_DIR}/empty-models")"
assert_contains "$out" "qwen3.6-27b-autoround-int4"
assert_contains "$out" "qwen3.6-27b-prism-eagle3/compressed"
assert_contains "$out" "hf download Ex0bit/Qwen3.6-27B-PRISM-EAGLE3"
assert_contains "$out" "WITH_PRISM_EAGLE3=1 bash scripts/setup.sh qwen3.6-27b"
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-autoround-int4" "${TMP_DIR}/models/qwen3.6-27b-prism-eagle3/compressed"
out="$(run_deps "$sglang_compose" "${TMP_DIR}/models")"
[[ -z "$out" ]]

# beellama.cpp DFlash: the image is NOT ggml-org/ik-llama (so it must still be
# detected as a llama.cpp-family GGUF server), and the drafter is named via
# --spec-draft-model (not -m). A present target + MISSING drafter must refuse
# with the drafter path + its hf-download hint — the #288 George report, where
# a missing drafter otherwise crashed cryptically in-container.
beellama_compose="${TMP_DIR}/beellama.yml"
cat > "$beellama_compose" <<'YAML'
services:
  beellama:
    image: ${BEELLAMA_IMAGE:-ghcr.io/anbeeld/beellama.cpp:server-cuda-v0.3.0-e0663be2713c}
    command: >-
      -m /models/${GGUF_FILE:-qwen3.6-27b-gguf/unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf}
      --spec-draft-model /models/${DRAFT_FILE:-qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf}
YAML
mkdir -p "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/unsloth-q5ks"
touch "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf"
out="$(expect_missing "$beellama_compose" "${TMP_DIR}/bl-models")"
assert_contains "$out" "qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
assert_contains "$out" "speculative drafter GGUF"
assert_contains "$out" "hf download Anbeeld/Qwen3.6-27B-DFlash-GGUF Qwen3.6-27B-DFlash-IQ4_XS.gguf"
# the target is present, so it must NOT be reported missing
assert_not_contains "$out" "unsloth-q5ks/Qwen3.6-27B-Q5_K_S.gguf (llama.cpp GGUF weights)"
# add the drafter → must pass
mkdir -p "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/anbeeld-dflash-iq4xs"
touch "${TMP_DIR}/bl-models/qwen3.6-27b-gguf/anbeeld-dflash-iq4xs/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
out="$(run_deps "$beellama_compose" "${TMP_DIR}/bl-models")"
[[ -z "$out" ]]

# A MODEL_DIR volume spelled with a DEFAULT, `${MODEL_DIR:-/abs}/sub`, plus a header
# comment that mentions `${MODEL_DIR}/sub`. Both were misread: the default resolved
# into the path and model_dir was prefixed again (a doubled path), and the comment
# was checked as a mount (with its trailing punctuation). Found on a local Strata
# recipe that boots fine with `docker compose` but was refused by switch.sh.
dflt_compose="${TMP_DIR}/defaulted-volume.yml"
cat > "$dflt_compose" <<'YAML'
# Model: downloaded to ${MODEL_DIR}/vendor-gguf/IQ3_S; setup checks it.
services:
  engine:
    image: example
    volumes:
      - "${MODEL_DIR:-/mnt/models/huggingface}/vendor-gguf/IQ3_S:/gguf:ro"
YAML
mkdir -p "${TMP_DIR}/dflt-models/vendor-gguf/IQ3_S"
out="$(run_deps "$dflt_compose" "${TMP_DIR}/dflt-models")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: defaulted volume present but reported: $out" >&2; exit 1; }
# NEGATIVE CONTROL: the folder absent → reported missing, ONCE, at the right single path.
rmdir "${TMP_DIR}/dflt-models/vendor-gguf/IQ3_S"
out="$(expect_missing "$dflt_compose" "${TMP_DIR}/dflt-models")"
assert_contains "$out" "${TMP_DIR}/dflt-models/vendor-gguf/IQ3_S (MODEL_DIR volume path)"
assert_not_contains "$out" "/mnt/models/huggingface/vendor-gguf"
assert_not_contains "$out" "IQ3_S;"

# bucko-vllm: the whole ${MODEL_DIR} is mounted at /models and the model (plus the
# Swift slug's drafter) are bare exec-form `command:` arguments. Nothing read them, so
# preflight passed with no weights on disk and the boot failed inside the container
# (Discussion #1700). Run against the REAL composes: a fixture would test the fixture.
bucko_main="${ROOT_DIR}/models/qwen3.8-flash-next/bucko-vllm/compose/dual/albucino-autoround-int4-fp8ple/base.yml"
bucko_swift="${ROOT_DIR}/models/qwen3.8-flash-next/bucko-vllm/compose/dual/ukisai-swift15-autoround-int4/base.yml"
mkdir -p "${TMP_DIR}/bucko-empty"
out="$(expect_missing "$bucko_main" "${TMP_DIR}/bucko-empty")"
assert_contains "$out" "${TMP_DIR}/bucko-empty/qwen3.8-flash-next-w4a16-fp8ple/config.json (HF model)"
# The hint must offer the main weights, not the drafter that shares their folder.
assert_contains "$out" "WEIGHT_KEY=qwen3.8-flash-next:albucino-autoround-int4-fp8ple"
out="$(expect_missing "$bucko_swift" "${TMP_DIR}/bucko-empty")"
assert_contains "$out" "${TMP_DIR}/bucko-empty/swift-1.5-qwen3.8-flash-next-w4a16-autoround/config.json (HF model)"
assert_contains "$out" "${TMP_DIR}/bucko-empty/qwen3.8-flash-next-w4a16-fp8ple/runtime/mtp-int4-g32/config.json (HF model)"
assert_contains "$out" "WEIGHT_KEY=qwen3.8-flash-next:albucino-mtp-int4-g32"
# POSITIVE CONTROL: both folders present → no complaint.
mkdir -p "${TMP_DIR}/bucko-ok/qwen3.8-flash-next-w4a16-fp8ple/runtime/mtp-int4-g32" \
         "${TMP_DIR}/bucko-ok/swift-1.5-qwen3.8-flash-next-w4a16-autoround"
touch "${TMP_DIR}/bucko-ok/qwen3.8-flash-next-w4a16-fp8ple/config.json" \
      "${TMP_DIR}/bucko-ok/qwen3.8-flash-next-w4a16-fp8ple/runtime/mtp-int4-g32/config.json" \
      "${TMP_DIR}/bucko-ok/swift-1.5-qwen3.8-flash-next-w4a16-autoround/config.json"
out="$(run_deps "$bucko_main" "${TMP_DIR}/bucko-ok")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: bucko weights present but reported: $out" >&2; exit 1; }
out="$(run_deps "$bucko_swift" "${TMP_DIR}/bucko-ok")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: bucko Swift weights present but reported: $out" >&2; exit 1; }
# A folder symlinked OUT of ${MODEL_DIR} is invisible in the container, because only
# ${MODEL_DIR} is mounted: refused with the symlink fix, not a re-download.
mkdir -p "${TMP_DIR}/bucko-link"
ln -s "${TMP_DIR}/bucko-ok/qwen3.8-flash-next-w4a16-fp8ple" "${TMP_DIR}/bucko-link/qwen3.8-flash-next-w4a16-fp8ple"
if out="$(run_deps "$bucko_main" "${TMP_DIR}/bucko-link")"; then
  echo "ASSERTION FAILED: a model folder symlinked out of MODEL_DIR passed preflight" >&2; exit 1
fi
assert_contains "$out" "resolve OUTSIDE \$MODEL_DIR through a symlink"
# A link that stays inside ${MODEL_DIR} (another name for the same folder) is fine.
mkdir -p "${TMP_DIR}/bucko-rel/elsewhere"
cp -r "${TMP_DIR}/bucko-ok/qwen3.8-flash-next-w4a16-fp8ple" "${TMP_DIR}/bucko-rel/elsewhere/"
ln -s "elsewhere/qwen3.8-flash-next-w4a16-fp8ple" "${TMP_DIR}/bucko-rel/qwen3.8-flash-next-w4a16-fp8ple"
out="$(run_deps "$bucko_main" "${TMP_DIR}/bucko-rel")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: an in-MODEL_DIR relative symlink was refused: $out" >&2; exit 1; }
# ExLlamaV3 / TabbyAPI: the whole ${MODEL_DIR} at /models and the model named as
# `--model-dir /models --model-name <folder>`, a folder name rather than a /models/<path>
# token, so no reader saw it: preflight passed with no weights on disk and `weights.sh
# show` listed no folder for the exl3 slugs. Run against the REAL composes.
exl3_305="${ROOT_DIR}/models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-3.05bpw/cpumoe.yml"
mkdir -p "${TMP_DIR}/exl3-empty"
out="$(expect_missing "$exl3_305" "${TMP_DIR}/exl3-empty")"
assert_contains "$out" "${TMP_DIR}/exl3-empty/qwen3.8-flash-next-exl3-3.05bpw/config.json (HF model)"
assert_contains "$out" "WEIGHT_KEY=qwen3.8-flash-next:exl3-3.05bpw"
# POSITIVE CONTROL: the folder present → no complaint.
mkdir -p "${TMP_DIR}/exl3-ok/qwen3.8-flash-next-exl3-3.05bpw"
touch "${TMP_DIR}/exl3-ok/qwen3.8-flash-next-exl3-3.05bpw/config.json"
out="$(run_deps "$exl3_305" "${TMP_DIR}/exl3-ok")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: exl3 weights present but reported: $out" >&2; exit 1; }
# The reader on its own: the inline `command: [...]` form works, and a --model-dir that
# isn't the /models mount gives nothing rather than a guessed host path.
tabby() { printf '%s\n' "services:" "  t:" "    volumes:" '      - "${MODEL_DIR:-x}:/models:ro"' "$1" > "${TMP_DIR}/tabby.yml"
  ( export ROOT_DIR; source "${ROOT_DIR}/scripts/preflight.sh"; preflight_compose_weight_paths "${TMP_DIR}/tabby.yml" ); }
got="$(tabby "    command: ['--model-dir', '/models', '--model-name', 'flow-model']")"
assert_contains "$got" $'command\tflow-model'
got="$(tabby "    command: ['--model-dir', '/elsewhere', '--model-name', 'other-model']")"
[[ -z "$got" ]] || { echo "ASSERTION FAILED: a --model-dir other than /models was mapped onto MODEL_DIR: $got" >&2; exit 1; }
# Only `command:` lists are read: a `/models/…` bind-mount TARGET (as SGLang's chat
# template mounts) must not be checked as a host path.
target_compose="${TMP_DIR}/mount-target.yml"
cat > "$target_compose" <<'YAML'
services:
  engine:
    image: example
    volumes:
      - "${MODEL_DIR}:/models:ro"
      - ./chat_template.jinja:/models/not-a-model/chat_template.jinja:ro
    environment:
      - M=/models/also-not-checked
YAML
out="$(run_deps "$target_compose" "${TMP_DIR}/bucko-empty")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: a /models mount target was checked as weights: $out" >&2; exit 1; }

# SGLang: the whole ${MODEL_DIR} is mounted at /models and a shell-script `command:`
# sets M=/models/<sub> before `--model-path "$$M"`. Neither reader above saw it, so
# every SGLang slug passed preflight with no weights on disk and the user got an
# engine traceback instead of the recipe. The REAL composes again: Nanbeige (no
# drafter), a Qwen3.8 DFlash2 slug (drafter behind ${DRAFTER_DIR:-…}, chat template
# mounted inside the model folder) and a ThinkingCap one (the model hint).
sgl_nanbeige="${ROOT_DIR}/models/nanbeige4.3-33b/sglang/compose/dual/autoround-int4/fp8.yml"
sgl_qwen="${ROOT_DIR}/models/qwen3.8-27b/sglang/compose/dual/autoround-int4/dflash2.yml"
sgl_tcap="${ROOT_DIR}/models/thinkingcap-qwen3.8-27b/sglang/compose/dual/fp8/mtp.yml"
mkdir -p "${TMP_DIR}/sgl-empty"
out="$(expect_missing "$sgl_nanbeige" "${TMP_DIR}/sgl-empty")"
assert_contains "$out" "${TMP_DIR}/sgl-empty/nanbeige4.3-33b-autoround-int4/config.json (HF model)"
assert_contains "$out" "hf download wasifb/Nanbeige4.3-33B-AutoRound-W4A16 --revision b87aa7fa5462094f8ea714767b7ac0fbbd4a239a"
assert_contains "$out" "WEIGHT_KEY=nanbeige4.3-33b:autoround-int4 bash scripts/setup.sh nanbeige4.3-33b"
out="$(expect_missing "$sgl_qwen" "${TMP_DIR}/sgl-empty")"
assert_contains "$out" "${TMP_DIR}/sgl-empty/qwen3.8-27b-autoround-int4/config.json (HF model)"
assert_contains "$out" "WEIGHT_KEY=qwen3.8-27b:autoround-int4 bash scripts/setup.sh qwen3.8-27b"
# The drafter is the script's own check (SPEC_N=0 switches it off), so it is not read.
assert_not_contains "$out" "dflash2-zlab-bf16"
assert_not_contains "$out" "thinkingcap"
out="$(expect_missing "$sgl_tcap" "${TMP_DIR}/sgl-empty")"
assert_contains "$out" "${TMP_DIR}/sgl-empty/thinkingcap-qwen3.8-27b-fp8/config.json (HF model)"
assert_contains "$out" "bash scripts/setup.sh thinkingcap-qwen3.8-27b"
# POSITIVE CONTROL: the model folders present → no complaint.
mkdir -p "${TMP_DIR}/sgl-ok/nanbeige4.3-33b-autoround-int4" "${TMP_DIR}/sgl-ok/qwen3.8-27b-autoround-int4" \
         "${TMP_DIR}/sgl-ok/thinkingcap-qwen3.8-27b-fp8"
touch "${TMP_DIR}/sgl-ok/nanbeige4.3-33b-autoround-int4/config.json" \
      "${TMP_DIR}/sgl-ok/qwen3.8-27b-autoround-int4/config.json" \
      "${TMP_DIR}/sgl-ok/thinkingcap-qwen3.8-27b-fp8/config.json"
for c in "$sgl_nanbeige" "$sgl_qwen" "$sgl_tcap"; do
  out="$(run_deps "$c" "${TMP_DIR}/sgl-ok")"
  [[ -z "$out" ]] || { echo "ASSERTION FAILED: SGLang weights present but reported for $c: $out" >&2; exit 1; }
done
# The spellings: `$${M}` as well as `$$M`, only the FIRST assignment (a later one sits
# behind a condition), a literal --model-path, and no other /models/… shell value.
sgl_fixture="${TMP_DIR}/sgl-script.yml"
cat > "$sgl_fixture" <<'YAML'
services:
  engine:
    image: example
    volumes:
      - "${MODEL_DIR:-../models-cache}:/models:ro"
    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        # M=/models/commented-out
        M=/models/fixture-main
        if [ -n "${ALT:-}" ]; then M=/models/fixture-conditional; fi
        D="${DRAFTER_DIR:-/models/fixture-drafter}"
        exec python3 -m sglang.launch_server \
          --model-path "$${M}" \
          --speculative-draft-model-path "$$D" \
          --tokenizer-path /models/fixture-tokenizer
  other:
    image: example
    command:
      - |
        exec python3 -m sglang.launch_server --model-path /models/fixture-literal
YAML
out="$(expect_missing "$sgl_fixture" "${TMP_DIR}/sgl-empty")"
assert_contains "$out" "${TMP_DIR}/sgl-empty/fixture-main/config.json (HF model)"
assert_contains "$out" "${TMP_DIR}/sgl-empty/fixture-literal/config.json (HF model)"
assert_not_contains "$out" "fixture-conditional"
assert_not_contains "$out" "commented-out"
assert_not_contains "$out" "fixture-drafter"
assert_not_contains "$out" "fixture-tokenizer"
mkdir -p "${TMP_DIR}/sgl-ok/fixture-main" "${TMP_DIR}/sgl-ok/fixture-literal"
touch "${TMP_DIR}/sgl-ok/fixture-main/config.json" "${TMP_DIR}/sgl-ok/fixture-literal/config.json"
out="$(run_deps "$sgl_fixture" "${TMP_DIR}/sgl-ok")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: SGLang fixture weights present but reported: $out" >&2; exit 1; }

# #1742, on the REAL composes. HyperQwen mounts the dir ABOVE its weights
# (qwen3.8-27b-hyperqwen/, where prepare also writes a -fast copy), and no weights
# entry is that dir, so preflight offered nothing for the 19 GB base. And the DFlash2
# drafter dir is an entry of both qwen3.8-27b and thinkingcap-qwen3.8-27b: the tie went
# to ThinkingCap, whose setup then suggested an SGLang ThinkingCap slug.
hq_compose="${ROOT_DIR}/models/qwen3.8-27b/vllm-hyperqwen/compose/single/hyperqwen-w4a16/dflash2.yml"
mkdir -p "${TMP_DIR}/hq-empty"
out="$(expect_missing "$hq_compose" "${TMP_DIR}/hq-empty")"
assert_contains "$out" "${TMP_DIR}/hq-empty/qwen3.8-27b-hyperqwen (MODEL_DIR volume path)"
assert_contains "$out" "--local-dir \$MODEL_DIR/qwen3.8-27b-hyperqwen/Qwen3.8-27B-W4A16-AutoRound"
assert_contains "$out" "WEIGHT_KEY=qwen3.8-27b:hyperqwen-w4a16 bash scripts/setup.sh qwen3.8-27b"
assert_contains "$out" "WITH_DFLASH_DRAFT=1 bash scripts/setup.sh qwen3.8-27b"
assert_not_contains "$out" "thinkingcap"
assert_not_contains "$out" "Check the compose header"
# The issue's second attempt: the drafter on disk, the base still missing.
mkdir -p "${TMP_DIR}/hq-draft/qwen3.8-27b-dflash2-w4a16"
out="$(expect_missing "$hq_compose" "${TMP_DIR}/hq-draft")"
assert_contains "$out" "WEIGHT_KEY=qwen3.8-27b:hyperqwen-w4a16 bash scripts/setup.sh qwen3.8-27b"
assert_not_contains "$out" "Check the compose header"
# POSITIVE CONTROL: the base where setup.sh puts it, plus the drafter → no complaint.
mkdir -p "${TMP_DIR}/hq-draft/qwen3.8-27b-hyperqwen/Qwen3.8-27B-W4A16-AutoRound"
out="$(run_deps "$hq_compose" "${TMP_DIR}/hq-draft")"
[[ -z "$out" ]] || { echo "ASSERTION FAILED: HyperQwen weights present but reported: $out" >&2; exit 1; }
# The shared drafter: a Qwen3.8-27B compose gets its own model's recipe, and a
# ThinkingCap compose still gets ThinkingCap's.
out="$(expect_missing "${ROOT_DIR}/models/qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2.yml" "${TMP_DIR}/hq-empty")"
assert_contains "$out" "WITH_DFLASH_DRAFT=1 bash scripts/setup.sh qwen3.8-27b"
assert_not_contains "$out" "thinkingcap"
out="$(expect_missing "${ROOT_DIR}/models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2.yml" "${TMP_DIR}/hq-empty")"
assert_contains "$out" "WITH_DFLASH_DRAFT=1 bash scripts/setup.sh thinkingcap-qwen3.8-27b"

# The fetch offer itself, which needs a terminal: the real HyperQwen compose under
# `script`, with setup.sh stubbed to log its call and create the dir it was asked for.
# It must fetch through the compose's model with SETUP_FOR_LAUNCH=1 (so setup.sh skips
# the next steps that name another slug), and a fetch whose re-check still fails must
# print the Skip line once: #1742 printed it twice.
if ! command -v script >/dev/null 2>&1; then
  echo "ASSERTION FAILED: util-linux 'script' is required for the fetch-offer coverage" >&2
  exit 1
fi
fake_root="${TMP_DIR}/fake-root"
mkdir -p "${fake_root}/scripts" "${TMP_DIR}/fake-bin"
ln -s "${ROOT_DIR}/scripts/lib" "${fake_root}/scripts/lib"
ln -s "${ROOT_DIR}/scripts/preflight.sh" "${fake_root}/scripts/preflight.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "${TMP_DIR}/fake-bin/hf"
cat > "${fake_root}/scripts/setup.sh" <<'STUB'
#!/usr/bin/env bash
echo "setup.sh $* WEIGHT_KEY=${WEIGHT_KEY:-} SETUP_FOR_LAUNCH=${SETUP_FOR_LAUNCH:-}" >> "${STUB_LOG}"
[[ "${STUB_FETCH}" == "1" ]] || exit 0
sub="$(python3 "$(dirname "$0")/lib/profiles/weights.py" entry "${WEIGHT_KEY}" | sed -n 's/^WEIGHT_SUBDIR=//p')"
mkdir -p "${MODEL_DIR}/${sub}"
STUB
chmod +x "${TMP_DIR}/fake-bin/hf" "${fake_root}/scripts/setup.sh"
run_fetch() {   # <model dir> <1 = the stub creates the dirs, 0 = it doesn't>
  printf 'y\n' | STUB_LOG="${TMP_DIR}/stub-$2.log" STUB_FETCH="$2" MODEL_DIR="$1" \
    ROOT_DIR="$fake_root" PATH="${TMP_DIR}/fake-bin:${PATH}" \
    script -qec "bash -c 'source \"${fake_root}/scripts/preflight.sh\"; preflight_compose_deps \"${hq_compose}\"; echo \"rc=\$?\"'" /dev/null 2>&1
}
mkdir -p "${TMP_DIR}/hq-fetch-ok" "${TMP_DIR}/hq-fetch-fail"
out="$(run_fetch "${TMP_DIR}/hq-fetch-ok" 1)"
assert_contains "$out" "rc=0"
log="$(cat "${TMP_DIR}/stub-1.log")"
assert_contains "$log" "setup.sh qwen3.8-27b WEIGHT_KEY=qwen3.8-27b:hyperqwen-w4a16 SETUP_FOR_LAUNCH=1"
assert_contains "$log" "setup.sh qwen3.8-27b WEIGHT_KEY=qwen3.8-27b:dflash2 SETUP_FOR_LAUNCH=1"
assert_not_contains "$log" "thinkingcap"
out="$(run_fetch "${TMP_DIR}/hq-fetch-fail" 0)"
assert_contains "$out" "rc=1"
skip_lines="$(command grep -c 'Skip this check' <<< "$out" || true)"
if [[ "$skip_lines" != "1" ]]; then
  echo "ASSERTION FAILED: expected the Skip line once after a failed fetch, got ${skip_lines}" >&2
  echo "--- output ---" >&2
  echo "$out" >&2
  exit 1
fi

echo "test-preflight-compose-deps: ok"
