#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
READER="${ROOT_DIR}/scripts/lib/profiles/weights.py"

load_entry() {
  local key="$1"
  local env_lines
  env_lines="$(python3 "$READER" entry "$key")"
  eval "$env_lines"
}

load_lookup() {
  local path="$1"
  local env_lines
  env_lines="$(python3 "$READER" lookup "$path")"
  eval "$env_lines"
}

assert_entry() {
  local key="$1" repo="$2" subdir="$3"
  load_entry "$key"
  if [[ "$WEIGHT_REPO" != "$repo" || "$WEIGHT_SUBDIR" != "$subdir" ]]; then
    echo "ASSERTION FAILED: bad entry for $key" >&2
    echo "  repo:   got '$WEIGHT_REPO' expected '$repo'" >&2
    echo "  subdir: got '$WEIGHT_SUBDIR' expected '$subdir'" >&2
    exit 1
  fi
}

assert_lookup() {
  local path="$1" expected="$2"
  load_lookup "$path"
  if [[ "$WEIGHT_KEY" != "$expected" ]]; then
    echo "ASSERTION FAILED: lookup for $path -> $WEIGHT_KEY, expected $expected" >&2
    exit 1
  fi
}

assert_entry qwen3.6-27b:autoround-int4 Lorbus/Qwen3.6-27B-int4-AutoRound qwen3.6-27b-autoround-int4
assert_entry qwen3.6-27b:dflash z-lab/Qwen3.6-27B-DFlash qwen3.6-27b-dflash
assert_entry qwen3.6-27b:prism_eagle3 Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 qwen3.6-27b-prism-eagle3
assert_entry qwen3.6-27b:unsloth-q4km unsloth/Qwen3.6-27B-MTP-GGUF qwen3.6-27b-gguf/unsloth-mtp-q4km
assert_entry qwen3.6-27b:gguf_mmproj_f16 unsloth/Qwen3.6-27B-GGUF qwen3.6-27b-gguf
assert_entry qwen3.6-27b:ubergarm-iq4ks ubergarm/Qwen3.6-27B-GGUF qwen3.6-27b-gguf/ubergarm-mtp-iq4ks
assert_entry qwen3.6-27b:hauhaucs-q4kp-mtp SummonGovernance/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-MTP qwen3.6-27b-gguf/hauhaucs-q4kp-mtp
assert_entry qwen3.6-35b-a3b:autoround-int4 Qwen/Qwen3-MoE-A3B-Instruct-AutoRound-Int4-mixed qwen3.6-35b-a3b-autoround-int4
assert_entry qwen3.6-35b-a3b:luffy-genesis-apex-compact LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-V2-APEX-MTP-GGUF qwen3.6-35b-a3b-gguf/luffy-genesis-apex-mtp
assert_entry qwen3.6-35b-a3b:carnice-apex-compact mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF qwen3.6-35b-a3b-gguf/carnice-apex-mtp
assert_entry qwen3.6-35b-a3b:qwopus-apex-compact mudler/Qwopus3.6-35B-A3B-v1-APEX-MTP-GGUF qwen3.6-35b-a3b-gguf/qwopus-apex-mtp
assert_entry gemma-4-31b:autoround-int4 Intel/gemma-4-31B-it-int4-AutoRound gemma-4-31b-autoround-int4
assert_entry gemma-4-31b:awq cyankiwi/gemma-4-31B-it-AWQ-4bit gemma-4-31b-it-AWQ-4bit
assert_entry gemma-4-31b:assistant google/gemma-4-31B-it-assistant gemma-4-31b-it-assistant
assert_entry gemma-4-31b:dflash z-lab/gemma-4-31b-it-dflash gemma-4-31b-it-dflash
assert_entry gemma-4-26b-a4b:autoround-int4-mixed Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound gemma-4-26b-a4b-autoround-int4-mixed
assert_entry gemma-4-26b-a4b:awq cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit gemma-4-26b-a4b-awq-4bit
assert_entry gemma-4-26b-a4b:assistant google/gemma-4-26B-A4B-it-assistant gemma-4-26b-a4b-it-assistant
assert_entry qwen3.6-27b:carnice-bf16mtp wasifb/Carnice_V2_27B_INT4_BF16MTP carnice-v2-27b-int4-recipe-d-bf16mtp

load_entry qwen3.6-27b:qwopus-bf16mtp
[[ -z "$WEIGHT_REPO" ]]
[[ "$WEIGHT_SUBDIR" == "qwopus3.6-27b-int4-recipe-d-bf16mtp" ]]
[[ -n "$WEIGHT_MANUAL_NOTE" ]]

# Legacy preflight aliases remain accepted for user-facing setup hints.
load_entry qwen3.6-27b-gguf-iq4ks
[[ "$WEIGHT_KEY" == "qwen3.6-27b:ubergarm-iq4ks" ]]

assert_lookup qwen3.6-27b-autoround-int4/config.json qwen3.6-27b:autoround-int4
assert_lookup qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf qwen3.6-27b:unsloth-q4km
assert_lookup qwen3.6-27b-gguf/mmproj-F16.gguf qwen3.6-27b:gguf_mmproj_f16
assert_lookup qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf qwen3.6-27b:ubergarm-iq4ks
assert_lookup qwen3.6-27b-gguf/hauhaucs-q4kp-mtp/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-MTP-Q4_K_P.gguf qwen3.6-27b:hauhaucs-q4kp-mtp
assert_lookup qwen3.6-35b-a3b-gguf/luffy-genesis-apex-mtp/Qwen3.6-35B-A3B-Uncensored-Genesis-MTP-APEX-Compact.gguf qwen3.6-35b-a3b:luffy-genesis-apex-compact
assert_lookup qwen3.6-35b-a3b-gguf/carnice-apex-mtp/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf qwen3.6-35b-a3b:carnice-apex-compact
assert_lookup qwen3.6-35b-a3b-gguf/qwopus-apex-mtp/Qwopus3.6-35B-A3B-v1-APEX-MTP-I-Compact.gguf qwen3.6-35b-a3b:qwopus-apex-compact
assert_lookup qwen3.6-27b-prism-eagle3/compressed qwen3.6-27b:prism_eagle3
assert_lookup carnice-v2-27b-int4-recipe-d-bf16mtp/chat_template.jinja qwen3.6-27b:carnice-bf16mtp
assert_lookup qwopus3.6-27b-int4-recipe-d-bf16mtp/config.json qwen3.6-27b:qwopus-bf16mtp

echo "test-model-weights-registry: ok"
