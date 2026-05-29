#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
ORIG_PATH="$PATH"
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

make_mock_tools() {
  mkdir -p "${TMP_DIR}/bin"
  cat > "${TMP_DIR}/bin/nvidia-smi" <<'MOCK_NVIDIA_SMI'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index,name,memory.total,compute_cap"*)
    printf '%s\n' "${MOCK_GPU_QUERY:?MOCK_GPU_QUERY not set}"
    ;;
  "-L")
    printf '%s\n' "${MOCK_GPU_QUERY:?MOCK_GPU_QUERY not set}" \
      | awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); print "GPU " $1 ": " $2}'
    ;;
  *)
    echo "unexpected nvidia-smi invocation: $*" >&2
    exit 2
    ;;
esac
MOCK_NVIDIA_SMI
  chmod +x "${TMP_DIR}/bin/nvidia-smi"

  cat > "${TMP_DIR}/switch-mock" <<'MOCK_SWITCH'
#!/usr/bin/env bash
echo "SWITCHED $* CUDA=${CUDA_VISIBLE_DEVICES:-} NVD=${NVIDIA_VISIBLE_DEVICES:-} TP=${TP:-} PP=${PP:-}"
MOCK_SWITCH
  chmod +x "${TMP_DIR}/switch-mock"

  export PATH="${TMP_DIR}/bin:${ORIG_PATH}"
}

set_rig() {
  export MOCK_GPU_QUERY="$1"
}

model_status() {
  local model="$1"
  (
    source "${ROOT_DIR}/scripts/lib/compose-meta.sh"
    compose_hw_model_status "$ROOT_DIR" "$model"
  )
}

assert_model_status() {
  local model="$1"
  local expected_prefix="$2"
  local expected_text="${3:-}"
  local status
  status="$(model_status "$model" || true)"
  if [[ "$status" != "${expected_prefix}"* ]]; then
    echo "ASSERTION FAILED: ${model} status expected prefix '${expected_prefix}', got '${status}'" >&2
    exit 1
  fi
  if [[ -n "$expected_text" ]]; then
    assert_contains "$status" "$expected_text"
  fi
}

make_mock_tools

# Matched 2x3090: Qwen and Gemma both have a viable compose.
set_rig $'0, NVIDIA GeForce RTX 3090, 24576, 8.6\n1, NVIDIA GeForce RTX 3090, 24576, 8.6'
assert_model_status "qwen3.6-27b" "ok|fits your rig"
assert_model_status "gemma-4-31b" "ok|fits your rig"

# Single 24 GB Ampere: Qwen fits; Gemma needs either 32 GB+ single-card or 2x24 GB.
set_rig $'0, NVIDIA GeForce RTX 3090, 24576, 8.6'
assert_model_status "qwen3.6-27b" "ok|fits your rig"
assert_model_status "gemma-4-31b" "no|" "needs 32 GB+ on single card OR 2× 24 GB"
assert_contains "$(model_status "gemma-4-31b")" "1× RTX 3090, 24 GB"

# Single 16 GB: neither shipped model has a viable compose.
set_rig $'0, NVIDIA RTX 4060 Ti, 16384, 8.9'
assert_model_status "qwen3.6-27b" "no|" "needs 20 GB+ VRAM"
assert_model_status "gemma-4-31b" "no|" "needs 32 GB+ on single card OR 2× 24 GB"

# Heterogeneous 16 + 24 GB: Qwen can run on the 24 GB card; Gemma dual cannot.
set_rig $'0, NVIDIA RTX 4060 Ti, 16384, 8.9\n1, NVIDIA GeForce RTX 3090, 24576, 8.6'
assert_model_status "qwen3.6-27b" "ok|fits your rig"
assert_model_status "gemma-4-31b" "no|" "RTX 4060 Ti, 16 GB + RTX 3090, 24 GB"

# 32 GB+ modern card: Gemma's single-card compose is eligible.
set_rig $'0, NVIDIA GeForce RTX 5090, 32768, 12.0'
assert_model_status "qwen3.6-27b" "ok|fits your rig"
assert_model_status "gemma-4-31b" "ok|fits your rig"

# Non-TTY no-arg setup fails fast with usage rather than hanging.
if out="$(echo | bash "${ROOT_DIR}/scripts/setup.sh" 2>&1)"; then
  echo "ASSERTION FAILED: non-TTY no-arg setup unexpectedly succeeded" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "Usage:"
assert_contains "$out" "Interactive picker available in a TTY shell"

# Positional setup path remains non-interactive and reaches the existing flow.
set_rig $'0, NVIDIA GeForce RTX 3090, 24576, 8.6'
out="$(MODEL_DIR="${TMP_DIR}/models" PREFLIGHT_DISK_GB=0 SKIP_GENESIS=1 SKIP_MODEL=1 bash "${ROOT_DIR}/scripts/setup.sh" qwen3.6-27b 2>&1)"
assert_not_contains "$out" "Which model to download?"
assert_contains "$out" "[model]   SKIP_MODEL=1"

# The launch wizard now picks model -> GPU set -> parallelism. Scripted flags
# skip prompts, select the expected variant, and export GPU / TP / PP envs.
mkdir -p "${TMP_DIR}/models/qwen3.6-27b-autoround-int4" \
         "${TMP_DIR}/models/gemma-4-31b-autoround-int4"
FAKE_8X3090='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6,2:RTX_3090:24576:8.6,3:RTX_3090:24576:8.6,4:RTX_3090:24576:8.6,5:RTX_3090:24576:8.6,6:RTX_3090:24576:8.6,7:RTX_3090:24576:8.6'

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0 --no-projection 2>&1)"
assert_contains "$out" "[launch] selected variant: vllm/default"
assert_contains "$out" "SWITCHED vllm/default CUDA=0 NVD=0 TP=1 PP=1"

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0,1 --no-projection 2>&1)"
assert_contains "$out" "[launch] Tensor parallel TP=2"
assert_not_contains "$out" "Topology:"
assert_contains "$out" "SWITCHED vllm/dual CUDA=0,1 NVD=0,1 TP=2 PP=1"
selected_count="$(grep -c "\[launch\] selected variant:" <<< "$out" || true)"
if [[ "$selected_count" != "1" ]]; then
  echo "ASSERTION FAILED: expected one selected-variant line, got ${selected_count}" >&2
  echo "$out" >&2
  exit 1
fi

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0 --workload fast-chat --no-projection 2>&1)"
assert_contains "$out" "[launch] selected variant: vllm/minimal"
assert_contains "$out" "SWITCHED vllm/minimal CUDA=0 NVD=0 TP=1 PP=1"

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0 --drafter off --no-projection 2>&1)"
assert_contains "$out" "[launch] selected variant: vllm/long-text-no-mtp"
assert_contains "$out" "SWITCHED vllm/long-text-no-mtp CUDA=0 NVD=0 TP=1 PP=1"

mkdir -p "${TMP_DIR}/models/qwen3.6-27b-gguf"
out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0 --stable --no-projection 2>&1)"
assert_contains "$out" "[launch] selected variant: llamacpp/default"
assert_contains "$out" "SWITCHED llamacpp/default CUDA=0 NVD=0 TP=1 PP=1"

if out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model gemma-4-31b --gpus 0 --no-projection 2>&1)"; then
  echo "ASSERTION FAILED: Gemma single-24GB launch unexpectedly succeeded" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "Gemma 4 31B does not fit on a single 24 GB card today"

out="$(CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6' \
  bash "${ROOT_DIR}/scripts/launch.sh" --topology 2>&1)"
assert_contains "$out" "Topology class: homogeneous"
assert_not_contains "$out" "Compute mismatch detected"

out="$(CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_4090:24576:8.9' \
  bash "${ROOT_DIR}/scripts/launch.sh" --topology 2>&1)"
assert_contains "$out" "Topology class: vram_matched_compute_mismatched"
assert_contains "$out" "Compute mismatch detected"
assert_contains "$out" "Estate planner"

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_4090:24576:8.9' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0,1 --no-projection 2>&1)"
assert_contains "$out" "Topology: vram_matched_compute_mismatched"
assert_contains "$out" "Compute mismatch detected"
assert_contains "$out" "SWITCHED vllm/dual CUDA=0,1 NVD=0,1 TP=2 PP=1"

if out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6,2:RTX_3090:24576:8.6,3:RTX_3090:24576:8.6,4:RTX_3090:24576:8.6,5:RTX_3090:24576:8.6' \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0,1,2,3,4,5 --tp 6 --no-projection 2>&1)"; then
  echo "ASSERTION FAILED: invalid Qwen TP=6 unexpectedly succeeded" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "Valid TP values: 1 2 4"

out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS="${FAKE_8X3090}" \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model gemma-4-31b --gpus 0,1,2,3,4,5,6,7 --tp 8 2>&1)"
assert_contains "$out" "[launch] Tensor parallel TP=8"
assert_contains "$out" "[launch] Suggested: vllm/gemma-mtp"
assert_contains "$out" "VRAM budget — per card"
assert_contains "$out" "Note: TP > 4 predictions are extrapolated"
assert_not_contains "$out" "KV projection skipped"
assert_contains "$out" "SWITCHED vllm/gemma-mtp CUDA=0,1,2,3,4,5,6,7 NVD=0,1,2,3,4,5,6,7 TP=8 PP=1"

if out="$(MODEL_DIR="${TMP_DIR}/models" CLUB3090_FAKE_GPUS="${FAKE_8X3090}" \
  SWITCH="${TMP_DIR}/switch-mock" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight --no-verify --model qwen3.6-27b --gpus 0,1,2,3,4,5,6,7 --tp 8 --no-projection 2>&1)"; then
  echo "ASSERTION FAILED: invalid Qwen TP=8 unexpectedly succeeded" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "num_kv_heads does not divide TP=8"
assert_contains "$out" "Valid TP values: 1 2 4"

# TTY-backed no-arg setup supports the cosmetic but real "Both" choice by
# dispatching through the positional path for both model families.
if ! command -v script >/dev/null 2>&1; then
  echo "ASSERTION FAILED: util-linux 'script' is required for TTY picker coverage" >&2
  exit 1
fi
set_rig $'0, NVIDIA GeForce RTX 3090, 24576, 8.6\n1, NVIDIA GeForce RTX 3090, 24576, 8.6'
export MODEL_DIR="${TMP_DIR}/models"
export PREFLIGHT_DISK_GB=0
export SKIP_GENESIS=1
export SKIP_MODEL=1
out="$(printf '3\n' | script -qec "bash '${ROOT_DIR}/scripts/setup.sh'" /dev/null 2>&1)"
assert_contains "$out" "[setup] Which model to download?"
assert_contains "$out" "Both"
assert_contains "$out" "[setup] downloading both supported models"
skip_count="$(grep -c "\[model\]   SKIP_MODEL=1" <<< "$out" || true)"
if [[ "$skip_count" != "2" ]]; then
  echo "ASSERTION FAILED: expected Both choice to dispatch two model setup runs, got ${skip_count}" >&2
  echo "--- output ---" >&2
  echo "$out" >&2
  exit 1
fi

echo "test-setup-picker: ok"
