#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

make_mock_nvidia_smi() {
  mkdir -p "${TMP_DIR}/bin"
  cat > "${TMP_DIR}/bin/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index,name,memory.total,compute_cap"*)
    printf '%s\n' "${MOCK_GPU_QUERY:?MOCK_GPU_QUERY not set}"
    ;;
  *"--query-gpu=index,memory.total"*)
    printf '%s\n' "${MOCK_GPU_MEM_QUERY:-${MOCK_GPU_QUERY:?}}" \
      | awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $3); print $1 ", " $3}'
    ;;
  *"--query-gpu=index,memory.free,memory.total"*)
    printf '%s\n' "${MOCK_GPU_FREE_QUERY:?MOCK_GPU_FREE_QUERY not set}"
    ;;
  "-L")
    printf '%s\n' "${MOCK_GPU_QUERY:?}" \
      | awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); print "GPU " $1 ": " $2}'
    ;;
  *)
    echo "unexpected nvidia-smi invocation: $*" >&2
    exit 2
    ;;
esac
MOCK
  chmod +x "${TMP_DIR}/bin/nvidia-smi"
  export PATH="${TMP_DIR}/bin:${PATH}"
}

make_compose() {
  local path="$1"
  local min_vram="$2"
  local min_gpu="$3"
  local tp="$4"
  local sm="${5:-}"

  {
    echo "# Hardware metadata (test fixture):"
    echo "# Requires-min-vram-gb: ${min_vram}"
    echo "# Requires-min-gpu-count: ${min_gpu}"
    echo "# Tensor-parallel: ${tp}"
    if [[ -n "$sm" ]]; then
      echo "# Requires-sm: ${sm}"
    fi
    echo "services: {}"
  } > "$path"
}

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

run_case() {
  local compose="$1"
  local variant="$2"
  local force="${3:-0}"
  (
    unset CLUB3090_GPU CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES FORCE
    source "${ROOT_DIR}/scripts/preflight.sh"
    if preflight_compose_hardware "$compose" "$variant" "$force"; then
      echo "STATUS=ok"
      echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
    else
      rc=$?
      echo "STATUS=fail:${rc}"
      exit "$rc"
    fi
  ) 2>&1
}

expect_failure() {
  local compose="$1"
  local variant="$2"
  local force="${3:-0}"
  local output

  if output="$(run_case "$compose" "$variant" "$force")"; then
    echo "ASSERTION FAILED: expected preflight failure for ${variant}" >&2
    echo "--- output ---" >&2
    echo "$output" >&2
    exit 1
  fi
  printf '%s' "$output"
}

make_mock_nvidia_smi

single_compose="${TMP_DIR}/single.yml"
dual_compose="${TMP_DIR}/dual.yml"
quad_compose="${TMP_DIR}/quad.yml"
gemma_single="${TMP_DIR}/gemma-single.yml"
missing_meta="${TMP_DIR}/missing.yml"

make_compose "$single_compose" 24 1 1
make_compose "$dual_compose" 24 2 2
make_compose "$quad_compose" 24 4 4
make_compose "$gemma_single" 32 1 1 "9.0+"
echo "services: {}" > "$missing_meta"

# 1. Matched 2x3090 + TP=1 compose: pass, deterministic GPU 0 selection.
MOCK_GPU_QUERY=$'0, NVIDIA GeForce RTX 3090, 24576, 8.6\n1, NVIDIA GeForce RTX 3090, 24576, 8.6'
MOCK_GPU_FREE_QUERY=$'0, 24000, 24576\n1, 24000, 24576'
export MOCK_GPU_QUERY MOCK_GPU_FREE_QUERY

out="$(run_case "$single_compose" "vllm/default")"
assert_contains "$out" "auto-selected GPU 0"
assert_contains "$out" "NVIDIA_VISIBLE_DEVICES=0"

# 2. Matched 2x3090 + TP=2 compose: pass.
out="$(run_case "$dual_compose" "vllm/dual")"
assert_contains "$out" "TP=2 requires 2 GPU(s)"
assert_contains "$out" "STATUS=ok"

# 3. Matched 2x3090 + TP=4 compose: hard fail.
out="$(expect_failure "$quad_compose" "vllm/dual4")"
assert_contains "$out" "requires 4 visible GPU(s)"

# 4. 1x3090 + TP=2 compose: hard fail.
MOCK_GPU_QUERY=$'0, NVIDIA GeForce RTX 3090, 24576, 8.6'
MOCK_GPU_FREE_QUERY=$'0, 24000, 24576'
export MOCK_GPU_QUERY MOCK_GPU_FREE_QUERY

out="$(expect_failure "$dual_compose" "vllm/dual")"
assert_contains "$out" "requires 2 visible GPU(s)"

# 5. 1x3090 + TP=1, 24 GB floor: pass.
out="$(run_case "$single_compose" "vllm/default")"
assert_contains "$out" "auto-selected GPU 0"
assert_contains "$out" "STATUS=ok"

# 6. 16 GB + 24 GB + TP=1: auto-select the 24 GB card.
MOCK_GPU_QUERY=$'0, RTX 4060 Ti, 16384, 8.9\n1, NVIDIA GeForce RTX 3090, 24576, 8.6'
MOCK_GPU_FREE_QUERY=$'0, 16000, 16384\n1, 24000, 24576'
export MOCK_GPU_QUERY MOCK_GPU_FREE_QUERY

out="$(run_case "$single_compose" "vllm/default")"
assert_contains "$out" "auto-selected GPU 1"
assert_contains "$out" "NVIDIA_VISIBLE_DEVICES=1"

# 7. 16 GB + 24 GB + TP=2: warn, then proceed for tuned sub-24 GB rigs.
out="$(run_case "$dual_compose" "vllm/dual")"
assert_contains "$out" "WARN:"
assert_contains "$out" "TP=2"
assert_contains "$out" "STATUS=ok"

# 8. 1x3090 + TP=1 compose with 32 GB + sm_9.0+ floor: hard fail.
MOCK_GPU_QUERY=$'0, NVIDIA GeForce RTX 3090, 24576, 8.6'
MOCK_GPU_FREE_QUERY=$'0, 24000, 24576'
export MOCK_GPU_QUERY MOCK_GPU_FREE_QUERY

out="$(expect_failure "$gemma_single" "vllm/gemma-mtp-tp1")"
assert_contains "$out" "requires one GPU with >=32 GB VRAM, sm_9.0+"

# 9. 1xH100 + TP=1 compose with sm_9.0+ floor: pass.
MOCK_GPU_QUERY=$'0, NVIDIA H100 80GB HBM3, 81920, 9.0'
MOCK_GPU_FREE_QUERY=$'0, 80000, 81920'
export MOCK_GPU_QUERY MOCK_GPU_FREE_QUERY

out="$(run_case "$gemma_single" "vllm/gemma-mtp-tp1")"
assert_contains "$out" "auto-selected GPU 0"
assert_contains "$out" "STATUS=ok"

# 10. --force skips the hardware gate.
out="$(run_case "$gemma_single" "vllm/gemma-mtp-tp1" 1)"
assert_contains "$out" "hardware: skipped"
assert_contains "$out" "STATUS=ok"

# 11. Missing metadata warns and allows.
out="$(run_case "$missing_meta" "vllm/local")"
assert_contains "$out" "no hardware metadata"
assert_contains "$out" "STATUS=ok"

echo "test-preflight-vram: ok"
