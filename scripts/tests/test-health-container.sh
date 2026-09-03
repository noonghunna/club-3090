#!/usr/bin/env bash
# CONTAINER= generalization for health.sh.
#
# health.sh used to hardcode the container match to
# `^(vllm-qwen36-27b|llama-cpp-qwen36-27b)`. The contract generalizes it:
#   1. CONTAINER=<name> targets ANY named container (exact match).
#   2. CONTAINER= unset broadens the auto-match to ANY recognized engine-prefix
#      container (vllm-/llama-cpp-/ik-llama-/sglang-/beellama-), not just qwen.
# COMPAT: with a qwen container running and CONTAINER= unset, the SAME container
# is selected as before (the qwen names still match the first two alternatives).
#
# The probe() emit is human-readable text, so we assert its SHAPE: which
# container name lands on the "✓ Container <name> ..." line. We mock docker /
# curl / nvidia-smi on PATH so the test is hermetic (no real engine needed).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HEALTH="$ROOT_DIR/scripts/health.sh"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

assert_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" == *"$needle"* ]] || note "${msg}: output lacks '${needle}'"
}
assert_not_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" != *"$needle"* ]] || note "${msg}: output unexpectedly contains '${needle}'"
}

# Mock binaries on a private PATH. MOCK_PS_NAMES is the newline list `docker ps
# --format '{{.Names}}'` returns; the selected container's inspect fields are
# fixed/harmless. curl always succeeds with a minimal /v1/models payload.
make_mocks() {
  mkdir -p "${TMP_DIR}/bin"

  cat > "${TMP_DIR}/bin/curl" <<'MOCK'
#!/usr/bin/env bash
# Only /v1/models is probed; return a minimal vLLM-shaped payload.
printf '%s\n' '{"data":[{"id":"mock-model","owned_by":"mock"}]}'
exit 0
MOCK

  cat > "${TMP_DIR}/bin/docker" <<'MOCK'
#!/usr/bin/env bash
case "$1" in
  ps)
    # `docker ps --format '{{.Names}}'`
    printf '%s\n' "${MOCK_PS_NAMES:-}"
    ;;
  inspect)
    # last arg is the container name; emit a fixed running state.
    case "$*" in
      *"{{.Id}}"*)        echo "abcdef0123456789" ;;
      *"{{.State.Status}}"*)    echo "running" ;;
      *"{{.State.StartedAt}}"*) echo "2026-06-18T00:00:00.000000000Z" ;;
      *) echo "" ;;
    esac
    ;;
  logs)
    # No log lines needed for the container-selection shape.
    printf '%s' ""
    ;;
  *) echo "" ;;
esac
exit 0
MOCK

  cat > "${TMP_DIR}/bin/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
# Pretend no GPU query is available; health.sh tolerates empty output.
exit 0
MOCK

  chmod +x "${TMP_DIR}/bin/curl" "${TMP_DIR}/bin/docker" "${TMP_DIR}/bin/nvidia-smi"
}

run_health() {
  # $1 = MOCK_PS_NAMES, $2 = CONTAINER (empty → unset)
  local names="$1" container="$2"
  if [[ -n "$container" ]]; then
    PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="$names" CONTAINER="$container" \
      bash "$HEALTH" 2>&1
  else
    PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="$names" \
      bash "$HEALTH" 2>&1
  fi
}

make_mocks

# --- 1. COMPAT: qwen container, CONTAINER= unset → same container selected ----
out="$(run_health $'vllm-qwen36-27b' '')"
assert_contains "$out" "Container vllm-qwen36-27b" \
  "qwen container auto-matched (compat)"
# Valid shape: reachable + serving + container line all present.
assert_contains "$out" "API reachable on /v1/models" "probe emits reachability line"
assert_contains "$out" "Serving model: mock-model" "probe emits served-model line"

# --- 2. Broadened auto-match: non-qwen engine container is now matched --------
for name in "vllm-gemma-4-31b" "ik-llama-something" "sglang-foo" "beellama-bar" "llama-cpp-other"; do
  out="$(run_health "$name" '')"
  assert_contains "$out" "Container ${name}" "broadened auto-match picks ${name}"
done

# --- 3. Auto-match ignores non-engine containers ------------------------------
out="$(run_health $'redis\npostgres' '')"
assert_contains "$out" "No matching container running" \
  "auto-match skips unrelated containers"

# --- 4. CONTAINER= targets ANY named container (exact match) ------------------
out="$(run_health $'vllm-qwen36-27b\nmy-custom-llm' 'my-custom-llm')"
assert_contains     "$out" "Container my-custom-llm" "CONTAINER= targets the named container"
assert_not_contains "$out" "Container vllm-qwen36-27b" "CONTAINER= overrides the auto-match"

# --- 5. CONTAINER= is an exact match, not a prefix/substring ------------------
out="$(run_health $'vllm-qwen36-27b' 'vllm-qwen')"
assert_contains "$out" "No matching container running" \
  "CONTAINER= does not substring-match"

if [[ "$fail" -ne 0 ]]; then
  echo "[health-container] FAIL" >&2
  exit 1
fi
echo "test-health-container: ok"
