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
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

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
# Only /v1/models is probed; return a minimal payload (owned_by from MOCK_OWNED_BY)
# and record which URL was asked, so the autodetected port can be asserted.
for a in "$@"; do [[ "$a" == http* ]] && echo "$a" >> "${MOCK_CURL_LOG:-/dev/null}"; done
printf '{"data":[{"id":"mock-model","owned_by":"%s"}]}\n' "${MOCK_OWNED_BY:-mock}"
exit 0
MOCK

  cat > "${TMP_DIR}/bin/docker" <<'MOCK'
#!/usr/bin/env bash
case "$1" in
  ps)
    # `docker ps --format '{{.Names}}|{{.Ports}}'` (the endpoint autodetect) gets
    # MOCK_PS_PORTS when set; `docker ps --format '{{.Names}}'` gets MOCK_PS_NAMES.
    if [[ "$*" == *".Ports"* ]]; then printf '%s\n' "${MOCK_PS_PORTS:-}"; else printf '%s\n' "${MOCK_PS_NAMES:-}"; fi
    ;;
  inspect)
    # last arg is the container name; emit a fixed running state.
    case "$*" in
      *"{{.Id}}"*)        echo "abcdef0123456789" ;;
      *"{{.State.Status}}"*)    echo "running" ;;
      *"{{.State.StartedAt}}"*) echo "2026-06-18T00:00:00.000000000Z" ;;
      *"{{.Config.Image}}"*)    echo "${MOCK_IMAGE:-}" ;;
      *) echo "" ;;
    esac
    ;;
  logs)
    printf '%s' "${MOCK_LOGS:-}"
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

# --- 5. The endpoint is autodetected like verify.sh (preflight_autodetect_endpoint)
# Before, health.sh probed qwen3.6-27b's default port whatever was serving, and
# reported "API not reachable" next to a healthy SGLang slug on :8142.
export MOCK_CURL_LOG="${TMP_DIR}/curl.log"
: > "$MOCK_CURL_LOG"
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="sglang-qwen38" MOCK_PS_PORTS="sglang-qwen38|0.0.0.0:8142->30000/tcp" bash "$HEALTH" 2>&1)"
assert_contains "$out" "Endpoint: http://localhost:8142" "autodetect probes the running container's port"
command grep -q 'localhost:8142/v1/models' "$MOCK_CURL_LOG" || note "autodetect: curl was not asked :8142 ($(tr '\n' ' ' < "$MOCK_CURL_LOG"))"
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="sglang-qwen38" MOCK_PS_PORTS="sglang-qwen38|0.0.0.0:8142->30000/tcp" URL=http://localhost:9999 bash "$HEALTH" 2>&1)"
assert_contains "$out" "Endpoint: http://localhost:9999" "URL= still wins over the autodetect"

# --- 6. The engine comes from engine-kind.sh (image > container name > owned_by)
# Before, anything whose owned_by wasn't llamacpp was labelled vLLM, and an SGLang
# server got a vLLM runtime section that could never find its log lines.
decode='[2026-09-29 00:00:00 TP0] Decode batch, #running-req: 2, #full token: 5632, full token usage: 0.12, mamba num: 4, mamba usage: 0.03, accept len: 3.12, accept rate: 0.56, cuda graph: True, gen throughput (token/s): 95.33, #queue-req: 1'
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="my-llm" CONTAINER=my-llm MOCK_IMAGE="lmsysorg/sglang:v0.5.20" MOCK_LOGS="$decode" bash "$HEALTH" 2>&1)"
assert_contains     "$out" "(engine: SGLang)"         "an SGLang image is named SGLang, even under a custom container name"
assert_contains     "$out" "SGLang runtime"           "SGLang gets its own runtime section"
assert_contains     "$out" "KV cache: 12%"            "SGLang KV usage comes from 'full token usage'"
assert_contains     "$out" "accept len last 5 = 3.12" "SGLang spec-decode comes from 'accept len'"
assert_contains     "$out" "Last gen throughput: 95.33 tokens/s" "SGLang throughput"
assert_not_contains "$out" "vLLM runtime"             "no vLLM section for SGLang"
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="my-llm" CONTAINER=my-llm MOCK_IMAGE="vllm/vllm-openai:v0.30.0" bash "$HEALTH" 2>&1)"
assert_contains "$out" "(engine: vLLM)" "a vLLM image is named vLLM"
assert_contains "$out" "vLLM runtime"   "vLLM keeps its runtime section"
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="my-llm" CONTAINER=my-llm MOCK_OWNED_BY=llamacpp bash "$HEALTH" 2>&1)"
assert_contains "$out" "(engine: llama.cpp)" "no image or recognised name: owned_by still decides"
out="$(PATH="${TMP_DIR}/bin:${PATH}" MOCK_PS_NAMES="my-llm" CONTAINER=my-llm bash "$HEALTH" 2>&1)"
assert_contains "$out" "(engine: unknown)" "no evidence at all: unknown, not a guessed vLLM"
assert_contains "$out" "runtime details aren't parsed" "unknown engine: says so instead of parsing the wrong logs"


# --- 5. CONTAINER= is an exact match, not a prefix/substring ------------------
out="$(run_health $'vllm-qwen36-27b' 'vllm-qwen')"
assert_contains "$out" "No matching container running" \
  "CONTAINER= does not substring-match"

if [[ "$fail" -ne 0 ]]; then
  echo "[health-container] FAIL" >&2
  exit 1
fi
echo "test-health-container: ok"
