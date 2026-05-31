#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

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

out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" vllm/dual 2>&1)"
assert_contains "$out" "Profile triage: vllm/dual"
assert_contains "$out" "[1/6] Compose registry entry exists"
assert_contains "$out" "[6/6] Vendored overlays applied"
assert_contains "$out" "Triage summary: GREEN"

out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" llamacpp/default 2>&1)"
assert_contains "$out" "Profile triage: llamacpp/default"
assert_contains "$out" "KV projection not available for non-vLLM engines"
assert_contains "$out" "Triage summary: GREEN"

if out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" not/a-compose 2>&1)"; then
  echo "ASSERTION FAILED: unknown compose unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
else
  rc=$?
fi
[[ "$rc" -eq 3 ]] || { echo "ASSERTION FAILED: unknown compose exit=$rc, expected 3" >&2; echo "$out" >&2; exit 1; }
assert_contains "$out" "not/a-compose not found"
assert_contains "$out" "available composes:"

if out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" vllm/dual --tp 2 2>&1)"; then
  echo "ASSERTION FAILED: compose plus free-form flag unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
else
  rc=$?
fi
[[ "$rc" -eq 3 ]] || { echo "ASSERTION FAILED: mixed-mode args exit=$rc, expected 3" >&2; echo "$out" >&2; exit 1; }
assert_contains "$out" "pass either a compose name or free-form flags"

if out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" \
  --model qwen3.6-27b --engine vllm-nightly-mtp --drafter qwen-mtp-builtin \
  --kv-format fp8_e5m2 --tp 8 --pp 1 --max-ctx 262144 2>&1)"; then
  echo "ASSERTION FAILED: invalid free-form combo unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
else
  rc=$?
fi
[[ "$rc" -eq 2 ]] || { echo "ASSERTION FAILED: invalid combo exit=$rc, expected 2" >&2; echo "$out" >&2; exit 1; }
assert_contains "$out" "C2: tp=8 not in model.valid_tp"
assert_contains "$out" "Triage summary: RED"

out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" \
  --model qwen3.6-27b --engine vllm-nightly-mtp --drafter qwen-mtp-builtin \
  --kv-format fp8_e5m2 --tp 2 --pp 1 --max-ctx 262144 2>&1)"
assert_contains "$out" "Profile triage: free-form combo"
assert_contains "$out" "Triage summary: GREEN"

out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" gemma-dual-int8 2>&1)"
assert_contains "$out" "Profile triage: vllm/gemma-int8-mtp"
assert_contains "$out" "vllm-pr40391-rebased"
assert_contains "$out" "VLLM_IMAGE resolves: vllm/vllm-openai:v0.22.0"
assert_contains "$out" "Triage summary: GREEN"

python3 - <<'PY' | while IFS= read -r compose; do
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
for name in sorted(COMPOSE_REGISTRY):
    print(name)
PY
  out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" "$compose" 2>&1)" || {
    echo "ASSERTION FAILED: diagnose-profile failed for $compose" >&2
    echo "$out" >&2
    exit 1
  }
  assert_contains "$out" "[6/6] Vendored overlays applied"
done

echo "test-diagnose-profile: ok"
