#!/usr/bin/env bash
# PR-B — <engine>/default resolver uses DEFAULTS + detected topology.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

assert_contains() {
  local haystack="$1" needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

fake_one='0:RTX_3090:24576:8.6'
fake_two='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6'

out="$(CLUB3090_FAKE_GPUS="$fake_one" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection vllm/default 2>&1)"
assert_contains "$out" "selected variant: vllm/minimal"
assert_contains "$out" "vllm/minimal"

out="$(CLUB3090_FAKE_GPUS="$fake_two" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection vllm/default 2>&1)"
assert_contains "$out" "selected variant: vllm/dual"
assert_contains "$out" "vllm/dual"

out="$(CLUB3090_FAKE_GPUS="$fake_one" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection vllm/dual/default 2>&1)"
assert_contains "$out" "selected variant: vllm/dual"

if out="$(CLUB3090_FAKE_GPUS="$fake_one" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection llamacpp/dual/default 2>&1)"; then
  echo "ASSERTION FAILED: bad topology default unexpectedly resolved" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "cannot resolve default variant 'llamacpp/dual/default'"
assert_contains "$out" "Available defaults"

out="$(NVIDIA_VISIBLE_DEVICES=0,1 FORCE=1 PREFLIGHT_NO_COMPOSE_DEPS=1 COMPOSE_BIN=: READY_TIMEOUT=1 bash scripts/switch.sh --no-wait vllm/default 2>&1 || true)"
assert_contains "$out" "bringing up: vllm/dual"

out="$(NVIDIA_VISIBLE_DEVICES=0 FORCE=1 PREFLIGHT_NO_COMPOSE_DEPS=1 COMPOSE_BIN=: READY_TIMEOUT=1 bash scripts/switch.sh --no-wait vllm/dual/default 2>&1 || true)"
assert_contains "$out" "bringing up: vllm/dual"

# PR-B: `<model>/default` token dispatch through launch.sh (engine-vs-model).
# Single rig: qwen3.6-27b/default → curated beellama (ENGINE_PREFERENCE #1, caveats);
# dual → vllm/dual.
out="$(CLUB3090_FAKE_GPUS="$fake_one" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection --variant qwen3.6-27b/default 2>&1)"
assert_contains "$out" "selected variant: beellama/dflash"
out="$(CLUB3090_FAKE_GPUS="$fake_two" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection --variant qwen3.6-27b/default 2>&1)"
assert_contains "$out" "selected variant: vllm/dual"
# gemma-4-31b/default dual → vllm/gemma-int8-mtp (model token overrides PRIMARY_MODEL).
out="$(CLUB3090_FAKE_GPUS="$fake_two" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection --variant gemma-4-31b/default 2>&1)"
assert_contains "$out" "selected variant: vllm/gemma-int8-mtp"
# Unknown X/default → clear error (neither engine nor model).
if out="$(CLUB3090_FAKE_GPUS="$fake_one" SWITCH=/bin/echo bash scripts/launch.sh --no-preflight --no-verify --no-projection --variant bogus/default 2>&1)"; then
  echo "ASSERTION FAILED: bogus/default unexpectedly resolved" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "neither a known engine nor a known model"

echo "test-default-resolver: ok"
