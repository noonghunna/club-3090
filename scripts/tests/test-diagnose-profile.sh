#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

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
  # ⚠️ EXIT CODES ARE A CONTRACT, NOT A BOOLEAN: GREEN=0, YELLOW=1, RED=2,
  # bad-args/crash=3. This loop used to treat ANY non-zero as a failure, which
  # made it assert something untrue — that every catalogued slug must triage
  # clean on the DEFAULT hardware profile (1x-rtx-3090). A slug whose documented
  # target class is a bigger card is YELLOW there BY DESIGN, and saying so is the
  # config being honest. RED and crashes stay hard failures for everyone.
  # `set -e` aborts an assignment whose command fails BEFORE `rc=$?` is read,
  # so the status must be captured with `|| rc=$?` — the exact trap the repo
  # guide calls out (never interpolate a status you did not capture).
  rc=0
  out="$(bash "${ROOT_DIR}/scripts/diagnose-profile.sh" "$compose" 2>&1)" || rc=$?
  case "$compose" in
    # EXPECTED-YELLOW REGISTER — each entry names WHY. Keep it short; a slug that
    # is yellow for any OTHER reason is a real regression and must still fail.
    #
    # vllm/qwen38-27b-single-nvfp4: 20.44 GiB of NVFP4 weights on a 24 GB card
    # leaves ~0 GB for the KV pool, so kv-calc [4/6] verdicts FAIL. Deliberate —
    # its registry note says "on a 3090 expect an OOM at KV init, which is the
    # config being honest rather than broken"; target class is 32 GB+ (5090 /
    # RTX 6000 Pro / H100), where the A4 groups also actually execute.
    vllm/qwen38-27b-single-nvfp4)
      if [[ "$rc" -eq 0 ]]; then
        echo "ASSERTION FAILED: $compose triaged GREEN on 1x-rtx-3090 — it is" >&2
        echo "  registered as expected-YELLOW (does not fit 24 GB). If the slug or" >&2
        echo "  kv-calc changed so it now fits, DROP it from this register." >&2
        exit 1
      fi
      [[ "$rc" -eq 1 ]] || {
        echo "ASSERTION FAILED: $compose exit=$rc, expected 1 (YELLOW)" >&2
        echo "$out" >&2; exit 1; }
      ;;
    *)
      [[ "$rc" -eq 0 ]] || {
        echo "ASSERTION FAILED: diagnose-profile exit=$rc for $compose (expected 0/GREEN)" >&2
        echo "$out" >&2
        exit 1
      }
      ;;
  esac
  assert_contains "$out" "[6/6] Vendored overlays applied"
done

echo "test-diagnose-profile: ok"
