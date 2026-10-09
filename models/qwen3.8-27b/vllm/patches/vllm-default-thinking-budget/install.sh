#!/usr/bin/env bash
# vllm-default-thinking-budget installer (patch D) — runs in the compose entrypoint before
# `vllm serve`, AFTER the effort_budget.py shell-env line has exported
# CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET (the budget of the compose's default effort).
#
# Makes SamplingParams fall back to that floor when a request set no thinking_token_budget,
# so endpoints without patch B's hook (Responses API, completions, /v1/chat/completions/batch)
# are bounded too. An explicit per-request budget wins. THINKING_BUDGETS=off unsets the env
# and the patched code then does nothing.
#
# Validates the env first (a non-integer floor refuses the boot), then inserts the floor
# (marker-gated: a second run is a verified no-op). Anchor missing or not unique -> exit 1,
# see README. Prints exactly one `[vllm-default-thinking-budget] …` line on success.
set -u
export PYTHONUTF8="${PYTHONUTF8:-1}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/patch_default_thinking_budget.py" patch || exit 1
