#!/usr/bin/env bash
# PR-C — hardware-aware --list topology filter.
#
# `switch.sh --list` should show only the topologies the machine can actually
# run (single on a 1-GPU box; single+dual on a 2-GPU box; everything on 4+),
# and `--list --all` / `--list-all` should always show everything. Detection
# reads CUDA_VISIBLE_DEVICES (what switch_topology_from_gpus consults first),
# so we simulate GPU counts by setting it. Fail-open: with no detection signal
# at all, show everything. Must not regress PR-A health markers/grouping or
# PR-B's Defaults view.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Keep the filter independent of whatever GPUs this host actually has and of a
# stray .env pin: pin MODEL_DIR + force a known model. The filter is about
# topology rank, not weights on disk.
export MODEL_DIR="${MODEL_DIR:-/mnt/models/huggingface}"

SWITCH="$ROOT_DIR/scripts/switch.sh"

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

# Count rows whose slug column matches a topology by grepping for the
# topology's compose-file path segment in the *visible* listing. We assert on
# slugs we know are bound to each topology in the shipped registry.
SINGLE_SLUG="ik-llama/iq4ks-mtp"      # single
DUAL_SLUG="vllm/dual"                 # dual
# NOTE: multi4 vLLM tier is empty post-#327 (dual4 + dual4-dflash archived); no MULTI_SLUG to assert.

run_list() {
  # $1 = CUDA_VISIBLE_DEVICES value (empty string → unset); rest = extra args.
  local cuda="$1"; shift
  if [[ -n "$cuda" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" NVIDIA_VISIBLE_DEVICES="" bash "$SWITCH" "$@" 2>&1
  else
    NVIDIA_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" bash "$SWITCH" "$@" 2>&1
  fi
}

# --- 1 GPU: only single slugs, with the note --------------------------------
out="$(run_list 0 --list)"
assert_contains     "$out" "$SINGLE_SLUG" "1-GPU shows single slugs"
assert_not_contains "$out" "$DUAL_SLUG"   "1-GPU hides dual slugs"
assert_contains     "$out" "1-GPU machine" "1-GPU prints the filter note"
assert_contains     "$out" "--list --all" "1-GPU note points at --all"
assert_contains     "$out" "hidden — --all" "1-GPU header tallies hidden count"
# PR-A / PR-B intact: health markers + grouping + Defaults view still render.
assert_contains "$out" "Health: bare max-ctx = production" "PR-A health legend present (1-GPU)"
assert_contains "$out" "(caveats," "PR-A caveats marker present (1-GPU)"
assert_contains "$out" "(NA:" "PR-A NA marker present (1-GPU)"
assert_contains "$out" "Defaults — what" "PR-B Defaults view present (1-GPU)"

# --- 2 GPUs: single + dual, no multi4 ---------------------------------------
out="$(run_list 0,1 --list)"
assert_contains     "$out" "$SINGLE_SLUG" "2-GPU shows single slugs"
assert_contains     "$out" "$DUAL_SLUG"   "2-GPU shows dual slugs"

# --- --all on a 1-GPU box: everything, NO note ------------------------------
out="$(run_list 0 --list --all)"
assert_contains     "$out" "$SINGLE_SLUG" "--all shows single slugs"
assert_contains     "$out" "$DUAL_SLUG"   "--all shows dual slugs"
assert_not_contains "$out" "hidden — --all" "--all suppresses the hidden tally"
assert_not_contains "$out" "GPU machine — use" "--all suppresses the filter note"

# --- --list-all alias behaves identically to --list --all -------------------
alias_out="$(run_list 0 --list-all)"
assert_contains     "$alias_out" "$DUAL_SLUG"  "--list-all shows dual slugs"
assert_not_contains "$alias_out" "GPU machine — use" "--list-all suppresses note"

# --- order independence: --all --list == --list --all -----------------------
reorder_out="$(run_list 0 --all --list)"
assert_contains "$reorder_out" "$DUAL_SLUG" "--all --list (reordered) shows multicard (dual)"

# --- --all without --list is rejected (not silently swallowed) --------------
if guard_out="$(bash "$SWITCH" --all 2>&1)"; then
  note "--all without --list unexpectedly succeeded"
else
  assert_contains "$guard_out" "--all only applies to --list" "--all-without-list error message"
fi

# --- fail-open: no selector + no nvidia-smi → show ALL, no note --------------
# Build a minimal PATH that deliberately omits nvidia-smi so detection has no
# signal; the filter must NOT hide dual/multi off a guess.
TMPBIN="$(mktemp -d)"
for b in bash sed awk sort cut tr wc env mktemp cat mv rm cp head tail printf grep python3 dirname docker curl jq; do
  src="$(command -v "$b" 2>/dev/null)" && ln -sf "$src" "$TMPBIN/$b"
done
failopen_out="$(PATH="$TMPBIN" CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" "$TMPBIN/bash" "$SWITCH" --list 2>&1)"
rm -rf "$TMPBIN"
assert_contains     "$failopen_out" "$DUAL_SLUG"  "fail-open shows dual slugs"
assert_not_contains "$failopen_out" "GPU machine — use" "fail-open prints no filter note"

if [[ "$fail" -ne 0 ]]; then
  echo "[list-topology-filter] FAIL" >&2
  exit 1
fi
echo "test-list-topology-filter: ok"
