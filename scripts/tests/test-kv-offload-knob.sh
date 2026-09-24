#!/usr/bin/env bash
# test-kv-offload-knob — the Qwen3.8-family dual MTP vLLM composes must expose a WORKING, off-by-default
# KV-offload knob, and only those composes may carry it.
#
# WHY THIS TEST EXISTS
# --------------------
# vLLM v0.30.0's native OffloadingConnector serves prefix hits back from host RAM on the MTP path
# (dual-fast, 2026-09-24: an 80K-token prompt 59.4 s cold -> 1.51 s after GPU eviction, 78,864 tokens
# loaded from RAM). On the DFlash tiers the same tier is WRITE-ONLY — it stores every prompt block and
# never serves one back (dual-superfast: 0 external hits, 58.8 s = cold) — so the knob there would only
# cost pinned RAM and PCIe. This guard keeps the knob on the five dual MTP composes, keeps it OFF the
# DFlash ones, and runs each compose's REAL entrypoint block to prove every documented combination:
#   unset = off (and PYTORCH_CUDA_ALLOC_CONF untouched), KV_OFFLOAD_GB -> --kv-offloading-size,
#   KV_OFFLOAD_DISK=1 -> a TieringOffloadingSpec fs tier at /kv-offload, and every malformed value
#   fails the boot instead of silently doing nothing. (LMCache is deliberately NOT wired here: when it
#   comes in, it is a separate container service, so its KV survives a model swap.)
# The SimpleCPUOffloadConnector is deliberately NOT exposed (vllm#53868: engine wedge at TP=2).
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fails=0
fail() { echo "✗ $*" >&2; fails=$((fails+1)); }

mapfile -t FILES < <(ls models/qwen3.8-27b/vllm/compose/dual/*/mtp.yml models/thinkingcap-qwen3.8-27b/vllm/compose/dual/*/mtp.yml 2>/dev/null | sort)
[[ ${#FILES[@]} -eq 5 ]] || fail "expected the 5 Qwen3.8-family dual MTP composes, found ${#FILES[@]}"
[[ -f kv-offload/.gitignore ]] || fail "kv-offload/.gitignore missing — the default disk-tier dir must exist and be gitignored"

# The real block: from `OFFLOAD_ARGS=()` to the `fi` after the summary echo, compose `$$` unescaped.
block_of() {
  awk '/^        OFFLOAD_ARGS=\(\)$/{f=1} f{print} f&&e&&/^        fi$/{exit} f&&/echo "\[kv-offload\] \$\$KV_OFFLOAD_GB GiB/{e=1}' "$1" | sed 's/\$\$/$/g'
}
resolve() {  # resolve <file> [VAR=VAL...] → "rc=<n>" line, then OFFLOAD_ARGS one per line, then PCAC=
  local f="$1"; shift
  env -i PATH="$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" bash -c \
    "( $(block_of "$f")"$'\n''printf "%s\n" "${OFFLOAD_ARGS[@]+"${OFFLOAD_ARGS[@]}"}"; echo "PCAC=${PYTORCH_CUDA_ALLOC_CONF:-unset}" ); echo "rc=$?"' 2>/dev/null
}

for f in "${FILES[@]}"; do
  for v in KV_OFFLOAD_GB KV_OFFLOAD_DISK; do
    command grep -qE "^      - ${v}=\\\$\\{${v}:-\\}\$" "$f" || fail "$f: $v not declared in environment: (docker would not forward it)"
  done
  command grep -q 'VLLM_USE_SIMPLE_KV_OFFLOAD' "$f" && fail "$f: exposes SimpleCPUOffloadConnector (vllm#53868 TP=2 wedge; not validated here)"
  command grep -qF -- '- ${KV_OFFLOAD_DIR:-../../../../../../kv-offload}:/kv-offload' "$f" || fail "$f: disk-tier mount with the repo kv-offload/ default missing"
  ex="$(command grep -c 'exec vllm serve' "$f")"; ox="$(command grep -c 'exec vllm serve.*OFFLOAD_ARGS\[@\]' "$f")"
  [[ "$ex" -gt 0 && "$ex" == "$ox" ]] || fail "$f: $ox of $ex 'exec vllm serve' lines pass OFFLOAD_ARGS"
  [[ -n "$(block_of "$f")" ]] || { fail "$f: OFFLOAD_ARGS block not found"; continue; }

  out="$(resolve "$f")"
  [[ "$out" == *"rc=0"* && "$out" != *--kv-* && "$out" == *"PCAC=expandable_segments:True"* ]] \
    || fail "$f: unset is not a clean no-op (got: ${out//$'\n'/ })"
  out="$(resolve "$f" KV_OFFLOAD_GB=64)"
  [[ "$out" == *$'--kv-offloading-size\n64\n--kv-offloading-backend\nnative'* && "$out" == *"PCAC=unset"* ]] \
    || fail "$f: KV_OFFLOAD_GB=64 did not resolve to native RAM with expandable_segments dropped (got: ${out//$'\n'/ })"
  out="$(resolve "$f" KV_OFFLOAD_GB=64 KV_OFFLOAD_DISK=1)"
  json="$(printf '%s\n' "$out" | command grep -E '^\{')"
  printf '%s' "$json" | python3 -c 'import json,sys; c=json.load(sys.stdin); e=c["kv_connector_extra_config"]
assert c["kv_connector"]=="OffloadingConnector" and e["spec_name"]=="TieringOffloadingSpec"
assert e["cpu_bytes_to_use"]==64*(1<<30) and e["secondary_tiers"]==[{"type":"fs","root_dir":"/kv-offload"}]' 2>/dev/null \
    || fail "$f: KV_OFFLOAD_DISK=1 did not resolve to a TieringOffloadingSpec fs tier (got: ${out//$'\n'/ })"
  for bad in "KV_OFFLOAD_GB=64G" "KV_OFFLOAD_GB=1.2.3" "KV_OFFLOAD_DISK=1" \
             "KV_OFFLOAD_GB=64 KV_OFFLOAD_DISK=yes" "KV_OFFLOAD_GB=-5"; do
    # shellcheck disable=SC2086
    out="$(resolve "$f" $bad)"
    [[ "$out" != *"rc=0"* ]] || fail "$f: [$bad] booted instead of failing (got: ${out//$'\n'/ })"
  done
  if docker compose version >/dev/null 2>&1; then
    src="$(env -u KV_OFFLOAD_DIR MODEL_DIR=/nonexistent docker compose -f "$f" config 2>/dev/null | command grep -B1 -E 'target: /kv-offload$' | command grep -oE 'source: .*' | sed 's/source: //')"
    [[ "$src" == "$ROOT/kv-offload" ]] || fail "$f: default disk-tier mount resolves to '$src', expected $ROOT/kv-offload"
    src="$(KV_OFFLOAD_DIR=/srv/kv MODEL_DIR=/nonexistent docker compose -f "$f" config 2>/dev/null | command grep -B1 -E 'target: /kv-offload$' | command grep -oE 'source: .*' | sed 's/source: //')"
    [[ "$src" == "/srv/kv" ]] || fail "$f: KV_OFFLOAD_DIR override renders '$src', expected /srv/kv"
  fi
done

# Scope: the knob belongs to the dual MTP composes only. DFlash is measured write-only on v0.30.0.
while IFS= read -r f; do
  printf '%s\n' "${FILES[@]}" | command grep -qxF "$f" && continue
  fail "$f: carries OFFLOAD_ARGS — only the dual MTP composes may (DFlash stores but never serves hits on v0.30.0)"
done < <(command grep -rlF 'OFFLOAD_ARGS' models/*/vllm/compose --include=*.yml | command grep -v _archive | sort)

# Registry <-> compose: the c3 catalog shows "kv opt" from the registry `kv_offload` facet, so the
# set of slugs declaring kv_offload=opt-in must be EXACTLY the set whose compose carries the knob.
reg="$(bash scripts/lib/registry-emit.sh --json 2>/dev/null | python3 -c '
import json, sys
for v in json.load(sys.stdin)["variants"]:
    print(v["compose_path"], v.get("kv_offload") or "-")' 2>/dev/null)"
[[ -n "$reg" ]] || fail "registry-emit.sh --json produced no variants (cannot check the kv_offload facet)"
while read -r cp kvo; do
  [[ -f "$cp" ]] || continue
  if command grep -qF 'OFFLOAD_ARGS' "$cp"; then
    [[ "$kvo" == "opt-in" ]] || fail "$cp: compose carries the knob but the registry kv_offload is '$kvo' (c3 would not show it)"
  else
    [[ "$kvo" == "-" ]] || fail "$cp: registry kv_offload='$kvo' but the compose has no KV-offload knob"
  fi
done <<< "$reg"

if [[ "$fails" -gt 0 ]]; then
  echo "test-kv-offload-knob: $fails failure(s) across ${#FILES[@]} composes" >&2
  exit 1
fi
echo "test-kv-offload-knob: ok (${#FILES[@]} dual MTP composes: off by default, RAM and RAM+disk resolve, 5 bad inputs refused, registry kv_offload in sync)"
