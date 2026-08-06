#!/usr/bin/env bash
# test-preflight-cpu-offload.sh — guards the CPU-offload preflight checks.
#
# WHY THIS EARNS ITS KEEP: these guards only ever fire on configs we DON'T ship.
# Our own composes are correct by construction, so the refusal paths are never
# exercised in normal use and would rot unnoticed. The two false-positive cases
# below are the ones that would do real damage:
#   • firing on a GPU-RESIDENT compose (=CUDA rules are not offload)
#   • refusing `tensor` on a compose that never offloads
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
fail=0
ok()  { echo "  ok   — $1"; }
bad() { echo "  FAIL — $1" >&2; fail=1; }

echo "== test-preflight-cpu-offload =="

# shellcheck source=/dev/null
source scripts/lib/compose-meta.sh
# shellcheck source=/dev/null
source scripts/preflight.sh 2>/dev/null

for fn in is_cpu_offload_compose preflight_offload_split_mode preflight_cpu_offload_ram; do
  declare -F "$fn" >/dev/null 2>&1 && ok "$fn defined" || bad "$fn missing"
done

Q8=models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-q8-kxl/offload.yml
IQ2=models/deepseek-v4-flash-0731/llama-cpp/compose/dual/unsloth-iq2-xxs/offload.yml
M4=models/deepseek-v4-flash-0731/llama-cpp/compose/multi4/unsloth-q8-kxl/offload.yml
NONOFF=models/tess-4-27b/llama-cpp/compose/dual/migtissera-q4km/mtp.yml

# ---- detector ----
for f in "$Q8" "$IQ2" "$M4"; do
  is_cpu_offload_compose "$f" && ok "detects offload: $(basename "$(dirname "$f")")" \
    || bad "MISSED offload compose $f"
done
is_cpu_offload_compose "$NONOFF" && bad "FALSE POSITIVE on a non-offload compose" \
  || ok "non-offload compose correctly ignored"

# ⚠️ the detector must key on =CPU, NOT on the presence of -ot: our offload composes
# carry 2-4 `-ot ...=CUDA*` RESIDENCY rules, which are the opposite of offload.
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
printf 'services:\n  x:\n    command: >-\n      -ngl 99 -ot blk\\.1\\.ffn_gate_exps\\.weight=CUDA0 -sm layer\n' > "$tmp"
is_cpu_offload_compose "$tmp" && bad "detector fires on a GPU-RESIDENT (=CUDA) -ot rule" \
  || ok "does not fire on =CUDA residency rules (the -ot false-positive trap)"

printf 'services:\n  x:\n    command: >-\n      --n-cpu-moe 20 -sm layer\n' > "$tmp"
is_cpu_offload_compose "$tmp" && ok "detects --n-cpu-moe" || bad "missed --n-cpu-moe"
printf 'services:\n  x:\n    command: >-\n      --cpu-moe -sm layer\n' > "$tmp"
is_cpu_offload_compose "$tmp" && ok "detects --cpu-moe" || bad "missed --cpu-moe"

# ---- split-mode guard ----
preflight_offload_split_mode "$Q8" >/dev/null 2>&1 \
  && ok "offload + layer passes" || bad "offload + layer wrongly refused"
for m in tensor row; do
  SPLIT_MODE="$m" preflight_offload_split_mode "$Q8" >/dev/null 2>&1 \
    && bad "offload + $m NOT refused" || ok "offload + $m refused"
done
SPLIT_MODE=tensor preflight_offload_split_mode "$NONOFF" >/dev/null 2>&1 \
  && ok "tensor allowed on a NON-offload compose" || bad "wrongly refused tensor without offload"

# the refusal must tell the user what to do instead, or they are stranded
msg="$(SPLIT_MODE=tensor preflight_offload_split_mode "$Q8" 2>&1)"
command grep -q -- "--split-mode layer" <<<"$msg" \
  && ok "refusal names the fix (--split-mode layer)" || bad "refusal does not name the fix"
command grep -qE "85%|305" <<<"$msg" \
  && ok "refusal cites the measurement" || bad "refusal has no evidence"

# ---- RAM guard ----
for f in "$Q8" "$IQ2" "$M4"; do
  v="$(compose_meta_get "$f" cpu-offload-host-ram-gb || true)"
  [[ "$v" =~ ^[0-9]+$ ]] && ok "declares CPU-Offload-Host-RAM-GB=$v" \
    || bad "$f missing/invalid CPU-Offload-Host-RAM-GB"
done
preflight_cpu_offload_ram "$NONOFF" >/dev/null 2>&1 \
  && ok "RAM guard is a no-op without the header" || bad "RAM guard errored on a non-offload compose"

# a host that cannot possibly satisfy it must be REFUSED, not warned
printf '# CPU-Offload-Host-RAM-GB: 999999\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
preflight_cpu_offload_ram "$tmp" >/dev/null 2>&1 \
  && bad "impossible RAM requirement was NOT refused" || ok "impossible RAM requirement refused"

# ---- wiring ----
command grep -q "preflight_cpu_offload_ram" scripts/switch.sh \
  && ok "switch.sh calls the RAM guard" || bad "switch.sh does not call the RAM guard"
command grep -q "preflight_offload_split_mode" scripts/switch.sh \
  && ok "switch.sh calls the split-mode guard" || bad "switch.sh does not call the split-mode guard"

[[ $fail -eq 0 ]] && echo "test-preflight-cpu-offload: ok" || echo "test-preflight-cpu-offload: FAIL"
exit $fail
