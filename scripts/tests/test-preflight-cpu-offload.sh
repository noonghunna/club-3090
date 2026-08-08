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

# ---- residency-aware RAM gate (the header is the ALL-on-CPU worst case; the ----
# ---- gate subtracts what THIS rig's VRAM will hold — clort81's #909 report) ----

# shared fit arithmetic must reproduce the MEASURED resident counts, or the gate
# and the boot disagree about placement (the 0.55 calibration is load-bearing)
declare -F _offload_fit_count >/dev/null 2>&1 && ok "_offload_fit_count defined" \
  || bad "_offload_fit_count missing"
[[ "$(_offload_fit_count 24576 18000 3264 21)" == "1" ]] \
  && ok "fit: Q8 dual 2x24 -> 1/card (measured 2 total)" || bad "fit arithmetic drifted (Q8 dual)"
[[ "$(_offload_fit_count 24576 14000 1840 21)" == "3" ]] \
  && ok "fit: IQ2 dual 2x24 -> 3/card (measured 6 total)" || bad "fit arithmetic drifted (IQ2 dual)"
[[ "$(_offload_fit_count 24576 12000 3264 10)" == "2" ]] \
  && ok "fit: Q8 multi4 4x24 -> 2/card (@milano's boot)" || bad "fit arithmetic drifted (multi4)"
[[ "$(_offload_fit_count 16384 12000 3264 10)" == "0" ]] \
  && ok "fit: 4x16 GB -> 0 bundles (the under-gate shape a pre-baked header missed)" \
  || bad "4x16 GB should fit ZERO bundles"

# deterministic VRAM via a stubbed nvidia-smi — the test must not depend on the
# rig it runs on (or on having GPUs at all)
stub="$(mktemp -d)"; trap 'rm -f "$tmp"; rm -rf "$stub"' EXIT
_mkstub() { printf '#!/usr/bin/env bash\nprintf "%s"\n' "$1" > "$stub/nvidia-smi"; chmod +x "$stub/nvidia-smi"; }

_mkstub '24576\n24576\n'
g="$(PATH="$stub:$PATH" offload_residency_grant_mib "$Q8")"
[[ "$g" == "6528" ]] && ok "grant: Q8 dual on 2x24 = 6528 MiB (2 bundles)" \
  || bad "Q8 dual grant '$g' != 6528"
g="$(PATH="$stub:$PATH" offload_residency_grant_mib "$IQ2")"
[[ "$g" == "11040" ]] && ok "grant: IQ2 dual on 2x24 = 11040 MiB (6 bundles)" \
  || bad "IQ2 dual grant '$g' != 11040"

_mkstub '24576\n24576\n24576\n24576\n'
g="$(PATH="$stub:$PATH" offload_residency_grant_mib "$M4")"
[[ "$g" == "26112" ]] && ok "grant: multi4 on 4x24 = 26112 MiB (8 bundles -> gate ~121 GB, @milano measured 120)" \
  || bad "multi4 grant '$g' != 26112"

_mkstub '24576\n'
g="$(PATH="$stub:$PATH" offload_residency_grant_mib "$Q8")"
[[ "$g" == "0" ]] && ok "grant: single GPU degrades to 0 (worst case — safe direction)" \
  || bad "single-GPU grant '$g' != 0"

# the injector itself, through the same shared helpers: known-good placement from
# the sizing session — card 0 pins layer 0, card 1 pins layer 42 (outer edges)
_mkstub '24576\n24576\n'
( PATH="$stub:$PATH" resolve_offload_residency "$Q8"
  [[ "${OT_G0:-}" == 'blk\.(0)\.ffn_(gate|up|down)_exps\.weight=CUDA0' ]] \
    && [[ "${OT_G1:-}" == 'blk\.(42)\.ffn_(gate|up|down)_exps\.weight=CUDA1' ]] ) \
  && ok "injector: 2x24 Q8 pins blk.0->CUDA0 + blk.42->CUDA1 (outer-edge)" \
  || bad "injector OT_G rules drifted from the known-good 2x24 placement"

# an explicit OT_G<i> from the user must NEVER be clobbered (the supported way to
# pin more residency than the calibrated sizer grants — #931), and the OTHER
# card's slot must still be auto-sized
( export OT_G0='blk\.(0|1|2|3)\.ffn_(gate|up|down)_exps\.weight=CUDA0'
  PATH="$stub:$PATH" resolve_offload_residency "$Q8" 2>/dev/null
  [[ "$OT_G0" == 'blk\.(0|1|2|3)\.ffn_(gate|up|down)_exps\.weight=CUDA0' ]] \
    && [[ "${OT_G1:-}" == 'blk\.(42)\.ffn_(gate|up|down)_exps\.weight=CUDA1' ]] ) \
  && ok "user OT_G0 wins; OT_G1 still auto-sized" \
  || bad "user OT_G override was clobbered (or blocked the other card's auto-size)"

# ---- user pins and the RAM gate must describe the SAME config (#931: paul's ----
# ---- Q8 attempt was gated on the auto ~12 GB while pinning 4x that)         ----

declare -F _offload_rule_layer_count >/dev/null 2>&1 && ok "_offload_rule_layer_count defined" \
  || bad "_offload_rule_layer_count missing"
[[ "$(_offload_rule_layer_count 'blk\.(0|1|2|3|4|5|6|7)\.ffn_(gate|up|down)_exps\.weight=CUDA0')" == "8" ]] \
  && ok "rule parser: 8-layer rule -> 8" || bad "rule parser miscounted an 8-layer rule"
[[ "$(_offload_rule_layer_count 'blk\.(42)\.ffn_(gate|up|down)_exps\.weight=CUDA1')" == "1" ]] \
  && ok "rule parser: single layer -> 1" || bad "rule parser miscounted a 1-layer rule"
[[ "$(_offload_rule_layer_count 'not-an-ot-rule')" == "0" ]] \
  && ok "rule parser: garbage -> 0 (worst case, safe)" || bad "rule parser did not zero on garbage"

# gate prices the user's ACTUAL pin: 4+4 layers = 8x3264 = 26112 MiB -> ~25 GB
# subtracted from the 999999 worst case
g="$(export OT_G0='blk\.(0|1|2|3)\.ffn_(gate|up|down)_exps\.weight=CUDA0' \
            OT_G1='blk\.(42|41|40|39)\.ffn_(gate|up|down)_exps\.weight=CUDA1'
     PATH="$stub:$PATH" offload_residency_grant_mib "$Q8")"
[[ "$g" == "26112" ]] && ok "grant follows user pins (8 layers = 26112 MiB, auto would say 6528)" \
  || bad "grant '$g' ignored user pins (expected 26112)"
printf '# CPU-Offload-Host-RAM-GB: 999999\n# CPU-Offload-Bundle-MiB: 3264\n# CPU-Offload-MoE-Layers: 43\n# CPU-Offload-First-MoE-Layer: 0\n# CPU-Offload-GPU-Reserve-MiB: 18000\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
msg="$(export OT_G0='blk\.(0|1|2|3)\.ffn_(gate|up|down)_exps\.weight=CUDA0' \
              OT_G1='blk\.(42|41|40|39)\.ffn_(gate|up|down)_exps\.weight=CUDA1'
       PATH="$stub:$PATH" preflight_cpu_offload_ram "$tmp" 2>&1)"
command grep -q "999974" <<<"$msg" \
  && ok "gate subtracts the USER grant (999999 - 25 GB)" \
  || bad "gate did not price the user pin (no 999974 in: $(head -1 <<<"$msg"))"

# an over-pin beyond even the card's naive free space must WARN (certain OOM —
# e.g. an IQ2-sized 8/card recipe applied to Q8's 3264 MiB bundles), and a sane
# pin must stay silent
w="$(export OT_G0='blk\.(0|1|2|3|4|5|6|7)\.ffn_(gate|up|down)_exps\.weight=CUDA0'
     PATH="$stub:$PATH" resolve_offload_residency "$Q8" 2>&1 >/dev/null)"
command grep -q "WARN: OT_G0 pins 8" <<<"$w" \
  && ok "over-pin (8x3264 on a 24 GB card) warns of certain OOM" \
  || bad "no WARN for a pin that cannot fit the card"
w="$(export OT_G0='blk\.(0)\.ffn_(gate|up|down)_exps\.weight=CUDA0'
     PATH="$stub:$PATH" resolve_offload_residency "$Q8" 2>&1 >/dev/null)"
command grep -q "WARN" <<<"$w" \
  && bad "sane 1-bundle pin wrongly warned" \
  || ok "sane pin stays silent"

# the gate must SUBTRACT the grant: worst-case 999999 minus 6528 MiB -> ~999993
printf '# CPU-Offload-Host-RAM-GB: 999999\n# CPU-Offload-Bundle-MiB: 3264\n# CPU-Offload-MoE-Layers: 43\n# CPU-Offload-First-MoE-Layer: 0\n# CPU-Offload-GPU-Reserve-MiB: 18000\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
msg="$(PATH="$stub:$PATH" preflight_cpu_offload_ram "$tmp" 2>&1)"
command grep -q "999993" <<<"$msg" \
  && ok "gate subtracts the detected residency grant (999999 - 6 GB)" \
  || bad "gate did not subtract residency (no 999993 in: $(head -1 <<<"$msg"))"

# same quant => same all-on-CPU worst case: dual Q8 and multi4 Q8 headers must be
# EQUAL. If someone re-bakes expected residency into one (the shape multi4 briefly
# shipped), this trips before a 4x16 GB rig thrashes.
q8_hdr="$(compose_meta_get "$Q8" cpu-offload-host-ram-gb)"
m4_hdr="$(compose_meta_get "$M4" cpu-offload-host-ram-gb)"
[[ "$q8_hdr" == "$m4_hdr" ]] \
  && ok "dual-Q8 and multi4-Q8 headers agree ($q8_hdr GB — same quant, same worst case)" \
  || bad "Q8 headers diverge (dual=$q8_hdr multi4=$m4_hdr): residency was pre-baked into one"

# registry host_ram_gb is the NOMINAL display figure and must never exceed the
# header's worst case (nominal = worst case minus residency, on any topology)
while IFS='|' read -r slug reg_gb; do
  case "$slug" in
    *dual-q8)   f="$Q8" ;;
    *dual-iq2)  f="$IQ2" ;;
    *multi4-q8) f="$M4" ;;
    *) continue ;;
  esac
  hdr="$(compose_meta_get "$f" cpu-offload-host-ram-gb)"
  [[ "$reg_gb" =~ ^[0-9]+$ && "$reg_gb" -le "$hdr" ]] \
    && ok "$slug: registry nominal ($reg_gb) <= header worst case ($hdr)" \
    || bad "$slug: registry host_ram_gb=$reg_gb exceeds header worst case $hdr"
done < <(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
for slug, e in COMPOSE_REGISTRY.items():
    if "deepseek-flash" in slug and e.get("host_ram_gb"):
        print(f"{slug}|{e['host_ram_gb']}")
PY
)

# ---- thread resolution (nproc/2 -- a SAFE limit for rigs whose core count we don't know) ----
declare -F resolve_offload_threads >/dev/null 2>&1 && ok "resolve_offload_threads defined" \
  || bad "resolve_offload_threads missing"
( unset THREADS; resolve_offload_threads "$Q8" >/dev/null 2>&1
  want=$(( $(nproc) / 2 )); [[ "${THREADS:-}" == "$want" ]] ) \
  && ok "resolves THREADS=nproc/2" || bad "THREADS != nproc/2"
( export THREADS=8; resolve_offload_threads "$Q8" >/dev/null 2>&1; [[ "$THREADS" == "8" ]] ) \
  && ok "explicit THREADS is never clobbered" || bad "user THREADS was overwritten"
( unset THREADS; resolve_offload_threads "$NONOFF" >/dev/null 2>&1; [[ -z "${THREADS:-}" ]] ) \
  && ok "no-op on a non-offload compose" || bad "set THREADS on a non-offload compose"
# the bare-compose fallback must be LOW, not the reference rig's number: over-subscribing
# measured -69% vs under-subscribing -9.6%, so err low when the core count is unknown.
for f in "$Q8" "$IQ2" "$M4"; do
  fb="$(command grep -oE 'THREADS:-[0-9]+' "$f" | command grep -oE '[0-9]+' | head -1)"
  [[ -n "$fb" && "$fb" -le 8 ]] && ok "$(basename "$(dirname "$f")"): safe THREADS fallback ($fb)" \
    || bad "$(basename "$(dirname "$f")"): THREADS fallback '$fb' is too high for an unknown rig"
done

# ---- wiring ----
command grep -q "preflight_cpu_offload_ram" scripts/switch.sh \
  && ok "switch.sh calls the RAM guard" || bad "switch.sh does not call the RAM guard"
command grep -q "preflight_offload_split_mode" scripts/switch.sh \
  && ok "switch.sh calls the split-mode guard" || bad "switch.sh does not call the split-mode guard"

[[ $fail -eq 0 ]] && echo "test-preflight-cpu-offload: ok" || echo "test-preflight-cpu-offload: FAIL"
exit $fail
