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
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
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
# The FORK (moe-cache) siblings. They are offload composes too, and they shipped
# for five days with NO CPU-Offload-Host-RAM-GB header at all -- so the RAM gate
# early-returned on the two slugs with the LARGEST host footprint in the catalog
# (they also hold the drafter in host RAM). A 128 GB owner got a raw
# cudaMallocHost failure instead of our refusal. Every list below includes them.
FORK_Q8=models/deepseek-v4-flash-0731/llamacpp-club3090/compose/dual/unsloth-q8-kxl/moecache.yml
FORK_M4=models/deepseek-v4-flash-0731/llamacpp-club3090/compose/multi4/unsloth-q8-kxl/moecache.yml
# ── Inkling-Small: the SAME defect, on a second model (#978). Both moe-cache
# composes carried only prose ("Needs ~120 GB of HOST RAM") and no header, so the
# gate early-returned again -- this time a dual-5090 / 128 GB owner's container
# was OOM-killed during load and his desktop froze. Prose is not a gate.
INK_CACHE=models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml
INK_M4=models/inkling-small/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml
# ⭐ the RESIDENCY sibling shipped by #978 -- the answer for a host-RAM-bound rig,
# and the one Inkling compose that legitimately DOES carry the bundle headers.
# Its presence is why the no-residency-headers assertions below are per-file.
INK_RES=models/inkling-small/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/residency.yml
# exl3/TabbyAPI — offloads by splitting EXPERT INDICES to the host and COMPUTING
# them there, so these carry host_ram_gb like any CPU-offload slug even though
# they pass no `-ot`.
EXL3_305=models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-3.05bpw/cpumoe.yml
EXL3_405=models/qwen3.8-flash-next/exllamav3/compose/dual/exl3-4.05bpw/cpumoe.yml
GLM_DUAL=models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq4xs/moecache.yml
GLM_M4=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq4xs/moecache.yml
GLM_M8=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq4xs/moecache.yml
GLM3_DUAL=models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq3xxs/moecache.yml
GLM3_M4=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/unsloth-ud-iq3xxs/moecache.yml
GLM3_M8=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/unsloth-ud-iq3xxs/moecache.yml
GLM_Q2K_DUAL=models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/moecache.yml
GLM_Q2K_M4=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/moecache.yml
GLM_Q2K_M8=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/moecache.yml
GLM_Q3KM_DUAL=models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/moecache.yml
GLM_Q3KM_M4=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/moecache.yml
GLM_Q3KM_M8=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/moecache.yml
GLM_Q2K_DUAL_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q2k/offload.yml
GLM_Q2K_M4_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q2k/offload.yml
GLM_Q2K_M8_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q2k/offload.yml
GLM_Q3KM_DUAL_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/offload.yml
GLM_Q3KM_M4_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/multi4/devquasar-q3km/offload.yml
GLM_Q3KM_M8_OFF=models/glm-5.3-flash/llamacpp-club3090/compose/multi8/devquasar-q3km/offload.yml
QWN_DUAL=models/qwen3.8-flash-next/llamacpp-club3090/compose/dual/unsloth-ud-q4kxl/moecache.yml
QWN_M4=models/qwen3.8-flash-next/llamacpp-club3090/compose/multi4/unsloth-ud-q4kxl/moecache.yml
QWN_M8=models/qwen3.8-flash-next/llamacpp-club3090/compose/multi8/unsloth-ud-q4kxl/moecache.yml
NONOFF=models/tess-4-27b/llama-cpp/compose/dual/migtissera-q4km/mtp.yml

# ---- detector ----
for f in "$Q8" "$IQ2" "$M4" "$FORK_Q8" "$FORK_M4" "$INK_CACHE" "$INK_M4" "$INK_RES"; do
  is_cpu_offload_compose "$f" && ok "detects offload: $(basename "$(dirname "$f")")/$(basename "$f")" \
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
# ⚠️ EVERY offload compose must declare the header. Without it the gate silently
# early-returns -- the failure mode is INVISIBLE (no warning, just no refusal),
# which is exactly how the two moe-cache slugs shipped ungated.
for f in "$Q8" "$IQ2" "$M4" "$FORK_Q8" "$FORK_M4" "$INK_CACHE" "$INK_M4" "$INK_RES"; do
  v="$(compose_meta_get "$f" cpu-offload-host-ram-gb || true)"
  [[ "$v" =~ ^[0-9]+$ ]] && ok "declares CPU-Offload-Host-RAM-GB=$v ($(basename "$(dirname "$f")")/$(basename "$f"))" \
    || bad "$f missing/invalid CPU-Offload-Host-RAM-GB"
done
preflight_cpu_offload_ram "$NONOFF" >/dev/null 2>&1 \
  && ok "RAM guard is a no-op without the header" || bad "RAM guard errored on a non-offload compose"

# a host that cannot possibly satisfy it must be REFUSED, not warned
printf '# CPU-Offload-Host-RAM-GB: 999999\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
preflight_cpu_offload_ram "$tmp" >/dev/null 2>&1 \
  && bad "impossible RAM requirement was NOT refused" || ok "impossible RAM requirement refused"

# ⚠️ the refusal's "buy more cards" hint must be TRUE for the compose it prints on.
# It only holds where residency can pin bundles back onto the GPUs — i.e. where the
# bundle header exists. On an all-experts-on-CPU compose (the moe-cache slugs) more
# GPUs change nothing, and telling a RAM-short user otherwise sends them shopping.
msg="$(preflight_cpu_offload_ram "$tmp" 2>&1)"
command grep -q "more GPUs will NOT lower" <<<"$msg" \
  && ok "refusal without a bundle header says more GPUs will NOT help" \
  || bad "no-residency refusal still promises more cards will help: $msg"
printf '# CPU-Offload-Host-RAM-GB: 999999\n# CPU-Offload-Bundle-MiB: 3264\n# CPU-Offload-MoE-Layers: 43\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
msg="$(preflight_cpu_offload_ram "$tmp" 2>&1)"
command grep -q "needs LESS host RAM" <<<"$msg" \
  && ok "refusal WITH a bundle header keeps the more-cards hint" \
  || bad "residency-capable refusal lost the more-cards hint: $msg"

# ---- residency-aware RAM gate (the header is the ALL-on-CPU worst case; the ----
# ---- gate subtracts what THIS rig's VRAM will hold — clort81's #909 report) ----

# shared fit arithmetic must reproduce the MEASURED resident counts, or the gate
# and the boot disagree about placement. The model is ADDITIVE (#931):
#   fit = (free − reserve_adjusted − margin) / bundle, margin default 1024,
# where reserve_adjusted = compose reserve (+768 first-card extra on card 0,
# added by the CALLER). Args here are (free, reserve_adjusted, bundle, cap).
declare -F _offload_fit_count >/dev/null 2>&1 && ok "_offload_fit_count defined" \
  || bad "_offload_fit_count missing"
[[ "$(_offload_fit_count 24576 18000 3264 21)" == "1" ]] \
  && ok "fit: Q8 dual 24 GB card -> 1 (measured)" || bad "fit arithmetic drifted (Q8 dual)"
[[ "$(_offload_fit_count 24576 18768 3264 21)" == "1" ]] \
  && ok "fit: Q8 dual 24 GB CARD 0 (+extra) -> still 1" || bad "fit drifted (Q8 card 0)"
[[ "$(_offload_fit_count 24576 16500 1840 21)" == "3" ]] \
  && ok "fit: IQ2 dual 24 GB card -> 3 (measured 6 total)" || bad "fit arithmetic drifted (IQ2 dual)"
[[ "$(_offload_fit_count 24576 14500 3264 10)" == "2" ]] \
  && ok "fit: Q8 multi4 24 GB card -> 2 (@milano's boot)" || bad "fit arithmetic drifted (multi4)"
[[ "$(_offload_fit_count 16384 14500 3264 10)" == "0" ]] \
  && ok "fit: 16 GB card -> 0 bundles (the under-gate shape a pre-baked header missed)" \
  || bad "16 GB card should fit ZERO bundles"
# ⭐ the #931 calibration points — 32 GB cards must NOT under-fill (old ×0.55
# left ~6 GB idle, −19% decode) and card 0 must NOT over-fill (8/8 died at
# 556 MiB free on a ~90K prefill; 7+8 survived a full 188K ladder at 896 MiB)
[[ "$(_offload_fit_count 32500 16500 1840 21)" == "8" ]] \
  && ok "fit: IQ2 32 GB bare card -> 8 (#931 GPU1, field-validated)" \
  || bad "32 GB bare card should fit 8 bundles"
[[ "$(_offload_fit_count 32500 17268 1840 21)" == "7" ]] \
  && ok "fit: IQ2 32 GB CARD 0 (+extra) -> 7 (#931 — the 8/8 death config must not return)" \
  || bad "32 GB card 0 should fit 7 bundles, not 8"
[[ "$(RESIDENCY_MARGIN_MB=4096 _offload_fit_count 24576 16500 1840 21)" == "2" ]] \
  && ok "fit: RESIDENCY_MARGIN_MB env widens the margin (4096 -> 2 bundles)" \
  || bad "RESIDENCY_MARGIN_MB override not respected"

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

# ⭐ #931 end-to-end: on 2x32 GB the IQ2 sizer must go ASYMMETRIC from IDENTICAL
# free readings — 7 on card 0 (first-card extra prices the larger drafter half +
# compute buffer), 8 on card 1 — matching the field-validated recipe (blk 0-6 /
# blk 35-42). The symmetric 8/8 it replaces died at 556 MiB free at ~90K depth.
_mkstub '32500\n32500\n'
w="$(unset OT_G0 OT_G1; PATH="$stub:$PATH" resolve_offload_residency "$IQ2" 2>&1 >/dev/null)"
command grep -q "card0: auto 7 bundles (blk 0|1|2|3|4|5|6)" <<<"$w" \
  && command grep -q "card1: auto 8 bundles (blk 42|41|40|39|38|37|36|35)" <<<"$w" \
  && ok "sizer: 2x32 GB IQ2 -> asymmetric 7+8, outer-edge layers (#931 field-validated)" \
  || bad "2x32 GB IQ2 expected 7+8 asymmetric, got: $w"
_mkstub '24576\n24576\n'   # restore the 2x24 stub for the legs below

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

# the boot line must SAY what was pinned and who decided it — three community
# debugging rounds (#931 twice, the multi4 first boot) needed docker-inspect
# forensics to answer exactly this
w="$(PATH="$stub:$PATH" resolve_offload_residency "$Q8" 2>&1 >/dev/null)"
command grep -q "card0: auto 1 bundles (blk 0)" <<<"$w" \
  && command grep -q "card1: auto 1 bundles (blk 42)" <<<"$w" \
  && ok "boot line reports auto residency per card" \
  || bad "no auto-residency boot line (got: $w)"
w="$(export OT_G0='blk\.(0|1|2|3)\.ffn_(gate|up|down)_exps\.weight=CUDA0'
     PATH="$stub:$PATH" resolve_offload_residency "$Q8" 2>&1 >/dev/null)"
command grep -q "card0: USER pin, 4 bundles" <<<"$w" \
  && ok "boot line marks a user pin as USER" \
  || bad "user pin not marked in the boot line (got: $w)"

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

# same again for the fork pair: same quant, same all-on-CPU `-ot`, same CPU
# drafter, and the CUDA_Host buffer is card-count-independent.
fq8_hdr="$(compose_meta_get "$FORK_Q8" cpu-offload-host-ram-gb)"
fm4_hdr="$(compose_meta_get "$FORK_M4" cpu-offload-host-ram-gb)"
[[ "$fq8_hdr" == "$fm4_hdr" ]] \
  && ok "fork dual and fork multi4 headers agree ($fq8_hdr GB)" \
  || bad "fork headers diverge (dual=$fq8_hdr multi4=$fm4_hdr)"

# ⭐ the fork slugs must ask for MORE than the stock ones on the same weights.
# `-devd none` puts the 10386.28 MiB DSpark model in HOST memory (the stock slug
# keeps it in VRAM), so a header merely COPIED from the stock sibling under-gates
# by ~10 GB. That copy is what shipped in the registry before this guard existed.
(( fq8_hdr > q8_hdr )) \
  && ok "fork Q8 header ($fq8_hdr) > stock Q8 header ($q8_hdr) — the CPU drafter is charged" \
  || bad "fork Q8 header ($fq8_hdr) does not exceed stock ($q8_hdr): the -devd none drafter is not charged"

# ⚠️ and they must carry NO residency headers. The fork composes' `-ot` is an
# unconditional all-experts->CPU catch-all with no ${OT_G*} slots, so nothing is
# ever pinned back onto the GPUs. Declaring CPU-Offload-Bundle-MiB would make the
# gate SUBTRACT a grant that never materialises -- under-gating every rig.
for f in "$FORK_Q8" "$FORK_M4" "$INK_CACHE" "$INK_M4"; do
  b="$(compose_meta_get "$f" cpu-offload-bundle-mib || true)"
  [[ -z "$b" ]] && ok "$(basename "$(dirname "$f")")/moecache.yml declares no bundle header (grant must stay 0)" \
    || bad "$f declares CPU-Offload-Bundle-MiB=$b but has no OT_G* slots — the gate would under-gate"
  command grep -q 'OT_G[0-9]' "$f" \
    && bad "$f now has OT_G* slots — it needs the residency headers after all" \
    || ok "$(basename "$(dirname "$f")")/moecache.yml has no OT_G* slots (all experts stay on CPU)"
done

# the Inkling cache pair: same weights, same all-on-CPU `-ot`, no drafter, and the
# CUDA_Host buffer is card-count-independent -- so dual and multi4 must AGREE, and
# the multi4 must NOT be discounted for "more cards". The cache is a COPY.
ic_hdr="$(compose_meta_get "$INK_CACHE" cpu-offload-host-ram-gb)"
im_hdr="$(compose_meta_get "$INK_M4" cpu-offload-host-ram-gb)"
[[ "$ic_hdr" == "$im_hdr" ]] \
  && ok "inkling dual and multi4 moe-cache headers agree ($ic_hdr GB)" \
  || bad "inkling headers diverge (dual=$ic_hdr multi4=$im_hdr): a cache slug's host RAM does not fall with card count"

# ---- the RESIDENCY compose is the mirror image: it MUST carry the full set ----
# A residency compose that declares OT_G* slots but omits the bundle headers gets
# grant 0 -- the launcher pins nothing, and the slug silently becomes an expensive
# copy of the cache sibling. Both halves must be present or neither.
command grep -q 'OT_G[0-9]' "$INK_RES" \
  && ok "inkling residency.yml has OT_G* slots (the launcher can pin)" \
  || bad "$INK_RES has no OT_G* slots — nothing will ever be pinned"
for k in bundle-mib moe-layers first-moe-layer gpu-reserve-mib first-card-extra-mib; do
  v="$(compose_meta_get "$INK_RES" "cpu-offload-$k" || true)"
  [[ "$v" =~ ^[0-9]+$ ]] && ok "inkling residency.yml declares CPU-Offload-$k=$v" \
    || bad "$INK_RES missing/invalid CPU-Offload-$k (grant would silently be 0)"
done
# ⚠️ the residency compose's host-RAM header must EQUAL its cache sibling's. It is
# the ALL-on-CPU worst case for BOTH -- preflight subtracts this rig's grant itself
# (offload_residency_grant_mib), so pre-baking expected residency here would
# DOUBLE-COUNT it and under-gate every rig.
ir_hdr="$(compose_meta_get "$INK_RES" cpu-offload-host-ram-gb)"
[[ "$ir_hdr" == "$ic_hdr" ]] \
  && ok "inkling residency header == cache header ($ir_hdr GB — worst case, grant subtracted at runtime)" \
  || bad "inkling residency header ($ir_hdr) != cache header ($ic_hdr): residency was pre-baked into the header"

# ⭐⭐ THE WINDOW MUST NOT CONTAIN AN OVERSIZED BUNDLE — the subtle one, and the
# reason CPU-Offload-MoE-Layers is 38 on a model with 40 MoE layers.
# _offload_layers_for_card() gives the LAST card outer-edge selection counting DOWN
# from the top of the window, so widening the window to 40 hands that card blk.41
# (3856 MiB) and blk.40 (3440 MiB) FIRST while the scalar gate prices both at 2848
# -- over-pinning it by 1,600 MiB and eating the deep-prefill margin. Nothing else
# would catch that: the compose still boots, the gate still prints a number, and
# the card just runs closer to the edge than #931's bracket allows.
# Cross-checked against the MEASURED per-layer table in the model profile.
while IFS='|' read -r line; do
  case "$line" in
    OK\|*)   ok "${line#OK|}" ;;
    FAIL\|*) bad "${line#FAIL|}" ;;
  esac
done < <(python3 - "$INK_RES" <<'PY'
import re
import sys

compose = open(sys.argv[1], encoding="utf-8").read()

def hdr(key):
    m = re.search(rf"^#\s*{key}:\s*(\d+)\s*$", compose, re.M)
    return int(m.group(1)) if m else None

first, count, scalar = hdr("CPU-Offload-First-MoE-Layer"), hdr("CPU-Offload-MoE-Layers"), hdr("CPU-Offload-Bundle-MiB")
prof = open("scripts/lib/profiles/models/inkling-small.yml", encoding="utf-8").read()
m = re.search(r"^\s*per_layer_mib_default:\s*(\d+)", prof, re.M)
default = int(m.group(1)) if m else None
m = re.search(r"^\s*per_layer_mib_overrides:\s*\{([^}]*)\}", prof, re.M)
overrides = {}
if m:
    for pair in m.group(1).split(","):
        if ":" in pair:
            k, v = pair.split(":")
            overrides[int(k.strip())] = int(v.strip())

if None in (first, count, scalar, default):
    print("FAIL|could not read the residency window or the profile's per-layer table")
    sys.exit(0)
if scalar != default:
    print(f"FAIL|compose bundle {scalar} MiB != profile per_layer_mib_default {default} MiB")
else:
    print(f"OK|compose bundle ({scalar} MiB) == profile per_layer_mib_default")
if not overrides:
    print("FAIL|profile declares no per_layer_mib_overrides — the outlier guard cannot run")
    sys.exit(0)
window = range(first, first + count)
bad_layers = sorted(l for l in overrides if l in window)
if bad_layers:
    detail = ", ".join(f"blk.{l}={overrides[l]}MiB" for l in bad_layers)
    print(f"FAIL|residency window blk.{first}-{first+count-1} contains OVERSIZED bundles ({detail}) "
          f"but the gate prices every layer at {scalar} MiB — narrow CPU-Offload-MoE-Layers")
else:
    print(f"OK|residency window blk.{first}-{first+count-1} holds only uniform {scalar} MiB bundles "
          f"(outliers {sorted(overrides)} correctly excluded)")
PY
)

# registry host_ram_gb is the NOMINAL display figure and must never exceed the
# header's worst case (nominal = worst case minus residency, on any topology)
# ⚠️ every deepseek-flash slug carrying host_ram_gb MUST map to a compose here.
# The `*-moecache` slugs used to fall through the `*)` arm and go UNCHECKED,
# which is how they kept a copied 146 while their compose needed ~10 GB more.
while IFS='|' read -r slug reg_gb; do
  case "$slug" in
    *dual-q8-moecache)      f="$FORK_Q8" ;;
    *multi4-q8-moecache)    f="$FORK_M4" ;;
    # ⚠️ MODEL-QUALIFIED, and these MUST come BEFORE the bare *-iq4xs-moecache arms.
    # Those are suffix-matched and silently claimed GLM's slugs the moment a second
    # iq4xs-moecache model existed — comparing GLM's registry figure against
    # INKLING's header (121) and failing for a reason unrelated to GLM. Any future
    # model sharing this suffix needs its own arm here too.
    *qwen38-flash-next-dual-q4kxl-moecache)   f="$QWN_DUAL" ;;
    *qwen38-flash-next-multi4-q4kxl-moecache) f="$QWN_M4" ;;
    *qwen38-flash-next-multi8-q4kxl-moecache) f="$QWN_M8" ;;
    *glm53-flash-dual-q2k-offload)              f="$GLM_Q2K_DUAL_OFF" ;;
    *glm53-flash-multi4-q2k-offload)            f="$GLM_Q2K_M4_OFF" ;;
    *glm53-flash-multi8-q2k-offload)            f="$GLM_Q2K_M8_OFF" ;;
    *glm53-flash-dual-q3km-offload)             f="$GLM_Q3KM_DUAL_OFF" ;;
    *glm53-flash-multi4-q3km-offload)           f="$GLM_Q3KM_M4_OFF" ;;
    *glm53-flash-multi8-q3km-offload)           f="$GLM_Q3KM_M8_OFF" ;;
    *glm53-flash-dual-q2k-moecache)             f="$GLM_Q2K_DUAL" ;;
    *glm53-flash-multi4-q2k-moecache)           f="$GLM_Q2K_M4" ;;
    *glm53-flash-multi8-q2k-moecache)           f="$GLM_Q2K_M8" ;;
    *glm53-flash-dual-q3km-moecache)            f="$GLM_Q3KM_DUAL" ;;
    *glm53-flash-multi4-q3km-moecache)          f="$GLM_Q3KM_M4" ;;
    *glm53-flash-multi8-q3km-moecache)          f="$GLM_Q3KM_M8" ;;
    *glm53-flash-dual-iq3xxs-moecache)   f="$GLM3_DUAL" ;;
    *glm53-flash-multi4-iq3xxs-moecache) f="$GLM3_M4" ;;
    *glm53-flash-multi8-iq3xxs-moecache) f="$GLM3_M8" ;;
    *glm53-flash-dual-iq4xs-moecache)   f="$GLM_DUAL" ;;
    *glm53-flash-multi4-iq4xs-moecache) f="$GLM_M4" ;;
    *glm53-flash-multi8-iq4xs-moecache) f="$GLM_M8" ;;
    *dual-iq4xs-moecache)   f="$INK_CACHE" ;;
    *multi4-iq4xs-moecache) f="$INK_M4" ;;
    *dual-iq4xs-residency)  f="$INK_RES" ;;
    *dual-exl3-305-cpumoe)  f="$EXL3_305" ;;
    *dual-exl3-405-cpumoe)  f="$EXL3_405" ;;
    *qwen3.8-flash-next-ple-dual)     f="models/qwen3.8-flash-next/bucko-vllm/compose/dual/albucino-autoround-int4-fp8ple/base.yml" ;;
    qengine/qwen38-flash-next-dual) f="models/qwen3.8-flash-next/qengine/compose/dual/qengine-int4/mtp.yml" ;;   # before the glob below, which it also matches
    qengine/qwen38-flash-next-multi3) f="models/qwen3.8-flash-next/qengine/compose/multi3/qengine-int4/mtp.yml" ;;
    qengine/qwen38-flash-next-multi4) f="models/qwen3.8-flash-next/qengine/compose/multi4/qengine-int4/mtp.yml" ;;
    *qwen38-flash-next-dual) f="models/qwen3.8-flash-next/vllm/compose/dual/albucino-autoround-int4-fp8ple/mtp.yml" ;;
    *qwen3.8-flash-next-swift15-dual) f="models/qwen3.8-flash-next/bucko-vllm/compose/dual/ukisai-swift15-autoround-int4/base.yml" ;;
    *qwen3.8-flash-next-swift15-abliterated-iq3s-multi4) f="models/qwen3.8-flash-next/strata/compose/multi4/sc117-swift15-abliterated-iq3s/base.yml" ;;
    *qwen3.8-flash-next-swift15-abliterated-iq3s-dual)   f="models/qwen3.8-flash-next/strata/compose/dual/sc117-swift15-abliterated-iq3s/base.yml" ;;
    *qwen3.8-flash-next-swift15-abliterated-iq3s-single) f="models/qwen3.8-flash-next/strata/compose/single/sc117-swift15-abliterated-iq3s/base.yml" ;;
    *qwen3.8-flash-next-iq3s-dual)    f="models/qwen3.8-flash-next/strata/compose/dual/ista-iq3s/base.yml" ;;
    *qwen3.8-flash-next-iq3s-single)  f="models/qwen3.8-flash-next/strata/compose/single/ista-iq3s/base.yml" ;;
    *qwen3.8-flash-next-ud-iq4xs-dual)  f="models/qwen3.8-flash-next/strata/compose/dual/unsloth-ud-iq4xs-v31/base.yml" ;;
    *qwen3.8-flash-next-ud-iq3xxs-dual) f="models/qwen3.8-flash-next/strata/compose/dual/unsloth-ud-iq3xxs/base.yml" ;;
    *glm53-flash-dual-maya-s) f="models/glm-5.3-flash/strata-maya/compose/dual/maya-s/base.yml" ;;
    *glm53-flash-dual-maya-s24) f="models/glm-5.3-flash/strata-maya/compose/dual/maya-s24/base.yml" ;;
    *glm53-flash-single-maya-s) f="models/glm-5.3-flash/strata-maya/compose/single/maya-s/base.yml" ;;
    *glm53-flash-multi4-maya-s) f="models/glm-5.3-flash/strata-maya/compose/multi4/maya-s/base.yml" ;;
    *glm53-flash-dual-maya-m) f="models/glm-5.3-flash/strata-maya/compose/dual/maya-m/base.yml" ;;
    *glm53-flash-single-maya-m) f="models/glm-5.3-flash/strata-maya/compose/single/maya-m/base.yml" ;;
    *glm53-flash-multi4-maya-m) f="models/glm-5.3-flash/strata-maya/compose/multi4/maya-m/base.yml" ;;
    *glm53-flash-dual-maya-l) f="models/glm-5.3-flash/strata-maya/compose/dual/maya-l/base.yml" ;;
    *glm53-flash-single-maya-l) f="models/glm-5.3-flash/strata-maya/compose/single/maya-l/base.yml" ;;
    *glm53-flash-multi4-maya-l) f="models/glm-5.3-flash/strata-maya/compose/multi4/maya-l/base.yml" ;;
    *deepseek41-flash-dual-ista-iq2-xxs) f="models/deepseek-v4.1-flash-1017/strata-maya/compose/dual/ista-iq2-xxs/dspark.yml" ;;
    *deepseek41-flash-dual-ud-iq3-s) f="models/deepseek-v4.1-flash-1017/strata-maya/compose/dual/ud-iq3-s/base.yml" ;;
    *deepseek41-flash-dual-maya-xs) f="models/deepseek-v4.1-flash-1017/strata-maya/compose/dual/maya-xs/dspark.yml" ;;
    *minimax31-flash-dual-maya-s) f="models/minimax-m3.1-flash/strata-maya/compose/dual/maya-s/dspark.yml" ;;
    *minimax31-flash-single-maya-s) f="models/minimax-m3.1-flash/strata-maya/compose/single/maya-s/dspark.yml" ;;
    *dual-q8)   f="$Q8" ;;
    *dual-iq2)  f="$IQ2" ;;
    *multi4-q8) f="$M4" ;;
    *) bad "$slug has host_ram_gb but no compose mapping in this test"; continue ;;
  esac
  hdr="$(compose_meta_get "$f" cpu-offload-host-ram-gb)"
  [[ "$reg_gb" =~ ^[0-9]+$ && "$reg_gb" -le "$hdr" ]] \
    && ok "$slug: registry nominal ($reg_gb) <= header worst case ($hdr)" \
    || bad "$slug: registry host_ram_gb=$reg_gb exceeds header worst case $hdr"
done < <(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
# EVERY slug carrying host_ram_gb, not just deepseek-flash: the filter used to be
# model-scoped, so the inkling slugs sailed past this check entirely and kept a
# hand-typed 120 that nothing compared against the compose (#978).
for slug, e in COMPOSE_REGISTRY.items():
    if e.get("host_ram_gb"):
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
for f in "$Q8" "$IQ2" "$M4" "$FORK_Q8" "$FORK_M4" "$INK_CACHE" "$INK_M4" "$INK_RES"; do
  # Two acceptable shapes, both safe on an unknown rig:
  #   (a) entrypoint resolution to nproc/2  — the CURRENT convention (adapts to the host)
  #   (b) a literal `THREADS:-N` with N <= 8 — the older bare-compose fallback
  # A hardcoded high number (e.g. the reference rig's 28) is what this rejects:
  # it over-subscribes a smaller box. CLI `-t` is separately forbidden because it
  # beats the entrypoint env.
  _lbl="$(basename "$(dirname "$f")")/$(basename "$f")"
  if command grep -qE '_np / 2|nproc/2' "$f"; then
    ok "$_lbl: resolves THREADS to nproc/2 in the entrypoint (portable)"
  else
    fb="$(command grep -oE 'THREADS:-[0-9]+' "$f" | command grep -oE '[0-9]+' | head -1)"
    [[ -n "$fb" && "$fb" -le 8 ]] && ok "$_lbl: safe THREADS fallback ($fb)" \
      || bad "$_lbl: THREADS fallback '$fb' is neither nproc/2 nor <=8 — unsafe on an unknown rig"
  fi
done

# ---- wiring ----
command grep -q "preflight_cpu_offload_ram" scripts/switch.sh \
  && ok "switch.sh calls the RAM guard" || bad "switch.sh does not call the RAM guard"
command grep -q "preflight_offload_split_mode" scripts/switch.sh \
  && ok "switch.sh calls the split-mode guard" || bad "switch.sh does not call the split-mode guard"

# --- Strata's n-gram (PLE) table: left out ONLY with =0 on an SSD (#1888) ------
# The Strata Flash-Next figures were measured with the 28.8 GB table in RAM (the
# reference rig's model disk is rotational). eddie's 92 GiB SSD rig with =0 was
# refused for the UD-IQ4_XS dual although the table stays on disk there. The
# subtraction must fire on exactly the cases where Strata's setup keeps the table
# on disk — an SSD, or a filesystem with no block device behind it, which setup
# also takes its SSD branch for — and on nothing else: a rotational disk, setup's
# own SSD pin point, model files that can't be found and any other value of the
# setting all keep the table in.
for fn in _strata_gguf_dir _strata_ple_disk _strata_ple_case; do
  declare -F "$fn" >/dev/null 2>&1 && ok "$fn defined" || bad "$fn missing"
done

# Every shipped Strata Flash-Next compose declares both headers, and its /gguf
# mount resolves under MODEL_DIR (no mount found = "unknown" = the table always
# counts, which would silently undo this fix).
for f in models/qwen3.8-flash-next/strata/compose/*/*/base.yml; do
  short="$(basename "$(dirname "$(dirname "$f")")")/$(basename "$(dirname "$f")")"
  hdr="$(compose_meta_get "$f" cpu-offload-host-ram-gb || true)"
  tbl="$(compose_meta_get "$f" strata-ple-table-gb || true)"
  pin="$(compose_meta_get "$f" strata-ple-setup-pins-from-gib || true)"
  if [[ "$tbl" =~ ^[0-9]+$ && "$pin" =~ ^[0-9]+$ && "$hdr" =~ ^[0-9]+$ ]] && (( tbl > 0 && tbl < hdr )); then
    ok "$short declares Strata-PLE-Table-GB=$tbl and Strata-PLE-Setup-Pins-From-GiB=$pin"
  else
    bad "$short: Strata-PLE-Table-GB='$tbl' / Strata-PLE-Setup-Pins-From-GiB='$pin' (RAM header '$hdr')"
  fi
  d="$(MODEL_DIR=/srv/m _strata_gguf_dir "$f" || true)"
  [[ "$d" == /srv/m/?* ]] && ok "$short mounts its weights at /gguf from MODEL_DIR ($d)" \
    || bad "$short: /gguf mount did not resolve under MODEL_DIR (got '$d')"
done

# the disk reader, on real files: procfs has no block device behind it (the
# btrfs/ZFS/NFS shape — setup.py's rotational_disk() returns None there too)
[[ "$(_strata_ple_disk /proc/meminfo)" == "nodev" ]] \
  && ok "disk reader: a procfs file is 'nodev' (no queue/rotational)" \
  || bad "disk reader on /proc/meminfo: '$(_strata_ple_disk /proc/meminfo)'"
r="$(_strata_ple_disk scripts/preflight.sh)"
[[ "$r" =~ ^(rotational|ssd)\ [^\ ]+$ || "$r" == nodev ]] \
  && ok "disk reader: a real file reads as '$r'" || bad "disk reader printed '$r'"
[[ "$(_strata_ple_disk /nonexistent/x.gguf)" == "nodev" ]] \
  && ok "disk reader: a file stat can't read is 'nodev' (setup's OSError → not rotational)" \
  || bad "disk reader on a missing file"
[[ "$(_strata_ple_disk /proc/meminfo /proc/meminfo)" == nodev ]] \
  && ok "disk reader: all no-device files -> 'nodev'" \
  || bad "disk reader: two no-device files did not read as 'nodev'"
# a rotational file anywhere in the set wins (the reference rig's model disk is one)
if [[ "$r" == rotational\ * ]]; then
  [[ "$(_strata_ple_disk /proc/meminfo scripts/preflight.sh)" == rotational\ * ]] \
    && ok "disk reader: no-device + rotational files -> rotational" \
    || bad "disk reader: a rotational file did not win over a no-device one"
fi

# A Strata-shaped compose under the gate. The 999999 figure makes every leg refuse,
# so the message says which figure the gate used; the disk is stubbed per leg.
pdir="$(mktemp -d)"; mkdir -p "$pdir/w"; : > "$pdir/w/m-00001-of-00001.gguf"
trap 'rm -f "$tmp"; rm -rf "$stub" "$pdir"' EXIT
_ple_compose() {  # <setup pin point GiB>
  printf '# CPU-Offload-Host-RAM-GB: 999999\n# Strata-PLE-Table-GB: 28\n# Strata-PLE-Setup-Pins-From-GiB: %s\nservices:\n  strata:\n    volumes:\n      - "${MODEL_DIR:-../m}/w:/gguf:ro"\n' "$1" > "$tmp"
}
_ple_gate() {  # <disk stub output> <STRATA_PLE_AUTO_RAM or "-" for unset>
  # (not named `disk`: _strata_ple_case declares a local of that name, which the
  #  stub would then read under dynamic scoping)
  local _ple_stub_disk="$1" auto="$2"
  ( _strata_ple_disk() { echo "$_ple_stub_disk"; }
    export MODEL_DIR="$pdir"
    if [[ "$auto" == "-" ]]; then unset STRATA_PLE_AUTO_RAM; else export STRATA_PLE_AUTO_RAM="$auto"; fi
    preflight_cpu_offload_ram "$tmp" 2>&1 )
}
_ple_compose 999999        # below setup's SSD pin point on any box

msg="$(_ple_gate "ssd nvme0n1" 0)"
command grep -q "needs ~999971 GB" <<<"$msg" && command grep -q "stays on disk" <<<"$msg" \
  && ok "=0 on an SSD: the 28 GB table is left out (999999 -> 999971)" \
  || bad "=0 on an SSD did not leave the table out: $msg"
msg="$(_ple_gate "rotational sdb" 0)"
command grep -q "needs ~999999 GB" <<<"$msg" && command grep -q "rotational disk (sdb)" <<<"$msg" \
  && ok "=0 on a rotational disk: the table still counts, and the gate says why" \
  || bad "rotational disk was priced without the table: $msg"
msg="$(_ple_gate "nodev" 0)"
command grep -q "needs ~999971 GB" <<<"$msg" && command grep -q "no block device" <<<"$msg" \
  && ok "=0 on a filesystem with no block device (btrfs/ZFS): left out, as setup's SSD branch" \
  || bad "=0 on a no-device filesystem kept the table (setup takes its SSD branch there): $msg"
for v in - 1 "" auto; do
  msg="$(_ple_gate "ssd nvme0n1" "$v")"
  command grep -q "needs ~999999 GB" <<<"$msg" \
    && ok "STRATA_PLE_AUTO_RAM='${v/-/<unset>}' on an SSD: the table still counts" \
    || bad "STRATA_PLE_AUTO_RAM='${v/-/<unset>}' left the table out: $msg"
done
# the refusal names the setting when it would have helped — eddie had to ask
msg="$(_ple_gate "ssd nvme0n1" -)"
command grep -q "STRATA_PLE_AUTO_RAM=0 keeps the ~28 GB n-gram table on disk and lowers this to ~999971 GB" <<<"$msg" \
  && ok "refusal on an SSD without =0 names STRATA_PLE_AUTO_RAM=0 and the lower figure" \
  || bad "refusal on an SSD does not point at STRATA_PLE_AUTO_RAM=0: $msg"
msg="$(_ple_gate "nodev" -)"
command grep -q "On this filesystem, STRATA_PLE_AUTO_RAM=0 keeps" <<<"$msg" \
  && ok "refusal on a no-device filesystem without =0 names the setting" \
  || bad "refusal on a no-device filesystem does not point at STRATA_PLE_AUTO_RAM=0: $msg"
msg="$(_ple_gate "rotational sdb" -)"
command grep -q "STRATA_PLE_AUTO_RAM=0" <<<"$msg" \
  && bad "refusal on a rotational disk suggests =0, which cannot help there: $msg" \
  || ok "refusal on a rotational disk does not suggest =0"

# setup's own SSD rule: from the pin point it writes --ple-io ram, which the
# engine keeps — so =0 cannot keep the table out there
_ple_compose 1             # every box is past a 1 GiB pin point
msg="$(_ple_gate "ssd nvme0n1" 0)"
command grep -q "needs ~999999 GB" <<<"$msg" && command grep -q "from 1 GiB of RAM" <<<"$msg" \
  && ok "=0 on an SSD past setup's pin point: the table still counts" \
  || bad "box past setup's SSD pin point was priced without the table: $msg"
msg="$(_ple_gate "nodev" 0)"
command grep -q "needs ~999999 GB" <<<"$msg" && command grep -q "from 1 GiB of RAM" <<<"$msg" \
  && ok "=0 on a no-device filesystem past setup's pin point: the table still counts" \
  || bad "no-device filesystem past setup's pin point was priced without the table: $msg"
# the boundary, in setup.py's units (MemTotal GiB >= the rounded-down point = pinned)
_ple_compose 92
( _strata_ple_disk() { echo "ssd nvme0n1"; }
  export MODEL_DIR="$pdir"
  [[ "$(_strata_ple_case "$tmp" $(( 91 * 1048576 + 943718 )))" == "ssd nvme0n1" ]] \
    && [[ "$(_strata_ple_case "$tmp" $(( 92 * 1048576 )))" == "setup-pins 92" ]] ) \
  && ok "pin point: 91.9 GiB (eddie's rig) is below 92, 92.0 GiB is at it" \
  || bad "the setup pin-point boundary moved"
# model files that can't be found: the table still counts
_ple_compose 999999
( rm -f "$pdir/w/"*.gguf
  msg="$(_ple_gate "ssd nvme0n1" 0)"
  : > "$pdir/w/m-00001-of-00001.gguf"
  command grep -q "needs ~999999 GB" <<<"$msg" && command grep -q "could not find the model's" <<<"$msg" ) \
  && ok "=0 with no model files in the /gguf folder: the table still counts" \
  || bad "a /gguf folder without model files lost the table"
# no /gguf mount, or no headers: unknown, so nothing changes
printf '# CPU-Offload-Host-RAM-GB: 999999\n# Strata-PLE-Table-GB: 28\n# Strata-PLE-Setup-Pins-From-GiB: 999999\nservices:\n  x:\n    command: >-\n      -ot a=CPU\n' > "$tmp"
msg="$(_ple_gate "ssd nvme0n1" 0)"
command grep -q "needs ~999999 GB" <<<"$msg" \
  && ok "no /gguf mount: the table still counts" || bad "a compose without a /gguf mount lost the table: $msg"
printf '# CPU-Offload-Host-RAM-GB: 999999\nservices:\n  strata:\n    volumes:\n      - "${MODEL_DIR:-../m}/w:/gguf:ro"\n' > "$tmp"
msg="$(_ple_gate "ssd nvme0n1" 0)"
command grep -q "needs ~999999 GB" <<<"$msg" && ! command grep -q "n-gram" <<<"$msg" \
  && ok "no Strata-PLE headers: the gate is unchanged and silent about the table" \
  || bad "a compose without the Strata headers was touched: $msg"

# --- unit contract: the -GB key is compared in DECIMAL GB --------------------
# The header key is `CPU-Offload-Host-RAM-GB` and every value authored against it
# is decimal GB. preflight_cpu_offload_ram() once computed `kb/1024/1024` (GiB)
# and compared THAT against it, so the gate ran ~7% stricter than its own
# documented contract on every offload slug — silent, surfacing only as a refusal
# on a rig that should have passed.
#
# ⚠️ Scoped to THIS function on purpose: preflight.sh has other `/1024/1024`
# sites (general RAM reporting) that are not the -GB path and must not be
# dragged along by a file-wide grep — an earlier version of this assertion
# matched one of those and reported a fixed file as broken.
_fn="$(awk '/^preflight_cpu_offload_ram\(\)/{f=1} f{print} f&&/^}/{exit}' scripts/preflight.sh)"
if command grep -q 'kb \* 1024 / 1000000000' <<<"$_fn"; then
  ok "preflight_cpu_offload_ram sizes host RAM in DECIMAL GB, matching the -GB key"
else
  bad "preflight_cpu_offload_ram host-RAM unit" "decimal GB (kb * 1024 / 1000000000)" \
      "$(command grep -oE 'kb [*/] 1024 [*/] [0-9]+' <<<"$_fn" | head -1)"
fi
# ⚠️ Strip comments first: the fix's own comment QUOTES the old expression to
# explain what changed, and a naive grep scores that as the bug still being
# present — a gate failing on its own documentation.
if command grep -vE '^\s*#' <<<"$_fn" | command grep -q 'kb / 1024 / 1024'; then
  bad "preflight_cpu_offload_ram still has a GiB expression in CODE" "none" "kb / 1024 / 1024"
else
  ok "no GiB expression remains in the -GB comparison path (comments excluded)"
fi

[[ $fail -eq 0 ]] && echo "test-preflight-cpu-offload: ok" || echo "test-preflight-cpu-offload: FAIL"
exit $fail
