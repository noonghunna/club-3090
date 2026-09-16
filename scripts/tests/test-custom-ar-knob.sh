#!/usr/bin/env bash
# DISABLE_CUSTOM_ALL_REDUCE=1 must keep the peer TRANSPORT up while turning
# vLLM's custom all-reduce kernel off — the #922 mitigation, which until #1332
# was only reachable by editing a shipped compose (NVLINK_MODE=force_off drops
# the transport too, and with it the prefill half of the win).
#
# The two decisions used to be fused into _NVLINK_ENABLED. This asserts the
# split end-to-end: detect_nvlink.sh's export, the compose's AR gate evaluated
# as the entrypoint evaluates it, the transport env, AND the round-trip through
# p2p_classify_engagement (a boot trail that still claimed "custom all-reduce ON"
# here would re-introduce the #922/#924 false verdict).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DETECT="${ROOT_DIR}/scripts/detect_nvlink.sh"
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT

# ⚠️ The link type and P2P matrix MUST honour FAKE_LINK/FAKE_P2P. An earlier cut
# hardcoded PHB/OK, so the "must not warn on NVLink" case was silently still
# testing PCIe and the assertion failed against correct code. A mock that
# ignores its own parameters tests one scenario N times.
cat > "$MOCK_DIR/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
n="${FAKE_GPUS:-2}"; args="$*"
case "$args" in
  "-L") for ((i=0;i<n;i++)); do echo "GPU $i: NVIDIA GeForce RTX 3090 (UUID: GPU-$i)"; done ;;
  "topo -m")
    printf "\t"; for ((j=0;j<n;j++)); do printf "GPU%d\t" "$j"; done; printf "\n"
    for ((i=0;i<n;i++)); do printf "GPU%d\t" "$i"
      for ((j=0;j<n;j++)); do [ "$i" = "$j" ] && printf "X\t" || printf "%s\t" "${FAKE_LINK:-PHB}"; done
      printf "\n"; done ;;
  "topo -p2p r")
    printf "\t"; for ((j=0;j<n;j++)); do printf "GPU%d\t" "$j"; done; printf "\n"
    for ((i=0;i<n;i++)); do printf "GPU%d\t" "$i"
      for ((j=0;j<n;j++)); do [ "$i" = "$j" ] && printf "X\t" || printf "%s\t" "${FAKE_P2P:-OK}"; done
      printf "\n"; done ;;
  "-q -d MEMORY")
    echo "Attached GPUs                       : $n"
    for ((i=0;i<n;i++)); do
      printf 'GPU 00000000:0%d:00.0\n' "$i"
      echo "    FB Memory Usage"; echo "        Total                      : 24576 MiB"
      echo "    BAR1 Memory Usage"; echo "        Total                      : 32768 MiB"
    done ;;
esac
MOCK
chmod +x "$MOCK_DIR/nvidia-smi"

fail() { echo "FAIL: $*" >&2; exit 1; }

# Run detect_nvlink under the mock and report the resolved state as one line.
# The AR expression is copied VERBATIM from the shipped composes (with compose's
# `$$` un-escaped) so this test fails if the two ever drift apart.
resolve() {
  # No opt-in marker: detect_nvlink.sh aliases _NVLINK_ENABLED to the kernel
  # decision, so every gate shape honours the knob by construction (§6).
  PATH="$MOCK_DIR:$PATH" DISABLE_CUSTOM_ALL_REDUCE="$1" NVLINK_MODE=auto \
  bash -c '
    trail="$(source '"$DETECT"' 2>&1)"
    source '"$DETECT"' >/dev/null 2>&1
    AR=""
    [ "${_CUSTOM_AR_ENABLED:-${_NVLINK_ENABLED:-0}}" = "1" ] || AR="--disable-custom-all-reduce"
    echo "ar=${AR:-<none>}"
    echo "custom_ar_enabled=${_CUSTOM_AR_ENABLED:-<unset>}"
    echo "nvlink_enabled=${_NVLINK_ENABLED:-<unset>}"
    echo "p2p_level=${NCCL_P2P_LEVEL:-<unset>}"
    echo "p2p_disable=${NCCL_P2P_DISABLE:-<unset>}"
    echo "trail=$trail"
  '
}

# ── 1. default (knob absent) — behaviour must be UNCHANGED: transport + kernel on
out="$(resolve 0)"
command grep -q '^ar=<none>$'             <<<"$out" || fail "default must NOT pass --disable-custom-all-reduce: $out"
command grep -q '^custom_ar_enabled=1$'   <<<"$out" || fail "default: the kernel must stay on when the knob is unset: $out"
command grep -q '^nvlink_enabled=1$'     <<<"$out" || fail "default: the aliased name must read 1 so legacy gates keep the kernel on: $out"
command grep -q '^p2p_level=PHB$'         <<<"$out" || fail "default: transport must be up: $out"
command grep -q 'custom all-reduce ON'    <<<"$out" || fail "default: trail must still claim AR ON: $out"
echo "  ✓ knob absent: unchanged — transport up, custom kernel on"

# ── 2. knob set — kernel OFF, transport STILL UP (the whole point)
out="$(resolve 1)"
command grep -q '^ar=--disable-custom-all-reduce$' <<<"$out" || fail "knob=1 must pass the flag: $out"
command grep -q '^custom_ar_enabled=0$'            <<<"$out" || fail "knob=1: _CUSTOM_AR_ENABLED must be 0: $out"
# ⚠️ NOT via _NVLINK_ENABLED: that name now carries the KERNEL decision, so it is
# correctly 0 here. Conflating the two is the bug this redesign removes. The
# transport's evidence is the NCCL env the script exports.
command grep -q '^nvlink_enabled=0$'               <<<"$out" || fail "knob=1: the aliased _NVLINK_ENABLED must read 0 so every legacy gate passes the flag: $out"
command grep -q '^p2p_level=PHB$'                  <<<"$out" || fail "knob=1: NCCL_P2P_LEVEL must survive: $out"
command grep -q '^p2p_disable=<unset>$'            <<<"$out" || fail "knob=1 must not set NCCL_P2P_DISABLE: $out"
echo "  ✓ knob set: custom kernel OFF, peer transport still UP"

# ── 3. the boot trail must not assert the opposite of what is running
command grep -q 'custom all-reduce ON' <<<"$out" \
  && fail "knob=1: trail still claims 'custom all-reduce ON' — this is the #922/#924 false-verdict substring: $out"
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/lib/p2p-state.sh"
r="$(printf '%s\n' "${out#*trail=}" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_operator" ]] || fail "knob=1 trail must classify nccl_only_operator (the OPERATOR cause, distinct from the engine's own vetoes), got '$r'"
echo "  ✓ boot trail classifies nccl_only, never a false custom-AR-ON"

# ── 4. a typo must HARD-ERROR, never silently leave the kernel on (#1332)
_bad="$(PATH="$MOCK_DIR:$PATH" DISABLE_CUSTOM_ALL_REDUCE=true NVLINK_MODE=auto \
        bash -c 'source "'"$DETECT"'"' 2>&1)" && fail "DISABLE_CUSTOM_ALL_REDUCE=true must be rejected, not silently ignored"
# A non-zero exit alone is satisfied by a MISSING script, so pin the message too.
command grep -q "invalid DISABLE_CUSTOM_ALL_REDUCE" <<<"$_bad" \
  || fail "rejection must name the variable; got: $_bad"
echo "  ✓ invalid value hard-errors (a typo cannot silently re-enable the kernel)"

# ── 5. EVERY compose that sources detect_nvlink.sh honours the split ─────────
# ⚠️ The scan set is "composes that source detect_nvlink.sh" — i.e. every compose
# that can have the kernel auto-enabled — NOT "composes containing the fix's own
# string". The first cut of this test used the latter, so the 33 composes the fix
# had NOT rewired were invisible to it and it reported full coverage over 12 of
# 45 (#1332 review). A scan set defined by the patch can only ever confirm the
# patch.
mapfile -t _composes < <(command grep -rl 'detect_nvlink.sh' "${ROOT_DIR}"/models/*/*/compose/*/*/*.yml 2>/dev/null | sort)
[ ${#_composes[@]} -ge 40 ] || fail "expected 40+ composes sourcing detect_nvlink.sh, found ${#_composes[@]} — the scan is broken, not the repo clean"

missing_env=()
for f in "${_composes[@]}"; do
  command grep -qE '^[[:space:]]*- DISABLE_CUSTOM_ALL_REDUCE=\$\{' "$f" || missing_env+=("$f")
done
[ ${#missing_env[@]} -eq 0 ] || fail "DISABLE_CUSTOM_ALL_REDUCE not declared (knob cannot arrive) in: ${missing_env[*]}"
echo "  ✓ all ${#_composes[@]} composes declare the knob so it reaches the container"

# ── 6. ⭐ the knob works on EVERY gate shape, including ones we do not ship ────
# detect_nvlink.sh aliases _NVLINK_ENABLED to the KERNEL decision, so any
# entrypoint that has ever gated the vLLM flag on it honours the knob by
# construction — the shipped shapes, the _archive composes, forks, hand-edits.
# This replaced an opt-in marker plus a boot refusal, which could only cover
# files we had edited and fired on host-side probes besides.
# Two shipped polarities. The first two mean "condition true => kernel ON"; the
# NVLink-requiring shape means the opposite (true => add the disable flag), so
# its expectations are inverted. Testing it with the wrong polarity would have
# it "fail" against correct code.
_gate_probe() {  # _gate_probe <link> <knob> <shape> <polarity:on|off>
  PATH="$MOCK_DIR:$PATH" FAKE_LINK="$1" FAKE_P2P=OK DISABLE_CUSTOM_ALL_REDUCE="$2" NVLINK_MODE=auto \
  bash -c 'source "'"$DETECT"'" >/dev/null 2>&1
    if '"$3"'; then [ "'"$4"'" = on ] && echo KERNEL_ON || echo KERNEL_OFF
    else            [ "'"$4"'" = on ] && echo KERNEL_OFF || echo KERNEL_ON; fi'
}
# PCIe peer path: the two straightforward shapes must follow the knob.
for _shape in '[ "${_NVLINK_ENABLED:-0}" = "1" ]' '[ "${_CUSTOM_AR_ENABLED:-${_NVLINK_ENABLED:-0}}" = "1" ]'; do
  [ "$(_gate_probe PHB 0 "$_shape" on)" = KERNEL_ON  ] || fail "shape <$_shape> knob=0 must keep the kernel on"
  [ "$(_gate_probe PHB 1 "$_shape" on)" = KERNEL_OFF ] || fail "shape <$_shape> knob=1 must turn the kernel off"
done
# The NVLink-requiring shape: always off on PCIe (by its own design), and on a
# real NVLink rig it must still follow the knob.
_NVL_SHAPE='[ "${_NVLINK_ENABLED:-0}" != 1 ] || [ "${NCCL_P2P_LEVEL:-}" != NVL ]'
[ "$(_gate_probe PHB 0 "$_NVL_SHAPE" off)" = KERNEL_OFF ] || fail "NVLink-requiring shape must stay off on a PCIe peer path"
[ "$(_gate_probe NV4 0 "$_NVL_SHAPE" off)" = KERNEL_ON  ] || fail "NVLink-requiring shape must run the kernel on real NVLink"
[ "$(_gate_probe NV4 1 "$_NVL_SHAPE" off)" = KERNEL_OFF ] || fail "NVLink-requiring shape must honour the knob on NVLink too"
unset _shape _NVL_SHAPE
echo "  ✓ the knob is honoured by every gate shape (legacy, split, NVLink-requiring) — by construction, not opt-in"

# ── 7. the risky-shape boot WARNING: fires where it should, silent elsewhere ──
# It exists because the knob is useless to someone who only learns it exists by
# crashing (#922, #1332). Scope: kernel actually running, over the PCIe peer
# path, at <=2 GPUs. NOT native NVLink, NOT when the operator already opted out.
# ⚠️ Passes a realistic entrypoint argv. detect_nvlink.sh reads the TP width from
# "$@" (a sourced script inherits it from `bash -c <script> -- <command...>`,
# which is how all 45 composes run) — not from the GPU count, because a
# single-card slug with `count: all` sees every card on the host. Omitting argv
# here means TP defaults to 1 and the warning correctly does not fire.
warn_out() {  # warn_out <FAKE_LINK> <FAKE_P2P> <knob> [gpus] [tp]
  PATH="$MOCK_DIR:$PATH" FAKE_LINK="$1" FAKE_P2P="$2" FAKE_GPUS="${4:-2}" \
  DISABLE_CUSTOM_ALL_REDUCE="$3" NVLINK_MODE=auto \
  bash -c 'source "'"$DETECT"'"' -- --model /x --tensor-parallel-size "${5:-2}" 2>&1
}
command grep -q 'NOTE: this rig is on the patched PCIe peer path' <<<"$(warn_out PHB OK 0)" \
  || fail "risky shape (2-GPU patched PCIe P2P, kernel on) must warn"
command grep -q 'DISABLE_CUSTOM_ALL_REDUCE=1' <<<"$(warn_out PHB OK 0)" \
  || fail "the warning must name the one-line mitigation"
command grep -q 'NOTE: this rig is on the patched PCIe peer path' <<<"$(warn_out NV4 OK 0)" \
  && fail "native NVLink is a different path with no reports — must NOT warn"
command grep -q 'NOTE: this rig is on the patched PCIe peer path' <<<"$(warn_out PHB OK 1)" \
  && fail "operator already opted out — must NOT warn"
command grep -q 'NOTE: this rig is on the patched PCIe peer path' <<<"$(warn_out PHB CNS 0)" \
  && fail "no peer access, kernel not on the peer path — must NOT warn"
# ⭐ R4: a single-card slug on a 2-GPU host. `count: all` makes the container see
# both cards, but TP=1 means no all-reduce runs at all — warning about a
# hard reset there is a false alarm.
command grep -q 'NOTE: this rig is on the patched PCIe peer path' <<<"$(warn_out PHB OK 0 2 1)" \
  && fail "TP=1 runs no all-reduce — must NOT warn even with 2 GPUs visible"
echo "  ✓ risky-shape warning: fires on patched PCIe P2P with the kernel live, silent on NVLink / opted-out / no-P2P"

# ── 8. ⚠️⚠️ the warning must not change what the CLASSIFIER reports ───────────
# The classifier keys on English phrases in this same log, so any line we add is
# a potential collision. The first cut of the warning ended "vs turning P2P off
# entirely" and `*"P2P off"*` is the transport-off key: a rig whose peer path had
# FAILED vLLM's own test was reclassified from nccl_only_degraded to off, i.e.
# told its transport was disabled and to set NVLINK_MODE=force_off. Wrong
# diagnosis, wrong remedy, on the rig shape the warning exists for.
#
# ⚠️ The previous version of this section tested the warning ALONE and passed.
# The collision only appears when the warning is COMBINED with an engine line,
# which is the only way it ever occurs in a real log. So: assert the warning is
# INVARIANT — for every engine-log case, classification must be identical with
# and without it. A phrase list in a comment cannot do this; it already failed to.
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/lib/p2p-state.sh"

VLLM_P2P_FAIL='Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.'
VLLM_W2_VETO='Custom allreduce is disabled because it'"'"'s not supported on more than two PCIe-only GPUs. To silence this warning, specify disable_custom_all_reduce=True explicitly.'
VLLM_NOLIB='Custom allreduce is disabled because of missing custom allreduce library'
VLLM_OPERATOR='disable_custom_all_reduce=True'

# ⚠️ Capture the WHOLE warning block. The first cut grepped a tag that only the
# first line carried, so the second line — which held the offending phrase — was
# never fed to the classifier and this whole section passed against the very
# regression it exists to catch. Both lines now carry the tag; assert the count
# so a reworded warning cannot silently shrink the capture again.
_warn_lines="$(warn_out PHB OK 0 | command grep -F '[nvlink] ⚠️' || true)"
_warn_n="$(command grep -c . <<<"$_warn_lines" || echo 0)"
[ "${_warn_n:-0}" -ge 2 ] || fail "captured ${_warn_n} warning line(s), expected the full block (>=2) — the invariance check below would be vacuous"

for _case in "P2P_FAIL:$VLLM_P2P_FAIL" "W2_VETO:$VLLM_W2_VETO" "NOLIB:$VLLM_NOLIB" "OPERATOR:$VLLM_OPERATOR" "NONE:"; do
  _name="${_case%%:*}"; _engine="${_case#*:}"
  _without="$(printf '%s\n%s' '[nvlink] P2P ENABLED — custom all-reduce ON' "$_engine" | p2p_classify_engagement)"
  _with="$(printf '%s\n%s\n%s' '[nvlink] P2P ENABLED — custom all-reduce ON' "$_warn_lines" "$_engine" | p2p_classify_engagement)"
  [[ "$_without" == "$_with" ]] || fail "the boot warning CHANGED classification for case $_name: without='$_without' with='$_with' — the warning text contains a phrase the classifier keys on"
done
unset _case _name _engine _without _with
echo "  ✓ the warning is classification-INVARIANT across every engine-log case (incl. P2P-test-failed)"

# And the states themselves must still be right on a warning-bearing boot.
r="$(warn_out PHB OK 0 | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "warning-bearing boot with the kernel live must classify 'on', got '$r'"
r="$(warn_out PHB OK 1 | p2p_classify_engagement)"
[[ "$r" == "nccl_only_operator" ]] || fail "opted-out boot must classify nccl_only_operator, got '$r'"
echo "  ✓ warning-bearing boots still classify correctly (on / nccl_only_operator)"

echo "test-custom-ar-knob: ok"
