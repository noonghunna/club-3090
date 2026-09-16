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

cat > "$MOCK_DIR/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
n=2; args="$*"
case "$args" in
  "-L") for ((i=0;i<n;i++)); do echo "GPU $i: NVIDIA GeForce RTX 3090 (UUID: GPU-$i)"; done ;;
  "topo -m")
    printf "\tGPU0\tGPU1\n"
    printf "GPU0\tX\tPHB\n"; printf "GPU1\tPHB\tX\n" ;;
  "topo -p2p r")
    printf "\tGPU0\tGPU1\n"
    printf "GPU0\tX\tOK\n"; printf "GPU1\tOK\tX\n" ;;
  "-q -d MEMORY")
    echo "Attached GPUs                       : 2"
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
command grep -q '^custom_ar_enabled=1$'   <<<"$out" || fail "default: _CUSTOM_AR_ENABLED must track _NVLINK_ENABLED: $out"
command grep -q '^p2p_level=PHB$'         <<<"$out" || fail "default: transport must be up: $out"
command grep -q 'custom all-reduce ON'    <<<"$out" || fail "default: trail must still claim AR ON: $out"
echo "  ✓ knob absent: unchanged — transport up, custom kernel on"

# ── 2. knob set — kernel OFF, transport STILL UP (the whole point)
out="$(resolve 1)"
command grep -q '^ar=--disable-custom-all-reduce$' <<<"$out" || fail "knob=1 must pass the flag: $out"
command grep -q '^custom_ar_enabled=0$'            <<<"$out" || fail "knob=1: _CUSTOM_AR_ENABLED must be 0: $out"
command grep -q '^nvlink_enabled=1$'               <<<"$out" || fail "knob=1 must NOT disable the transport: $out"
command grep -q '^p2p_level=PHB$'                  <<<"$out" || fail "knob=1: NCCL_P2P_LEVEL must survive: $out"
command grep -q '^p2p_disable=<unset>$'            <<<"$out" || fail "knob=1 must not set NCCL_P2P_DISABLE: $out"
echo "  ✓ knob set: custom kernel OFF, peer transport still UP"

# ── 3. the boot trail must not assert the opposite of what is running
command grep -q 'custom all-reduce ON' <<<"$out" \
  && fail "knob=1: trail still claims 'custom all-reduce ON' — this is the #922/#924 false-verdict substring: $out"
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/lib/p2p-state.sh"
r="$(printf '%s\n' "${out#*trail=}" | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "knob=1 trail must classify nccl_only, got '$r'"
echo "  ✓ boot trail classifies nccl_only, never a false custom-AR-ON"

# ── 4. a typo must HARD-ERROR, never silently leave the kernel on (#1332)
if PATH="$MOCK_DIR:$PATH" DISABLE_CUSTOM_ALL_REDUCE=true NVLINK_MODE=auto \
   bash -c 'source "'"$DETECT"'"' >/dev/null 2>&1; then
  fail "DISABLE_CUSTOM_ALL_REDUCE=true must be rejected, not silently ignored"
fi
echo "  ✓ invalid value hard-errors (a typo cannot silently re-enable the kernel)"

# ── 5. every shipped compose must gate on the SPLIT variable and declare the var
# Scoped to compose trees on purpose: a bare -r over models/ walks the
# torch_compile cache, where root-owned files make grep noisy and, worse, make a
# silent partial scan look like a clean one.
mapfile -t _composes < <(
  command grep -rl 'AR="--disable-custom-all-reduce"' "${ROOT_DIR}"/models/*/vllm/compose 2>/dev/null | sort
)
[ ${#_composes[@]} -gt 0 ] || fail "found NO composes with the AR gate — the scan is broken, not the repo clean"
missing_gate=(); missing_env=()
for f in "${_composes[@]}"; do
  command grep -q '_CUSTOM_AR_ENABLED' "$f"             || missing_gate+=("$f")
  command grep -q 'DISABLE_CUSTOM_ALL_REDUCE=\${' "$f"  || missing_env+=("$f")
done
[ ${#missing_gate[@]} -eq 0 ] || fail "composes still gate AR on _NVLINK_ENABLED only: ${missing_gate[*]}"
# Declaring it in environment: is the DELIVERY path — docker forwards nothing else.
[ ${#missing_env[@]} -eq 0 ] || fail "composes do not declare DISABLE_CUSTOM_ALL_REDUCE (knob would be inert): ${missing_env[*]}"
echo "  ✓ all ${#_composes[@]} composes gate on the split var and declare it"

echo "test-custom-ar-knob: ok"
