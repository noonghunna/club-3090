#!/usr/bin/env bash
# detect_nvlink.sh `auto` mode must auto-enable PCIe P2P when nvidia-smi reports
# P2P=OK between all pairs (a patched consumer-GPU driver on a P2P-capable layout),
# and MUST stay off when P2P is unavailable (stock driver → CNS, or cards on
# separate root complexes). NVLink detection still wins when present.
#
# nvidia-smi is mocked (no real GPUs); FAKE_GPUS / FAKE_LINK / FAKE_P2P drive it.
set -euo pipefail

# ⚠️ Pin the knob, like NVLINK_MODE. Nothing else scrubs it, so an operator with
# DISABLE_CUSTOM_ALL_REDUCE exported in their shell — which our own boot warning
# tells them to do — would red this suite for reasons unrelated to the code
# under test (#1332 review).
export DISABLE_CUSTOM_ALL_REDUCE=0


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DETECT="${ROOT_DIR}/scripts/detect_nvlink.sh"
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT

cat > "$MOCK_DIR/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
n="${FAKE_GPUS:-2}"; args="$*"
case "$args" in
  "-L")
    for ((i=0;i<n;i++)); do echo "GPU $i: NVIDIA GeForce RTX 3090 (UUID: GPU-$i)"; done ;;
  "topo -m")
    printf "\t"; for ((j=0;j<n;j++)); do printf "GPU%d\t" "$j"; done; printf "\n"
    for ((i=0;i<n;i++)); do printf "GPU%d\t" "$i"
      for ((j=0;j<n;j++)); do [ "$i" = "$j" ] && printf "X\t" || printf "%s\t" "${FAKE_LINK:-PHB}"; done
      printf "\n"; done ;;
  "topo -p2p r")
    printf "\t"; for ((j=0;j<n;j++)); do printf "GPU%d\t" "$j"; done; printf "\n"
    for ((i=0;i<n;i++)); do printf "GPU%d\t" "$i"
      for ((j=0;j<n;j++)); do
        if [ "$i" = "$j" ]; then printf "X\t"
        elif [ "${FAKE_P2P:-CNS}" = "MIXED" ] && [ "$i" = 0 ] && [ "$j" = 1 ]; then printf "CNS\t"
        else printf "%s\t" "${FAKE_P2P:-CNS}"; fi
      done; printf "\n"; done ;;
  "-q -d MEMORY")
    # FAKE_BAR1: MiB per card (unset => omit the whole query, i.e. a driver that
    # doesn't report it — the fail-open case). VRAM is fixed at 24576 MiB.
    [ -n "${FAKE_BAR1:-}" ] || exit 0
    echo "Attached GPUs                       : $n"
    for ((i=0;i<n;i++)); do
      printf 'GPU 00000000:0%d:00.0\n' "$i"
      echo "    FB Memory Usage"
      echo "        Total                      : 24576 MiB"
      echo "        Used                       : 1 MiB"
      echo "    BAR1 Memory Usage"
      echo "        Total                      : ${FAKE_BAR1} MiB"
      echo "        Used                       : 4 MiB"
      echo "    Conf Compute Protected Memory Usage"
      echo "        Total                      : 0 MiB"
    done ;;
esac
exit 0
MOCK
chmod +x "$MOCK_DIR/nvidia-smi"

fail=0
# $1=gpus $2=link $3=p2p -> "DISABLE=<>|LEVEL=<>"
run() {
  PATH="$MOCK_DIR:$PATH" FAKE_GPUS="$1" FAKE_LINK="$2" FAKE_P2P="$3" NVLINK_MODE=auto \
    bash -c "source '$DETECT' >/dev/null 2>&1; printf 'DISABLE=%s|LEVEL=%s' \"\${NCCL_P2P_DISABLE:-}\" \"\${NCCL_P2P_LEVEL:-}\""
}
check() {
  local desc="$1" gpus="$2" link="$3" p2p="$4" want="$5" got
  got="$(run "$gpus" "$link" "$p2p")"
  if [ "$got" != "$want" ]; then echo "[autop2p] FAIL: $desc — got '$got' want '$want'" >&2; fail=1
  else echo "[autop2p] ok: $desc"; fi
}

# 2-GPU, no NVLink, P2P=OK (patched driver, shared root) -> AUTO-ENABLE PCIe P2P
check "2gpu PHB + P2P OK -> enabled@PHB" 2 PHB OK  "DISABLE=|LEVEL=PHB"
# 2-GPU, no NVLink, P2P=CNS (this rig: stock driver / separate root complex) -> OFF
check "2gpu PHB + P2P CNS -> off"        2 PHB CNS "DISABLE=1|LEVEL="
# 2-GPU NVLink present -> NVLink wins (NVL), regardless of P2P matrix
check "2gpu NVLink -> enabled@NVL"       2 NV4 CNS "DISABLE=|LEVEL=NVL"
# >2 GPUs, no NVLink, all pairs OK -> auto-enable
check "4gpu PHB + all P2P OK -> enabled" 4 PHB OK  "DISABLE=|LEVEL=PHB"
# >2 GPUs, one pair NOT OK -> stay off (custom all-reduce needs full P2P)
check "4gpu PHB + mixed P2P -> off"      4 PHB MIXED "DISABLE=1|LEVEL="
# >2 GPUs, no P2P at all -> off
check "4gpu PHB + no P2P -> off"         4 PHB CNS "DISABLE=1|LEVEL="

# ── pcie_p2p FORCE mode: config forced on either way, message must be HONEST ──
# The #688 false-"engaged" fix: forcing NVLINK_MODE=pcie_p2p applies the NCCL /
# all-reduce config regardless (escape hatch), but the boot line only says
# "ENABLED" when topo -p2p confirms; "REQUESTED (UNVERIFIED)" when it doesn't.
pcie_text() {  # $1=gpus $2=p2p(OK/CNS) -> combined [nvlink] stdout+stderr
  PATH="$MOCK_DIR:$PATH" FAKE_GPUS="$1" FAKE_LINK=PHB FAKE_P2P="$2" NVLINK_MODE=pcie_p2p \
    bash -c "source '$DETECT' 2>&1"
}
pcie_env() {   # $1=gpus $2=p2p -> resolved NCCL env (config must be forced ON both ways)
  PATH="$MOCK_DIR:$PATH" FAKE_GPUS="$1" FAKE_LINK=PHB FAKE_P2P="$2" NVLINK_MODE=pcie_p2p \
    bash -c "source '$DETECT' >/dev/null 2>&1; printf 'DISABLE=%s|LEVEL=%s' \"\${NCCL_P2P_DISABLE:-}\" \"\${NCCL_P2P_LEVEL:-}\""
}
ptext_ok="$(pcie_text 2 OK)"
case "$ptext_ok" in
  *"P2P ENABLED"*) [[ "$ptext_ok" == *UNVERIFIED* ]] \
      && { echo "[autop2p] FAIL: pcie_p2p + topo OK must not also say UNVERIFIED" >&2; fail=1; } \
      || echo "[autop2p] ok: pcie_p2p + topo OK -> ENABLED (verified)" ;;
  *) echo "[autop2p] FAIL: pcie_p2p + topo OK should say ENABLED — got: $ptext_ok" >&2; fail=1 ;;
esac
ptext_cns="$(pcie_text 2 CNS)"
case "$ptext_cns" in
  *"P2P ENABLED"*) echo "[autop2p] FAIL: pcie_p2p + topo CNS must NOT falsely say ENABLED (#688 bug)" >&2; fail=1 ;;
  *"P2P REQUESTED (UNVERIFIED)"*) echo "[autop2p] ok: pcie_p2p + topo CNS -> UNVERIFIED (honest)" ;;
  *) echo "[autop2p] FAIL: pcie_p2p + topo CNS should say UNVERIFIED — got: $ptext_cns" >&2; fail=1 ;;
esac
# config forced ON in BOTH cases — the escape hatch is preserved
for p2p in OK CNS; do
  penv="$(pcie_env 2 "$p2p")"
  [ "$penv" = "DISABLE=|LEVEL=PHB" ] \
    && echo "[autop2p] ok: pcie_p2p ($p2p) forces config on (DISABLE unset, LEVEL=PHB)" \
    || { echo "[autop2p] FAIL: pcie_p2p ($p2p) must force config on — got '$penv'" >&2; fail=1; }
done

# ── BAR1 sanity warning (#873) ────────────────────────────────────────────────
# The patched-driver P2P path maps the full VRAM aperture through BAR1. When BAR1
# is far smaller than VRAM, peer access can be ADVERTISED (topo -p2p: OK) yet fail
# in practice — and NCCL HANGS at init instead of falling back. Warn, never gate:
# the config must come out identical with and without the warning.
bar1_text() {   # $1=mode $2=p2p $3=bar1MiB("" => driver doesn't report it) $4=link
  PATH="$MOCK_DIR:$PATH" FAKE_GPUS=2 FAKE_LINK="${4:-PHB}" FAKE_P2P="$2" FAKE_BAR1="$3" \
    NVLINK_MODE="$1" bash -c "source '$DETECT' 2>&1"
}
bar1_env() {    # same args -> resolved NCCL env
  PATH="$MOCK_DIR:$PATH" FAKE_GPUS=2 FAKE_LINK="${4:-PHB}" FAKE_P2P="$2" FAKE_BAR1="$3" \
    NVLINK_MODE="$1" bash -c "source '$DETECT' >/dev/null 2>&1; printf 'DISABLE=%s|LEVEL=%s' \"\${NCCL_P2P_DISABLE:-}\" \"\${NCCL_P2P_LEVEL:-}\""
}
warns() { [[ "$1" == *"BAR1 is far smaller than VRAM"* ]]; }

# small BAR1 (256 MiB vs 24576 MiB VRAM) + P2P advertised -> WARN
if warns "$(bar1_text auto OK 256)"; then echo "[autop2p] ok: auto + P2P OK + 256MiB BAR1 -> BAR1 warning"
else echo "[autop2p] FAIL: small BAR1 on the auto P2P path must warn" >&2; fail=1; fi
# ...and the same on the explicit force path
if warns "$(bar1_text pcie_p2p OK 256)"; then echo "[autop2p] ok: pcie_p2p + 256MiB BAR1 -> BAR1 warning"
else echo "[autop2p] FAIL: small BAR1 on the pcie_p2p path must warn" >&2; fail=1; fi
# full-size BAR1 (ReBAR / static mapping live) -> silent
if warns "$(bar1_text auto OK 32768)"; then echo "[autop2p] FAIL: full-size BAR1 must NOT warn" >&2; fail=1
else echo "[autop2p] ok: auto + P2P OK + 32768MiB BAR1 -> no warning"; fi
# driver doesn't report BAR1 at all -> fail open, no warning
if warns "$(bar1_text auto OK "")"; then echo "[autop2p] FAIL: unreported BAR1 must fail open (no warning)" >&2; fail=1
else echo "[autop2p] ok: BAR1 unreported -> fails open, no warning"; fi
# NVLink path never depends on BAR1 -> silent even with a tiny aperture
if warns "$(bar1_text auto CNS 256 NV4)"; then echo "[autop2p] FAIL: NVLink path must not run the BAR1 probe" >&2; fail=1
else echo "[autop2p] ok: NVLink (NVL) + tiny BAR1 -> no warning"; fi
# P2P off -> nothing to warn about
if warns "$(bar1_text auto CNS 256)"; then echo "[autop2p] FAIL: P2P-off path must not warn about BAR1" >&2; fail=1
else echo "[autop2p] ok: P2P unavailable + tiny BAR1 -> no warning"; fi
# WARN, NEVER GATE: identical env whether BAR1 is tiny or full-size
b_small="$(bar1_env auto OK 256)"; b_big="$(bar1_env auto OK 32768)"
if [ "$b_small" = "DISABLE=|LEVEL=PHB" ] && [ "$b_small" = "$b_big" ]; then
  echo "[autop2p] ok: BAR1 warning does not change the resolved config (warn, not gate)"
else
  echo "[autop2p] FAIL: BAR1 warning must not alter config — small='$b_small' big='$b_big'" >&2; fail=1
fi

if [ "$fail" -ne 0 ]; then echo "[autop2p] FAILED" >&2; exit 1; fi
echo "[autop2p] PASS: auto promotes to PCIe P2P only when topo -p2p reports OK; pcie_p2p force reports honestly (ENABLED vs UNVERIFIED) while always applying the config; undersized BAR1 warns on the PCIe-P2P path without changing it"
