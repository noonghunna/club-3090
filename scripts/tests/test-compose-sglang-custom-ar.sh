#!/usr/bin/env bash
# test-compose-sglang-custom-ar — the SGLang composes' custom all-reduce gate (#2008).
#
# SGLang runs its own all-reduce kernel (V2) only where every card pair is NVLink; over PCIe it falls
# back to V1, which measured no gain on the reference rig's P2P pair (2026-09-28), and vLLM's PCIe kernel
# harmed patched peer paths (#922, #1332). So every multi-GPU SGLang compose drops
# --disable-custom-all-reduce ONLY when detect_nvlink.sh finds real NVLink (a full mesh above two cards)
# and DISABLE_CUSTOM_ALL_REDUCE isn't 1. Two halves:
#   1. static: every registered multi-GPU SGLang compose carries the gate, mounts detect_nvlink.sh,
#      forwards NVLINK_MODE and DISABLE_CUSTOM_ALL_REDUCE with defaults, passes `$$_AR` where the flag
#      was, and hands the gate the same width as --tp-size. No unconditional flag is left;
#   2. functional: the gate, cut VERBATIM out of a shipped compose ($$ un-escaped), run against a mocked
#      nvidia-smi: NVLink pair -> on; PCIe P2P granted -> off; NVLink + DISABLE_CUSTOM_ALL_REDUCE=1 -> off;
#      four cards with pairwise bridges -> off; a full four-card NVLink mesh -> on; a PCIe rig with
#      NCCL_P2P_LEVEL=NVL set by hand -> off (the trap _NVLINK_FOUND exists for); NVLINK_MODE=force_off
#      on NVLink -> off.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
DETECT="${ROOT_DIR}/scripts/detect_nvlink.sh"
fail=0
note() { echo "FAIL: $*" >&2; fail=1; }

# ── 1. static ────────────────────────────────────────────────────────────────
static_out="$(python3 - <<'PY'
import pathlib, re, yaml
entries = yaml.safe_load(open("scripts/lib/profiles/registry.yaml", encoding="utf-8"))["entries"]
paths = sorted({e["compose_path"] for e in entries.values()
                if "/sglang/compose/" in e.get("compose_path", "") and "/single/" not in e["compose_path"]})
bad = []
for p in paths:
    s = pathlib.Path(p).read_text(encoding="utf-8")
    if re.search(r"^\s+--disable-custom-all-reduce \\$", s, re.M):
        bad.append(f"{p}: an unconditional --disable-custom-all-reduce is still on the command line")
    if not re.search(r"^\s+\$\$_AR \\$", s, re.M):
        bad.append(f"{p}: no $$_AR argument")
    if "/scripts/detect_nvlink.sh:/etc/club3090/detect_nvlink.sh:ro" not in s:
        bad.append(f"{p}: detect_nvlink.sh not mounted")
    for k in ("NVLINK_MODE=${NVLINK_MODE:-auto}", "DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-0}"):
        if f"- {k}" not in s:
            bad.append(f"{p}: {k.split('=')[0]} not forwarded with its default")
    tp = re.findall(r"^\s+--tp-size (\d+) \\$", s, re.M)
    gate = re.findall(r"detect_nvlink\.sh --tensor-parallel-size (\d+) ", s)
    if len(tp) != 1 or gate != tp:
        bad.append(f"{p}: gate width {gate} != --tp-size {tp}")
print(f"checked {len(paths)}")
for b in bad:
    print("BAD " + b)
PY
)"
n_checked="$(sed -n 's/^checked //p' <<<"$static_out")"
[[ "${n_checked:-0}" -ge 20 ]] || note "static: expected the multi-GPU SGLang composes, checked ${n_checked:-0}"
while IFS= read -r line; do [[ "$line" == BAD* ]] && note "static: ${line#BAD }"; done <<<"$static_out"

# ── 2. functional ────────────────────────────────────────────────────────────
MOCK_DIR="$(mktemp -d)"; trap 'rm -rf "$MOCK_DIR"' EXIT
# FAKE_LINK applies to every off-diagonal cell; FAKE_PAIRWISE=1 makes (0,1) and (2,3) NV4 and the rest PHB
cat > "$MOCK_DIR/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
n="${FAKE_GPUS:-2}"; args="$*"
cell() { local i=$1 j=$2
  if [ "${FAKE_PAIRWISE:-0}" = 1 ]; then
    if [ $((i/2)) = $((j/2)) ]; then printf 'NV4'; else printf 'PHB'; fi
  else printf '%s' "${FAKE_LINK:-PHB}"; fi; }
case "$args" in
  "-L") for ((i=0;i<n;i++)); do echo "GPU $i: NVIDIA GeForce RTX 3090 (UUID: GPU-$i)"; done ;;
  "topo -m")
    printf "\t"; for ((j=0;j<n;j++)); do printf "GPU%d\t" "$j"; done; printf "\n"
    for ((i=0;i<n;i++)); do printf "GPU%d\t" "$i"
      for ((j=0;j<n;j++)); do [ "$i" = "$j" ] && printf "X\t" || { cell "$i" "$j"; printf "\t"; }; done
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

# The gate, cut from a shipped compose: from the _AR assignment through its `fi`. The width is replaced
# per case, the mount point by the repo's script, and compose's $$ by $.
gate_of() {   # $1 = compose path
  python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r'^( +)_AR="--disable-custom-all-reduce"\n.*?^\1fi\n', s, re.M | re.S)
print(m.group(0) if m else "", end="")
PY
}
GATE="$(gate_of models/qwen3.8-27b/sglang/compose/multi4/autoround-int4/mtp.yml)"
[[ -n "$GATE" ]] || { note "functional: could not cut the gate out of the multi4 compose"; GATE='_AR=MISSING'; }
GATE="${GATE//\$\$/\$}"
GATE="${GATE//\/etc\/club3090\/detect_nvlink.sh/$DETECT}"

resolve() {   # $1 = tp width; the case's variables are already in the environment
  local g="${GATE//--tensor-parallel-size 4/--tensor-parallel-size $1}"
  PATH="$MOCK_DIR:$PATH" bash -c "set -euo pipefail
$g
echo \"AR=[\$_AR]\"" 2>/dev/null
}
expect() {   # $1 = label, $2 = want (on|off), $3 = tp width, rest = VAR=value for this case
  local label="$1" want="$2" tp="$3" out; shift 3
  out="$(unset FAKE_GPUS FAKE_LINK FAKE_PAIRWISE FAKE_P2P NCCL_P2P_LEVEL NCCL_P2P_DISABLE \
                DISABLE_CUSTOM_ALL_REDUCE NVLINK_MODE
         export NVLINK_MODE=auto DISABLE_CUSTOM_ALL_REDUCE=0 "$@"
         resolve "$tp")"
  case "$want:$out" in
    "on:AR=[]") ;;
    "off:AR=[--disable-custom-all-reduce]") ;;
    *) note "functional: $label: want $want, got '$out'" ;;
  esac
}
expect "NVLink pair"                     on  2 FAKE_GPUS=2 FAKE_LINK=NV4 FAKE_P2P=OK
expect "PCIe P2P granted"                off 2 FAKE_GPUS=2 FAKE_LINK=PHB FAKE_P2P=OK
expect "NVLink + operator veto"          off 2 FAKE_GPUS=2 FAKE_LINK=NV4 FAKE_P2P=OK DISABLE_CUSTOM_ALL_REDUCE=1
expect "4 cards, pairwise bridges"       off 4 FAKE_GPUS=4 FAKE_PAIRWISE=1 FAKE_P2P=OK
expect "4 cards, full NVLink mesh"       on  4 FAKE_GPUS=4 FAKE_LINK=NV12 FAKE_P2P=OK
expect "PCIe rig, NCCL_P2P_LEVEL=NVL"    off 2 FAKE_GPUS=2 FAKE_LINK=PHB FAKE_P2P=OK NCCL_P2P_LEVEL=NVL
expect "no P2P at all"                   off 2 FAKE_GPUS=2 FAKE_LINK=PHB FAKE_P2P=CNS
expect "NVLink, NVLINK_MODE=force_off"   off 2 FAKE_GPUS=2 FAKE_LINK=NV4 FAKE_P2P=OK NVLINK_MODE=force_off

if [[ "$fail" -ne 0 ]]; then echo "test-compose-sglang-custom-ar: FAILED" >&2; exit 1; fi
echo "test-compose-sglang-custom-ar: ok (${n_checked} composes, 8 gate cases)"
