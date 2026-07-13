#!/usr/bin/env bash
# test-p2p-state — the interconnect verdict matrix (scripts/lib/p2p-state.sh)
# + a consistency guard that runs the DECIDER (detect_nvlink.sh) and the
# AUDITOR (p2p_host_capability) against the same faked nvidia-smi and asserts
# they agree — the two parse the same probes and must not drift.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || { echo "FAIL: missing '$2' in: $1" >&2; exit 1; }; }
assert_empty() { [[ -z "$1" ]] || { echo "FAIL: expected silence, got: $1" >&2; exit 1; }; }

source scripts/lib/p2p-state.sh

# ── 1. pure verdict matrix ────────────────────────────────────────────────────
assert_empty "$(p2p_verdict 1 nvlink off)"                          # single GPU -> silent
assert_empty "$(p2p_verdict 2 none off)"                            # stock PCIe -> silent
assert_empty "$(p2p_verdict 2 none unknown)"
out="$(p2p_verdict 2 nvlink on)";     assert_contains "$out" "✓ interconnect: NVLink engaged"
out="$(p2p_verdict 2 pcie_p2p on)";   assert_contains "$out" "PCIe P2P engaged"
out="$(p2p_verdict 2 nvlink off)";    assert_contains "$out" "⚠ interconnect WARN"
assert_contains "$out" "NVLINK_MODE=force_on"
out="$(p2p_verdict 2 nvlink unknown)"; assert_contains "$out" "⚠ interconnect WARN"
out="$(p2p_verdict 2 pcie_p2p off)";  assert_contains "$out" "ℹ interconnect"
assert_contains "$out" "NVLINK_MODE=pcie_p2p"
out="$(p2p_verdict 4 nvlink off)";    assert_contains "$out" "WARN"   # multi-GPU too
# forced-but-unverified pcie_p2p — WARN even when capability probe says none (#688).
out="$(p2p_verdict 2 none requested)"; assert_contains "$out" "⚠ interconnect WARN"
assert_contains "$out" "topo -p2p rw"
assert_contains "$out" "forced PCIe P2P on"
out="$(p2p_verdict 4 none requested)"; assert_contains "$out" "⚠ interconnect WARN"  # multi-GPU too
assert_empty "$(p2p_verdict 1 none requested)"                       # single GPU still silent

# ── 2. engagement classifier (pure, stdin fixtures) ───────────────────────────
r="$(echo '[nvlink] detected NVLink (NV4) between GPU0-GPU1 — enabling NVLink mode' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "nvlink boot line -> on (got $r)"
r="$(echo '[nvlink] NVLINK_MODE=pcie_p2p — forcing PCIe P2P; driver confirms peer access (nvidia-smi topo -p2p: OK) — NCCL_P2P_LEVEL=PHB, custom all-reduce ON' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "verified pcie_p2p boot line -> on (got $r)"
# forced-but-UNVERIFIED pcie_p2p -> requested, NOT on (the #688 false-engaged guard).
r="$(echo '[nvlink] P2P REQUESTED (UNVERIFIED) — NCCL_P2P_LEVEL=PHB + custom all-reduce configured as forced, but peer access is UNCONFIRMED (topo -p2p ≠ OK; see the warning above)' | p2p_classify_engagement)"
[[ "$r" == "requested" ]] || fail "unverified pcie_p2p -> requested (got $r)"
# combined warn(stderr)+trail as report.sh greps them together -> still requested
r="$(printf '%s\n%s' '[nvlink] WARNING: NVLINK_MODE=pcie_p2p set, but nvidia-smi topo -p2p does NOT report peer access as OK' '[nvlink] P2P REQUESTED (UNVERIFIED) — custom all-reduce configured as forced' | p2p_classify_engagement)"
[[ "$r" == "requested" ]] || fail "combined warn+trail -> requested (got $r)"
r="$(echo '[nvlink] PCIe topology (PHB), P2P not available (topo -p2p: no OK) — using PCIe mode' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "pcie-mode boot line -> off (got $r)"
r="$(echo '[nvlink] NVLINK_MODE=force_off — forcing PCIe mode (P2P off)' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "force_off boot line -> off (got $r)"
r="$(echo 'NCCL_P2P_DISABLE=1' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "env disable -> off (got $r)"
r="$(echo 'NCCL_P2P_LEVEL=NVL' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "env level -> on (got $r)"
r="$(echo '' | p2p_classify_engagement)"
[[ "$r" == "unknown" ]] || fail "empty -> unknown (got $r)"
# boot trail beats env: a trail that resolved OFF wins over a leftover LEVEL var
r="$(printf '%s\n%s' '[nvlink] NVLINK_MODE=force_off — forcing PCIe mode (P2P off)' 'NCCL_P2P_LEVEL=PHB' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "trail-over-env precedence (got $r)"

# ── 3. capability probes via faked nvidia-smi ────────────────────────────────
mk_smi() { cat > "$TMP/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  -L) printf '%b' "$1" ;;
  "topo -m") printf '%b' "$2" ;;
  "topo -p2p r") printf '%b' "$3" ;;
esac
EOF
chmod +x "$TMP/nvidia-smi"; }

L2='GPU 0: RTX 3090\nGPU 1: RTX 3090\n'
TOPO_NV='\tGPU0\tGPU1\nGPU0\t X \tNV4\nGPU1\tNV4\t X \n'
TOPO_PHB='\tGPU0\tGPU1\nGPU0\t X \tPHB\nGPU1\tPHB\t X \n'
P2P_OK=' \tGPU0\tGPU1\nGPU0\tX\tOK\nGPU1\tOK\tX\n'
P2P_CNS=' \tGPU0\tGPU1\nGPU0\tX\tCNS\nGPU1\tCNS\tX\n'

mk_smi "$L2" "$TOPO_NV" "$P2P_CNS"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "nvlink" ]] || fail "NV topo -> nvlink (got $r)"

mk_smi "$L2" "$TOPO_PHB" "$P2P_OK"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "pcie_p2p" ]] || fail "PHB + p2p OK -> pcie_p2p (got $r)"

mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "none" ]] || fail "stock PCIe -> none (got $r)"

mk_smi 'GPU 0: RTX 3090\n' "$TOPO_PHB" "$P2P_OK"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "none" ]] || fail "single GPU -> none even with p2p OK (got $r)"

# ── 4. decider↔auditor consistency: detect_nvlink.sh on the same fixtures ────
run_decider() { PATH="$TMP:$PATH" NVLINK_MODE=auto bash scripts/detect_nvlink.sh 2>/dev/null || true; }
mk_smi "$L2" "$TOPO_NV" "$P2P_CNS"
d="$(run_decider)"; assert_contains "$d" "enabling NVLink mode"       # decider: nvlink
mk_smi "$L2" "$TOPO_PHB" "$P2P_OK"
d="$(run_decider)"; assert_contains "$d" "P2P=OK"                     # decider: pcie_p2p
mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS"
d="$(run_decider)"; assert_contains "$d" "using PCIe mode"            # decider: none
echo "  ✓ decider (detect_nvlink.sh) and auditor (p2p-state.sh) agree on all 3 fixtures"

# ── 5. driver-flavor probe via faked modinfo ─────────────────────────────────
# open modules (nvidia-open / aikitoria fork) report "Dual MIT/GPL"; closed
# reports "NVIDIA". We report open-vs-proprietary; the fork is NOT fingerprintable.
mk_modinfo() { printf '#!/usr/bin/env bash\nprintf %%s %q\n' "$1" > "$TMP/modinfo"; chmod +x "$TMP/modinfo"; }
mk_modinfo "NVIDIA"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_driver_flavor')"
[[ "$r" == "proprietary" ]] || fail "license NVIDIA -> proprietary (got $r)"
mk_modinfo "Dual MIT/GPL"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_driver_flavor')"
[[ "$r" == "open" ]] || fail "license Dual MIT/GPL -> open (got $r)"
mk_modinfo ""
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_driver_flavor')"
[[ "$r" == "unknown" ]] || fail "empty license -> unknown (got $r)"
echo "  ✓ driver-flavor probe: NVIDIA->proprietary, Dual MIT/GPL->open, empty->unknown"

echo "test-p2p-state: ok"
