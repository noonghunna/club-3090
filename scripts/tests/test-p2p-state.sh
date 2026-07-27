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
assert_not_contains() { [[ "$1" != *"$2"* ]] || { echo "FAIL: must NOT contain '$2': $1" >&2; exit 1; }; }

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

# ── 6. #786: vLLM custom-AR veto at world>2 → nccl_only, never a false "ON" ──
VLLM_WARN='Custom allreduce is disabled because it'"'"'s not supported on more than two PCIe-only GPUs. To silence this warning, specify disable_custom_all_reduce=True'
# pre-#786 trail (asserts AR ON) + vLLM veto line -> nccl_only, NOT on
r="$(printf '%s\n%s' '[nvlink] 4 GPUs — no NVLink, but nvidia-smi reports P2P=OK — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=PHB, custom all-reduce ON)' "$VLLM_WARN" | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "old trail + vLLM veto -> nccl_only (got $r)"
# bare stack (no trail) + vLLM veto + NCCL level env -> nccl_only (was: unknown, alesha's false negative)
r="$(printf '%s\n%s' "$VLLM_WARN" 'NCCL_P2P_LEVEL=PHB' | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "bare stack + veto + level -> nccl_only (got $r)"
# vLLM veto but P2P explicitly disabled -> off (AR off AND P2P off)
r="$(printf '%s\n%s' "$VLLM_WARN" 'NCCL_P2P_DISABLE=1' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "veto + P2P disabled -> off (got $r)"
# post-#786 decider trail wording alone -> nccl_only
r="$(echo '[nvlink] 4 GPUs — no NVLink, but nvidia-smi reports P2P=OK (patched driver / P2P-capable layout) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=PHB, custom all-reduce engine-gated (vLLM disables its custom kernel at >2 PCIe-only GPUs — P2P runs via NCCL; #786))' | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "engine-gated trail -> nccl_only (got $r)"
# requested still wins over the veto line (#688 precedence preserved)
r="$(printf '%s\n%s' '[nvlink] P2P REQUESTED (UNVERIFIED) — custom all-reduce configured as forced' "$VLLM_WARN" | p2p_classify_engagement)"
[[ "$r" == "requested" ]] || fail "requested beats veto (got $r)"
# 2-GPU trail with true AR ON and no veto line -> still on (unchanged)
r="$(echo '[nvlink] PCIe topology (PHB) but nvidia-smi reports P2P=OK (patched driver / shared root complex) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=PHB, custom all-reduce ON)' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "2-GPU AR-on trail -> on (got $r)"
# verdict lines for the new state
out="$(p2p_verdict 4 pcie_p2p nccl_only)"
assert_contains "$out" "via NCCL"
assert_contains "$out" "NOT a misconfiguration"
assert_not_contains "$out" "custom all-reduce ON"
out="$(p2p_verdict 4 nvlink nccl_only)";  assert_contains "$out" "NCCL"
assert_empty "$(p2p_verdict 1 pcie_p2p nccl_only)"                   # single GPU -> silent
assert_empty "$(p2p_verdict 2 none nccl_only)"                       # no capability -> silent
echo "  ✓ #786: vLLM AR veto -> nccl_only state; verdict never claims custom-AR-ON"

# ── 7. #786 decider wording: >2-GPU PCIe-P2P boot must not assert AR ON ──────
L4='GPU 0: RTX 3090\nGPU 1: RTX 3090\nGPU 2: RTX 3090\nGPU 3: RTX 3090\n'
TOPO_PHB4='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tPHB\tPHB\tPHB\nGPU1\tPHB\t X \tPHB\tPHB\nGPU2\tPHB\tPHB\t X \tPHB\nGPU3\tPHB\tPHB\tPHB\t X \n'
P2P_OK4=' \tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\tX\tOK\tOK\tOK\nGPU1\tOK\tX\tOK\tOK\nGPU2\tOK\tOK\tX\tOK\nGPU3\tOK\tOK\tOK\tX\n'
mk_smi "$L4" "$TOPO_PHB4" "$P2P_OK4"
d="$(run_decider)"
assert_contains "$d" "engine-gated"
assert_not_contains "$d" "custom all-reduce ON"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "pcie_p2p" ]] || fail "4-GPU PHB + p2p OK -> pcie_p2p (got $r)"
# decider trail round-trips through the auditor to nccl_only
r="$(printf '%s' "$d" | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "4-GPU decider trail -> nccl_only (got $r)"
# 2-GPU P2P boot keeps the true claim
mk_smi "$L2" "$TOPO_PHB" "$P2P_OK"
d="$(run_decider)"
assert_contains "$d" "custom all-reduce ON"
# 4x 3090 with PAIRWISE bridges (2 bridges on 4 cards) — NVLink present but the
# mesh is not fully connected, so vLLM's custom AR is off just like PCIe >2.
TOPO_NV4_PAIR='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tNV4\tPHB\tPHB\nGPU1\tNV4\t X \tPHB\tPHB\nGPU2\tPHB\tPHB\t X \tNV4\nGPU3\tPHB\tPHB\tNV4\t X \n'
mk_smi "$L4" "$TOPO_NV4_PAIR" "$P2P_OK4"
d="$(run_decider)"
assert_contains "$d" "NOT fully connected"
assert_contains "$d" "engine-gated"
assert_not_contains "$d" "custom all-reduce ON"
r="$(printf '%s' "$d" | p2p_classify_engagement)"
[[ "$r" == "nccl_only" ]] || fail "pairwise-NVLink 4-GPU trail -> nccl_only (got $r)"
# full 1-hop mesh (NVSwitch/SXM class) — the only >2-GPU case where AR ON is true
TOPO_NV4_FULL='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tNV12\tNV12\tNV12\nGPU1\tNV12\t X \tNV12\tNV12\nGPU2\tNV12\tNV12\t X \tNV12\nGPU3\tNV12\tNV12\tNV12\t X \n'
mk_smi "$L4" "$TOPO_NV4_FULL" "$P2P_OK4"
d="$(run_decider)"
assert_contains "$d" "NVLink full mesh"
assert_contains "$d" "custom all-reduce ON"
r="$(printf '%s' "$d" | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "full-mesh 4-GPU trail -> on (got $r)"
echo "  ✓ #786 decider: >2-GPU PCIe-P2P AND pairwise-NVLink print engine-gated (round-trip nccl_only); 2-GPU + full-mesh keep AR ON"

# ── 8. transfer-verified P2P cache (read-only tier folded from #787) ─────────
j='{"0->1": true, "1->0": true, "0->2": true, "2->0": true, "1->2": true, "2->1": true}'
r="$(printf '%s' "$j" | p2p_transfer_cache_parse)"
[[ "$r" == "6 6" ]] || fail "full-OK cache -> 6 6 (got $r)"
j='{"0->1": true, "1->0": false, "0->2": true, "2->0": true}'
r="$(printf '%s' "$j" | p2p_transfer_cache_parse)"
[[ "$r" == "3 4" ]] || fail "partial cache -> 3 4 (got $r)"
r="$(printf '%s' '{"not": "a cache"}' | p2p_transfer_cache_parse)"
[[ -z "$r" ]] || fail "garbage JSON -> silent (got $r)"
out="$(p2p_transfer_verdict 6 6 '~/.cache/vllm/gpu_p2p_access_cache_for_0,1,2.json')"
assert_contains "$out" "verified by TRANSFER"
assert_contains "$out" "6/6"
out="$(p2p_transfer_verdict 3 4 'container:/root/.cache/vllm/x.json')"
assert_contains "$out" "WARN"
assert_contains "$out" "3/4"
assert_empty "$(p2p_transfer_verdict 0 0 x)"                          # no data -> silent
echo "  ✓ transfer-cache parse + verdict: full OK / partial WARN / garbage silent"

echo "test-p2p-state: ok"
