#!/usr/bin/env bash
# test-p2p-state — the interconnect verdict matrix (scripts/lib/p2p-state.sh)
# + a consistency guard that runs the DECIDER (detect_nvlink.sh) and the
# AUDITOR (p2p_host_capability) against the same faked nvidia-smi and asserts
# they agree — the two parse the same probes and must not drift.
set -euo pipefail

# ⚠️ Pin the knob, like NVLINK_MODE. Nothing else scrubs it, so an operator with
# DISABLE_CUSTOM_ALL_REDUCE exported in their shell — which our own boot warning
# tells them to do — would red this suite for reasons unrelated to the code
# under test (#1332 review).
export DISABLE_CUSTOM_ALL_REDUCE=0


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
  "-q -d MEMORY") printf '%b' "${4:-}" ;;
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
# ⚠️ Passes a TP width. The decider derives `gate=` (its prediction that vLLM will
# veto its own kernel) from the TP world size, read from the entrypoint argv —
# NOT from the visible GPU count, because a 4-visible container running TP=2 has
# no veto coming. A decider fixture with no argv therefore resolves to TP=1 and
# predicts nothing, which is correct behaviour and would silently weaken every
# >2-GPU assertion here. Default 2 matches the 2-GPU fixtures; pass 4 for the
# 4-GPU ones.
run_decider() { PATH="$TMP:$PATH" NVLINK_MODE=auto bash scripts/detect_nvlink.sh -- --tensor-parallel-size "${1:-2}" 2>/dev/null || true; }
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
[[ "$r" == "nccl_only_gated" ]] || fail "old trail + vLLM veto -> nccl_only (got $r)"
# bare stack (no trail) + vLLM veto + NCCL level env -> nccl_only (was: unknown, alesha's false negative)
r="$(printf '%s\n%s' "$VLLM_WARN" 'NCCL_P2P_LEVEL=PHB' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "bare stack + veto + level -> nccl_only (got $r)"
# vLLM veto but P2P explicitly disabled -> off (AR off AND P2P off)
r="$(printf '%s\n%s' "$VLLM_WARN" 'NCCL_P2P_DISABLE=1' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "veto + P2P disabled -> off (got $r)"
# operator flag / knob -> the operator state, distinct from the engine's veto
for _sig in 'disable_custom_all_reduce=True' '--disable-custom-all-reduce' 'custom all-reduce OFF by operator request'; do
  r="$(printf '%s\n%s' '[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=PHB' "$_sig" | p2p_classify_engagement)"
  [[ "$r" == "nccl_only_operator" ]] || fail "operator signal '$_sig' -> nccl_only_operator (got $r)"
done
unset _sig
# post-#786 decider trail wording alone -> nccl_only
r="$(echo '[nvlink] 4 GPUs — no NVLink, but nvidia-smi reports P2P=OK (patched driver / P2P-capable layout) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=PHB, custom all-reduce engine-gated (vLLM disables its custom kernel at >2 PCIe-only GPUs — P2P runs via NCCL; #786))' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "engine-gated trail -> nccl_only (got $r)"
# requested still wins over the veto line (#688 precedence preserved)
r="$(printf '%s\n%s' '[nvlink] P2P REQUESTED (UNVERIFIED) — custom all-reduce configured as forced' "$VLLM_WARN" | p2p_classify_engagement)"
[[ "$r" == "requested" ]] || fail "requested beats veto (got $r)"
# 2-GPU trail with true AR ON and no veto line -> still on (unchanged)
r="$(echo '[nvlink] PCIe topology (PHB) but nvidia-smi reports P2P=OK (patched driver / shared root complex) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=PHB, custom all-reduce ON)' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "2-GPU AR-on trail -> on (got $r)"
# verdict lines for the new state
out="$(p2p_verdict 4 pcie_p2p nccl_only_gated)"
assert_contains "$out" "via NCCL"
assert_contains "$out" "NOT a misconfiguration"
assert_not_contains "$out" "custom all-reduce ON"
out="$(p2p_verdict 4 nvlink nccl_only_gated)";  assert_contains "$out" "NCCL"
assert_empty "$(p2p_verdict 1 pcie_p2p nccl_only_gated)"                   # single GPU -> silent
assert_empty "$(p2p_verdict 2 none nccl_only_gated)"                       # no capability -> silent
# #1332: the CAUSE of a custom-AR-off state is read from the signal, never
# inferred from GPU count. The first cut of this branched on `count <= 2` and so
# mislabelled vLLM's P2P-TEST-FAILED veto — also 2 GPUs, also custom-AR off — as
# a healthy deliberate operator choice, on a rig whose peer path had just failed
# vLLM's own transfer test. Assert each cause renders as itself at EVERY count.
VLLM_P2P_FAIL='Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.'
r="$(printf '%s\n%s' '[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=PHB, custom all-reduce ON' "$VLLM_P2P_FAIL" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_degraded" ]] || fail "vLLM P2P-test-failed veto -> nccl_only_degraded (got $r)"
r="$(printf '%s\n%s' "$VLLM_P2P_FAIL" 'NCCL_P2P_DISABLE=1' | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "P2P-test-failed + P2P disabled -> off (got $r)"
r="$(echo 'Custom allreduce is disabled because of missing custom allreduce library' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_nolib" ]] || fail "missing-library -> nccl_only_nolib (got $r)"
# The degraded state must WARN at every count and every capability, and must
# never read as a deliberate choice.
for _n in 2 4; do for _cap in pcie_p2p nvlink; do
  out="$(p2p_verdict "$_n" "$_cap" nccl_only_degraded)"
  assert_contains "$out" "WARN"
  assert_contains "$out" "P2P TEST FAILED"
  # It must not CLAIM an operator choice or a healthy state. It may — and does —
  # say the opposite of both, so assert on the claiming phrases, not the words.
  assert_contains "$out" "not an operator choice"
  assert_not_contains "$out" "operator request"
  assert_not_contains "$out" "deliberate"
  assert_not_contains "$out" "✓ interconnect"
  assert_not_contains "$out" "expected at >2"
done; done
# Operator-disabled must say so at every count — including >2, where the old
# count-based branch fell through to the #786 text on a 2-GPU container whose
# HOST has 4 cards (report.sh/bench.sh pass the host count).
for _n in 2 4; do for _cap in pcie_p2p nvlink; do
  out="$(p2p_verdict "$_n" "$_cap" nccl_only_operator)"
  assert_contains "$out" "operator request"
  assert_not_contains "$out" "expected at >2"
  assert_not_contains "$out" "not fully connected"
done; done
unset _cap _n
echo "  ✓ #786: vLLM AR veto -> nccl_only state; verdict never claims custom-AR-ON"
echo "  ✓ #1332: custom-AR-off CAUSE read from the signal (operator / gated / degraded / nolib), never inferred from GPU count"

# ── 7. #786 decider wording: >2-GPU PCIe-P2P boot must not assert AR ON ──────
L4='GPU 0: RTX 3090\nGPU 1: RTX 3090\nGPU 2: RTX 3090\nGPU 3: RTX 3090\n'
TOPO_PHB4='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tPHB\tPHB\tPHB\nGPU1\tPHB\t X \tPHB\tPHB\nGPU2\tPHB\tPHB\t X \tPHB\nGPU3\tPHB\tPHB\tPHB\t X \n'
P2P_OK4=' \tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\tX\tOK\tOK\tOK\nGPU1\tOK\tX\tOK\tOK\nGPU2\tOK\tOK\tX\tOK\nGPU3\tOK\tOK\tOK\tX\n'
mk_smi "$L4" "$TOPO_PHB4" "$P2P_OK4"
d="$(run_decider 4)"
assert_contains "$d" "engine-gated"
assert_not_contains "$d" "custom all-reduce ON"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_host_capability')"
[[ "$r" == "pcie_p2p" ]] || fail "4-GPU PHB + p2p OK -> pcie_p2p (got $r)"
# decider trail round-trips through the auditor to nccl_only
r="$(printf '%s' "$d" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "4-GPU decider trail -> nccl_only (got $r)"
# 2-GPU P2P boot keeps the true claim
mk_smi "$L2" "$TOPO_PHB" "$P2P_OK"
d="$(run_decider)"
assert_contains "$d" "custom all-reduce ON"
# 4x 3090 with PAIRWISE bridges (2 bridges on 4 cards) — NVLink present but the
# mesh is not fully connected, so vLLM's custom AR is off just like PCIe >2.
TOPO_NV4_PAIR='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tNV4\tPHB\tPHB\nGPU1\tNV4\t X \tPHB\tPHB\nGPU2\tPHB\tPHB\t X \tNV4\nGPU3\tPHB\tPHB\tNV4\t X \n'
mk_smi "$L4" "$TOPO_NV4_PAIR" "$P2P_OK4"
d="$(run_decider 4)"
assert_contains "$d" "NOT fully connected"
assert_contains "$d" "engine-gated"
assert_not_contains "$d" "custom all-reduce ON"
r="$(printf '%s' "$d" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "pairwise-NVLink 4-GPU trail -> nccl_only (got $r)"
# full 1-hop mesh (NVSwitch/SXM class) — the only >2-GPU case where AR ON is true
TOPO_NV4_FULL='\tGPU0\tGPU1\tGPU2\tGPU3\nGPU0\t X \tNV12\tNV12\tNV12\nGPU1\tNV12\t X \tNV12\tNV12\nGPU2\tNV12\tNV12\t X \tNV12\nGPU3\tNV12\tNV12\tNV12\t X \n'
mk_smi "$L4" "$TOPO_NV4_FULL" "$P2P_OK4"
d="$(run_decider 4)"
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

# ── 9. P2P opportunity hint (#873): tiered, achievability-gated, never nags ──
# The naive "you could enable P2P!" is WRONG for most rigs that would see it.
# What matters here is that each rig class gets the message that is TRUE for it,
# and that the firmware gate outranks the driver gate — the reference 2x3090 is
# CNS *and* BAR1-capped, so a precedence bug would tell the maintainer's own rig
# to go build a DKMS module that provably cannot work.
TOPO_SYS='\tGPU0\tGPU1\nGPU0\t X \tSYS\nGPU1\tSYS\t X \n'
_mem() { printf 'GPU 00000000:0%d:00.0\n    FB Memory Usage\n        Total : 24576 MiB\n        Used  : 1 MiB\n    BAR1 Memory Usage\n        Total : %d MiB\n        Used  : 4 MiB\n    Conf Compute Protected Memory Usage\n        Total : 0 MiB\n' "$1" "$2"; }
MEM_SMALL="$(_mem 1 256)$(_mem 3 256)"      # aperture never grew (ReBAR off / capped VBIOS)
MEM_BIG="$(_mem 1 32768)$(_mem 3 32768)"    # full-size aperture

hint() { PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_opportunity_hint "$(p2p_gpu_count)" "$(p2p_host_capability)"'; }

# (a) separate root complexes -> unreachable; must NOT sell the patched module
mk_smi "$L2" "$TOPO_SYS" "$P2P_CNS" "$MEM_BIG"
out="$(hint)"
assert_contains "$out" "SEPARATE root complexes"
[[ "$out" != *"§5"* ]] || fail "split-topology hint must not point at the patched-module path"

# (b) same root + CNS + SMALL BAR1 -> firmware gate wins over the driver gate
mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS" "$MEM_SMALL"
out="$(hint)"
assert_contains "$out" "BAR1 is 256 MiB against 24576 MiB"
assert_contains "$out" "§4"
[[ "$out" != *"§5"* ]] || fail "BAR1-capped hint must not recommend the patched module (it cannot work)"

# (c) same root + CNS + full-size BAR1 -> the one actionable case
mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS" "$MEM_BIG"
out="$(hint)"
assert_contains "$out" "§5"
assert_contains "$out" "CNS"
assert_contains "$out" "DKMS"          # the cost is stated, not just the gain
# The advisory must state the gain HONESTLY so a user can decline. This used to
# assert "+2% narrative" — a custom-all-reduce-dependent decode figure that is
# unreachable on any rig which has to disable that kernel (#922). The reliable,
# transport-only gain is PREFILL, so that is what the hint now leads with.
assert_contains "$out" "PREFILL"
assert_contains "$out" "14.4%"         # the transport-isolated measurement
assert_contains "$out" "AUTO-ENABLE"   # our default turns the kernel ON for P2P rigs — say so
assert_contains "$out" "ENGINE MATTERS" # llama.cpp layer split gains ~nothing (disc #921)
assert_contains "$out" "UNSETTLED"     # the wrong-data cause is contested (#751 vs #922) — do not overclaim
# ...and it must not sell a patched module without its sharpest edge.
assert_contains "$out" "WRONG DATA"
assert_contains "$out" "disable-custom-all-reduce"

# (d) BAR1 unreported by the driver -> fail open to the driver-gate message
mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS" ""
assert_contains "$(hint)" "§5"

# (e) silence where there is nothing to say
mk_smi "$L2" "$TOPO_PHB" "$P2P_OK" "$MEM_BIG"
assert_empty "$(hint)"                                  # P2P already on
mk_smi "$L2" "$TOPO_NV" "$P2P_CNS" "$MEM_BIG"
assert_empty "$(hint)"                                  # NVLink rig
mk_smi 'GPU 0: RTX 3090\n' "$TOPO_PHB" "$P2P_CNS" "$MEM_SMALL"
assert_empty "$(hint)"                                  # single card
echo "  ✓ opportunity hint: split/firmware/driver tiers, BAR1 outranks CNS, silent when moot"

# ── 10. p2p_gpu_count is EXACTLY one integer line, driver up or down (#1279) ─
# `grep -c` prints 0 AND exits 1 on no match, so the old `|| echo 0` tail fired
# too and the function returned "0\n0". Every consumer does arithmetic on the
# result, so report.sh emitted `[[: 0 0: syntax error in expression` into the
# diagnostic a user was about to send us. "Contains 0" would have passed against
# the broken version — these assert the LINE COUNT and the arithmetic.
count_under() {  # count_under <nvidia-smi stub body>
  printf '#!/usr/bin/env bash\n%s\n' "$1" > "$TMP/nvidia-smi"
  chmod +x "$TMP/nvidia-smi"
  PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh; p2p_gpu_count'
}

# (a) driver query FAILS outright (no GPU / driver not loaded) -> exactly "0"
out="$(count_under 'exit 1')"
[[ "$(printf '%s' "$out" | wc -l)" -eq 0 ]] \
  || fail "failing nvidia-smi: expected ONE line, got $(printf '%s\n' "$out" | wc -l): $(printf '%q' "$out")"
[[ "$out" =~ ^[0-9]+$ ]] || fail "failing nvidia-smi: not an integer: $(printf '%q' "$out")"
[[ "$out" == "0" ]]      || fail "failing nvidia-smi: expected 0, got $(printf '%q' "$out")"

# (b) nvidia-smi present but prints nothing matching (empty output) -> "0"
out="$(count_under 'exit 0')"
[[ "$out" == "0" ]] || fail "empty nvidia-smi -L: expected 0, got $(printf '%q' "$out")"

# (c) the happy path still counts
out="$(count_under "printf 'GPU 0: RTX 3090\\nGPU 1: RTX 3090\\n'")"
[[ "$out" == "2" ]] || fail "two GPUs: expected 2, got $(printf '%q' "$out")"

# (d) the DOWNSTREAM shape that broke: arithmetic on the result, driver down.
# report.sh:566 runs `[[ "$(p2p_gpu_count)" -ge 2 ]]`; with "0\n0" bash aborts
# the test with a syntax error ON STDERR and a non-zero status. Assert stderr
# is clean and the comparison evaluates.
printf '#!/usr/bin/env bash\nexit 1\n' > "$TMP/nvidia-smi"; chmod +x "$TMP/nvidia-smi"
err="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh
if [[ "$(p2p_gpu_count)" -ge 2 ]]; then echo many; else echo few; fi' 2>&1 >/dev/null)"
[[ -z "$err" ]] || fail "arithmetic on p2p_gpu_count emitted to stderr: $err"
r="$(PATH="$TMP:$PATH" bash -c 'source scripts/lib/p2p-state.sh
if [[ "$(p2p_gpu_count)" -ge 2 ]]; then echo many; else echo few; fi' 2>/dev/null)"
[[ "$r" == "few" ]] || fail "driver-down arithmetic should take the <2 branch (got $r)"

# (e) and it must not trip a pipefail caller (bench.sh runs with -o pipefail)
r="$(PATH="$TMP:$PATH" bash -c 'set -euo pipefail; source scripts/lib/p2p-state.sh; p2p_gpu_count')" \
  || fail "p2p_gpu_count returned non-zero under set -euo pipefail"
[[ "$r" == "0" ]] || fail "pipefail caller: expected 0, got $(printf '%q' "$r")"
echo "  ✓ p2p_gpu_count: exactly one integer line, driver up or down; arithmetic-safe"

# ── 9. STATE line: boot epoch, catch-all veto, legacy fallback ───────────────
# The three properties the redesign exists for. Each has a concrete failure that
# motivated it; each assertion below fails against the pre-redesign classifier.
S_ON='[nvlink] STATE v=1 transport=pcie_p2p kernel=on cause=none verified=yes gate=none mode=auto world=2'
S_OP='[nvlink] STATE v=1 transport=pcie_p2p kernel=off cause=operator verified=yes gate=none mode=auto world=2'
# ⚠️ NOT hand-written: taken from the decider itself. The previous fixture was
# invented, and the decider did not emit a transport=off STATE line at all — so
# this arm tested a string nothing produced and passed while the feature was
# absent. Any fixture claiming to be decider output must come from the decider.
S_OFF="$(NVLINK_MODE=force_off PATH="$TMP:$PATH" bash -c 'source scripts/detect_nvlink.sh' 2>/dev/null | command grep -F '[nvlink] STATE v=1 ' | tail -1)"
[ -n "$S_OFF" ] || fail "the decider emits NO STATE line on a transport-off boot — the transport=off arm would be dead"
case "$S_OFF" in *"transport=off"*) ;; *) fail "force_off boot must report transport=off, got: $S_OFF" ;; esac
V_FAIL='Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed.'
# ⚠️ vLLM's REAL text, advice sentence included. Without it this fixture scored
# `gated` even with the operator arm mis-ordered ahead of the catch-all, so it
# passed against a mutation that genuinely misclassifies this exact line.
V_WORLD='Custom allreduce is disabled due to an unsupported world size: 3. Supported world sizes: [2, 4, 6, 8, 16]. To silence this warning, specify disable_custom_all_reduce=True explicitly.'

r="$(printf '%s' "$S_ON" | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "STATE kernel=on alone -> on (got $r)"
r="$(printf '%s\n%s' "$S_OP" 'noise' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_operator" ]] || fail "STATE cause=operator -> nccl_only_operator (got $r)"
r="$(printf '%s' "$S_OFF" | p2p_classify_engagement)"
[[ "$r" == "off" ]] || fail "STATE transport=off -> off (got $r)"

# ⭐ BOOT EPOCH. `restart: unless-stopped` leaves several boots in one log.
# Boot 1 had a failed peer path; boot 2 is clean. Reading the whole log naively
# carries boot 1's veto into boot 2 and condemns a healthy rig — and the mirror
# case (veto in boot 2 only) reports a broken rig as healthy. Only lines AFTER
# the LAST STATE describe the current boot.
r="$(printf '%s\n%s\n%s' "$S_ON" "$V_FAIL" "$S_ON" | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "boot1 veto must NOT leak into a clean boot2 (got $r)"
r="$(printf '%s\n%s\n%s' "$S_ON" "$S_ON" "$V_FAIL" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_degraded" ]] || fail "boot2's own veto must be seen (got $r)"

# ⭐ CATCH-ALL. vLLM has more veto reasons than we enumerate. An unenumerated one
# must never resolve to "the engine said nothing" -> our configured kernel=on.
r="$(printf '%s\n%s' "$S_ON" "$V_WORLD" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "an UNENUMERATED vLLM veto must not read as kernel-on (got $r)"
r="$(printf '%s\n%s' "$S_ON" 'Custom allreduce is disabled because of some future reason we never coded for' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_gated" ]] || fail "catch-all must cover an unknown future veto (got $r)"

# ⭐ LEGACY. No STATE line -> the old cascade, which is permanent for SGLang and
# llama.cpp and for containers started before this shipped.
r="$(printf '%s' '[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=PHB, custom all-reduce ON' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "legacy trail (no STATE) must still classify (got $r)"
r="$(printf '%s\n%s' '[nvlink] P2P ENABLED — custom all-reduce ON' "$V_FAIL" | p2p_classify_engagement)"
[[ "$r" == "nccl_only_degraded" ]] || fail "legacy path must still see the degraded veto (got $r)"
# ⭐ EVERY boot emits exactly one STATE line. #1 shipped green because nothing
# asserted the emitter ran; it sat inside the transport-ON branch, so every
# transport-off boot silently fell back to the legacy cascade.
# ⚠️ Driven through mk_smi. An earlier cut set FAKE_P2P directly, but mk_smi is
# what writes the mock and it ignores that variable — and the ambient mock at
# this point is an exit-1 stub, so `auto` never reached a transport-on branch and
# the OK/CNS arms were byte-identical. Four shapes were being tested as eight.
for _m in auto force_off pcie_p2p force_on; do
  for _p2p in OK CNS; do
    if [ "$_p2p" = OK ]; then mk_smi "$L2" "$TOPO_PHB" "$P2P_OK"; else mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS"; fi
    _n="$(NVLINK_MODE="$_m" PATH="$TMP:$PATH" bash scripts/detect_nvlink.sh -- --tensor-parallel-size 2 2>/dev/null \
          | command grep -cF '[nvlink] STATE v=1 ' || true)"
    [ "${_n:-0}" -eq 1 ] || fail "NVLINK_MODE=$_m p2p=$_p2p emitted ${_n} STATE lines, expected exactly 1"
  done
done
unset _m _p2p _n

# ⭐ transport/verified LABELS, per branch. `_NVLINK_FOUND` is set on three
# branches and must not be set on the others; nothing asserted the resulting
# label, so dropping it from a branch (NVLink rigs silently relabelled pcie_p2p)
# or over-setting it left every suite green.
_label() {  # _label <mode> <topo> <p2p-matrix> -> "transport=… verified=…"
  mk_smi "$L2" "$2" "$3"
  PATH="$TMP:$PATH" NVLINK_MODE="$1" bash scripts/detect_nvlink.sh -- --tensor-parallel-size 2 2>/dev/null \
    | command grep -F 'STATE v=1' | head -1 \
    | command grep -oE 'transport=[a-z0-9_]+ kernel=[a-z]+ cause=[a-z_]+ verified=[a-z-]+' \
    | command grep -oE 'transport=[a-z0-9_]+|verified=[a-z-]+' | paste -sd' ' -
}
for _c in \
  "auto:$TOPO_NV:$P2P_CNS:transport=nvlink verified=yes" \
  "auto:$TOPO_PHB:$P2P_OK:transport=pcie_p2p verified=yes" \
  "auto:$TOPO_PHB:$P2P_CNS:transport=off verified=n-a" \
  "force_off:$TOPO_NV:$P2P_OK:transport=off verified=n-a" \
  "pcie_p2p:$TOPO_PHB:$P2P_CNS:transport=pcie_p2p verified=no" \
  "force_on:$TOPO_PHB:$P2P_CNS:transport=nvlink verified=n-a" ; do
  _m="${_c%%:*}"; _rest="${_c#*:}"; _t="${_rest%%:*}"; _rest="${_rest#*:}"
  _p="${_rest%%:*}"; _want="${_rest#*:}"
  _got="$(_label "$_m" "$_t" "$_p")"
  [[ "$_got" == "$_want" ]] || fail "NVLINK_MODE=$_m labels: got '$_got', want '$_want'"
done
unset _c _m _t _p _want _got
echo "  ✓ STATE labels correct per branch (nvlink only on a real NVLink finding; force_on never claims verified)"

# ⭐ F3 / mutation A: `gate=` is a restatement of vLLM's rule, which is on the TP
# WORLD SIZE — not on how many cards the container can see. A 4-visible container
# running TP=2 (CDI, or `count: all`) has no veto coming, and predicting one made
# the verdict say "vLLM auto-disabled its kernel" in the same boot whose warning
# said the kernel was live. Keyed on GPU count this pair is indistinguishable.
mk_smi "$L4" "$TOPO_PHB4" "$P2P_OK4"
_st4_tp2="$(PATH="$TMP:$PATH" NVLINK_MODE=auto bash scripts/detect_nvlink.sh -- --tensor-parallel-size 2 2>/dev/null | command grep -F 'STATE v=1' | head -1)"
_st4_tp4="$(PATH="$TMP:$PATH" NVLINK_MODE=auto bash scripts/detect_nvlink.sh -- --tensor-parallel-size 4 2>/dev/null | command grep -F 'STATE v=1' | head -1)"
case "$_st4_tp2" in *"gate=none"*) ;; *) fail "4 visible GPUs but TP=2: no veto is coming, expected gate=none — got: $_st4_tp2" ;; esac
case "$_st4_tp4" in *"gate=expected_veto"*) ;; *) fail "4 visible GPUs and TP=4 on PCIe: vLLM will veto, expected gate=expected_veto — got: $_st4_tp4" ;; esac
# The human-readable prose must agree with the machine-readable field.
_pr4_tp2="$(PATH="$TMP:$PATH" NVLINK_MODE=auto bash scripts/detect_nvlink.sh -- --tensor-parallel-size 2 2>/dev/null)"
case "$_pr4_tp2" in *engine-gated*) fail "boot prose says engine-gated while STATE says gate=none (TP=2, 4 visible)" ;; esac
unset _st4_tp2 _st4_tp4 _pr4_tp2

# ⭐ F1: our STATE says what WE decided; it cannot know a compose passed the flag
# itself. models/qwen3.8-27b/vllm/compose/dual/fp8/dflash2.yml does exactly that
# on every non-NVLink rig, so the engine's own argv must be consulted — otherwise
# a shipped slug reports "custom all-reduce ON" with the kernel off.
r="$(printf '%s\n%s' "$S_ON" 'Initializing a V1 LLM engine with config: disable_custom_all_reduce=True, seed=0' | p2p_classify_engagement)"
[[ "$r" == "nccl_only_operator" ]] || fail "operator-passed flag with STATE kernel=on -> nccl_only_operator (got $r)"
# ...but a veto's ADVICE sentence contains that same substring and must not win.
r="$(printf '%s\n%s' "$S_ON" "$V_FAIL To silence this warning, specify disable_custom_all_reduce=True explicitly." | p2p_classify_engagement)"
[[ "$r" == "nccl_only_degraded" ]] || fail "a veto's advice sentence must not read as an operator choice (got $r)"

# ⭐ F2: #688 — a forced PCIe path the driver refused must never read as healthy.
# ⚠️ Round-tripped through the DECIDER, not hand-written. A hand-written STATE
# line tests the classifier's reading of `verified=no` but not that the decider
# ever EMITS it — so forcing the emitter to `verified=yes` left every suite green
# while removing the #688 guard entirely.
mk_smi "$L2" "$TOPO_PHB" "$P2P_CNS"
_forced="$(PATH="$TMP:$PATH" NVLINK_MODE=pcie_p2p bash scripts/detect_nvlink.sh -- --tensor-parallel-size 2 2>&1)"
case "$_forced" in *"verified=no"*) ;; *) fail "a forced-but-unconfirmed PCIe boot must emit verified=no; got: $(printf '%s' "$_forced" | command grep -F 'STATE v=1')" ;; esac
r="$(printf '%s' "$_forced" | p2p_classify_engagement)"
[[ "$r" == "requested" ]] || fail "a forced-but-unconfirmed boot must classify requested, never healthy (got $r)"
unset _forced

# ⭐ F6/E: an unrecognised cause is a parse failure, not a ✓ state.
r="$(printf '%s' '[nvlink] STATE v=1 transport=pcie_p2p kernel=off cause=something_new verified=yes gate=none mode=auto world=2' | p2p_classify_engagement)"
[[ "$r" == "unknown" ]] || fail "unrecognised cause must be unknown, not a healthy state (got $r)"

# ⭐ F6/D: an unknown STATE version must fall back, not be parsed under v=1 rules.
r="$(printf '%s\n%s' '[nvlink] STATE v=2 transport=off kernel=off' '[nvlink] P2P ENABLED — custom all-reduce ON' | p2p_classify_engagement)"
[[ "$r" == "on" ]] || fail "a v=2 STATE line must fall through to legacy (got $r)"
echo "  ✓ STATE: emitted exactly once on every boot mode, epoch sliced, veto catch-all, legacy fallback intact"

echo "test-p2p-state: ok"
