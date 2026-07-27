#!/usr/bin/env bash
# p2p-state.sh — interconnect capability × engagement VERDICT (the #488/#158
# triage matrix). Read-only AUDITOR; the boot-time DECIDER is
# scripts/detect_nvlink.sh. The capability probes here mirror the decider's
# semantics on purpose — test-p2p-state.sh runs BOTH against shared fixtures
# so they cannot drift apart silently.
#
# Verdict matrix (report.sh renders it; preflight prints the capability line):
#   <2 GPUs, or no capability          -> silent (nothing useful to say)
#   capability + engagement ON         -> one OK line
#   capability + engagement NCCL-ONLY  -> OK line stating vLLM vetoed its custom
#                                        all-reduce (NVLink-only gate at
#                                        world>2 — #786); P2P still live via NCCL
#   NVLink bridge + engagement OFF     -> WARN (hardware idle; ~15% decode
#                                        left on the table per the #77 A/B)
#   PCIe-P2P-capable + engagement OFF  -> INFO (launcher boots auto-enable;
#                                        direct compose users can opt in)
#   pcie_p2p forced, grant UNVERIFIED  -> WARN (looked engaged, wasn't — #688;
#                                        driver didn't confirm peer access)

# GPU count (host).
p2p_gpu_count() {
  nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0
}

# Host capability: "nvlink" | "pcie_p2p" | "none".
# NVLink probe matches detect_nvlink.sh's auto path (topo -m, \bNV<n>\b).
p2p_host_capability() {
  local count="${1:-$(p2p_gpu_count)}"
  if [[ "${count:-0}" -lt 2 ]]; then
    echo none
    return 0
  fi
  if nvidia-smi topo -m 2>/dev/null | grep -qP '\bNV[0-9]+\b'; then
    echo nvlink
    return 0
  fi
  if _p2p_pairs_ok; then
    echo pcie_p2p
    return 0
  fi
  echo none
}

# True when nvidia-smi reports working P2P between ALL GPU pairs. Parser
# mirrors detect_nvlink.sh `_pcie_p2p_available` (stock GeForce drivers report
# CNS; only a patched driver on a P2P-capable layout reports OK).
_p2p_pairs_ok() {
  nvidia-smi topo -p2p r 2>/dev/null | awk '
    $1 ~ /^GPU[0-9]+$/ {
      hasX = 0
      for (i = 2; i <= NF; i++) if ($i == "X") hasX = 1
      if (!hasX) next
      rows++
      for (i = 2; i <= NF; i++) if ($i != "X" && $i != "OK") bad = 1
    }
    END { exit (rows > 0 && !bad) ? 0 : 1 }
  '
}

# NVIDIA kernel-module flavor: "open" | "proprietary" | "unknown". The open
# kernel modules (`nvidia-open`, OR a fork like aikitoria/open-gpu-kernel-modules)
# report license "Dual MIT/GPL"; the closed/proprietary module reports "NVIDIA".
# This is the reliable, actionable signal for GeForce P2P triage — a proprietary
# module REFUSES peer access; the open modules can GRANT it. It is host-side
# (needs /lib/modules → modinfo), so it belongs here / in report.sh, not in the
# in-container detect_nvlink.sh boot path.
#   IMPORTANT: this canNOT fingerprint the aikitoria patch specifically. That
#   fork is the open modules with the P2P block removed — identical license,
#   version, and filename to stock nvidia-open. The functional proof a
#   P2P-enabling module is actually working is `topo -p2p rw = OK`
#   (_p2p_pairs_ok), NOT this flavor probe. So we report open-vs-proprietary and
#   pair it with the topo result; we never claim "aikitoria detected".
p2p_driver_flavor() {
  local lic
  lic="$(modinfo -F license nvidia 2>/dev/null)"
  case "$lic" in
    *NVIDIA*)    echo proprietary ;;
    *GPL*|*MIT*) echo open ;;
    *)           echo unknown ;;
  esac
}

# Engagement classifier — PURE (takes the boot-trail/env text on stdin so the
# caller decides where it comes from and tests can feed fixtures).
# report.sh already gathers exactly this text: the container's `[nvlink]`
# boot lines + resolved NCCL_P2P*/NVLINK_MODE env + vLLM's own custom-AR
# gate line when present.
# Prints: "on" | "nccl_only" | "off" | "unknown" | "requested".
p2p_classify_engagement() {
  local text
  text="$(cat)"
  # A forced PCIe-P2P request whose driver grant we could NOT confirm: the
  # NCCL/all-reduce config is applied but peer access is unverified. Must NOT
  # read as "on" — that's the false-engaged bug from #688. Checked FIRST so it
  # wins over the "custom all-reduce" signal the forced path also carries.
  case "$text" in
    *"P2P REQUESTED (UNVERIFIED)"*) echo requested; return 0 ;;
  esac
  # vLLM runtime veto (#786): at world_size>2 without NVLink, vLLM disables its
  # custom all-reduce regardless of peer access — its gate queries NVML for
  # NVLink only. A pre-#786 boot trail still asserts "custom all-reduce ON" in
  # that case, so this check must run BEFORE the "on" match. P2P itself stays
  # live via NCCL peer transfers → a distinct state, not "on" and not "off".
  # Matches vLLM's log line AND the post-#786 decider trail wording.
  case "$text" in
    *"Custom allreduce is disabled"*|*"custom all-reduce engine-gated"*)
      case "$text" in
        *"P2P off"*|*"using PCIe mode"*|*"forcing PCIe mode"*|*NCCL_P2P_DISABLE=1*)
          echo off; return 0 ;;
      esac
      echo nccl_only; return 0 ;;
  esac
  # The [nvlink] decision trail is authoritative (it states what the boot
  # resolved, post-override); env is the fallback for pre-trail entrypoints.
  case "$text" in
    *"custom all-reduce ON"*|*"enabling NVLink mode"*) echo on; return 0 ;;
  esac
  case "$text" in
    *"P2P off"*|*"using PCIe mode"*|*"forcing PCIe mode"*) echo off; return 0 ;;
  esac
  case "$text" in
    *NCCL_P2P_DISABLE=1*) echo off; return 0 ;;
    *NCCL_P2P_LEVEL=*)    echo on;  return 0 ;;
  esac
  echo unknown
}

# Pure verdict matrix: p2p_verdict <gpu_count> <capability> <engagement>.
# Prints zero or one line; silent cases print nothing (exit 0 always).
p2p_verdict() {
  local count="$1" cap="$2" eng="$3" state
  [[ "${count:-0}" -ge 2 ]] || return 0
  # A forced-but-unverified P2P request is worth flagging EVEN when the
  # capability probe says "none": the request asked for P2P, the driver didn't
  # confirm it. This is the #688 case (looked engaged, wasn't) — never silent.
  if [[ "$eng" == "requested" ]]; then
    echo "⚠ interconnect WARN: NVLINK_MODE=pcie_p2p forced PCIe P2P on, but nvidia-smi does not report peer access as OK — the driver likely refused it (a closed GeForce driver disables P2P; the open kernel modules or a patched module + a P2P-capable board enable it). NCCL falls back, so throughput ≈ P2P-off. Verify: nvidia-smi topo -p2p rw. Guide: docs/PCIE_P2P.md"
    return 0
  fi
  [[ "$cap" != "none" ]] || return 0
  case "$eng" in
    on|nccl_only) state="" ;;
    off)     state="is running with P2P OFF" ;;
    unknown) state="shows no P2P engagement signal (no [nvlink] boot line / NCCL env)" ;;
    *)       return 0 ;;
  esac
  case "$cap:$eng" in
    nvlink:on)
      echo "✓ interconnect: NVLink engaged (custom all-reduce ON)" ;;
    pcie_p2p:on)
      echo "✓ interconnect: PCIe P2P engaged (patched driver, custom all-reduce ON)" ;;
    pcie_p2p:nccl_only)
      echo "✓ interconnect: PCIe P2P engaged via NCCL peer transfers. vLLM auto-disabled its custom all-reduce kernel — expected at >2 PCIe-only GPUs (its gate checks NVLink, not peer access; #786), NOT a misconfiguration. P2P is still active on the NCCL path." ;;
    nvlink:nccl_only)
      echo "✓ interconnect: P2P engaged via NCCL peer transfers, but vLLM disabled its custom all-reduce — the NVLink mesh is not fully connected across all GPUs (vLLM requires full 1-hop connectivity at world>2). Peer transfers remain active." ;;
    nvlink:*)
      echo "⚠ interconnect WARN: an NVLink bridge is present on this host but the serving container ${state} — the bridge is idle. On modern vLLM the NVLink win is workload-shaped: small on decode (~3-5%) but large on prefill / long-context (+35-49%) (BENCHMARKS #698; the ~15% #77 figure was the older v7.72.2 image). Boot via launch.sh/switch.sh (auto-detects) or set NVLINK_MODE=force_on; if auto-detect misses on your rig, please file it. Full guide: docs/PCIE_P2P.md" ;;
    pcie_p2p:*)
      echo "ℹ interconnect: this driver reports PCIe P2P available (patched driver / P2P-capable layout) but the serving container ${state}. Launcher boots auto-enable it; for direct docker compose set NVLINK_MODE=pcie_p2p (+10–22% code TPS measured on DUAL-card rigs, #91/#295; at >2 GPUs the gain path is NCCL-only — see docs/PCIE_P2P.md). Full guide: docs/PCIE_P2P.md" ;;
  esac
}

# ── Transfer-verified P2P (folded from #787) ─────────────────────────────────
# Every other signal in this file bottoms out in a driver ASSERTION (topo -p2p,
# modinfo, a clean boot). vLLM ships a functional check that actually moves
# bytes — `gpu_p2p_access_check()` (enabled by VLLM_SKIP_P2P_CHECK=0): IPC
# write/read-back across every directed pair, cached as JSON. We READ that
# cache when it exists; we never run the check ourselves (it's off by default
# upstream). Advertised-but-broken peer access fails it — the class no driver
# query can see (#688's cousin, closed by construction).

# Parse a gpu_p2p_access_cache JSON on stdin ({"0->1": true, ...}).
# Prints "ok total" (directed pairs); silent when no pair entries parse.
p2p_transfer_cache_parse() {
  # grep exits 1 on a cache with no pair entries — that's "nothing to report",
  # not an error, and must not trip a pipefail caller.
  { grep -oE '"[0-9]+->[0-9]+"[[:space:]]*:[[:space:]]*(true|false)' || true; } | awk '
    { total++; if ($0 ~ /true/) ok++ }
    END { if (total > 0) printf "%d %d\n", ok, total }'
}

# Newest host-side cache file, if any (container-side lookup is the caller's
# job — report.sh checks the serving container too).
p2p_transfer_cache_file() {
  ls -t "$HOME"/.cache/vllm/gpu_p2p_access_cache_for_*.json 2>/dev/null | head -1 || true
}

# Pure formatter: p2p_transfer_verdict <ok> <total> <src> — one line or nothing.
p2p_transfer_verdict() {
  local ok="${1:-0}" total="${2:-0}" src="${3:-}"
  [[ "${total:-0}" -gt 0 ]] || return 0
  if [[ "$ok" -eq "$total" ]]; then
    echo "✓ interconnect: P2P verified by TRANSFER — ${ok}/${total} directed pairs OK (vLLM gpu_p2p_access_check cache: ${src}). Measured data-path result, not a driver assertion."
  else
    echo "⚠ interconnect WARN: P2P transfer check has only ${ok}/${total} directed pairs OK — peer access is advertised but BROKEN on some pairs (vLLM cache: ${src}). NCCL silently falls back on failing pairs → throughput ≈ P2P-off there. Guide: docs/PCIE_P2P.md"
  fi
}
