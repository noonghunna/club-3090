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
#   NVLink bridge + engagement OFF     -> WARN (hardware idle; ~15% decode
#                                        left on the table per the #77 A/B)
#   PCIe-P2P-capable + engagement OFF  -> INFO (launcher boots auto-enable;
#                                        direct compose users can opt in)

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

# Engagement classifier — PURE (takes the boot-trail/env text on stdin so the
# caller decides where it comes from and tests can feed fixtures).
# report.sh already gathers exactly this text: the container's `[nvlink]`
# boot lines + resolved NCCL_P2P*/NVLINK_MODE env.
# Prints: "on" | "off" | "unknown".
p2p_classify_engagement() {
  local text
  text="$(cat)"
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
  [[ "$cap" != "none" ]] || return 0
  case "$eng" in
    on)      state="" ;;
    off)     state="is running with P2P OFF" ;;
    unknown) state="shows no P2P engagement signal (no [nvlink] boot line / NCCL env)" ;;
    *)       return 0 ;;
  esac
  case "$cap:$eng" in
    nvlink:on)
      echo "✓ interconnect: NVLink engaged (custom all-reduce ON)" ;;
    pcie_p2p:on)
      echo "✓ interconnect: PCIe P2P engaged (patched driver, custom all-reduce ON)" ;;
    nvlink:*)
      echo "⚠ interconnect WARN: an NVLink bridge is present on this host but the serving container ${state} — the bridge is idle, leaving ~15% decode on the table (controlled A/B, BENCHMARKS #77). Boot via launch.sh/switch.sh (auto-detects) or set NVLINK_MODE=force_on; if auto-detect misses on your rig, please file it. Full guide: docs/PCIE_P2P.md" ;;
    pcie_p2p:*)
      echo "ℹ interconnect: this driver reports PCIe P2P available (patched driver / P2P-capable layout) but the serving container ${state}. Launcher boots auto-enable it; for direct docker compose set NVLINK_MODE=pcie_p2p (+10–22% code TPS measured, #91/#295). Full guide: docs/PCIE_P2P.md" ;;
  esac
}
