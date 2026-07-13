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
# boot lines + resolved NCCL_P2P*/NVLINK_MODE env.
# Prints: "on" | "off" | "unknown".
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
