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

# GPU count (host). Prints EXACTLY one line: a non-negative integer.
#
# The `|| echo 0` this used to carry was not a fallback, it was the bug (#1279):
# `grep -c` PRINTS `0` *and* exits 1 when it matches nothing, so on any rig where
# `nvidia-smi -L` fails or lists no GPUs, BOTH sides fired and the function
# returned the two-line string "0\n0". Every consumer does arithmetic on it
# (`[[ "$(p2p_gpu_count)" -ge 2 ]]`), which then dies with
# `[[: 0 0: syntax error in expression` — pasted straight into the diagnostic
# report a user is about to send us. A caller-side `|| echo 0` cannot rescue it
# either: the function already exited 0, its last command having been the echo.
#
# So: capture the count, keep grep's no-match exit off `set -e`/pipefail, and
# print one integer whatever happened. `command grep` because the interactive
# shell shims grep to ugrep.
p2p_gpu_count() {
  local n
  n="$(nvidia-smi -L 2>/dev/null | command grep -c '^GPU ')" || true
  [[ "$n" =~ ^[0-9]+$ ]] || n=0
  printf '%s\n' "$n"
}

# Host capability: "nvlink" | "pcie_p2p" | "none".
# NVLink probe matches detect_nvlink.sh's auto path (topo -m, \bNV<n>\b).
p2p_host_capability() {
  local count="${1:-$(p2p_gpu_count)}"
  if [[ "${count:-0}" -lt 2 ]]; then
    echo none
    return 0
  fi
  if nvidia-smi topo -m 2>/dev/null | command grep -qP '\bNV[0-9]+\b'; then
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
# Prints: "on" | "nccl_only_operator" | "nccl_only_gated" | "nccl_only_degraded"
#         | "nccl_only_nolib" | "off" | "unknown" | "requested".
# Reads detect_nvlink.sh's machine-readable STATE line for OUR configuration and
# vLLM's own messages for what the ENGINE did; our prose is only consulted on the
# legacy path (containers predating STATE, SGLang, llama.cpp). See #1332.
p2p_classify_engagement() {
  local text state engine_verdict transport kernel cause verified
  text="$(cat)"

  # ── Step 0: slice to the CURRENT BOOT ────────────────────────────────────
  # `restart: unless-stopped` means one `docker logs` can hold several boots.
  # Combining our state from one boot with vLLM's veto from another reports the
  # wrong rig — and does so silently. detect_nvlink.sh's STATE line marks the
  # start of a boot's record, so the LAST one wins and only engine lines after
  # it describe the current boot. (Callers must not pre-filter with head/-m1.)
  # ⚠️ LITERAL matching throughout. The marker contains "[nvlink]", which every
  # regex engine reads as a bracket expression matching one of n/v/l/i/k — so a
  # regex slice silently never matches and the epoch is not applied at all. Use
  # grep -F to find it and awk index() to slice: take the log reversed, emit up
  # to and including the first marker found (= the LAST one in real order), then
  # un-reverse. That yields exactly the current boot's record.
  # ⚠️ v=1 specifically, not "STATE v=". A future v=2 with different fields would
  # otherwise be parsed under v=1 rules and silently mis-scored; an unknown version
  # must fall through to the legacy cascade instead.
  state="$(printf '%s\n' "$text" | command grep -F '[nvlink] STATE v=1 ' | tail -n 1)"
  if [ -n "$state" ]; then
    text="$(printf '%s\n' "$text" | tac \
            | awk 'index($0,"[nvlink] STATE v=1 ")>0 {print; exit} {print}' | tac)"
  fi

  # ── Step 1: a forced-but-unconfirmed request outranks everything (#688) ───
  # Checked before the STATE fields: the config was applied but the driver never
  # confirmed peer access, and reporting that as a healthy configured state is
  # the false-engaged bug.
  case "$text" in
    *"P2P REQUESTED (UNVERIFIED)"*) echo requested; return 0 ;;
  esac
  case "$state" in *" verified=no"*) echo requested; return 0 ;; esac

  # ── Step 2: what the ENGINE actually did (FOREIGN prose — vLLM's, not ours) ──
  # ⚠️ The specific messages are matched first, then a CATCH-ALL on the shared
  # prefix. vLLM has more veto reasons than we enumerate ("unsupported world
  # size", "spans across nodes", and whatever a future release adds); an
  # enumerated list silently scores those as "the engine said nothing", which
  # resolves to our configured value and reports the kernel ON while it is off.
  # The catch-all is the whole point — never replace it with a closed list.
  engine_verdict=""
  case "$text" in
    *"lacks GPU P2P capability"*|*"P2P test failed"*)       engine_verdict=degraded ;;
    *"missing custom allreduce library"*)                   engine_verdict=nolib ;;
    *"not supported on"*"PCIe-only GPUs"*)                  engine_verdict=gated ;;
    *"Custom allreduce is disabled"*)                       engine_verdict=gated ;;
    # ⚠️ LAST, and only reachable when no veto matched above — every veto message
    # ends "To silence this warning, specify disable_custom_all_reduce=True",
    # which contains this substring and must not be read as an operator choice.
    #
    # This arm is why the engine's argv is consulted at all. Our STATE line says
    # what detect_nvlink.sh DECIDED; it cannot know that a compose passed the flag
    # on its own. `models/qwen3.8-27b/vllm/compose/dual/fp8/dflash2.yml` does
    # exactly that on every non-NVLink rig, so without this the verdict reads
    # "custom all-reduce ON" while the kernel is off — the #922 false-ON, on a
    # shipped slug, and a regression against the legacy cascade.
    *"disable_custom_all_reduce=True"*) engine_verdict=operator ;;
  esac

  # ── Step 3: OUR configured state, as data ────────────────────────────────
  if [ -n "$state" ]; then
    transport="$(printf '%s' "$state" | sed -n 's/.* transport=\([a-z0-9_-]*\).*/\1/p')"
    kernel="$(printf '%s'    "$state" | sed -n 's/.* kernel=\([a-z0-9_-]*\).*/\1/p')"
    cause="$(printf '%s'     "$state" | sed -n 's/.* cause=\([a-z0-9_-]*\).*/\1/p')"
    [ "$transport" = "off" ] && { echo off; return 0; }
    # The engine overrides our intent when it spoke; it is the one that ran.
    case "$engine_verdict" in
      degraded) echo nccl_only_degraded; return 0 ;;
      nolib)    echo nccl_only_nolib;    return 0 ;;
      gated)    echo nccl_only_gated;    return 0 ;;
      operator) echo nccl_only_operator; return 0 ;;
    esac
    # The engine has not spoken yet. If we predicted it will veto (>2 PCIe-only
    # GPUs, or a partial NVLink mesh), say so rather than claiming the kernel is
    # running — claiming AR ON there is the #786 bug.
    case "$state" in *" gate=expected_veto"*) echo nccl_only_gated; return 0 ;; esac
    if [ "$kernel" = "on" ]; then echo on; return 0; fi
    case "$cause" in
      operator)      echo nccl_only_operator; return 0 ;;
      transport_off) echo off;               return 0 ;;
      # An unrecognised cause is a PARSE failure, not a healthy state. Defaulting
      # it to nccl_only_gated rendered a ✓ verdict for a line we did not understand.
      *)             echo unknown;           return 0 ;;
    esac
  fi

  # ── Step 4: LEGACY fallback — no STATE line ──────────────────────────────
  # Reached by containers started before this shipped, by SGLang and llama.cpp
  # (which never emit one), and after log rotation. It is the old prose cascade,
  # unchanged, and it is PERMANENT for the non-vLLM engines — do not delete it.
  # ⚠️ Our own trail phrases are load-bearing ONLY on this path; changing them
  # breaks classification of older containers.
  case "$text" in
    *"lacks GPU P2P capability"*|*"P2P test failed"*)
      case "$text" in *"P2P off"*|*"using PCIe mode"*|*"forcing PCIe mode"*|*NCCL_P2P_DISABLE=1*) echo off; return 0 ;; esac
      echo nccl_only_degraded; return 0 ;;
  esac
  case "$text" in
    *"missing custom allreduce library"*) echo nccl_only_nolib; return 0 ;;
  esac
  case "$text" in
    *"Custom allreduce is disabled"*|*"custom all-reduce engine-gated"*)
      case "$text" in *"P2P off"*|*"using PCIe mode"*|*"forcing PCIe mode"*|*NCCL_P2P_DISABLE=1*) echo off; return 0 ;; esac
      echo nccl_only_gated; return 0 ;;
  esac
  case "$text" in
    *"disable_custom_all_reduce=True"*|*"--disable-custom-all-reduce"*|*"custom all-reduce OFF by operator request"*)
      case "$text" in *"P2P off"*|*"using PCIe mode"*|*"forcing PCIe mode"*|*NCCL_P2P_DISABLE=1*) echo off; return 0 ;; esac
      echo nccl_only_operator; return 0 ;;
  esac
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
  # ⚠️ The cause of a custom-AR-off state is READ FROM THE SIGNAL, never inferred
  # from GPU count. An earlier cut of this branched on `count <= 2` to decide
  # that the operator must have disabled it — which mislabelled vLLM's
  # P2P-TEST-FAILED veto (also 2 GPUs, also custom-AR off) as a healthy
  # deliberate choice, and still misattributed an operator-disabled 2-GPU
  # container on a 4-GPU host, because report.sh/bench.sh pass the HOST count,
  # not the container's world size. The classifier now emits the cause directly.
  case "$eng" in
    nccl_only_degraded)
      echo "⚠️ interconnect WARN: the engine disabled its custom all-reduce because peer access is MISSING or its own P2P TEST FAILED — this is not an operator choice and not a healthy state. The driver may still advertise peer access (topo -p2p: OK) while transfers do not work; that is the #873 class. Validate with scripts/p2p-validate.sh, and see docs/PCIE_P2P.md §4a/§7. NCCL is falling back, so throughput will sit at P2P-off levels."
      return 0 ;;
    nccl_only_operator)
      echo "✓ interconnect: P2P engaged via NCCL peer transfers, custom all-reduce OFF by operator request (--disable-custom-all-reduce / DISABLE_CUSTOM_ALL_REDUCE=1 — the #922 mitigation). A deliberate, healthy state: you keep the NCCL transport and skip the custom kernel. If you did NOT set it, check the engine log."
      return 0 ;;
    nccl_only_nolib)
      echo "ℹ interconnect: P2P transport is up, but this image has no custom all-reduce library to run (non-GPU or stripped build). Peer transfers still go via NCCL."
      return 0 ;;
  esac
  case "$eng" in
    on|nccl_only_gated) state="" ;;
    off)     state="is running with P2P OFF" ;;
    unknown) state="shows no P2P engagement signal (no [nvlink] boot line / NCCL env)" ;;
    *)       return 0 ;;
  esac
  case "$cap:$eng" in
    nvlink:on)
      echo "✓ interconnect: NVLink engaged (custom all-reduce ON)" ;;
    pcie_p2p:on)
      echo "✓ interconnect: PCIe P2P engaged (patched driver, custom all-reduce ON)" ;;
    pcie_p2p:nccl_only_gated)
      echo "✓ interconnect: PCIe P2P engaged via NCCL peer transfers. vLLM auto-disabled its custom all-reduce kernel — expected at >2 PCIe-only GPUs (its gate checks NVLink, not peer access; #786), NOT a misconfiguration. P2P is still active on the NCCL path." ;;
    nvlink:nccl_only_gated)
      echo "✓ interconnect: P2P engaged via NCCL peer transfers, but vLLM disabled its custom all-reduce — the NVLink mesh is not fully connected across all GPUs (vLLM requires full 1-hop connectivity at world>2). Peer transfers remain active." ;;
    nvlink:*)
      echo "⚠ interconnect WARN: an NVLink bridge is present on this host but the serving container ${state} — the bridge is idle. On modern vLLM the NVLink win is workload-shaped: small on decode (~3-5%) but large on prefill / long-context (+35-49%) (BENCHMARKS #698; the ~15% #77 figure was the older v7.72.2 image). Boot via launch.sh/switch.sh (auto-detects) or set NVLINK_MODE=force_on; if auto-detect misses on your rig, please file it. Full guide: docs/PCIE_P2P.md" ;;
    pcie_p2p:*)
      echo "ℹ interconnect: this driver reports PCIe P2P available (patched driver / P2P-capable layout) but the serving container ${state}. Every multi-GPU vLLM compose auto-enables P2P from its own entrypoint — launcher AND raw 'docker compose' alike — so a capable driver plus a P2P-off container usually means one of: NVLINK_MODE=force_off is set (.env or environment), the container predates the running compose (recreate it), or this engine's compose doesn't do interconnect detection at all (the llama.cpp-family and SGLang composes don't). SGLang uses NCCL, so NCCL_P2P_DISABLE=0 opts it in. llama.cpp uses NCCL only for --split-mode row/tensor all-reduce (the mainline image links libnccl); the default --split-mode layer path uses no NCCL and calls cudaDeviceEnablePeerAccess() directly, gated on the GGML_CUDA_P2P env var, which is OPT-IN and OFF by default: set GGML_CUDA_P2P=1 to enable peer access, unset it to disable (that pair is also the only valid way to A/B P2P on llama.cpp — NVLINK_MODE and NCCL_P2P_DISABLE are both no-ops there). Note that with the default --split-mode layer almost nothing crosses between GPUs (~7 MB/s vs ~1400 MB/s on row/tensor), so even with GGML_CUDA_P2P=1 there is little for P2P to accelerate and a null A/B result is expected. To skip the diagnosis and force it on any vLLM slug: NVLINK_MODE=pcie_p2p. +10–22% code TPS measured on DUAL-card rigs, #91/#295 — those are vLLM numbers; at >2 GPUs the gain path is NCCL-only. ⚠️ On llama.cpp the measured gain is ZERO: a community 3×3090 rig with transfer-verified P2P (p2pBandwidthLatencyTest: 6.08 → 13.17 GB/s on its best pair, a 2.2x jump) saw NO TPS change at all with GGML_CUDA_P2P=1 vs unset on --split-mode tensor at ~150 TPS — single-stream llama.cpp simply is not interconnect-bound, so P2P has no headroom to give back. Don't expect the vLLM numbers to transfer. Full guide: docs/PCIE_P2P.md" ;;
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
# ── P2P opportunity hint (#873) ──────────────────────────────────────────────
# Everything above answers "is P2P on?". This answers the question a user with
# two cards and no P2P actually has: "could it be, and is it worth it?".
#
# Deliberately tiered rather than one line, because the naive "you could enable
# P2P!" is WRONG for most rigs that would see it: cards on separate root
# complexes can never peer regardless of driver, and a BAR1 smaller than VRAM
# cannot back the patched module's static full-VRAM mapping (#734) — telling
# either of those users to build a DKMS module wastes their evening. The
# reference 2x3090 is itself the BAR1 case, which is why B outranks A.
#
# Prints ONE informational line, or nothing. Never a warning: not having P2P is
# not a fault, and this must never read as "your rig is misconfigured".

# GPU-GPU link class from `topo -m`: "same_root" (PHB/PIX/PXB — peer traffic can
# stay below the CPU) | "split" (SYS/NODE — crosses the SMP/host-bridge boundary,
# unreachable for P2P) | "unknown". Off-diagonal cells only.
p2p_topology_class() {
  nvidia-smi topo -m 2>/dev/null | awk '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) ngpu++; next }
    $1 ~ /^GPU[0-9]+$/ {
      for (i = 2; i <= ngpu + 1; i++) {
        if ($i == "X") continue
        if ($i ~ /^(PHB|PIX|PXB)$/) near = 1
        else if ($i ~ /^(SYS|NODE)$/) far = 1
      }
    }
    END {
      if (far) print "split"
      else if (near) print "same_root"
      else print "unknown"
    }'
}

# Smallest BAR1 and its card's VRAM, as "<bar1MiB> <fbMiB>"; empty when the
# driver does not report the fields. Mirrors detect_nvlink.sh _bar1_undersized,
# but reports the numbers rather than a verdict (report.sh wants to print them).
p2p_bar1_min() {
  nvidia-smi -q -d MEMORY 2>/dev/null | awk '
    /^GPU /             { idx++; sect = ""; next }
    /FB Memory Usage/   { sect = "fb";   next }
    /BAR1 Memory Usage/ { sect = "bar1"; next }
    /Memory Usage/      { sect = "";     next }
    sect != "" && $1 == "Total" && $3 ~ /^[0-9]+$/ {
      if (sect == "fb") fb[idx] = $3; else bar1[idx] = $3
      sect = ""
    }
    END {
      for (i = 1; i <= idx; i++)
        if (fb[i] > 0 && bar1[i] > 0 && (best == 0 || bar1[i] < best)) { best = bar1[i]; bfb = fb[i] }
      if (best > 0) printf "%d %d\n", best, bfb
    }'
}

# True (0) when `topo -p2p r` reports CNS on any pair — the stock GeForce
# driver's software refusal, the one gate a patched module actually lifts.
p2p_reports_cns() {
  nvidia-smi topo -p2p r 2>/dev/null | command grep -qE '(^|[[:space:]])CNS([[:space:]]|$)'
}

# p2p_opportunity_hint <gpu_count> <host_capability>
# One line when there is something true to say; silent otherwise.
p2p_opportunity_hint() {
  local count="${1:-0}" cap="${2:-none}"
  [[ "${count:-0}" -ge 2 ]] || return 0        # single card: meaningless
  [[ "$cap" == "none" ]] || return 0           # nvlink / pcie_p2p: already covered

  local topo; topo="$(p2p_topology_class)"
  if [[ "$topo" == "split" ]]; then
    echo "ℹ interconnect: ${count} GPUs, but they sit on SEPARATE root complexes (topo -m: SYS/NODE) — peer-to-peer is unreachable on this layout regardless of driver or BIOS, so there is nothing to enable here. If you can re-slot them onto the same root complex, docs/PCIE_P2P.md §1-§2 covers what to aim for."
    return 0
  fi
  [[ "$topo" == "same_root" ]] || return 0     # unknown topology: say nothing

  local bar1 fb; read -r bar1 fb <<<"$(p2p_bar1_min)"
  if [[ -n "${bar1:-}" ]] && [[ "${bar1:-0}" -gt 0 ]] && (( bar1 * 2 < fb )); then
    echo "ℹ interconnect: ${count} GPUs on a P2P-capable layout (same root complex) with peer access OFF, and BAR1 is ${bar1} MiB against ${fb} MiB of VRAM. The patched-driver P2P path maps the FULL VRAM aperture through BAR1, so it cannot work until that changes — this is the gate to fix first, not the driver. Check which one you're behind: the BIOS toggles (Above 4G Decoding + Re-Size BAR, docs/PCIE_P2P.md §4), or a pre-ReBAR VBIOS ceiling — \`sudo lspci -vv -s <bus> | grep -A4 'Physical Resizable BAR'\`, and if \`supported:\` tops out at 256MB it is firmware (#734)."
    return 0
  fi

  if p2p_reports_cns; then
    echo "ℹ interconnect: ${count} GPUs on a P2P-capable layout (same root complex) with peer access OFF — \`topo -p2p\` reports CNS, which is the stock driver refusing P2P on GeForce cards rather than a hardware limit. A patched kernel module can unlock it."
      echo "  PREFILL is the transport gain, and it scales with world size: +14.4% @10K on dual 3090 at TP=2 (#922), +74.7% Qwen / +84.9% Ling at TP=4 on four cards (disc #921). Repeatable, CV <=1%."
      echo "  DECODE comes from a DIFFERENT mechanism — vLLM's custom all-reduce kernel, NOT the transport. Three rigs measure +7.7% to +12.5% with that kernel on; with it forced off the decode delta sits inside run-to-run noise. Our composes AUTO-ENABLE it when P2P is detected, so on this stack you get both halves."
      echo "  ⚠️ ENGINE MATTERS. Those gains are vLLM tensor-parallel numbers. On the SHIPPED llama.cpp image P2P buys essentially NOTHING at any split mode: +0.2% decode / ~+1% prefill with \`-sm layer\` (disc #921), and ~noise even with \`-sm tensor\` (disc #903). Two separate reasons: layer split forwards activations with plain copies so there is no all-reduce at all, AND the ggml-org image is built WITHOUT NCCL — it logs 'NCCL not compiled in; falling back to internal AllReduce' and that fallback does not use the peer path, so even tensor split cannot benefit. Verified: zero NCCL symbols in server-cuda-b10236. If you serve only GGUF, this lever is not worth a DKMS rebuild unless you also rebuild llama.cpp with -DGGML_CUDA_NCCL=ON."
      echo "  ⚠️ One Ampere rig saw the custom all-reduce return WRONG DATA over a patched peer path — fast, plausible TPS, garbage text (#922). CAUSE IS UNSETTLED: the same signature appeared on a rig with P2P OFF and was fixed by reseating cables (#751), and two other patched rigs run the kernel cleanly. If you hit it, \`--disable-custom-all-reduce\` is the one-line test — it keeps the prefill win. Read real generated output before trusting any verdict line."
      echo "  It is an optional enthusiast lever shipped as a custom DKMS module you rebuild on every driver bump. If you want it: docs/PCIE_P2P.md §5 (setup) and §7a (failure modes)."
  fi
}

