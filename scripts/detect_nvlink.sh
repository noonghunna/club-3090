#!/bin/bash
# NVLink / PCIe-P2P detection + override. Sources NVLINK_MODE from env (default: auto).
# Exports: _NVLINK_ENABLED (0/1) — ⚠️ this is the CUSTOM ALL-REDUCE decision, not
# the transport. It has always been consumed as "should the entrypoint pass
# --disable-custom-all-reduce?", so DISABLE_CUSTOM_ALL_REDUCE is applied here and
# every gate that reads this name honours it by construction, including composes
# this repo does not ship (#1332). The TRANSPORT is NCCL_P2P_LEVEL /
# NCCL_P2P_DISABLE, which this script also exports; read those, not this, if you
# want to know whether peer transfers are on.
# Also exports _CUSTOM_AR_ENABLED (same value, unambiguous name) and emits one
# machine-readable `[nvlink] STATE v=1 …` line per boot, which is what
# scripts/lib/p2p-state.sh parses. Our prose is NOT a classification input.
# Handles 2-GPU setups (single NVLink bridge) and N-GPU setups (e.g. 2 bridges on 4 cards).
#
# NVLINK_MODE values:
#   auto       — detect a fast P2P interconnect via nvidia-smi (default): NVLink (topo -m)
#                OR, failing that, PCIe P2P that `nvidia-smi topo -p2p r` reports as OK
#                between all pairs — i.e. a patched consumer-GPU driver (tinygrad/geohot/
#                aikitoria) on a P2P-capable layout (shared root complex / switch). Neither
#                => P2P off. NOTE: stock GeForce drivers software-disable P2P (report CNS),
#                and cards on separate root complexes can't P2P — both correctly stay off.
#   force_on   — assert NVLink present (NCCL_P2P_LEVEL=NVL).
#   force_off  — no P2P at all (NCCL_P2P_DISABLE=1).
#   pcie_p2p   — FORCE the PCIe P2P config on (NCCL_P2P_LEVEL=PHB or your own,
#                custom all-reduce ON), bypassing auto-detect. It ALSO probes
#                `topo -p2p` and reports honestly: "P2P ENABLED" when the driver
#                confirms peer access, "P2P REQUESTED (UNVERIFIED)" + a warning
#                when it doesn't (a closed GeForce driver refuses P2P; the open
#                kernel modules / a patched module + a P2P-capable board enable
#                it). The config is forced either way — this is the escape hatch
#                for a rig whose probe is stricter than reality — but we never
#                claim "engaged" without evidence. See club-3090 #290, #688.
#
# DISABLE_CUSTOM_ALL_REDUCE=1 — ORTHOGONAL operator override: keep the peer
#   TRANSPORT exactly as resolved above, but turn vLLM's own custom all-reduce
#   kernel off. This is the #922 mitigation, and until #1332 the only way to
#   reach it was editing a shipped compose — NVLINK_MODE=force_off drops the
#   transport too and throws away the prefill half of the win (a measured
#   +10.7% prefill / -8.6% TTFT on the reporting rig). Three rigs have now been
#   harmed by the custom kernel over a PATCHED PCIe peer path — two with silent
#   wrong output (#922), one with a hard machine reset under sustained decode
#   (#1332) — while other rigs run it clean, so this stays an opt-in, not a
#   default. Transport and kernel are two decisions; this is the second one.
#   Exports _CUSTOM_AR_ENABLED (0/1) — that, not _NVLINK_ENABLED, is what a
#   compose must gate --disable-custom-all-reduce on.
#
# On every path that ends in PCIe P2P (not NVLink) we also check BAR1 against VRAM
# and warn when the aperture is too small to back the patched driver's static-BAR1
# mapping — the one failure mode that HANGS NCCL instead of degrading. See #873.

# ── World size, read from the entrypoint's OWN argv ──────────────────────────
# A sourced script inherits "$@", and every compose runs `bash -c <script> --
# <command...>` with `--tensor-parallel-size` in the command list. We cannot use
# ${TP:-}: zero composes forward TP into `environment:` (compose interpolates it
# at render time), so it never reaches the container. And we must not use the
# GPU COUNT — a single-card slug with `count: all` sees every card on the host,
# which would claim an all-reduce that never runs. vLLM's default is 1.
# Custom all-reduce operates over the TP group, so TP is the relevant width.
_world_size() {
  local i prev=""
  for i in "$@"; do
    case "$prev" in --tensor-parallel-size|-tp) printf '%s' "$i"; return 0 ;; esac
    case "$i" in
      --tensor-parallel-size=*) printf '%s' "${i#*=}"; return 0 ;;
      -tp=*)                    printf '%s' "${i#*=}"; return 0 ;;
    esac
    prev="$i"
  done
  printf '1'
}
_WORLD="$(_world_size "$@")"
case "$_WORLD" in
  ""|*[!0-9]*)
    # Do not silently default. A misparsed width suppresses both the risky-shape
    # warning and the engine-veto prediction, and every symptom of that is an
    # ABSENCE — exactly the failure that cannot be told from success.
    echo "[nvlink] WARNING: could not read a numeric tensor-parallel size from the entrypoint arguments (saw '${_WORLD}'); assuming 1. The custom-all-reduce risk warning and the >2-GPU veto prediction are suppressed for this boot." >&2
    _WORLD=1 ;;
esac

NVLINK_MODE="${NVLINK_MODE:-auto}"
# A typo here must never silently resolve to "kernel stays on" — that is the
# dangerous direction (it is the arm that reset a machine in #1332), and a knob
# whose failure is indistinguishable from its success is not a knob. Hard-error,
# same contract as an invalid NVLINK_MODE.
case "${DISABLE_CUSTOM_ALL_REDUCE:-0}" in
  0|1) ;;
  *) echo "[nvlink] ERROR: invalid DISABLE_CUSTOM_ALL_REDUCE='${DISABLE_CUSTOM_ALL_REDUCE}' (must be 0 or 1)" >&2; exit 1 ;;
esac
_P2P_LEVEL=NVL   # NCCL_P2P_LEVEL used when the interconnect is up (overridden by pcie_p2p)
_NVLINK_FOUND=0  # 1 only when NVLink is actually detected or force_on'd — NOT when
                 # NCCL_P2P_LEVEL merely says NVL, which a user can set on a PCIe rig
_GPU_COUNT=$(nvidia-smi -L 2>/dev/null | command grep -c 'GPU' || echo 0)

# What may we honestly claim about vLLM's custom all-reduce? Only "ON" for <=2
# GPUs, or for a FULLY-CONNECTED NVLink mesh: vLLM hard-disables its custom
# kernel at world_size>2 unless every GPU pair is 1-hop NVLink (its gate
# queries NVML for NVLink, never peer access). Two consequences (club-3090
# #786): a >2-GPU PCIe-P2P boot must not assert it, and neither may a >2-GPU
# rig with PAIRWISE bridges — consumer 3090-class cards bridge exactly two
# cards, so 4x 3090 = 2 separate bridges = never a full mesh. P2P/NVLink still
# runs via NCCL in both cases. The auditor (p2p-state.sh) keys on both wordings.
# ⚠️ Keys on _WORLD (TP width), matching the STATE line's gate= field and vLLM's
# own world_size gate. Keyed on _GPU_COUNT it disagreed with STATE on any
# container that sees more cards than it uses (CDI, `count: all`), so the boot
# log said "engine-gated" while the machine-readable line said gate=none.
_ar_claim() {
  if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
    # Must NOT contain the literal "custom all-reduce ON" — p2p_classify_engagement
    # keys "on" off that substring, and the whole point of this branch is that the
    # kernel is off. (The classifier also sees vLLM's own disable_custom_all_reduce=True
    # and that check runs first, but the boot trail must not say the opposite either.)
    printf 'custom all-reduce OFF by operator request (DISABLE_CUSTOM_ALL_REDUCE=1; peer transport stays up — #922/#1332)'
  elif [ "${_WORLD:-1}" -gt 2 ] && [ "${_P2P_LEVEL:-NVL}" != "NVL" ]; then
    printf 'custom all-reduce engine-gated (vLLM disables its custom kernel at >2 PCIe-only GPUs — P2P runs via NCCL; #786)'
  elif [ "${_WORLD:-1}" -gt 2 ] && [ "${_NVLINK_PARTIAL:-0}" -eq 1 ]; then
    printf 'custom all-reduce engine-gated (NVLink mesh not fully connected — pairwise bridges; vLLM requires full 1-hop connectivity at world>2, so its kernel is off and NVLink/P2P runs via NCCL; #786)'
  else
    printf 'custom all-reduce ON'
  fi
}

# Full-mesh NVLink probe (>2 GPUs): every off-diagonal GPU-GPU cell in
# `topo -m` must be NV<n>. Pairwise 3090 bridges (2 bridges on 4 cards) fail
# this; NVSwitch/SXM meshes pass. Exit 0 = full mesh.
_nvlink_full_mesh() {
  nvidia-smi topo -m 2>/dev/null | awk '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) ngpu++ ; next }
    $1 ~ /^GPU[0-9]+$/ {
      rows++
      for (i = 2; i <= ngpu + 1; i++)
        if ($i != "X" && $i !~ /^NV[0-9]+$/) partial = 1
    }
    END { exit (rows > 0 && !partial) ? 0 : 1 }
  '
}

# True (0) when nvidia-smi reports working P2P between ALL GPU pairs — e.g. a patched
# consumer-GPU driver (NVIDIA's stock driver software-disables P2P → reports "CNS") on a
# P2P-capable PCIe layout. Parses `topo -p2p r`: a data row carries the self-"X" (header /
# legend rows don't, so they're skipped); ANY off-diagonal cell that isn't OK => unavailable.
_pcie_p2p_available() {
  nvidia-smi topo -p2p r 2>/dev/null | awk '
    $1 ~ /^GPU[0-9]+$/ {
      hasX = 0
      for (i = 2; i <= NF; i++) if ($i == "X") hasX = 1
      if (!hasX) next                                  # header row (no self-X) — skip
      rows++
      for (i = 2; i <= NF; i++) if ($i != "X" && $i != "OK") bad = 1
    }
    END { exit (rows > 0 && !bad) ? 0 : 1 }
  '
}

# Cards whose BAR1 aperture is far smaller than their VRAM, as "GPU<n> BAR1=<x>MiB
# VRAM=<y>MiB" lines (empty when every card is fine). The patched-consumer-driver
# P2P path maps the WHOLE VRAM aperture through BAR1 (static BAR1 mapping) and DMAs
# straight to the peer's physical addresses, so a small BAR1 cannot back it — the
# hard prerequisite already documented in docs/PCIE_P2P.md §5. Real values are
# bimodal (~256 MiB when the aperture never grew, >= VRAM when it did), so half-VRAM
# discriminates with room to spare and never trips on accounting slop.
#
# This WARNS, it does not gate: peer access being advertised (topo -p2p: OK) while
# BAR1 is small is a combination we have inferred but not yet confirmed on a real
# rig, and silently withholding P2P from a rig where it works would be the worse
# error. Promote to a gate only with a confirmed case. Fails open — a driver that
# doesn't report these fields yields no lines and no warning.
#
# ⚠️ Necessary, not sufficient — do NOT read silence here as "P2P will work". The
# #873 rig reported BAR1 Total 32768 MiB against 32607 MiB of VRAM (so this probe
# stays quiet) and still hung in NCCL init until the driver was forced onto the
# static mapping via NVreg_RegistryDwords. What this catches is the *aperture*
# being too small (the firmware-gated #734 class) — one cause of an unusable
# mapping, ahead of the hang rather than after it, not all of them.
_bar1_undersized() {
  nvidia-smi -q -d MEMORY 2>/dev/null | awk '
    /^GPU /             { idx++; sect = ""; next }
    /FB Memory Usage/   { sect = "fb";   next }
    /BAR1 Memory Usage/ { sect = "bar1"; next }
    /Memory Usage/      { sect = "";     next }   # e.g. Conf Compute Protected
    sect != "" && $1 == "Total" && $3 ~ /^[0-9]+$/ {
      if (sect == "fb") fb[idx] = $3; else bar1[idx] = $3
      sect = ""
    }
    END {
      for (i = 1; i <= idx; i++)
        if (fb[i] > 0 && bar1[i] > 0 && bar1[i] * 2 < fb[i])
          printf "GPU%d BAR1=%dMiB VRAM=%dMiB ", i - 1, bar1[i], fb[i]
    }
  '
}

case "$NVLINK_MODE" in
  force_on)
    _NVLINK_ENABLED=1
    _NVLINK_FOUND=1   # a real NVLink finding (or force_on), not a PCIe peer path
    echo "[nvlink] NVLINK_MODE=force_on — enabling NVLink mode"
    ;;
  force_off)
    _NVLINK_ENABLED=0
    echo "[nvlink] NVLINK_MODE=force_off — forcing PCIe mode (P2P off)"
    ;;
  pcie_p2p)
    # Explicit opt-in for PCIe P2P (no NVLink) — a patched module OR a driver/board
    # that grants peer access (some server boards + the open kernel modules do).
    # Force the config on either way (the escape hatch for a rig whose topo -p2p
    # probe is stricter than reality), but VERIFY peer access and flag it honestly
    # when the driver didn't grant it — else we'd falsely report "P2P engaged" on a
    # closed/stock driver that silently refuses it (club-3090 #688).
    _NVLINK_ENABLED=1
    _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
    if _pcie_p2p_available; then
      echo "[nvlink] NVLINK_MODE=pcie_p2p — forcing PCIe P2P; driver confirms peer access (nvidia-smi topo -p2p: OK) — NCCL_P2P_LEVEL=$_P2P_LEVEL, $(_ar_claim)"
    else
      _P2P_UNVERIFIED=1
      echo "[nvlink] WARNING: NVLINK_MODE=pcie_p2p set, but nvidia-smi topo -p2p does NOT report peer access as OK — the driver likely refused P2P (a closed GeForce driver disables it; the open kernel modules or a patched module + a P2P-capable board are what enable it). Forcing the NCCL/all-reduce config on as requested, but NCCL will silently fall back → throughput ≈ P2P-off. Verify with: nvidia-smi topo -p2p rw. Guide: docs/PCIE_P2P.md" >&2
    fi
    ;;
  auto)
    GPU_COUNT="$_GPU_COUNT"
    if [ "$GPU_COUNT" -gt 2 ]; then
      # Check topology matrix for any NVLink connections (e.g. 2 bridges on 4 cards).
      if nvidia-smi topo -m 2>/dev/null | command grep -qP '\bNV[0-9]+\b'; then
        _NVLINK_ENABLED=1
        _NVLINK_FOUND=1   # a real NVLink finding (or force_on), not a PCIe peer path
        if _nvlink_full_mesh; then
          echo "[nvlink] $GPU_COUNT GPUs detected — NVLink full mesh, enabling NVLink mode"
        else
          _NVLINK_PARTIAL=1
          echo "[nvlink] $GPU_COUNT GPUs detected — NVLink found on GPU pairs but the mesh is NOT fully connected (pairwise bridges, e.g. 2 bridges on 4 cards) — enabling NVLink mode; NCCL uses NVLink per bridged pair"
        fi
      elif _pcie_p2p_available; then
        _NVLINK_ENABLED=1; _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
        echo "[nvlink] $GPU_COUNT GPUs — no NVLink, but nvidia-smi reports P2P=OK (patched driver / P2P-capable layout) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=$_P2P_LEVEL, $(_ar_claim))"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] $GPU_COUNT GPUs detected — no NVLink, no P2P — using PCIe mode"
      fi
    elif [ "$GPU_COUNT" -eq 2 ]; then
      LINK=$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $3}')
      if [[ "$LINK" =~ ^NV[0-9]+$ ]]; then
        _NVLINK_ENABLED=1
        _NVLINK_FOUND=1   # a real NVLink finding (or force_on), not a PCIe peer path
        echo "[nvlink] detected NVLink ($LINK) between GPU0-GPU1 — enabling NVLink mode ($(_ar_claim))"
      elif _pcie_p2p_available; then
        _NVLINK_ENABLED=1; _P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
        # $(_ar_claim), NOT a hardcoded "custom all-reduce ON": this is the exact
        # rig shape (2 GPUs, patched PCIe P2P) where DISABLE_CUSTOM_ALL_REDUCE is
        # the #922/#1332 mitigation, and a trail that asserts the kernel is on
        # while it is off is the #924 false-verdict bug all over again.
        echo "[nvlink] PCIe topology ($LINK) but nvidia-smi reports P2P=OK (patched driver / shared root complex) — auto-enabling PCIe P2P (NCCL_P2P_LEVEL=$_P2P_LEVEL, $(_ar_claim))"
      else
        _NVLINK_ENABLED=0
        echo "[nvlink] PCIe topology ($LINK), P2P not available (topo -p2p: no OK) — using PCIe mode (no P2P; for a patched driver on a P2P-capable layout this auto-enables, or set NVLINK_MODE=pcie_p2p to force)"
      fi
    else
      _NVLINK_ENABLED=0
      echo "[nvlink] $GPU_COUNT GPU(s) — skipping NVLink detection"
    fi
    ;;
  *)
    echo "[nvlink] ERROR: invalid NVLINK_MODE=$NVLINK_MODE (must be auto|force_on|force_off|pcie_p2p)" >&2
    exit 1
    ;;
esac

# Apply environment overrides based on detection result.
# _NVLINK_ENABLED=1 means a fast P2P interconnect is available (NVLink OR patched PCIe
# P2P) — P2P stays on and the compose entrypoint enables custom all-reduce. The level is
# NVL for NVLink, PHB (or the user's value) for pcie_p2p.
# ── The two decisions, split ────────────────────────────────────────────────
# _P2P_TRANSPORT answers "is the peer transport up?" (internal — the NCCL env
# that depends on it is exported by this script, so no entrypoint needs it).
# _CUSTOM_AR_ENABLED answers "does vLLM run its own all-reduce kernel on top?".
_P2P_TRANSPORT="$_NVLINK_ENABLED"
if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
  _CUSTOM_AR_ENABLED=0
else
  _CUSTOM_AR_ENABLED="$_NVLINK_ENABLED"
fi
export _CUSTOM_AR_ENABLED

# ⭐ And ALIAS the legacy name to the KERNEL decision, deliberately.
# `_NVLINK_ENABLED` has always meant "fast interconnect available -> custom
# all-reduce ON" (see this file's header), and every entrypoint we have ever
# shipped gates the vLLM flag on it. Pointing it at the kernel decision makes
# DISABLE_CUSTOM_ALL_REDUCE effective on ALL of them — including the `_archive`
# composes, any fork, and any hand-edited file we will never see — *by
# construction*, instead of requiring each one to opt in.
#
# This replaces an earlier opt-in marker plus a boot-time refusal. That shape
# was wrong twice over: it could only refuse when the variable actually reached
# the container (so old or un-updated composes stayed silently inert anyway),
# and the refusal fired on host-side probes, where `launch.sh` sources this file
# inside `( ... ) 2>/dev/null || true` purely to read the value — swallowing the
# exit and yielding an empty answer.
#
# ⚠️ The transport is NOT this variable any more. Anything that wants to know
# whether peer transport is up must read NCCL_P2P_DISABLE / NCCL_P2P_LEVEL, or
# _P2P_TRANSPORT inside this script.
_NVLINK_ENABLED="$_CUSTOM_AR_ENABLED"
export _NVLINK_ENABLED

if [ "$_P2P_TRANSPORT" -eq 1 ]; then
  export NCCL_P2P_LEVEL="${_P2P_LEVEL:-NVL}"
  unset NCCL_P2P_DISABLE 2>/dev/null || true
  # custom all-reduce is ON here. expandable_segments backs allocations with a
  # cuMemMap VA range, and cudaIpcGetMemHandle on that range fails during graph-
  # buffer registration (custom_all_reduce.cuh "invalid argument") — so it MUST
  # be off on this path. Dual composes inject
  # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,... for the PCIe path, so a
  # plain ${VAR:-default} would keep that crashing value. Strip ONLY the
  # expandable_segments token and preserve any other knobs the user set
  # (max_split_size_mb, garbage_collection_threshold, ...). See docs/UPSTREAM.md.
  _alloc="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
  _alloc="$(printf '%s' "$_alloc" | sed -E 's/(^|,)expandable_segments:[^,]*//g; s/^,+//; s/,+$//; s/,+/,/g')"
  [ -n "$_alloc" ] || _alloc="max_split_size_mb:512"
  export PYTORCH_CUDA_ALLOC_CONF="$_alloc"
  # BAR1 sanity — PCIe-P2P path only (NVLink peer traffic doesn't ride BAR1, so an
  # NVL level skips the probe and its nvidia-smi call entirely). Advertised-but-
  # unmappable peer access does NOT degrade gracefully: NCCL hangs inside
  # ncclCommInitRank instead of falling back, which reads as a boot that stops dead
  # right after "vLLM is using nccl". Name the cause before that happens (#873).
  if [ "$NCCL_P2P_LEVEL" != "NVL" ]; then
    _BAR1_BAD="$(_bar1_undersized)"
    if [ -n "$_BAR1_BAD" ]; then
      echo "[nvlink] WARNING: enabling PCIe P2P, but BAR1 is far smaller than VRAM on: ${_BAR1_BAD}— the patched-driver P2P path maps the FULL VRAM aperture through BAR1 (static BAR1 mapping), which a BAR1 this small cannot back. Peer access can still be ADVERTISED by the driver (topo -p2p: OK) while transfers fail, and NCCL then HANGS during init rather than falling back — if this boot stops right after 'using nccl', this is why. Fix: enable Above 4G Decoding + Re-Size BAR in BIOS; if lspci says the card's Physical Resizable BAR tops out at 256MB it is a VBIOS ceiling, not a BIOS setting (docs/PCIE_P2P.md §4). NOTE: the NVreg static-BAR1 override in §5 does NOT help here — that addresses a driver refusing to USE a full-size aperture, not an aperture that is too small. To boot now without P2P: NVLINK_MODE=force_off." >&2
    fi
    unset _BAR1_BAD
  fi
  if [ "${_P2P_UNVERIFIED:-0}" -eq 1 ]; then
    # pcie_p2p forced, but topo -p2p didn't confirm peer access. Config is applied;
    # engagement is NOT proven. Say so — never print "ENABLED" without evidence (#688).
    echo "[nvlink] P2P REQUESTED (UNVERIFIED) — NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL + $(_ar_claim), but peer access is UNCONFIRMED (topo -p2p ≠ OK; see the warning above). If the driver refused P2P, NCCL falls back and throughput ≈ P2P-off — verify with nvidia-smi topo -p2p rw. expandable_segments stripped (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
  else
    echo "[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL, $(_ar_claim), expandable_segments stripped (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
  fi
  # ⚠️ Tell people about the risk BEFORE it bites, not after. Three community
  # rigs have been harmed by the custom all-reduce kernel running over a PATCHED
  # PCIe peer path — two with silent wrong output (#922: @juslex Z390, @fkrutko
  # Z690) and one with a HARD MACHINE RESET under sustained decode (#1332,
  # @leo-3889 AM5). All three are consumer boards whose 16 CPU lanes are split
  # x8/x8; the rigs that run it cleanly are HEDT/server at full x16. The knob is
  # useless to someone who only learns it exists by crashing, and nothing else
  # in the boot path mentions it.
  #
  # Scope: the PCIe peer path only, and only where the kernel actually RUNS.
  # Native NVLink is a different path with no reports against it, vLLM vetoes
  # its own kernel above 2 PCIe-only GPUs (#786), and someone who already set
  # the knob has opted out and does not need telling.
  #
  # ⚠️ We deliberately do NOT narrow this to x8/x8 links. The platform
  # correlation is confounded with speculative decoding and the mechanism is
  # unestablished (#1332), so narrowing on an unproven mechanism would silence
  # the warning for someone it should reach. Over-warning costs a log line;
  # under-warning costs a filesystem.
  # ⚠️ _WORLD (TP), not _GPU_COUNT: a single-card slug with `count: all` sees
  # every card on the host and would otherwise be warned about an all-reduce it
  # never performs. And only on a VERIFIED peer path — claiming "patched PCIe
  # peer path" on a driver that refused peer access is just wrong.
  if [ "${_CUSTOM_AR_ENABLED:-0}" = "1" ] \
     && [ "$NCCL_P2P_LEVEL" != "NVL" ] \
     && [ "${_P2P_UNVERIFIED:-0}" != "1" ] \
     && [ "${_WORLD:-1}" -eq 2 ]; then
    # ⚠️⚠️ Wording is load-bearing: p2p-state.sh's classifier reads this same log
    # and keys on English phrases. A warning about the kernel must not contain
    # any of them, or it silently changes what we report the kernel is doing.
    # This is not hypothetical — the first cut of this warning ended "vs turning
    # P2P off entirely", and `*"P2P off"*` is the classifier's TRANSPORT-OFF key,
    # so on a rig whose peer path had actually failed vLLM's own test the verdict
    # flipped from "peer path broken" to "P2P is off, set NVLINK_MODE=force_off"
    # — the wrong diagnosis AND the wrong remedy, on the exact rig shape this
    # warning exists to protect.
    #
    # ⚠️ Do NOT rely on the comment you are reading to keep this safe: it listed
    # three phrases to avoid and missed the one that mattered. The guard is
    # BEHAVIOURAL — test-custom-ar-knob.sh §8 asserts classification is identical
    # with and without these lines, against every engine-log case. Add a case
    # there rather than a phrase here.
    echo "[nvlink] ⚠️  NOTE: this rig is on the patched PCIe peer path with the engine's extra all-reduce kernel active. On three community rigs that exact combination has misbehaved — two produced silent WRONG OUTPUT, one HARD-RESET the machine a few seconds into sustained generation (club-3090 #922, #1332). Other rigs run it without trouble and the cause is not established, so this is a heads-up, not a diagnosis." >&2
    echo "[nvlink] ⚠️  If you see garbage output, or an unexplained reboot under load, launch with DISABLE_CUSTOM_ALL_REDUCE=1 — that keeps the fast peer transport and drops only the kernel, and measured +10.7% prefill against disabling the peer path altogether. Details: docs/PCIE_P2P.md." >&2
  fi
else
  export NCCL_P2P_DISABLE=1
  unset NCCL_P2P_LEVEL 2>/dev/null || true
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  echo "[nvlink] P2P DISABLED — NCCL_P2P_DISABLE=1, custom all-reduce OFF, expandable_segments ON"
fi

# ── The machine-readable state line ───────────────────────────────────────
# ⭐ THIS, not our prose, is what p2p-state.sh parses for OUR half of the
# picture. We know these values exactly at the moment we decide them, so we
# emit them as data instead of letting a classifier re-derive them by grepping
# English out of a log. That is what made every previous defect in this area
# possible: any sentence we added could collide with a key, and one did — a
# warning ending "P2P off" reclassified a broken peer path as a disabled one.
#
# ⭐⭐ It also defines the BOOT EPOCH. `restart: unless-stopped` means a single
# `docker logs` can hold several boots' worth of trail, and combining our state
# from one boot with vLLM's veto from another is unsound — it silently reports
# the wrong rig. The contract: the LAST STATE line in the log is the current
# boot, and only engine lines AFTER it describe that boot. Readers must not
# use `head`/`grep -m1`, which take the FIRST match.
#
# Fields are a closed vocabulary, stdout (stderr is not in `docker logs c 2>&-`
# pipelines), v= so the format can change without silently mis-parsing.
#   transport = nvlink | pcie_p2p | off      what NCCL is configured to use
#   kernel    = on | off                     what we passed to vLLM. NEVER a
#                                            prediction of what vLLM will do.
#   cause     = none | operator | transport_off
#   verified  = yes | no | n-a               did the driver confirm peer access
#   mode      = the resolved NVLINK_MODE
#   world     = TP width from the entrypoint argv
_state_kernel=off; _state_cause=transport_off
[ "${_CUSTOM_AR_ENABLED:-0}" = "1" ] && { _state_kernel=on; _state_cause=none; }
# Only when the transport is actually up: on a transport-off boot the kernel is
# off because there is no peer path, whatever the knob says.
if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ] && [ "${_P2P_TRANSPORT:-0}" = "1" ]; then _state_cause=operator; fi
# An unconfirmed grant is reported through verified=no (which maps to `requested`),
# NOT through cause= — cause must never overwrite `operator`. The `refused` value
# was named for a boot refusal that no longer exists.
# `nvlink` requires an actual NVLink finding, not just the NVL level — a user can
# set NCCL_P2P_LEVEL=NVL on a PCIe rig and would otherwise be labelled NVLink.
_state_transport=off
if [ "${_P2P_TRANSPORT:-0}" = "1" ]; then
  if [ "${_NVLINK_FOUND:-0}" = "1" ]; then _state_transport=nvlink; else _state_transport=pcie_p2p; fi
fi
# ⚠️ Keyed on _P2P_UNVERIFIED alone, NOT on the transport label. Deriving it only
# for `transport=pcie_p2p` meant `NVLINK_MODE=pcie_p2p NCCL_P2P_LEVEL=NVL` on a
# rig whose driver refused peer access produced transport=nvlink / verified=n-a,
# and classified as a healthy `on` — the #688 false-engaged bug with a ✓.
_state_verified=n-a
if [ "${_P2P_TRANSPORT:-0}" = "1" ] && [ "$NVLINK_MODE" != "force_on" ]; then
  # force_on ASSERTS NVLink; nothing is probed, so `yes` would be an over-claim.
  _state_verified=yes
  [ "${_P2P_UNVERIFIED:-0}" = "1" ] && _state_verified=no
fi
# `gate=` is a PREDICTION about what vLLM will do; `kernel=` is a FACT about
# what we passed. They are separate fields on purpose. Folding the prediction
# into kernel= would make our own state line lie about our own argv; dropping
# it would make us claim "custom all-reduce ON" on a >2-GPU PCIe rig in the
# window before vLLM logs its veto, which is exactly the #786 false claim.
# When the engine actually speaks, its word wins over this field.
# ⚠️ Keyed on _WORLD (the TP width), NOT the visible GPU count. vLLM's gate is on
# world_size, so a 4-visible container running TP=2 has no veto coming — predicting
# one there made the classifier report "vLLM auto-disabled its kernel" in the very
# boot whose warning said the kernel was live. A prediction is sound only as a
# restatement of the engine's actual rule, so it is derived from world here.
_state_gate=none
if [ "${_WORLD:-1}" -gt 2 ] \
   && { [ "${_state_transport}" = "pcie_p2p" ] || [ "${_NVLINK_PARTIAL:-0}" = "1" ]; }; then
  _state_gate=expected_veto
fi
echo "[nvlink] STATE v=1 transport=${_state_transport} kernel=${_state_kernel} cause=${_state_cause} verified=${_state_verified} gate=${_state_gate} mode=${NVLINK_MODE} world=${_WORLD}"
unset _state_kernel _state_cause _state_transport _state_verified _state_gate


unset -f _pcie_p2p_available _ar_claim _nvlink_full_mesh _bar1_undersized _world_size 2>/dev/null || true   # don't leak the probes into the sourcing shell
