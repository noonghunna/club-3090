#!/usr/bin/env bash
#
# Transparent-hugepage check/apply for CPU-offloaded MoE models (moe-cache engine).
#
# WHY THIS EXISTS
#   With `-ot …exps=CPU`, expert weights are allocated by CUDA as PINNED HOST
#   memory, which the kernel accounts as **Shmem** — NOT anonymous memory. So the
#   familiar knob `transparent_hugepage/enabled` governs only the process heap and
#   does nothing for the model. A ~105 GiB working set then runs on ~27 M x 4 KiB
#   pages against a ~2 K-entry L2 TLB, and almost every expert read costs a page
#   walk. There is no error and no visible symptom.
#
#   The knob that matters is `transparent_hugepage/shmem_enabled` (default: never).
#   VERIFY WITH ShmemHugePages, NOT AnonHugePages — the latter will climb and look
#   like success while the model stays on small pages.
#
# WHAT IT ACTUALLY BUYS (measured 2026-08-30, GLM-5.3-Flash, 2x RTX 3090)
#   first-token latency (short prompts) : ~593 ms -> ~262 ms   (-56%)
#   decode throughput                   : -0.2%   (no change)
#   prefill @10K / @90K                 : +0.9% / -2.7%  (no change)
#   ⇒ This is a LATENCY optimisation. It does NOT increase tokens/sec.
#
# USAGE
#   scripts/hugepages.sh            # check + report coverage (no root needed)
#   scripts/hugepages.sh --apply    # set the knobs for this boot (needs root)
#   scripts/hugepages.sh --persist  # --apply + survive reboot (GRUB + tmpfiles.d)
#
# Style: one "[hugepages] ..." line per check; failures get a "Fix:" hint.
set -uo pipefail

THP=/sys/kernel/mm/transparent_hugepage
MODE="${1:-check}"

say() { printf '[hugepages] %s\n' "$*"; }
val() { [[ -r "$1" ]] && sed -n 's/.*\[\([a-z_+]*\)\].*/\1/p' "$1" || echo "n/a"; }
mib() { awk -v k="$1" '$1==k":"{print $2/1024}' /proc/meminfo 2>/dev/null; }

[[ -d "$THP" ]] || { say "ERROR: $THP not present — kernel has no THP support."; exit 1; }

enabled=$(val "$THP/enabled"); shmem=$(val "$THP/shmem_enabled"); defrag=$(val "$THP/defrag")
say "enabled=$enabled  shmem_enabled=$shmem  defrag=$defrag"

# --- coverage: the only honest measure. A knob set to 'always' can still yield 0%
#     under fragmentation, so report the ratio, not the setting.
sh=$(mib Shmem); shp=$(mib ShmemHugePages)
if [[ -n "${sh:-}" && -n "${shp:-}" ]] && awk "BEGIN{exit !($sh>1024)}"; then
  pct=$(awk "BEGIN{printf \"%.1f\", $shp/$sh*100}")
  say "coverage: ShmemHugePages $(awk "BEGIN{printf \"%.1f\",$shp/1024}") GiB of Shmem $(awk "BEGIN{printf \"%.1f\",$sh/1024}") GiB = ${pct}%"
  awk "BEGIN{exit !($pct<50)}" && {
    say "WARN: a model appears loaded but hugepage coverage is low."
    say "Fix: run '$0 --apply', then RESTART the container (existing 4 KiB mappings do not merge)."
  }
else
  say "coverage: no large Shmem allocation present — load a model, then re-run to measure."
fi

# --- contiguity: hugepages need physically contiguous memory; a long-uptime box
#     can be too fragmented to supply them even with the knobs set correctly.
if [[ -r /proc/buddyinfo ]]; then
  # buddyinfo: fields 5..15 are orders 0..10, so order 9 (2 MiB) starts at field 14.
  # Counting from field 13 would include order-8 (1 MiB) blocks and over-report.
  gib=$(awk '/Normal/{s=0; for(i=14;i<=NF;i++){s+=$i*(2^(i-5))*4/1048576} printf "%.1f", s; exit}' /proc/buddyinfo)
  say "free 2 MiB-contiguous memory right now: ${gib:-?} GiB"
  say "  (naturally low once a model is resident — this matters BEFORE a load, not after)"
fi

case "$MODE" in
  check) say "check only. Use --apply to set, --persist to also survive reboot."; exit 0 ;;
  --apply|--persist) : ;;
  *) say "ERROR: unknown argument '$MODE'"; say "Fix: use no argument, --apply, or --persist."; exit 2 ;;
esac

[[ $EUID -eq 0 ]] || command -v sudo >/dev/null || { say "ERROR: need root."; exit 1; }
SUDO=""; [[ $EUID -eq 0 ]] || SUDO="sudo"

say "applying: enabled=always shmem_enabled=always defrag=defer"
echo always | $SUDO tee "$THP/enabled"       >/dev/null || { say "ERROR: could not write enabled";       exit 1; }
echo always | $SUDO tee "$THP/shmem_enabled" >/dev/null || { say "ERROR: could not write shmem_enabled"; exit 1; }
echo defer  | $SUDO tee "$THP/defrag"        >/dev/null || true
say "now: enabled=$(val "$THP/enabled") shmem_enabled=$(val "$THP/shmem_enabled") defrag=$(val "$THP/defrag")"
say "NOTE: restart the model container — memory already mapped at 4 KiB does not merge."

if [[ "$MODE" == "--persist" ]]; then
  # GRUB carries `enabled` only; shmem_enabled has NO kernel parameter, so it needs
  # a post-boot write. tmpfiles.d is the least invasive mechanism for that.
  say "persisting via tmpfiles.d (shmem_enabled + defrag)"
  $SUDO tee /etc/tmpfiles.d/thp-moecache.conf >/dev/null <<'EOF'
# club-3090: CPU-offloaded MoE experts are CUDA pinned host memory (Shmem), so
# transparent_hugepage=always alone does not cover them. Verify ShmemHugePages.
w /sys/kernel/mm/transparent_hugepage/shmem_enabled - - - - always
w /sys/kernel/mm/transparent_hugepage/defrag        - - - - defer
EOF
  say "wrote /etc/tmpfiles.d/thp-moecache.conf"
  if command -v update-grub >/dev/null 2>&1 && [[ -w /etc/default/grub || -n "$SUDO" ]]; then
    if ! command grep -q 'transparent_hugepage=' /etc/default/grub; then
      $SUDO cp /etc/default/grub "/etc/default/grub.bak-$(date +%Y%m%d-%H%M%S)"
      $SUDO sed -i 's|^\(GRUB_CMDLINE_LINUX_DEFAULT="\)|\1transparent_hugepage=always |' /etc/default/grub
      $SUDO update-grub >/dev/null 2>&1 && say "added transparent_hugepage=always to GRUB (backup taken)"
    else
      say "GRUB already carries a transparent_hugepage= setting — left alone."
    fi
  else
    say "NOTE: no update-grub here; set transparent_hugepage=always via your bootloader."
  fi
fi
say "done. Re-run with no arguments after loading a model to confirm coverage."
