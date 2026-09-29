# membw.sh — sourced by report.sh: measured host memory bandwidth, and what it is
# measured against. The probe itself is scripts/lib/membw_probe.c (see its header for
# what varies between rigs and how it copes).
#
#   club_membw_rated      < dmidecode -t memory text
#       prints "<GB/s> <channels> <MT/s>" — the rated peak, populated channels x MT/s x
#       8 bytes — or nothing. Only when EVERY populated DIMM names its channel
#       explicitly (a "CHANNEL …" token in its Locator or Bank Locator: "P0 CHANNEL A",
#       "Controller0-ChannelA-DIMM0", "P0_Node0_Channel0_Dimm0") and reports a speed.
#       Slots are not channels (2 DIMMs per channel is common), so without an explicit
#       channel name there is no honest peak to divide by, and none is printed.
#   club_membw_lines      [dmidecode text]
#       the report bullet(s). Never fails the caller.
#
# Test seams: MEMBW_FAKE_RESULT (a probe output line: skip compiling and running),
# MEMBW_LOADAVG / MEMBW_VIRT (fake /proc/loadavg first field / systemd-detect-virt).

export PYTHONUTF8="${PYTHONUTF8:-1}"
_CLUB_MEMBW_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/membw_probe.c"

club_membw_rated() {
  awk '
    BEGIN { RS = ""; FS = "\n" }
    /Memory Device/ {
      size = ""; spd = ""; loc = ""; bank = ""
      for (i = 1; i <= NF; i++) {
        f = $i; v = f; sub(/^[^:]*:[[:space:]]*/, "", v)
        if (f ~ /^[[:space:]]*Size:/)                    size = v
        if (f ~ /^[[:space:]]*Configured Memory Speed:/) spd  = v
        if (f ~ /^[[:space:]]*Locator:/)                 loc  = v
        if (f ~ /^[[:space:]]*Bank Locator:/)            bank = v
      }
      if (size == "" || size ~ /No Module|Not Installed|Unknown/) next
      key = ""
      if (match(bank, /[Cc][Hh][Aa][Nn][Nn][Ee][Ll][ _-]?[A-Za-z0-9]+/))     key = "B:" substr(bank, 1, RSTART + RLENGTH - 1)
      else if (match(loc, /[Cc][Hh][Aa][Nn][Nn][Ee][Ll][ _-]?[A-Za-z0-9]+/)) key = "L:" substr(loc, 1, RSTART + RLENGTH - 1)
      sp = spd + 0
      if (key == "" || sp <= 0) { unknown = 1; next }
      if (!(key in chan)) { chan[key] = 1; n++ }
      if (min == 0 || sp < min) min = sp
    }
    END { if (!unknown && n > 0) printf "%.1f %d %d\n", n * min * 8 / 1000, n, min }'
}

club_membw_lines() {
  local dmi="${1:-}" result="" load virt
  load="${MEMBW_LOADAVG:-$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)}"
  if [[ -n "${MEMBW_VIRT+x}" ]]; then virt="$MEMBW_VIRT"
  else virt="$(systemd-detect-virt 2>/dev/null || true)"; fi
  if [[ -n "${MEMBW_FAKE_RESULT:-}" ]]; then
    result="$MEMBW_FAKE_RESULT"
  else
    local cc="" c d
    for c in cc gcc clang; do command -v "$c" >/dev/null 2>&1 && { cc="$c"; break; }; done
    if [[ -z "$cc" ]]; then
      echo "- **Memory bandwidth:** not measured (no C compiler) — install \`gcc\` for a STREAM Triad figure"
      return 0
    fi
    d="$(mktemp -d 2>/dev/null)" || return 0
    if "$cc" -O3 -march=native -pthread -o "$d/p" "$_CLUB_MEMBW_SRC" >/dev/null 2>&1 \
       || "$cc" -O3 -pthread -o "$d/p" "$_CLUB_MEMBW_SRC" >/dev/null 2>&1; then
      result="$(timeout 300 "$d/p" 2>/dev/null | tail -n 1)"
    fi
    rm -rf "$d"
  fi
  local -A r=()
  local kv
  for kv in $result; do r["${kv%%=*}"]="${kv#*=}"; done
  if [[ -n "${r[skip]:-}" ]]; then
    case "${r[skip]}" in
      low-memory) echo "- **Memory bandwidth:** not measured (only ${r[avail_mib]:-?} MiB of memory available; the probe needs at least ~400 MiB free)" ;;
      *)          echo "- **Memory bandwidth:** not measured (probe: ${r[skip]})" ;;
    esac
    return 0
  fi
  if [[ -z "${r[triad]:-}" || "${r[triad]}" == 0.0 ]]; then
    echo "- **Memory bandwidth:** probe failed to build/run — the rated speed above is NOT a substitute"
    return 0
  fi
  local rated="" pct=""
  [[ -n "$dmi" ]] && rated="$(printf '%s\n' "$dmi" | club_membw_rated)"
  local line="- **Memory bandwidth (measured):** ${r[triad]} GB/s STREAM Triad · ${r[read]} GB/s read"
  line+=" (best with ${r[threads]} thread(s) on ${r[cores]} physical core(s); 3 × ${r[array_mib]} MiB arrays)"
  if [[ -n "$rated" ]]; then
    local gbs ch mts
    read -r gbs ch mts <<<"$rated"
    pct="$(awk -v m="${r[triad]}" -v p="$gbs" 'BEGIN { printf "%d", m / p * 100 + 0.5 }')"
    line+=" = **${pct}% of ${gbs} GB/s rated** (${ch} channel(s) × ${mts} MT/s × 8 B; STREAM Triad typically reaches ~70–85% of rated)"
  fi
  line+=" — *this*, not the rated MT/s above, is what sets CPU-offload decode throughput"
  echo "$line"
  if [[ -n "$virt" && "$virt" != none ]]; then
    echo "  - Measured inside a VM (\`${virt}\`): this is what the guest gets; the host itself may measure higher."
  fi
  if awk -v l="${load:-0}" 'BEGIN { exit !(l >= 1.0) }'; then
    echo "  - ⚠️ The host was busy while measuring (load average ${load}): other work competes for memory bandwidth, so this can read low. Rerun when idle."
  fi
  if [[ "${r[capped]:-0}" == 1 ]]; then
    echo "  - Arrays were limited by free memory; a figure this small may partly fit in the CPU caches and read high."
  fi
  if [[ "${r[nt]:-1}" == 0 ]]; then
    echo "  - Triad used plain stores (no x86 non-temporal stores on this CPU); where writes allocate, it can read up to ~25% low. The read figure is unaffected."
  fi
  return 0
}
