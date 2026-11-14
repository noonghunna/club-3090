# results-dir.sh — sourced. Where a run's results go when the checkout's results/ can't take them.
#
#   club_results_dir <path> [ROOT]
#       <path> created and writable → prints <path>.
#       Otherwise the same path under the per-user data dir (CLUB3090_DATA_DIR, saved or set, else
#       ~/.local/share/club-3090), created, with one note on stderr naming both; prints that.
#       Neither writable → prints <path> and returns 1, so the caller's own mkdir reports the error.
#
# Why (#2006, D#2002): a checkout owned by another account (or mounted read-only) has a results/ the
# user can't write. soak-test.sh stopped at `mkdir: cannot create directory 'results/…': Permission
# denied`, report.sh's soak stage then read as Overall FAIL, and concurrency-probe.sh's sweep lost its
# card. The run itself was fine; only where it wrote was not. ROOT (default: this checkout) is
# stripped from an absolute <path> so the fallback keeps the results/<…> shape under the data dir.
club_results_dir() {
  local want="${1:-}" root="${2:-}" rel data alt
  [[ -n "$want" ]] || return 1
  if mkdir -p -- "$want" 2>/dev/null && [[ -w "$want" ]]; then
    printf '%s\n' "$want"
    return 0
  fi
  [[ -n "$root" ]] || root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  rel="${want#"$root"/}"
  rel="${rel#./}"
  case "$rel" in /*) rel="results/$(basename -- "$rel")" ;; esac
  # The data dir can be a saved setting: read it through the one settings loader, in a subshell so
  # sourcing it changes nothing in the caller.
  data="$(
    # shellcheck source=club-config.sh
    source "$(dirname -- "${BASH_SOURCE[0]}")/club-config.sh"
    if [[ -z "${CLUB3090_DATA_DIR:-}" ]]; then
      v="$(club_config_get CLUB3090_DATA_DIR "$root" 2>/dev/null || true)"
      [[ -n "$v" ]] && CLUB3090_DATA_DIR="$v"
    fi
    club_config_data_dir
  )"
  alt="${data}/${rel}"
  if [[ -n "$data" ]] && mkdir -p -- "$alt" 2>/dev/null && [[ -w "$alt" ]]; then
    echo "[results] ${want} isn't writable here; writing to ${alt} instead" >&2
    printf '%s\n' "$alt"
    return 0
  fi
  printf '%s\n' "$want"
  return 1
}
