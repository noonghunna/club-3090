# engine-cache.sh — sourced. Hands a compose the host directories for its compile
# caches and its KV-offload disk tier (club-3090#1466, phases 4a/4b). The rules live in
# scripts/lib/engine_cache.py (see its docstring); this is the launchers' bash door.
#
#   club_engine_cache_env <compose_dir> <compose_file> [engine_cache.py prepare options]
#       Creates, as you, every directory the compose mounts under
#       ${CLUB3090_ENGINE_CACHE_DIR} / ${CLUB3090_DATA_DIR}, and prints the
#       CLUB3090_ENGINE_CACHE_DIR=… / CLUB3090_DATA_DIR=… lines for that compose's
#       environment. Prints nothing, and costs nothing, for a compose that mounts
#       neither. Never fails: on any problem it says why, prints nothing, and the
#       compose falls back to its in-repo default for that launch.
#   club_engine_cache_export <compose_dir> <compose_file> [options]
#       The same, exported into this shell (switch.sh).
#
# Why the launcher and not the compose: the cache is keyed by the engine image, which
# compose can't compute, and `sudo docker compose` (gpu-mode) loses HOME. Why before
# compose runs: docker would create a missing bind-mount source as root:root.

export PYTHONUTF8="${PYTHONUTF8:-1}"   # engine_cache.py (#779)
_CLUB_ENGINE_CACHE_PY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/engine_cache.py"

club_engine_cache_env() {
  local dir="$1" file="$2"; shift 2
  command grep -qE '\$\{CLUB3090_(ENGINE_CACHE|DATA)_DIR' "$dir/$file" 2>/dev/null || return 0
  python3 "$_CLUB_ENGINE_CACHE_PY" prepare --compose "$dir/$file" "$@" || true
}

club_engine_cache_export() {
  local line
  # Recomputed on every launch: an inherited value must never outlive a fallback.
  command grep -qE '\$\{CLUB3090_ENGINE_CACHE_DIR' "$1/$2" 2>/dev/null && unset CLUB3090_ENGINE_CACHE_DIR
  while IFS= read -r line; do
    case "$line" in
      CLUB3090_ENGINE_CACHE_DIR=*|CLUB3090_DATA_DIR=*) export "$line" ;;
    esac
  done < <(club_engine_cache_env "$@")
}
