# slug-label.sh — sourced. The launchers' bash door to scripts/lib/slug_label.py:
# a compose override that labels every service club3090.slug=<slug>, so a running
# container says which registry slug it is (two slugs can share one compose file).
#
#   club_slug_label_override SLUG COMPOSE_PATH [ROOT]
#       prints the override's path, or nothing (unknown slug, or a slug whose
#       registered compose isn't COMPOSE_PATH). Never fails the caller.

export PYTHONUTF8="${PYTHONUTF8:-1}"   # slug_label.py (#779)
_CLUB_SLUG_LABEL_PY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/slug_label.py"

club_slug_label_override() {
  python3 "$_CLUB_SLUG_LABEL_PY" override --slug "$1" --compose "$2" ${3:+--root "$3"} 2>/dev/null || true
}
