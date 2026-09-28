# club-config.sh — sourced; the ONE bash loader for club-3090 settings (club-3090#1466).
#
# Bash twin of scripts/lib/club_config.py, which documents the store, the parsing
# rules and the precedence. scripts/tests/test-club-config.sh holds the two to
# byte-identical output, and test-config-single-parser.sh fails any other code that
# parses these files. Change a rule here and there, or not at all.
#
#   club_config_dir                 the per-user config directory
#   club_config_cache_dir           the per-user cache dir (compile caches)
#   club_config_data_dir            the per-user data dir (the KV-offload disk tier)
#   club_config_resolve [ROOT]      KEY<TAB>SOURCE<TAB>VALUE for every configured key
#   club_config_load [ROOT]         export every key the environment doesn't already
#                                   set; CLUB3090_CONFIG_SOURCE[KEY] says which file
#   club_config_get KEY [ROOT]      one setting's effective value, or exit 1
#   club_config_compose_env_file [ROOT]   0600 temp file for docker compose --env-file
#   club_config_set [--file global|secrets] KEY=VALUE...   the ONE writer (python)
#   club_config_unset [--file global|secrets] [--root ROOT] KEY...
#                                   --root also clears ROOT/.env (read last, so an old
#                                   copy there would otherwise come back into effect)
#   club_config_migrate_notice ROOT [PREFIX]   one-time "settings still in this
#                                   checkout" notice on stderr (switch/launch/gpu-mode)
#
# ROOT is a repo checkout whose legacy .env is read last. Precedence, highest
# first: the shell > club3090.env > secrets.env > ROOT/.env. A variable already
# set in the shell wins even when empty (the rule switch.sh's own loader already used).

export PYTHONUTF8="${PYTHONUTF8:-1}"   # club_config_set/unset run python3 (#779)

club_config_dir() {
  if [[ -n "${CLUB3090_CONFIG_DIR:-}" ]]; then
    printf '%s\n' "$CLUB3090_CONFIG_DIR"
  else
    printf '%s/club-3090\n' "${XDG_CONFIG_HOME:-${HOME:-~}/.config}"
  fi
}

# The per-user cache and data dirs (#1466 phase 4) — club_config.py cache_dir /
# data_dir, same rule as the config dir. Both can be saved settings, so call them
# after club_config_load when that matters. Not created here: the launchers create
# the directories a compose mounts (scripts/lib/engine_cache.py).
club_config_cache_dir() {
  if [[ -n "${CLUB3090_CACHE_DIR:-}" ]]; then
    printf '%s\n' "$CLUB3090_CACHE_DIR"
  else
    printf '%s/club-3090\n' "${XDG_CACHE_HOME:-${HOME:-~}/.cache}"
  fi
}
club_config_data_dir() {
  if [[ -n "${CLUB3090_DATA_DIR:-}" ]]; then
    printf '%s\n' "$CLUB3090_DATA_DIR"
  else
    printf '%s/club-3090\n' "${XDG_DATA_HOME:-${HOME:-~}/.local/share}"
  fi
}

# <file> <label> → fills the caller's _CC_VAL / _CC_SRC. Within a file the last
# assignment wins; a later call (higher precedence) overwrites an earlier one.
_club_config_parse_into() {
  local LC_ALL=C                       # [[:space:]] = the ASCII set club_config.py strips
  local file="$1" label="$2" line key value q
  [[ -r "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == '#'* ]] && continue
    if [[ "$line" == "export "* ]]; then
      line="${line#export }"; line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"
    if (( ${#value} >= 2 )); then
      q="${value:0:1}"
      if [[ ( "$q" == '"' || "$q" == "'" ) && "${value: -1}" == "$q" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    _CC_VAL["$key"]="$value"; _CC_SRC["$key"]="$label"
  done < "$file"
}

club_config_resolve() {
  # Locals carry a __cc_ prefix: bash functions see their callers' variables, so
  # `${!__cc_k+x}` must not mistake one of OUR locals for a shell setting.
  local __cc_root="${1:-}" __cc_dir __cc_k
  local -A _CC_VAL=() _CC_SRC=()
  __cc_dir="$(club_config_dir)"
  [[ -n "$__cc_root" ]] && _club_config_parse_into "$__cc_root/.env" "repo .env"
  _club_config_parse_into "$__cc_dir/secrets.env" "secrets.env"
  _club_config_parse_into "$__cc_dir/club3090.env" "club3090.env"
  for __cc_k in "${!_CC_VAL[@]}"; do
    if [[ -n "${!__cc_k+x}" ]]; then
      printf '%s\tshell\t%s\n' "$__cc_k" "${!__cc_k}"
    else
      printf '%s\t%s\t%s\n' "$__cc_k" "${_CC_SRC[$__cc_k]}" "${_CC_VAL[$__cc_k]}"
    fi
  done | LC_ALL=C sort
}

# Same rules as club_config.py: a secret is anything from secrets.env or a
# credential-looking name; an "expansion" is $VAR, ${VAR} or a leading ~.
_club_config_is_secret() {
  [[ "$2" == secrets.env || "$1" =~ (TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|MASTER_KEY|_KEY)$ ]]
}

club_config_load() {
  local __cc_k __cc_src __cc_val
  declare -gA CLUB3090_CONFIG_SOURCE=()
  while IFS=$'\t' read -r __cc_k __cc_src __cc_val; do
    [[ -z "$__cc_k" || "$__cc_src" == shell ]] && continue
    export "$__cc_k=$__cc_val"
    CLUB3090_CONFIG_SOURCE["$__cc_k"]="$__cc_src"
    # launch.sh/report.sh/setup.sh used to `source` the repo .env, which expanded
    # these. Values are literal now; say so (key and file only, never the value).
    if ! _club_config_is_secret "$__cc_k" "$__cc_src" \
       && [[ "$__cc_val" =~ \$\{?[A-Za-z_] || "$__cc_val" =~ ^~(/|$) ]]; then
      echo "[config] WARN: $__cc_k (from $__cc_src) contains '\$VAR' or a leading '~'. Settings are read literally now, not expanded the way 'source .env' did — write the full path." >&2
    fi
  done < <(club_config_resolve "${1:-}")
}

# One setting's effective value (the shell wins), or exit 1. For scripts that need
# a key or two without exporting every setting into their environment.
club_config_get() {
  local __cc_key="$1" __cc_line
  [[ -n "${!__cc_key+x}" ]] && { printf '%s\n' "${!__cc_key}"; return 0; }
  __cc_line="$(club_config_resolve "${2:-}" | awk -F'\t' -v k="$__cc_key" '$1 == k { print; exit }')"
  [[ -n "$__cc_line" ]] || return 1
  __cc_line="${__cc_line#*$'\t'}"; printf '%s\n' "${__cc_line#*$'\t'}"
}

# A 0600 temp file of the resolved settings for `docker compose --env-file`, for
# callers whose environment doesn't reach compose (`sudo` strips it). Prints the
# path; the caller removes it. Written by club_config.py (compose quoting rules).
club_config_compose_env_file() { _club_config_py compose-env-file ${1:+--root "$1"}; }

_club_config_py() {
  python3 "$(dirname -- "${BASH_SOURCE[0]}")/club_config.py" "$@"
}
club_config_set()   { _club_config_py set "$@"; }
# club_config_migrate_notice ROOT [PREFIX] — the one-time "your settings still live in
# this checkout" notice on stderr (#1466; club_config.migrate_notice). Once per set
# of pending items; never fails the caller.
club_config_migrate_notice() { _club_config_py migrate-notice --root "$1" ${2:+--prefix "$2"} || true; }
# club_config_migrate_pending ROOT — what `settings.sh migrate` would still copy, one
# line ("; "-joined phrases, no values); empty when nothing. setup.sh's offer uses it.
club_config_migrate_pending() { _club_config_py migrate-pending --root "$1" 2>/dev/null || true; }
club_config_unset() { _club_config_py unset "$@"; }
