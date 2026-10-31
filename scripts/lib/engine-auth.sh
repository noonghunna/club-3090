#!/usr/bin/env bash
# engine-auth.sh — send an engine's API key with our own requests to it.
#
# Source it; it defines functions and runs nothing at source time.
#
# Why this exists: Strata 0.2+ serves nothing but GET /health without a key (`strata launch
# --access lan`, which its Docker entrypoint always uses; only `--access local` is keyless, and
# inside a container that is the container's own loopback). Our composes set the key as the
# container's API_KEY, so for a local endpoint the key comes from the container itself: it is
# always the one the server uses, whether the compose default or a key of the user's own. No key
# is built in here, and a container without an API_KEY (Strata 0.1.x, Project Maya) gets no header.
#
# Precedence:
#   1. ENGINE_API_KEY            any engine, e.g. a remote URL= with no container here
#   2. the container's API_KEY   a Strata container (engine-kind.sh decides) that has one
#   3. STRATA_API_KEY            a Strata container without one
#
# Usage, once the script knows its container (empty or "none" for a URL with none here):
#   source scripts/lib/engine-auth.sh
#   club_engine_auth_enable "$CONTAINER"
# After that, every `curl` in this shell and its subshells sends `Authorization: Bearer <key>`,
# read from a process substitution so the key is never on a command line; child processes (the
# python helpers) get it as CLUB_ENGINE_API_KEY. With no key that applies, nothing changes.

[[ "$(type -t engine_kind_from_image)" == function ]] \
  || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/engine-kind.sh"

# club_engine_api_key [CONTAINER] — print the key that applies to CONTAINER, or nothing.
club_engine_api_key() {
  local c="${1:-}" kind img key
  if [[ -n "${ENGINE_API_KEY:-}" ]]; then printf '%s' "$ENGINE_API_KEY"; return 0; fi
  [[ -n "$c" && "$c" != none ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  img="$(docker inspect -f '{{.Config.Image}}' "$c" 2>/dev/null)" || return 0
  kind="$(engine_kind_from_image "$img")"
  [[ "$kind" == unknown ]] && kind="$(engine_kind_from_container "$c")"
  [[ "$kind" == strata ]] || return 0
  key="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null | sed -n 's/^API_KEY=//p' | head -1)"
  [[ -z "$key" ]] && key="${STRATA_API_KEY:-}"
  printf '%s' "$key"
}

# club_engine_auth_enable [CONTAINER] — when a key applies, make this shell's curl send it.
club_engine_auth_enable() {
  CLUB_ENGINE_KEY_="$(club_engine_api_key "${1:-}")"
  [[ -n "$CLUB_ENGINE_KEY_" ]] || return 0
  export CLUB_ENGINE_API_KEY="$CLUB_ENGINE_KEY_"
  curl() { command curl -H @<(printf 'Authorization: Bearer %s\n' "$CLUB_ENGINE_KEY_") "$@"; }
}

# club_engine_auth_enable_for_url URL — the same, for scripts that know only their endpoint:
# club_container_for_url answers WHICH container publishes it (#1584), then the key rule
# decides WHICH key. Scripts never resolve a container or a key themselves — the URL is the
# only input, so no path through the repo hardcodes either.
club_engine_auth_enable_for_url() {
  declare -F club_container_for_url >/dev/null 2>&1 \
    || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/club-containers.sh"
  club_engine_auth_enable "$(club_container_for_url "${1:-}")"
}
