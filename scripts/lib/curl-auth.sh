#!/usr/bin/env bash
# curl-auth.sh — authenticate curl against a VLLM_API_KEY-secured endpoint.
#
# Source this (don't execute it) near the top of any script that curls the
# model's /v1/* endpoints, then call `club3090_curl_auth_setup "$ROOT_DIR"`.
#
# What it does:
#   1. Resolves the bearer token from $VLLM_API_KEY / $CLUB3090_API_TOKEN, and
#      if neither is set in the environment, from the repo-root .env. This is
#      the fix for the direct-invocation case: `bash scripts/verify-full.sh`
#      when the key lives ONLY in .env — switch.sh loads .env into its child
#      `docker compose`, not into a sibling verify/bench/soak run, so those
#      scripts otherwise see no key and 401.
#   2. If a token is found, points curl's default config at a private temp
#      .curlrc (0700 dir, 0600 file) that adds `Authorization: Bearer <token>`
#      to EVERY curl in the sourcing process via $CURL_HOME — no per-curl edits.
#   3. Installs an EXIT trap to delete the temp file — UNLESS the caller already
#      has an EXIT trap (we don't clobber it). A caller with its own EXIT trap
#      MUST call `club3090_curl_auth_cleanup` from its handler (see soak-test.sh).
#
# No token found → no-op: nothing written, no header sent, behaviour is
# byte-identical to an unauthenticated run. The header is harmless on endpoints
# that don't require it (/health and llama.cpp /props ignore it).
#
# Why a shared lib: a secured compose (VLLM_API_KEY set) breaks EVERY /v1/*
# consumer — verify-full, verify-stress, bench, soak, health — not just one.
# Threading a token through dozens of inline curls is error-prone; one $CURL_HOME
# seam authenticates them all.

club3090_curl_auth_cleanup() {
  [[ -n "${CLUB3090_CURL_AUTH_DIR:-}" ]] && rm -rf "${CLUB3090_CURL_AUTH_DIR}"
  CLUB3090_CURL_AUTH_DIR=""
}

# club3090_curl_auth_setup [repo_root]
#   repo_root (optional): where to look for .env when the key isn't in the env.
club3090_curl_auth_setup() {
  local root="${1:-}"
  local key="${VLLM_API_KEY:-${CLUB3090_API_TOKEN:-}}"
  if [[ -z "$key" && -n "$root" && -f "${root}/.env" ]]; then
    # Source .env in a subshell (like launch.sh's loader) so its quoting is
    # honoured and its side effects don't leak into the calling script.
    key="$(set -a; . "${root}/.env" >/dev/null 2>&1; printf '%s' "${VLLM_API_KEY:-${CLUB3090_API_TOKEN:-}}")"
  fi
  [[ -z "$key" ]] && return 0

  # Expose the resolved token for consumers that authenticate a CHILD PROCESS
  # rather than curl — e.g. quality-test.sh handing $BENCHLOCAL_API_KEY to
  # benchlocal-cli. CURL_HOME only covers this process's own curls.
  export CLUB3090_RESOLVED_API_KEY="$key"

  CLUB3090_CURL_AUTH_DIR="$(mktemp -d)"                 # mktemp -d is 0700
  ( umask 077; printf 'header = "Authorization: Bearer %s"\n' "$key" \
      > "${CLUB3090_CURL_AUTH_DIR}/.curlrc" )
  export CURL_HOME="${CLUB3090_CURL_AUTH_DIR}"          # curl reads $CURL_HOME/.curlrc first
  # Auto-install cleanup only when the caller has no EXIT trap of its own.
  if [[ -z "$(trap -p EXIT)" ]]; then
    trap club3090_curl_auth_cleanup EXIT
  fi
}
