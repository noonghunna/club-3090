#!/usr/bin/env bash
# listen-scope.sh — what a TCP port on THIS host is listening on, and therefore
# whether a Docker container can reach it through host.docker.internal (#1578).
# Sourced by quality-test.sh's hermes container-reachability preflight (#960).
#
# hermesagent-20's sandbox reaches a loopback URL through host.docker.internal,
# i.e. the Docker bridge gateway (172.17.0.1 by default). A server is reachable
# that way only if it listens on a wildcard address or on the bridge address
# itself. When the preflight's container probe fails, the listeners tell us WHY,
# and the fix differs per cause; the old message assumed "bound to loopback" for
# every failure, which is wrong for a firewalled or LAN-IP-only server.
#
#   listen_addrs PORT
#     the local addresses (port stripped) of TCP listeners on PORT, one per line.
#     Nothing if none, or if `ss` is unavailable.
#   listen_scope PORT [GATEWAY]
#     one word: wildcard | bridge | loopback | specific | none | unknown
#       wildcard  — 0.0.0.0 / :: / * among the listeners: the bind is not the problem
#       bridge    — listening on GATEWAY (the Docker bridge address): reachable
#       loopback  — every listener is 127.x / ::1 (incl. a `-p 127.0.0.1:PORT:…` publish)
#       specific  — only non-loopback addresses that aren't the bridge (e.g. a LAN IP)
#       none      — nothing listening on PORT
#       unknown   — `ss` unavailable
#   docker_bridge_gateway
#     the default bridge's gateway address, or nothing.
#   container_serves_url CONTAINER URL
#     exit 0 when URL names THIS host (loopback, 0.0.0.0, one of its own
#     addresses, or a name resolving to either) and CONTAINER publishes URL's
#     port; 1 otherwise (#1579).
#     preflight's autodetect binds the first engine-port container even when
#     URL= points somewhere else, so a remote run would have recorded the local
#     container's sampling defaults and topology as its own.

listen_addrs() {
  local port="$1"
  command -v ss >/dev/null 2>&1 || return 0
  ss -ltnH "sport = :${port}" 2>/dev/null | awk '{print $4}' | sed -E 's/:[0-9]+$//; s/%[^]]*$//' | sort -u
}

docker_bridge_gateway() {
  command -v docker >/dev/null 2>&1 || return 0
  docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true
}

listen_scope() {
  local port="$1" gw="${2:-}" a seen=0 loop=0 bridge=0 wild=0 other=0
  command -v ss >/dev/null 2>&1 || { echo unknown; return 0; }
  while IFS= read -r a; do
    [[ -n "$a" ]] || continue
    seen=1
    case "$a" in
      0.0.0.0|'*'|'[::]'|'::'|'[::ffff:0.0.0.0]') wild=1 ;;
      127.*|'[::1]'|'::1'|'[::ffff:127.'*) loop=1 ;;
      *) if [[ -n "$gw" && ( "$a" == "$gw" || "$a" == "[$gw]" ) ]]; then bridge=1; else other=1; fi ;;
    esac
  done < <(listen_addrs "$port")
  if   [[ $seen -eq 0 ]];   then echo none
  elif [[ $wild -eq 1 ]];   then echo wildcard
  elif [[ $bridge -eq 1 ]]; then echo bridge
  elif [[ $other -eq 1 ]];  then echo specific
  else echo loopback
  fi
}

# A name counts when any address it resolves to does — `URL=http://<this rig's
# hostname>:PORT` must keep its container, or --thinking-budget refuses.
_host_is_this_machine() {
  local host="$1" own a
  case "$host" in localhost|127.*|::1|0.0.0.0) return 0 ;; esac
  own=" $(hostname -I 2>/dev/null) "
  [[ "$own" == *" ${host} "* ]] && return 0
  [[ "$host" =~ ^[0-9.]+$ || "$host" == *:* ]] && return 1   # an address that is not ours
  while read -r a _; do
    case "$a" in 127.*|::1) return 0 ;; esac
    [[ "$own" == *" ${a} "* ]] && return 0
  done < <(getent ahosts "$host" 2>/dev/null)
  return 1
}

container_serves_url() {
  local container="$1" url="$2" rest hostport host port
  [[ -n "$container" && -n "$url" ]] || return 1
  command -v docker >/dev/null 2>&1 || return 1
  rest="${url#*://}"; hostport="${rest%%/*}"; hostport="${hostport##*@}"
  if [[ "$hostport" == \[* ]]; then
    host="${hostport#[}"; host="${host%%]*}"
    port="${hostport##*]}"; port="${port#:}"
  else
    host="${hostport%%:*}"
    port=""; [[ "$hostport" == *:* ]] && port="${hostport##*:}"
  fi
  if [[ -z "$port" ]]; then
    if [[ "$url" == https://* ]]; then port=443; else port=80; fi
  fi
  _host_is_this_machine "$host" || return 1
  # `docker port` prints one mapping per line: "8000/tcp -> 0.0.0.0:8020".
  docker port "$container" 2>/dev/null | sed -nE 's/.*:([0-9]+)[[:space:]]*$/\1/p' | command grep -qx -- "$port"
}
