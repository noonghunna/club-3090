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
