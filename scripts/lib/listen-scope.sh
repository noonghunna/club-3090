#!/usr/bin/env bash
# listen-scope.sh — what a TCP port on THIS host is listening on, and therefore
# whether a Docker container can reach it through host.docker.internal (#1578).
# Sourced by quality-test.sh's hermes container-reachability preflight (#960)
# and by preflight.sh's endpoint autodetect (#1584).
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
#   ports_serve_url PORTS URL
#     exit 0 when a mapping in PORTS (a `docker ps` "Ports" string) serves URL on
#     THIS host: same host port, and the URL's host is an address the mapping is
#     published on — any of this host's addresses for 0.0.0.0 / ::, loopback for
#     a 127.x / ::1 publish, that exact address otherwise. A hostname counts by
#     what it resolves to. 1 otherwise (#1584). Used by preflight's endpoint
#     autodetect, which bound the first engine-port container even when URL=
#     pointed somewhere else, so a run against another machine read the local
#     container's logs, sampling defaults and topology as its own.
#   url_host_port URL
#     "host port", the port defaulted from the scheme.
#   url_on_this_host URL
#     exit 0 when URL's host is this machine — loopback, 0.0.0.0, one of its own
#     addresses, or a name resolving to one. concurrency-probe.sh uses it to tell
#     a host-process server (measure this rig's GPUs) from a remote one (don't).

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

# URL → "host port". The port defaults from the scheme (80 / 443).
url_host_port() {
  local url="$1" rest hostport host port
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
  printf '%s %s\n' "$host" "$port"
}

# HOST → the addresses it names, one per line: a literal is itself, a name is
# whatever `getent ahosts` resolves it to. `URL=http://<this rig's hostname>:PORT`
# has to count as this host, or a correct container would be dropped.
_host_addrs() {
  local host="$1"
  case "$host" in
    localhost|0.0.0.0) printf '127.0.0.1\n::1\n' ;;
    *:*)               printf '%s\n' "$host" ;;
    *[!0-9.]*)         getent ahosts "$host" 2>/dev/null | awk '{print $1}' | sort -u ;;
    *)                 printf '%s\n' "$host" ;;
  esac
}

url_on_this_host() {
  local host port own a
  read -r host port < <(url_host_port "$1")
  own=" $(hostname -I 2>/dev/null) "
  while IFS= read -r a; do
    [[ -n "$a" ]] || continue
    [[ "$a" == 127.* || "$a" == ::1 || "$own" == *" ${a} "* ]] && return 0
  done < <(_host_addrs "$host")
  return 1
}

ports_serve_url() {
  local ports="$1" url="$2" host port addrs own m bind hp lo hi a
  read -r host port < <(url_host_port "$url")
  [[ "$port" =~ ^[0-9]+$ ]] || return 1
  addrs="$(_host_addrs "$host")"
  [[ -n "$addrs" ]] || return 1
  own=" $(hostname -I 2>/dev/null) "
  while IFS= read -r m; do
    m="${m# }"
    [[ "$m" == *"->"* ]] || continue          # an exposed-only port, not published
    m="${m%%->*}"                              # 0.0.0.0:8020 · [::]:8020 · 127.0.0.1:6333-6334
    bind="${m%:*}"; hp="${m##*:}"
    bind="${bind#[}"; bind="${bind%]}"
    lo="${hp%-*}"; hi="${hp#*-}"
    [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ ]] || continue
    (( port >= lo && port <= hi )) || continue
    while IFS= read -r a; do
      [[ -n "$a" ]] || continue
      case "$bind" in
        0.0.0.0|::|'') [[ "$a" == 127.* || "$a" == ::1 || "$own" == *" ${a} "* ]] && return 0 ;;
        127.*|::1)      [[ "$a" == 127.* || "$a" == ::1 ]] && return 0 ;;
        *)              [[ "$a" == "$bind" ]] && return 0 ;;
      esac
    done <<<"$addrs"
  done < <(tr ',' '\n' <<<"$ports")
  return 1
}
