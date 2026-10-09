#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-preflight-autodetect — guards #1584: endpoint autodetect binds the
# container that actually serves the endpoint, and nothing that merely shares a
# port number with an engine.
#
#   1. ports_serve_url (scripts/lib/listen-scope.sh) — does a `docker ps` Ports
#      string serve URL on this host? Bind address, port ranges, hostnames,
#      against a fake `hostname -I` / `getent`.
#   2. preflight_autodetect_endpoint against a mocked `docker ps`:
#        nothing set — SearXNG / Open WebUI on 8080 are NOT engines (the #1584
#                      repro), a BYO llama.cpp image on 8080 IS, ours win;
#        URL= set    — the container publishing that URL, else CONTAINER=none;
#        CONTAINER=  — the URL is that container's own port.
#      Every negative has a positive leg in the same mock world.
#   3. soak-test.sh, which picks its container itself: a URL no local container
#      publishes runs in host mode and never inspects the local container; the
#      matching URL does (positive control). Asserted at the docker layer.
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
bin="$tmp/bin"; mkdir -p "$bin"

bash -n scripts/lib/listen-scope.sh || fail "bash -n: listen-scope.sh"
bash -n scripts/preflight.sh || fail "bash -n: preflight.sh"

cat > "$bin/hostname" <<'EOF'
#!/usr/bin/env bash
echo "192.168.1.5 172.17.0.1 "
EOF
cat > "$bin/getent" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == "ahosts" ]] || exit 2
case "$2" in
  rig.lan)   printf '192.168.1.5     STREAM rig.lan\n192.168.1.5     DGRAM\n' ;;
  myrig)     printf '127.0.1.1       STREAM myrig\n' ;;
  other.lan) printf '10.9.9.9        STREAM other.lan\n' ;;
  *) exit 2 ;;
esac
EOF
# `docker ps` prints MOCK_PS (one `name|ports|image` line per entry) whatever the
# --format; nothing else is called by the code under test.
cat > "$bin/docker" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == "ps" ]] && printf '%s\n' "${MOCK_PS:-}"
exit 0
EOF
chmod +x "$bin"/*
export PATH="$bin:$PATH"

# ---------------------------------------------------------------------------
echo "--- 1. ports_serve_url ---"
serves() { bash -c 'source scripts/lib/listen-scope.sh; ports_serve_url "$1" "$2"' _ "$1" "$2"; }
yes_() { serves "$1" "$2" || fail "expected [$1] to serve $2"; pass "serves:  $2 ← $1"; }
no_()  { if serves "$1" "$2"; then fail "expected [$1] NOT to serve $2"; fi; pass "refuses: $2 ← $1"; }

W="0.0.0.0:8020->8000/tcp, [::]:8020->8000/tcp"
yes_ "$W" http://localhost:8020
no_  "$W" http://localhost:8021
yes_ "$W" http://127.0.0.1:8020/v1
yes_ "$W" "http://[::1]:8020"
yes_ "$W" http://192.168.1.5:8020            # this host's own LAN address
no_  "$W" http://10.9.9.9:8020               # same port, another machine
yes_ "$W" http://172.17.0.1:8020             # the bridge gateway is this host too
no_  "$W" http://localhost                   # no port → 80
yes_ "$W" http://rig.lan:8020                # a name resolving to this host's LAN address
yes_ "$W" http://myrig:8020                  # this host's own name (Debian: 127.0.1.1)
no_  "$W" http://other.lan:8020              # a name resolving to another machine
no_  "$W" http://unresolvable:8020
L="127.0.0.1:8031->8000/tcp"                  # BIND_HOST=127.0.0.1
yes_ "$L" http://localhost:8031
no_  "$L" http://192.168.1.5:8031            # loopback-only publish is not reachable on the LAN address
S="192.168.1.5:8040->8000/tcp"                # published on one address only
yes_ "$S" http://192.168.1.5:8040
no_  "$S" http://localhost:8040
yes_ "0.0.0.0:6333-6334->6333-6334/tcp" http://localhost:6334   # a published range
no_  "8000/tcp" http://localhost:8000          # exposed, not published

# ---------------------------------------------------------------------------
echo "--- 2. preflight_autodetect_endpoint ---"
SEARXNG="searxng|0.0.0.0:8088->8080/tcp, [::]:8088->8080/tcp|searxng/searxng:latest"
WEBUI="open-webui|0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp|ghcr.io/open-webui/open-webui:main"
GATEWAY="litellm|0.0.0.0:4000->4000/tcp|ghcr.io/berriai/litellm:main"
VLLM_A="vllm-qwen38-27b-dual|0.0.0.0:8020->8000/tcp|vllm/vllm-openai:v0.31.0"
VLLM_B="vllm-gemma-4-31b-dual|0.0.0.0:8021->8000/tcp|vllm/vllm-openai:v0.31.0"
BYO="my-llm|0.0.0.0:8099->8080/tcp|ghcr.io/ggml-org/llama.cpp:server-cuda"

# detect MOCK_PS URL CONTAINER → "URL=… CONTAINER=… AUTODET=…" on stdout, notices in $tmp/err
detect() {
  MOCK_PS="$1" URL_IN="$2" CONTAINER_IN="$3" bash -c '
    unset URL CONTAINER PREFLIGHT_ENDPOINT_AUTODETECTED PREFLIGHT_NO_AUTODETECT
    source scripts/preflight.sh
    [[ -n "$URL_IN" ]] && URL="$URL_IN"
    [[ -n "$CONTAINER_IN" ]] && CONTAINER="$CONTAINER_IN"
    preflight_autodetect_endpoint
    echo "URL=${URL:-} CONTAINER=${CONTAINER:-} AUTODET=${PREFLIGHT_ENDPOINT_AUTODETECTED:-}"
  ' 2>"$tmp/err"
}
expect() {
  local label="$1" want="$2" got="$3"
  [[ "$got" == "$want" ]] || { cat "$tmp/err" >&2; fail "$label: expected '$want', got '$got'"; }
  pass "$label → $got"
}
notice() { command grep -qF -- "$1" "$tmp/err" || { cat "$tmp/err" >&2; fail "missing notice: $1"; }; }

# nothing set
expect "#1584 repro: only SearXNG + Open WebUI up → nothing bound" \
  "URL= CONTAINER= AUTODET=" "$(detect "$SEARXNG"$'\n'"$WEBUI"$'\n'"$GATEWAY" "" "")"
expect "positive control: an engine beside them is found" \
  "URL=http://localhost:8020 CONTAINER=vllm-qwen38-27b-dual AUTODET=1" \
  "$(detect "$SEARXNG"$'\n'"$WEBUI"$'\n'"$VLLM_A" "" "")"
expect "a BYO llama.cpp image on 8080 still counts" \
  "URL=http://localhost:8099 CONTAINER=my-llm AUTODET=1" "$(detect "$SEARXNG"$'\n'"$BYO" "" "")"
expect "ours by name on 8080, any image" \
  "URL=http://localhost:8030 CONTAINER=llama-cpp-qwen38 AUTODET=1" \
  "$(detect "$SEARXNG"$'\n'"llama-cpp-qwen38|0.0.0.0:8030->8080/tcp|example/custom:1" "" "")"

# URL= set
expect "URL= another machine → host-only" \
  "URL=http://10.9.9.9:8020 CONTAINER=none AUTODET=" "$(detect "$VLLM_A" http://10.9.9.9:8020 "")"
notice "no running container publishes http://10.9.9.9:8020 on an engine port here"
expect "URL= a local port nothing publishes → host-only" \
  "URL=http://localhost:9999 CONTAINER=none AUTODET=" "$(detect "$VLLM_A" http://localhost:9999 "")"
expect "URL= the gateway (4000 is not an engine port) → host-only" \
  "URL=http://localhost:4000 CONTAINER=none AUTODET=" "$(detect "$VLLM_A"$'\n'"$GATEWAY" http://localhost:4000 "")"
expect "URL= the SECOND engine's port → that engine, not the first" \
  "URL=http://localhost:8021 CONTAINER=vllm-gemma-4-31b-dual AUTODET=1" \
  "$(detect "$VLLM_A"$'\n'"$VLLM_B" http://localhost:8021 "")"
expect "URL= by this rig's hostname → kept" \
  "URL=http://rig.lan:8020 CONTAINER=vllm-qwen38-27b-dual AUTODET=1" "$(detect "$VLLM_A" http://rig.lan:8020 "")"
expect "URL= you named needs no evidence: whatever publishes it on an engine port" \
  "URL=http://localhost:8088 CONTAINER=searxng AUTODET=1" "$(detect "$SEARXNG"$'\n'"$VLLM_A" http://localhost:8088 "")"

# CONTAINER= set
expect "CONTAINER= the second engine → ITS port" \
  "URL=http://localhost:8021 CONTAINER=vllm-gemma-4-31b-dual AUTODET=1" \
  "$(detect "$VLLM_A"$'\n'"$VLLM_B" "" vllm-gemma-4-31b-dual)"
expect "CONTAINER= with no engine port → URL left for the caller's default" \
  "URL= CONTAINER=litellm AUTODET=" "$(detect "$VLLM_A"$'\n'"$GATEWAY" "" litellm)"
notice "container 'litellm' publishes no engine port"

# both set / opted out → untouched
expect "both set → untouched" \
  "URL=http://10.9.9.9:8020 CONTAINER=mine AUTODET=" "$(detect "$VLLM_A" http://10.9.9.9:8020 mine)"
expect "PREFLIGHT_NO_AUTODETECT=1 → untouched" \
  "URL= CONTAINER= AUTODET=" "$(PREFLIGHT_NO_AUTODETECT_IN=1 MOCK_PS="$VLLM_A" bash -c '
    unset URL CONTAINER; source scripts/preflight.sh; PREFLIGHT_NO_AUTODETECT=1 preflight_autodetect_endpoint
    echo "URL=${URL:-} CONTAINER=${CONTAINER:-} AUTODET=${PREFLIGHT_ENDPOINT_AUTODETECTED:-}"' 2>/dev/null)"

# club_container_for_url — the shared answer preflight and soak both use
cfu() { MOCK_PS="$1" bash -c 'source scripts/lib/club-containers.sh; club_container_for_url "$1"' _ "$2"; }
expect "club_container_for_url: the second engine's URL" "vllm-gemma-4-31b-dual" "$(cfu "$VLLM_A"$'\n'"$VLLM_B" http://localhost:8021)"
expect "club_container_for_url: a remote URL → nothing" "" "$(cfu "$VLLM_A" http://10.9.9.9:8020)"
expect "club_container_for_url: the gateway's own port → nothing" "" "$(cfu "$GATEWAY" http://localhost:4000)"

# ---------------------------------------------------------------------------
echo "--- 3. soak-test.sh decides its container by the URL too ---"
sbin="$tmp/sbin"; mkdir -p "$sbin"
cat > "$sbin/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MOCK_LOG"
case "$1" in
  ps)      printf '%s\n' "vllm-mock|0.0.0.0:8020->8000/tcp|vllm/vllm-openai:v0" ;;
  inspect) [[ "$*" == *"State.Running"* ]] && echo true
           [[ "$*" == *vllm-mock* ]] || exit 1 ;;
esac
exit 0
EOF
# /v1/models never answers, so soak stops right after choosing its container.
printf '#!/usr/bin/env bash\nexit 7\n' > "$sbin/curl"
chmod +x "$sbin"/*
soak() {   # soak URL → the run's output; docker calls land in $tmp/soak-docker.log
  : > "$tmp/soak-docker.log"
  PATH="$sbin:$PATH" MOCK_LOG="$tmp/soak-docker.log" URL="$1" SOAK_OUTPUT="$tmp/soak-out" \
    timeout 120 bash scripts/soak-test.sh --quick 2>&1 || true
}
out="$(soak http://10.9.9.9:8020)"
command grep -q "host mode: CONTAINER=none" <<<"$out" || { printf '%s\n' "$out" >&2; fail "soak: a remote URL did not run in host mode"; }
if command grep -q "inspect.*vllm-mock" "$tmp/soak-docker.log"; then fail "soak: a remote URL inspected the local container"; fi
pass "soak: remote URL → host mode, local container never inspected"
out="$(soak http://localhost:8020)"
if command grep -q "host mode" <<<"$out"; then printf '%s\n' "$out" >&2; fail "soak: the serving container's own URL went to host mode"; fi
command grep -q "inspect.*vllm-mock" "$tmp/soak-docker.log" || fail "soak: the serving container was not inspected — the remote leg proves nothing"
pass "soak: the container's own URL → that container (positive control)"

echo "test-preflight-autodetect: ok"
