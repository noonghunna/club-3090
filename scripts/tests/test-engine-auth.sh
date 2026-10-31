#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-engine-auth — scripts/lib/engine-auth.sh sends an engine's API key with our requests, and only
# where one applies. Strata 0.2+ answers nothing but GET /health without its key.
#
#   1. club_engine_api_key against a mocked `docker inspect`: a Strata container's own API_KEY;
#      STRATA_API_KEY for a Strata container without one; nothing for a Strata 0.1.x / Maya
#      container, a non-Strata container (even one with an API_KEY), a missing container, or
#      none; ENGINE_API_KEY over all of them.
#   2. club_engine_auth_enable: curl then sends exactly that key (a local server hashes the
#      Authorization header it receives), in subshells too, and children get CLUB_ENGINE_API_KEY;
#      with no key, curl is untouched.
#   3. preflight_autodetect_endpoint turns it on for the CONTAINER it settles on, also when URL= and
#      CONTAINER= are given (its early return) and with PREFLIGHT_NO_AUTODETECT=1.
#   4. The other delivery paths: every python request in the bench / verify-stress / bench-agentic /
#      concurrency_probe / soak-helper / compaction / deep-context chain goes through
#      scripts/lib/engine_http.py — which resolves the key from the URL through the helpers
#      (401-lazily, §4b) — never a re-derived key; strata_metrics.py, switch.sh's wait_ready and
#      Open WebUI registration read the key the helper found; the chain's shells enable it for the
#      URL or CONTAINER they settle on; quality-test.sh's --api-key fallback reads CLUB_ENGINE_API_KEY.
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

tmp="$(mktemp -d)"
srv="" srv2=""
trap '[[ -n "$srv" ]] && kill "$srv" 2>/dev/null; [[ -n "$srv2" ]] && kill "$srv2" 2>/dev/null; rm -rf "$tmp"' EXIT
bin="$tmp/bin"; mkdir -p "$bin"

bash -n scripts/lib/engine-auth.sh || fail "bash -n: engine-auth.sh"

# `docker inspect -f <fmt> <name>`: the image and the env of a few fake containers; anything
# else (or an unknown name) fails like docker does. `docker ps` answers with strata-new
# publishing FAKE_ENGINE_PORT — the URL→container half of the rule (#1584).
cat > "$bin/docker" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == ps ]] && { echo "strata-new|0.0.0.0:${FAKE_ENGINE_PORT:-18018}->8000/tcp"; exit 0; }
[[ "$1" == inspect && "$2" == -f ]] || exit 1
fmt="$3"; name="$4"
case "$name" in
  strata-new)    img=ghcr.io/niko1221/strata:0.2.1-cuda13;          env='API_KEY=key-from-the-container-0001' ;;
  strata-nokey)  img=ghcr.io/niko1221/strata:0.2.1-cuda13;          env='' ;;
  strata-old)    img=ghcr.io/noonghunna/strata-club3090:v0.1.45;     env='' ;;
  maya)          img=ghcr.io/noonghunna/strata-maya-club3090:v1.1.17; env='MAYA_GPUS=0,1' ;;
  vllm-keyed)    img=vllm/vllm-openai:v0.32.0;                       env='API_KEY=not-for-us-0000000000' ;;
  *) echo "Error: No such object: $name" >&2; exit 1 ;;
esac
case "$fmt" in
  '{{.Config.Image}}') echo "$img" ;;
  *Config.Env*)        printf 'PATH=/usr/bin\n%s\n' "$env" ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$bin/docker"
export PATH="$bin:$PATH"

# --- 1. which key applies ----------------------------------------------------------------------
key_for() { bash -c 'source scripts/lib/engine-auth.sh; club_engine_api_key "$1"' _ "$1"; }
[[ "$(key_for strata-new)" == key-from-the-container-0001 ]] || fail "a Strata container's own API_KEY"
pass "Strata 0.2 container: its own API_KEY"
[[ -z "$(key_for strata-nokey)" ]] || fail "a Strata container without API_KEY and no STRATA_API_KEY gets a key"
[[ "$(STRATA_API_KEY=from-settings-0000001 key_for strata-nokey)" == from-settings-0000001 ]] \
  || fail "STRATA_API_KEY for a Strata container without one"
pass "Strata container without API_KEY: STRATA_API_KEY when set, else nothing"
for c in strata-old maya vllm-keyed no-such-container "" none; do
  [[ -z "$(key_for "$c")" ]] || fail "'${c:-<empty>}' must get no key"
done
[[ -z "$(STRATA_API_KEY=from-settings-0000001 key_for vllm-keyed)" ]] || fail "STRATA_API_KEY must not reach a vLLM container"
pass "no key for Strata 0.1.x, Maya, a keyed vLLM container, a missing container, none"
[[ "$(ENGINE_API_KEY=explicit-key-00000001 key_for strata-new)" == explicit-key-00000001 ]] || fail "ENGINE_API_KEY wins"
[[ "$(ENGINE_API_KEY=explicit-key-00000001 key_for "")" == explicit-key-00000001 ]] || fail "ENGINE_API_KEY with no container"
pass "ENGINE_API_KEY over every other source, with or without a container"

# --- 2. curl sends it ---------------------------------------------------------------------------
cat > "$tmp/srv.py" <<'EOF'
import hashlib, http.server, socket, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        a = self.headers.get("Authorization")
        b = (hashlib.sha256(a.encode()).hexdigest() if a else "none").encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(s.server_address[1]))
s.serve_forever()
EOF
python3 "$tmp/srv.py" "$tmp/port" & srv=$!
for _ in $(seq 50); do [[ -s "$tmp/port" ]] && break; sleep 0.1; done
[[ -s "$tmp/port" ]] || fail "the test server did not start"
U="http://127.0.0.1:$(cat "$tmp/port")/"
want() { printf 'Bearer %s' "$1" | sha256sum | cut -d' ' -f1; }

got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable strata-new; curl -s "$1"' _ "$U")"
[[ "$got" == "$(want key-from-the-container-0001)" ]] || fail "curl did not send the container's key ($got)"
got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable strata-new; ( curl -s "$1" )' _ "$U")"
[[ "$got" == "$(want key-from-the-container-0001)" ]] || fail "a subshell's curl did not send the key"
got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable strata-new; bash -c "printf %s \"\$CLUB_ENGINE_API_KEY\""')"
[[ "$got" == key-from-the-container-0001 ]] || fail "a child process does not get CLUB_ENGINE_API_KEY"
pass "after enable: curl sends exactly the key (subshells too); children get CLUB_ENGINE_API_KEY"
got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable vllm-keyed; printf "%s|%s" "$(type -t curl)" "$(curl -s "$1")"' _ "$U")"
[[ "$got" == "file|none" ]] || fail "with no key, curl must stay the real curl and send nothing ($got)"
pass "with no key (a vLLM container): curl untouched, no header"
bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable strata-new; declare -f curl' | command grep -q -F '@<(printf' \
  || fail "the header must come from a process substitution, never the command line"
pass "the key reaches curl through a process substitution, not its command line"

# --- 3. through preflight_autodetect_endpoint ---------------------------------------------------
pf() {  # <env...> -- prints what curl received after the autodetect
  env "$@" bash -c 'source scripts/preflight.sh; preflight_autodetect_endpoint >/dev/null 2>&1; curl -s "$U"'
}
export U
[[ "$(pf URL=http://localhost:18018 CONTAINER=strata-new)" == "$(want key-from-the-container-0001)" ]] \
  || fail "preflight with URL= and CONTAINER= set must enable the key"
[[ "$(pf PREFLIGHT_NO_AUTODETECT=1 CONTAINER=strata-new)" == "$(want key-from-the-container-0001)" ]] \
  || fail "preflight with PREFLIGHT_NO_AUTODETECT=1 must still enable the key"
[[ "$(pf URL=http://localhost:18018 CONTAINER=strata-old)" == none ]] || fail "preflight sent a key to a keyless Strata 0.1.x"
pass "preflight_autodetect_endpoint enables it for the CONTAINER it settles on (and not for a keyless one)"

# --- 4. the other delivery paths ----------------------------------------------------------------
# One rule, one entry point: every python request in the chain goes through engine_http, and
# no path re-derives the key (no CLUB_ENGINE_API_KEY reads, no bare urlopen, no hand-built auth).
for f in scripts/bench.sh scripts/bench-agentic.sh scripts/verify-stress.sh \
         scripts/lib/concurrency_probe.py scripts/soak-helper.py \
         scripts/lib/compaction_probe.py scripts/lib/deep_context_probe.py; do
  n="$(command grep -c -E 'urllib\.request\.Request\(' "$f" || true)"
  m="$(command grep -c -E 'headers=engine_http' "$f" || true)"
  [[ "$n" -ge 1 && "$n" == "$m" ]] || fail "$f: every urllib Request must carry engine_http headers ($m of $n)"
  command grep -q -E 'import engine_http' "$f" || fail "$f: its python requests bypass scripts/lib/engine_http.py"
  ! command grep -q -E 'urllib\.request\.urlopen\(' "$f" || fail "$f: a request bypasses engine_http.urlopen"
  ! command grep -q -E 'CLUB_ENGINE_API_KEY' "$f" || fail "$f: re-derives the engine key instead of using engine_http"
done
pass "all seven chain scripts: every python request goes through engine_http, none re-derives the key"
CLUB_ENGINE_API_KEY=k-metrics-0000000001 python3 - "$U" <<'PY' || fail "strata_metrics.py did not send CLUB_ENGINE_API_KEY"
import hashlib, sys, urllib.request
sys.path.insert(0, "scripts/lib")
import strata_metrics
seen = {}
class Fake:
    def __init__(self, req): seen["auth"] = req.get_header("Authorization")
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return b"{}"
strata_metrics.urllib.request.urlopen = lambda req, timeout=None: Fake(req)
try:
    strata_metrics.fetch("http://127.0.0.1:1")
except BaseException:
    pass
sys.exit(0 if seen.get("auth") == "Bearer k-metrics-0000000001" else 1)
PY
pass "strata_metrics.py sends CLUB_ENGINE_API_KEY"
command grep -q -E 'club_engine_auth_enable "\$container"' scripts/switch.sh || fail "switch.sh wait_ready must enable the key"
command grep -q -F 'docker exec -i -e CLUB_ENGINE_API_KEY' scripts/lib/owui-register.sh \
  || fail "owui-register.sh must pass CLUB_ENGINE_API_KEY into its exec by name"
pass "switch.sh's wait_ready enables it; Open WebUI registration gets it by name"

# --- 4b. engine_http: the URL is the only input --------------------------------------------------
# A server that demands a key (401 without one) + the fake docker's ps arm (FAKE_ENGINE_PORT →
# strata-new): engine_http must resolve the key through the bash helpers after the 401 and retry.
cat > "$tmp/srv401.py" <<'EOF'
import hashlib, http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def _r(self):
        a = self.headers.get("Authorization")
        if not a:
            self.send_response(401); self.send_header("Content-Length", "0"); self.end_headers(); return
        b = hashlib.sha256(a.encode()).hexdigest().encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    do_GET = _r; do_POST = _r
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(s.server_address[1]))
s.serve_forever()
EOF
python3 "$tmp/srv401.py" "$tmp/port401" & srv2=$!
for _ in $(seq 50); do [[ -s "$tmp/port401" ]] && break; sleep 0.1; done
[[ -s "$tmp/port401" ]] || fail "the 401 server did not start"
U2="http://127.0.0.1:$(cat "$tmp/port401")/"
export FAKE_ENGINE_PORT="$(cat "$tmp/port401")"

got="$(python3 - "$U2" <<'PY'
import sys; sys.path.insert(0, "scripts/lib"); import engine_http
print(engine_http.urlopen(sys.argv[1], timeout=5).read().decode())
PY
)"
[[ "$got" == "$(want key-from-the-container-0001)" ]] || fail "engine_http: the 401 did not trigger the URL's key resolution ($got)"
pass "engine_http: a 401 resolves the URL's container key through the bash helpers, retries once"
got="$(CLUB_ENGINE_API_KEY=env-key-0000000000001 python3 - "$U2" <<'PY'
import sys; sys.path.insert(0, "scripts/lib"); import engine_http
print(engine_http.urlopen(sys.argv[1], timeout=5).read().decode())
PY
)"
[[ "$got" == "$(want env-key-0000000000001)" ]] || fail "engine_http ignored CLUB_ENGINE_API_KEY from the parent shell ($got)"
pass "engine_http: a parent shell's CLUB_ENGINE_API_KEY is used without re-resolving"
got="$(python3 - "$U" <<'PY'
import sys; sys.path.insert(0, "scripts/lib"); import engine_http
print(engine_http.urlopen(sys.argv[1], timeout=5).read().decode())
PY
)"
[[ "$got" == none ]] || fail "engine_http sent a key to a server that never asked for one ($got)"
pass "engine_http: a server that does not 401 sees the identical keyless request"

# club_engine_auth_enable_for_url: the shell half — URL in, container+key resolved by the helpers.
got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable_for_url "$1"; curl -s "$1"' _ "$U2")"
[[ "$got" == "$(want key-from-the-container-0001)" ]] || fail "club_engine_auth_enable_for_url did not send the URL's container key ($got)"
got="$(bash -c 'source scripts/lib/engine-auth.sh; club_engine_auth_enable_for_url "$1"; curl -s "$1"' _ "$U")"
[[ "$got" == none ]] || fail "club_engine_auth_enable_for_url sent a key for a URL no container publishes ($got)"
pass "club_engine_auth_enable_for_url: keys our URL, pays nothing for a URL that is not ours"

# --- 4c. the rebench chain's shells ---------------------------------------------------------------
# 2026-10-30: a key-custom Strata container 401'd every one of these.
for f in scripts/bench-agentic.sh scripts/verify-stress.sh scripts/compaction-probe.sh scripts/deep-context-probe.sh; do
  command grep -q -E 'club_engine_auth_enable_for_url "\$URL"' "$f" \
    || fail "$f: must enable the key for the URL it settles on"
done
pass "bench-agentic / verify-stress / compaction-probe / deep-context-probe shells enable it for their URL"
command grep -q -E 'club_engine_auth_enable "\$\{CONTAINER:-\}"' scripts/concurrency-probe.sh \
  || fail "concurrency-probe.sh must enable the key for its resolved container"
command grep -q -E 'club_engine_auth_enable "\$\{CONTAINER:-\}"' scripts/soak-test.sh \
  || fail "soak-test.sh must enable the key for its resolved container"
pass "concurrency-probe.sh and soak-test.sh enable it for the CONTAINER they settle on"
command grep -q -E 'API_KEY=.*CLUB_ENGINE_API_KEY' scripts/quality-test.sh \
  || fail "quality-test.sh: the API_KEY fallback chain must include CLUB_ENGINE_API_KEY"
pass "quality-test.sh's external benchlocal-cli gets the key through the env fallback"

echo "✓ engine-auth: one rule, one entry point per language, every path covered"
