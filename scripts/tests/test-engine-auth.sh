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
#   4. The other delivery paths: bench.sh's python requests, strata_metrics.py, switch.sh's
#      wait_ready and Open WebUI registration read the key the helper found.
#   4b. The rest of the rebench chain (#1878): every python request in verify-stress.sh,
#      bench-agentic.sh, concurrency_probe.py, soak-helper.py and deep_context_probe.py carries the key
#      (static), concurrency-probe.sh and soak-test.sh enable it themselves, and soak-helper.py's run plus concurrency_probe.py's
#      GETs reach a server that refuses everything without it (functional, with a no-key control).
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

tmp="$(mktemp -d)"
srv=""; srv2=""
trap '[[ -n "$srv" ]] && kill "$srv" 2>/dev/null; [[ -n "$srv2" ]] && kill "$srv2" 2>/dev/null; rm -rf "$tmp"' EXIT
bin="$tmp/bin"; mkdir -p "$bin"

bash -n scripts/lib/engine-auth.sh || fail "bash -n: engine-auth.sh"

# `docker inspect -f <fmt> <name>`: the image and the env of a few fake containers; anything
# else (or an unknown name) fails like docker does.
cat > "$bin/docker" <<'EOF'
#!/usr/bin/env bash
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
n="$(command grep -c -E 'urllib\.request\.Request\(' scripts/bench.sh)"
m="$(command grep -c -E 'headers=_HEADERS\)' scripts/bench.sh)"
[[ "$n" -ge 1 && "$n" == "$m" ]] || fail "bench.sh: every urllib Request must use _HEADERS ($m of $n)"
command grep -q -E 'CLUB_ENGINE_API_KEY' scripts/bench.sh || fail "bench.sh: _HEADERS must read CLUB_ENGINE_API_KEY"
pass "bench.sh: all $n python requests carry _HEADERS, which reads CLUB_ENGINE_API_KEY"
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

# --- 4b. the rest of the rebench chain (#1878) -----------------------------------------------------
# Static: each file defines _auth (reading CLUB_ENGINE_API_KEY), every urllib Request it builds carries
# _auth(...), and every urlopen gets such a Request (never a bare URL, which would go out keyless).
python3 - scripts/verify-stress.sh scripts/bench-agentic.sh scripts/lib/concurrency_probe.py \
  scripts/soak-helper.py scripts/lib/deep_context_probe.py <<'PY' || fail "a rebench-chain python request goes out without the key"
import re, sys
bad = []
for f in sys.argv[1:]:
    t = open(f, encoding="utf-8").read()
    if not re.search(r"def _auth\(.*?CLUB_ENGINE_API_KEY", t, re.S):
        bad.append(f"{f}: no _auth() reading CLUB_ENGINE_API_KEY")
    for m in re.finditer(r"urllib\.request\.Request\(", t):
        if "_auth(" not in t[m.end():m.end() + 300].split("\n\n")[0]:
            bad.append(f"{f}:{t[:m.start()].count(chr(10)) + 1}: a Request without _auth(...)")
    for m in re.finditer(r"urlopen\(\s*", t):
        if not re.match(r"(urllib\.request\.Request\(|req\b|http_req\b|_post\()", t[m.end():]):
            bad.append(f"{f}:{t[:m.start()].count(chr(10)) + 1}: urlopen of something other than an authed Request")
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY
for f in scripts/concurrency-probe.sh scripts/soak-test.sh; do
  command grep -q -E 'club_engine_auth_enable "\$CONTAINER"' "$f" \
    || fail "$f must enable the key for its container (it doesn't go through preflight_autodetect_endpoint)"
done
pass "verify-stress / bench-agentic / concurrency_probe / soak-helper / deep_context_probe: every request carries _auth(); concurrency-probe.sh and soak-test.sh enable the key"

# Functional: a server that answers 401 to anything without the right bearer, as Strata 0.2+ does.
KEY4B=k-chain-0000000001
port4b="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
KEY4B="$KEY4B" python3 - "$port4b" > /dev/null 2>&1 <<'PY' &
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
KEY = "Bearer " + os.environ["KEY4B"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _ok(self):
        if self.headers.get("Authorization") != KEY:
            self.send_response(401); self.end_headers(); return False
        return True
    def do_GET(self):
        if not self._ok(): return
        if self.path == "/v1/models":
            b = json.dumps({"data": [{"id": "m", "max_model_len": 4321}]}).encode()
        elif self.path == "/metrics":
            b = b"# ok\n"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not self._ok(): return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for c in ({"choices": [{"delta": {"content": "hi"}}]},
                  {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}):
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
srv2=$!
for _ in $(seq 1 50); do curl -s -m 1 -o /dev/null "http://127.0.0.1:$port4b/v1/models" && break; sleep 0.1; done
U4B="http://127.0.0.1:$port4b"
printf '%s' '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' > "$tmp/req4b.json"
CLUB_ENGINE_API_KEY="$KEY4B" python3 scripts/soak-helper.py run "$U4B" "$tmp/req4b.json" 30 "$tmp/m4b-key.json" >/dev/null 2>&1 || true
env -u CLUB_ENGINE_API_KEY python3 scripts/soak-helper.py run "$U4B" "$tmp/req4b.json" 30 "$tmp/m4b-nokey.json" >/dev/null 2>&1 || true
s_key="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$tmp/m4b-key.json" 2>/dev/null || echo none)"
s_nokey="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$tmp/m4b-nokey.json" 2>/dev/null || echo none)"
[[ "$s_key" == 200 && "$s_nokey" == 401 ]] || fail "soak-helper.py run: with the key status=$s_key (want 200), without status=$s_nokey (want 401)"
n_key="$(cd scripts/lib && CLUB_ENGINE_API_KEY="$KEY4B" URL="$U4B" CONTAINER="" python3 concurrency_probe.py --served-max-len 2>/dev/null || true)"
n_nokey="$(cd scripts/lib && env -u CLUB_ENGINE_API_KEY URL="$U4B" CONTAINER="" python3 concurrency_probe.py --served-max-len 2>/dev/null || true)"
[[ "$n_key" == 4321 && "$n_nokey" != 4321 ]] || fail "concurrency_probe.py --served-max-len: with the key '$n_key' (want 4321), without '$n_nokey' (want anything else)"
pass "against a server that refuses keyless requests: soak-helper.py run 200 with the key / 401 without; concurrency_probe.py reads /v1/models with it, not without"

echo "PASS: test-engine-auth"
