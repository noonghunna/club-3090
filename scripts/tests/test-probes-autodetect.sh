#!/usr/bin/env bash
# test-probes-autodetect — concurrency-probe.sh takes the URL from CONTAINER=, and deep-context-probe.sh
# resolves MODEL from the endpoint, as the other serving scripts do (#1584, #372).
#
# Why this exists (2026-10-31): concurrency-probe.sh took its URL from URL= or the registry's default
# port, never from CONTAINER=, though the repo guide says CONTAINER= sets it. A run told
# CONTAINER=<the QEngine Flash-Next slug> probed the default port, and every request errored.
# deep-context-probe.sh refused to run at all without MODEL=.
#
#   1. club_url_for_container: the host port a container publishes on an engine port (docker stubbed);
#   2. concurrency-probe.sh with CONTAINER= and no URL= announces and uses that port; URL= still wins;
#   3. deep-context-probe.sh with no MODEL= asks the endpoint (a stub server) for it.
# ⚠️ Against the pre-fix tree this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail() { echo "FAIL: test-probes-autodetect — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }
tmp="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$tmp"; }
trap cleanup EXIT

# docker: `ps --filter name=^NAME$ --format {{.Ports}}` answers for three fake containers; anything else
# fails like a docker with nothing running.
bin="$tmp/bin"; mkdir -p "$bin"
cat > "$bin/docker" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == ps ]]; then
  f=""; for a in "$@"; do case "$a" in name=*) f="$a" ;; esac; done
  case "$f" in
    'name=^qengine-x$') echo "9393/tcp, 0.0.0.0:8198->8080/tcp, [::]:8198->8080/tcp" ;;
    'name=^strata-y$')  echo "127.0.0.1:8024->8080/tcp" ;;
    'name=^noport-z$')  echo "9393/tcp" ;;
  esac
  exit 0
fi
exit 1
EOF
chmod +x "$bin/docker"

# --- 1. club_url_for_container -----------------------------------------------------------
u() { PATH="$bin:$PATH" bash -c 'source scripts/lib/club-containers.sh; club_url_for_container "$1"' _ "$1"; }
[[ "$(u qengine-x)" == "http://localhost:8198" ]] || fail "qengine-x: got '$(u qengine-x)'"
[[ "$(u strata-y)" == "http://localhost:8024" ]] || fail "a 127.0.0.1-bound container: got '$(u strata-y)'"
[[ -z "$(u noport-z)" && -z "$(u none)" && -z "$(u missing)" ]] || fail "no engine port / none / missing must give nothing"
pass "club_url_for_container: the published engine port, nothing without one"

# --- 2. concurrency-probe.sh: CONTAINER= sets the URL ----------------------------------------
out="$(env -u URL PATH="$bin:$PATH" CONTAINER=qengine-x MODEL=m timeout 60 bash scripts/concurrency-probe.sh --sweep --dry --n 1 2>&1 || true)"
[[ "$out" == *"[concurrency-probe] URL=http://localhost:8198 (the port qengine-x publishes"* ]] \
  || { echo "$out" | head -20 >&2; fail "CONTAINER=qengine-x with no URL= did not use its port"; }
out="$(PATH="$bin:$PATH" URL=http://127.0.0.1:1 CONTAINER=qengine-x MODEL=m timeout 60 bash scripts/concurrency-probe.sh --sweep --dry --n 1 2>&1 || true)"
[[ "$out" != *"the port qengine-x publishes"* ]] || fail "an explicit URL= must win over CONTAINER="
pass "concurrency-probe.sh: CONTAINER= with no URL= probes that container's port; URL= still wins"

# --- 3. deep-context-probe.sh: MODEL from the endpoint ------------------------------------
python3 -c '
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            b = json.dumps({"object": "list", "data": [{"id": "stub-model", "object": "model"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(s.server_address[1], flush=True); s.serve_forever()' > "$tmp/port" 2>/dev/null </dev/null &
PIDS+=("$!")
port=""; for _ in $(seq 1 50); do port="$(head -1 "$tmp/port" 2>/dev/null || true)"; [[ -n "$port" ]] && break; sleep 0.1; done
[[ -n "$port" ]] || fail "the stub server did not start"
# MODE=sideways stops the script right after MODEL is settled, before it would run the probe itself.
set +e
out="$(env -u MODEL PATH="$bin:$PATH" URL="http://127.0.0.1:$port" MODE=sideways bash scripts/deep-context-probe.sh 2>&1)"; rc=$?
set -e
[[ "$out" == *"[deep-context] model=stub-model"* ]] || { echo "$out" >&2; fail "MODEL unset: not resolved from the endpoint"; }
[[ $rc -eq 2 && "$out" == *"MODE must be"* ]] || fail "expected the MODE refusal after MODEL resolved (rc=$rc)"
pass "deep-context-probe.sh: MODEL unset resolves to what the endpoint serves"

echo "test-probes-autodetect: ok"
