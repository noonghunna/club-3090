#!/usr/bin/env bash
# test-gateway-key-apply — `gpu-mode gateway` and `gateway-key.sh rotate --apply` (#1467).
#
# WHY THIS TEST EXISTS
# --------------------
# A new gateway key does nothing until the gateway is recreated, and gpu-mode had no
# way to recreate ONLY the gateway: every mode also starts or stops models. So
# `rotate` could only print a hand-made `sudo docker compose` line. `gpu-mode gateway`
# is that restart through gpu-mode's own start path, and `rotate --apply` runs it and
# then re-runs the agent setups. What must hold, and would fail silently otherwise:
#   1. gpu-mode gateway renders the routes FIRST, then recreates the litellm container
#      and nothing else (no model, no other service), through sudo docker compose with
#      the per-call settings file — the one place the stored key reaches compose;
#      --force-recreate, since a container keeps the env it was created with; waits
#      for /health/liveliness, then reports; a failed start or a gateway that never
#      answers exits 1.
#   2. Run as a symlink (/usr/local/bin/gpu-mode on the reference rig) it still finds
#      its lib/ — the route sync used to be looked up next to the symlink.
#   3. rotate --apply: the NEW key reaches the recreate's env file, the gateway comes
#      up on it, and only then are the setups on this rig's gateway re-run (pi and
#      Hermes read the gateway's routes with the key); one pointed at another box's
#      gateway is left alone; the key is never printed. If the recreate fails, no
#      setup is touched. Without --apply nothing is started.
#
# Offline: a scratch clone (copies of the scripts; lib/litellm-sync.sh is a stub),
# `docker` / `sudo` / `curl` PATH shims prepended INLINE on every call, a stub gateway
# on a free 127.0.0.1 port, a stub `hermes`, temp HOME and config dirs. The curl shim
# refuses :4000 and :8080 outright, so the rig's real gateway and Open WebUI can't be
# reached. No container is started, stopped or touched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; STUB_PID=""
trap '[ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; rm -rf "$T"' EXIT
fail=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; fail=1; }
DEFAULT="sk-litellm-master-key"
KEY="test-key-$(printf '%s' "$$-$RANDOM" | md5sum | cut -c1-24)"     # a throwaway test key
unset LITELLM_MASTER_KEY HF_TOKEN PI_CODING_AGENT_DIR HERMES_HOME CLUB3090_GATEWAY_URL CLUB3090_DIR LITELLM_LOG GPU_MODE_GATEWAY_WAIT_S   # a real token in this shell stays out of anything resolved here
export HOME="$T/home"; mkdir -p "$HOME"
REAL_CURL="$(command -v curl)"

# ── a scratch clone ──────────────────────────────────────────────────────────
C="$T/club"; mkdir -p "$C/scripts/lib" "$C/services/litellm"
cp "$ROOT/scripts/"{gpu-mode.sh,gateway-key.sh,omp-setup.sh,pi-setup.sh,hermes-setup.sh} "$C/scripts/"
cp "$ROOT/scripts/lib/club-config.sh" "$ROOT/scripts/lib/club_config.py" "$C/scripts/lib/"
cp "$ROOT/services/litellm/docker-compose.yml" "$C/services/litellm/"
# The real sync would re-render (and probe for) this checkout's routes: a stub that logs.
printf '#!/usr/bin/env bash\necho "SYNC $*" >> "$ORDER_LOG"\n' > "$C/scripts/lib/litellm-sync.sh"
chmod +x "$C/scripts/"*.sh "$C/scripts/lib/litellm-sync.sh"

# ── a stub gateway: liveliness once "recreated", routes only for the key in $T/gw-key ──
cat > "$T/gw.py" <<'PY'
import http.server, json, os, sys
T = sys.argv[1]
INFO = json.dumps({"data": [
    {"model_group": "qwen3.8-27b", "max_input_tokens": 163840.0},
    {"model_group": "gemma-4-31b", "max_input_tokens": 131072.0},
]}).encode()
MODELS = json.dumps({"data": [{"id": "qwen3.8-27b"}]}).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health/liveliness":
            up = os.path.exists(os.path.join(T, "gw-up"))
            self.send_response(200 if up else 503); self.end_headers(); return
        body = {"/model_group/info": INFO, "/v1/models": MODELS}.get(self.path)
        if body is None:
            self.send_response(404); self.end_headers(); return
        want = "Bearer " + open(os.path.join(T, "gw-key")).read().strip()
        code = 200 if self.headers.get("Authorization", "") == want else 400
        with open(os.environ["ORDER_LOG"], "a") as fh:
            fh.write(f"READ {self.path} {code}\n")
        self.send_response(code); self.end_headers()
        self.wfile.write(body if code == 200 else b'{"error":"No connected db."}')
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(os.path.join(T, "gw-port"), "w").write(str(s.server_port))
s.serve_forever()
PY
export ORDER_LOG="$T/order.log"; : > "$ORDER_LOG"
printf '%s\n' "$DEFAULT" > "$T/gw-key"; touch "$T/gw-up"
python3 "$T/gw.py" "$T" & STUB_PID=$!
for _ in $(seq 1 50); do [ -s "$T/gw-port" ] && break; sleep 0.1; done
GW="http://127.0.0.1:$(cat "$T/gw-port")"

# ── shims ────────────────────────────────────────────────────────────────────
mkdir -p "$T/bin" "$T/stub" "$T/tmp"
cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
echo "SUDO $*" >> "$STUB_LOG"
while [[ "${1:-}" == *=* ]]; do shift; done
exec "$@"
EOF
# docker: logs every call with the compose dir; `compose … up` copies its --env-file,
# and — as a recreated gateway would — takes LITELLM_MASTER_KEY from it.
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "[$(basename "$PWD")] docker $*" >> "$STUB_LOG"
case "$1" in
  info) exit 0 ;;
  compose)
    if [[ " $* " == *" config "* ]]; then
      printf '{"name":"litellm","services":{"litellm":{"image":"ghcr.io/berriai/litellm:v1.100.3","container_name":"litellm"}}}\n'; exit 0
    fi
    if [[ " $* " == *" up "* ]]; then
      [[ -n "${FAIL_UP:-}" ]] && exit 1
      n=$(( $(cat "$STUB_DIR/ups" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/ups"
      envf=""; prev=""; for a in "$@"; do [[ "$prev" == --env-file ]] && envf="$a"; prev="$a"; done
      key="$DEFAULT_KEY"
      if [[ -n "$envf" ]]; then
        cp "$envf" "$STUB_DIR/envfile.$n"
        k="$(sed -n "s/^LITELLM_MASTER_KEY='\(.*\)'\$/\1/p" "$envf")"; [[ -n "$k" ]] && key="$k"
      fi
      rm -f "$GW_DIR/gw-up"; printf '%s\n' "$key" > "$GW_DIR/gw-key"
      echo "UP $(basename "$PWD")" >> "$ORDER_LOG"
      [[ -z "${NO_LIVE:-}" ]] && touch "$GW_DIR/gw-up"
    fi
    exit 0 ;;
  inspect)
    case "$4" in
      *Config.Image*)  echo "${STUB_RUNNING_IMAGE:-ghcr.io/berriai/litellm:v1.100.3}" ;;
      *State.Running*) echo true ;;
    esac ;;
  *) exit 0 ;;
esac
EOF
# curl: the real one, except that the rig's gateway (:4000) and Open WebUI (:8080) are refused.
cat > "$T/bin/curl" <<EOF
#!/usr/bin/env bash
for a in "\$@"; do
  case "\$a" in *:4000|*:4000/*|*:8080|*:8080/*) echo "REFUSED \$*" >> "\$CURL_LOG"; printf 000; exit 7 ;; esac
done
exec "$REAL_CURL" "\$@"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/nvidia-smi"
chmod +x "$T/bin/"*
for b in docker sudo curl; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done
export STUB_DIR="$T/stub" GW_DIR="$T" DEFAULT_KEY="$DEFAULT" STUB_LOG="$T/docker.log" CURL_LOG="$T/curl.log"
reset_logs() { : > "$STUB_LOG"; : > "$ORDER_LOG"; : > "$CURL_LOG"; rm -f "$STUB_DIR/ups" "$STUB_DIR/envfile."*; }

CFG="$T/cfg"; mkdir -p "$CFG"; chmod 700 "$CFG"
printf 'LITELLM_MASTER_KEY=%s\n' "$KEY" > "$CFG/secrets.env"; chmod 600 "$CFG/secrets.env"

# ── 1. gpu-mode gateway ──────────────────────────────────────────────────────
echo "1. gpu-mode gateway"
reset_logs; printf '%s\n' "$DEFAULT" > "$T/gw-key"
out="$(CLUB3090_CONFIG_DIR="$CFG" CLUB3090_GATEWAY_URL="$GW" GPU_MODE_GATEWAY_WAIT_S=20 TMPDIR="$T/tmp" PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" gateway 2>&1)"; rc=$?
[[ $rc -eq 0 ]] && ok "exits 0 when the gateway comes up" || bad "gpu-mode gateway rc=$rc: $out"
calls="$(command grep -E '^\[[^]]*\] docker compose' "$STUB_LOG" | command grep -v ' config ' || true)"
[[ "$(wc -l <<<"$calls")" == 1 && "$calls" == "[litellm] docker compose --env-file "*" -f docker-compose.yml up -d --force-recreate" ]] \
  && ok "recreates ONLY litellm: one compose call, 'up -d --force-recreate' in services/litellm" \
  || bad "expected exactly one 'up -d --force-recreate' in services/litellm, got: $calls"
command grep -qE 'docker (compose .*(down|stop|rm|kill|restart)|stop|rm|kill|restart)( |$)' "$STUB_LOG" \
  && bad "gpu-mode gateway stopped or removed something: $(command grep -E 'down|stop|rm |kill|restart' "$STUB_LOG")" \
  || ok "starts or stops nothing else (no down / stop / rm / restart, no model compose)"
command grep -q '^SUDO .*docker compose .* up -d --force-recreate' "$STUB_LOG" && ok "…through sudo docker compose, gpu-mode's start path" \
  || bad "the recreate did not go through sudo: $(cat "$STUB_LOG")"
[[ "$(head -2 "$ORDER_LOG")" == "SYNC --no-restart --quiet"$'\n'"UP litellm" ]] \
  && ok "renders the gateway's routes first (litellm-sync --no-restart), then recreates" \
  || bad "expected SYNC --no-restart then UP: $(tr '\n' '|' < "$ORDER_LOG")"
command grep -qxF "LITELLM_MASTER_KEY='$KEY'" "$STUB_DIR/envfile.1" 2>/dev/null \
  && ok "the per-call settings file handed to compose carries the stored key" \
  || bad "the compose --env-file lacks the stored key"
[[ -z "$(ls -A "$T/tmp")" ]] && ok "the settings file is removed afterwards" || bad "left in TMPDIR: $(ls -A "$T/tmp")"
[[ "$out" == *"health/liveliness...up"* && "$out" == *"Running gateway ($GW): accepts it."* ]] \
  && ok "waits for /health/liveliness, then prints gateway-key.sh status (accepts it)" || bad "no liveliness wait / status: $out"
[[ "$out" != *"$KEY"* ]] && ok "never prints the key" || bad "gpu-mode gateway printed the key"
[[ ! -s "$CURL_LOG" ]] || bad "something tried the rig's :4000 / :8080: $(cat "$CURL_LOG")"

reset_logs
out="$(CLUB3090_CONFIG_DIR="$CFG" CLUB3090_GATEWAY_URL="$GW" GPU_MODE_GATEWAY_WAIT_S=20 FAIL_UP=1 PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" gateway 2>&1)"; rc=$?
[[ $rc -ne 0 && "$out" == *"FAILED to start: litellm"* && "$out" != *"liveliness"* ]] \
  && ok "a failed recreate exits 1 and doesn't wait for the gateway" || bad "FAIL_UP: rc=$rc $out"
reset_logs
out="$(CLUB3090_CONFIG_DIR="$CFG" CLUB3090_GATEWAY_URL="$GW" GPU_MODE_GATEWAY_WAIT_S=2 NO_LIVE=1 PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" gateway 2>&1)"; rc=$?
[[ $rc -ne 0 && "$out" == *"no answer after 2s"* && "$out" == *"did not come up"* ]] \
  && ok "a gateway that never answers /health/liveliness exits 1" || bad "NO_LIVE: rc=$rc $out"
touch "$T/gw-up"

# Run as a symlink, as /usr/local/bin/gpu-mode is on the reference rig.
mkdir -p "$T/usrbin"; ln -s "$C/scripts/gpu-mode.sh" "$T/usrbin/gpu-mode"
reset_logs
out="$(CLUB3090_CONFIG_DIR="$CFG" CLUB3090_GATEWAY_URL="$GW" GPU_MODE_GATEWAY_WAIT_S=20 PATH="$T/bin:$PATH" bash "$T/usrbin/gpu-mode" gateway 2>&1)"; rc=$?
[[ $rc -eq 0 && "$(head -1 "$ORDER_LOG")" == "SYNC --no-restart --quiet" ]] && command grep -q '^\[litellm\] docker compose' "$STUB_LOG" \
  && ok "run through a symlink it still finds its lib/ (the sync runs) and this clone's services/litellm" \
  || bad "via symlink: rc=$rc order=$(tr '\n' '|' < "$ORDER_LOG") $out"
reset_logs
out="$(CLUB3090_CONFIG_DIR="$CFG" STUB_RUNNING_IMAGE=ghcr.io/berriai/litellm:v1.99.0 PATH="$T/bin:$PATH" bash "$T/usrbin/gpu-mode" upgrade 2>&1)"
[[ "$(head -1 "$ORDER_LOG")" == "SYNC --no-restart --quiet" ]] \
  && ok "…and so does every other gateway start through a symlink (start_service litellm, via upgrade)" \
  || bad "start_service litellm via symlink skipped the route sync: $(tr '\n' '|' < "$ORDER_LOG")"

out="$(PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" help 2>&1)"
[[ "$out" == *"  gateway            Recreate ONLY the LiteLLM gateway"* ]] && ok "usage lists 'gateway'" || bad "usage lacks gateway"
out="$(PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" --list-modes --json 2>&1)"
python3 -c 'import json,sys; sys.exit(0 if all(r["name"] != "gateway" for r in json.loads(sys.argv[1])) else 1)' "$out" \
  && ok "--list-modes leaves it out (c3 would offer it as a scene)" || bad "--list-modes lists gateway"

# ── 2. gateway-key.sh rotate --apply ─────────────────────────────────────────
echo "2. gateway-key.sh rotate --apply"
G="$T/gk"; mkdir -p "$G"     # nothing stored: the gateway and the agents are on the default
mkdir -p "$T/hbin"; cat > "$T/hbin/hermes" <<'PY'
#!/usr/bin/env python3
import json, os, sys
path = os.path.join(os.environ.get("HERMES_HOME") or os.path.join(os.environ["HOME"], ".hermes"), "config.yaml")
doc = json.load(open(path)) if os.path.exists(path) else {}
verb, key = sys.argv[2], sys.argv[3]
parts = key.split(".")
cur = doc
for p in parts[:-1]:
    cur = cur.get(p) if isinstance(cur, dict) else None
if verb == "get":
    if not isinstance(cur, dict) or parts[-1] not in cur:
        print(f"Config key not set: {key}"); sys.exit(1)
    v = cur[parts[-1]]
    print(json.dumps(v) if isinstance(v, (dict, list)) else v); sys.exit(0)
if verb == "set":
    cur = doc
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = json.loads(sys.argv[4])
    json.dump(doc, open(path, "w"))
PY
chmod +x "$T/hbin/hermes"; export HERMES_BIN="$T/hbin/hermes"
mkdir -p "$HOME/.omp/agent" "$HOME/.pi/agent" "$HOME/.hermes"
# omp on this rig's default gateway, Hermes on this rig's (the stub), pi on another box.
CLUB3090_CONFIG_DIR="$G" PATH="$T/bin:$PATH" bash "$C/scripts/omp-setup.sh" >/dev/null 2>&1 || bad "fixture: omp-setup failed"
CLUB3090_CONFIG_DIR="$G" PATH="$T/bin:$PATH" bash "$C/scripts/hermes-setup.sh" --gateway "$GW/v1" >/dev/null 2>&1 || bad "fixture: hermes-setup failed"
CLUB3090_CONFIG_DIR="$G" PATH="$T/bin:$PATH" bash "$C/scripts/pi-setup.sh" --gateway http://gpu-box.invalid:4000/v1 >/dev/null 2>&1 || bad "fixture: pi-setup failed"
omp_key() { sed -n 's/^[[:space:]]*apiKey:[[:space:]]*//p' "$HOME/.omp/agent/models.yml" | head -1; }
pi_key()  { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["providers"]["club"]["apiKey"])' "$HOME/.pi/agent/models.json"; }
hermes_out() { python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["providers"]["club"]; print(c["api_key"], ",".join(sorted(c["models"])))' "$HOME/.hermes/config.yaml"; }
stored() { sed -n 's/^LITELLM_MASTER_KEY=//p' "$G/secrets.env" 2>/dev/null; }
[[ "$(omp_key)" == "$DEFAULT" && "$(pi_key)" == "$DEFAULT" ]] || bad "fixture: agents should start on the default key"
gk() { CLUB3090_CONFIG_DIR="$G" CLUB3090_GATEWAY_URL="$GW" GPU_MODE_GATEWAY_WAIT_S=20 TMPDIR="$T/tmp" PATH="$T/bin:$PATH" bash "$C/scripts/gateway-key.sh" "$@"; }

# Plain rotate: unchanged behaviour — prints the steps, starts nothing.
reset_logs; printf '%s\n' "$DEFAULT" > "$T/gw-key"
out="$(gk rotate 2>&1)"; rc=$?; k0="$(stored)"
[[ $rc -eq 0 && ! -s "$STUB_LOG" && "$(omp_key)" == "$DEFAULT" ]] && ok "rotate without --apply stores the key and starts nothing" \
  || bad "plain rotate: rc=$rc docker calls: $(cat "$STUB_LOG")"
[[ "$out" == *" $C/scripts/gpu-mode.sh gateway"* && "$out" == *"rotate --apply does steps 1 and 2"* ]] \
  && ok "…and points at gpu-mode gateway and rotate --apply" || bad "plain rotate's steps: $out"

reset_logs; printf '%s\n' "$DEFAULT" > "$T/gw-key"
out="$(gk rotate --apply 2>&1)"; rc=$?; k1="$(stored)"
[[ $rc -eq 0 && "$k1" =~ ^sk-club-[0-9a-f]{32}$ && "$k1" != "$k0" ]] && ok "rotate --apply stores a new key and exits 0" || bad "rotate --apply: rc=$rc $out"
command grep -qxF "LITELLM_MASTER_KEY='$k1'" "$STUB_DIR/envfile.1" 2>/dev/null && [[ "$(cat "$T/gw-key")" == "$k1" ]] \
  && ok "the NEW key reaches the gateway recreate's --env-file" || bad "the recreate did not get the new key"
[[ "$(command grep -cE '^\[[^]]*\] docker .* up ' "$STUB_LOG")" == 1 ]] && command grep -q '^\[litellm\] docker compose .* up -d --force-recreate' "$STUB_LOG" \
  && ok "…through gpu-mode gateway (one recreate of litellm, nothing else)" || bad "recreate calls: $(cat "$STUB_LOG")"
[[ "$(omp_key)" == "$k1" ]] && ok "omp (this rig's gateway) re-run: holds the new key" || bad "omp not refreshed with the new key"
[[ "$(hermes_out)" == "$k1 gemma-4-31b,qwen3.8-27b" ]] && ok "Hermes (this rig's gateway) re-run: new key, and it read the gateway's routes with it" \
  || bad "Hermes not refreshed with the new key + routes: $(hermes_out | sed "s/$k1/<new>/")"
[[ "$(pi_key)" == "$DEFAULT" && "$out" == *"pi points at http://gpu-box.invalid:4000/v1, not this rig's gateway: left out."* ]] \
  && ok "pi (another box's gateway) is left alone" || bad "pi was touched or not reported as left out"
# Every read of the gateway comes after the recreate, and every one is accepted: a
# setup run before step 1 reads the old gateway with the new key (400) and writes a
# fallback snapshot.
up_line="$(command grep -n '^UP litellm' "$ORDER_LOG" | head -1 | cut -d: -f1)"
first_read="$(command grep -n '^READ ' "$ORDER_LOG" | head -1 | cut -d: -f1)"
[[ -n "$up_line" && -n "$first_read" && "$up_line" -lt "$first_read" ]] && ! command grep -q '^READ .* 400$' "$ORDER_LOG" \
  && ok "the setups ran only after the gateway was up on the new key (every read accepted)" \
  || bad "order: $(tr '\n' '|' < "$ORDER_LOG")"
[[ "$out" == *"Claude Code"* && "$out" == *"Open WebUI"* && "$out" == *"Admin → Settings → Connections"* ]] \
  && ok "prints what is left by hand: Claude Code, an existing Open WebUI connection" || bad "manual steps missing: $out"
[[ "$out" != *"$k1"* ]] && ok "never prints the key" || bad "rotate --apply printed the key"
[[ ! -s "$CURL_LOG" ]] || bad "something tried the rig's :4000 / :8080: $(cat "$CURL_LOG")"

# The recreate fails: the key is stored, no setup is touched, exit non-zero.
reset_logs
out="$(FAIL_UP=1 gk rotate --apply 2>&1)"; rc=$?; k2="$(stored)"
[[ $rc -ne 0 && "$k2" != "$k1" && "$(omp_key)" == "$k1" && "$out" == *"NOT recreated"* ]] \
  && ok "a failed recreate stops there: exit 1, no setup re-run, says what to do" || bad "FAIL_UP apply: rc=$rc omp=$( [[ "$(omp_key)" == "$k1" ]] && echo kept || echo changed) $out"

# Refusals still come first and start nothing.
reset_logs
out="$(LITELLM_MASTER_KEY=key-in-shell gk rotate --apply 2>&1)"; rc=$?
[[ $rc -ne 0 && ! -s "$STUB_LOG" && "$(stored)" == "$k2" ]] && ok "a shell LITELLM_MASTER_KEY still refuses rotate --apply, before anything starts" \
  || bad "shell-key refusal: rc=$rc $(cat "$STUB_LOG")"
out="$(gk rotate --apply --now 2>&1)"; rc=$?
[[ $rc -eq 2 ]] && ok "rotate takes only --apply" || bad "rotate --apply --now: rc=$rc"

[ "$fail" -eq 0 ] && echo "test-gateway-key-apply: ok (gpu-mode gateway: sync first, recreates only litellm via sudo + the settings file, waits, reports, fails loudly; works via symlink; rotate --apply: new key to the recreate, then the local setups, foreign left alone, never printed)"
exit "$fail"
