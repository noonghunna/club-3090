#!/usr/bin/env bash
# test-gateway-key — the gateway key is per install and every reader follows it (#1467).
#
# WHY THIS TEST EXISTS
# --------------------
# The LiteLLM gateway's key used to be a literal in services/litellm/docker-compose.yml,
# and omp-setup.sh, pi-setup.sh and hermes-setup.sh found it by sed-ing that line. Once
# the compose reads `${LITELLM_MASTER_KEY:-sk-litellm-master-key}`, that sed quietly
# falls back to the public default and the agents get a key the gateway refuses
# (`400 No connected db.`) — and nothing fails. So every reader is checked through its
# real entry point with the key stored ONLY in a temporary secrets.env:
#   1. docker compose interpolates it (and still falls back to the default);
#   2. omp / pi / Hermes setup write it, and pi / Hermes use it to read the gateway;
#   3. gateway-key.sh stores a random key 0600, never prints it, refuses to write one
#      that a higher-precedence setting would shadow, and lists the agent setups to re-run;
#   4. gpu-mode's status probe sends it (on stdin, never argv), and gpu-mode warns when
#      it starts the gateway on the public default.
# Offline: temp HOME / config dirs, a stub gateway, a stub `hermes`, and for gpu-mode a
# scratch copy with docker / sudo / curl shims. No container is started or touched.
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
unset LITELLM_MASTER_KEY PI_CODING_AGENT_DIR HERMES_HOME
export HOME="$T/home"; mkdir -p "$HOME"

# A config dir holding only the key, in secrets.env (0600), as gateway-key.sh leaves it.
CFG="$T/cfg"; mkdir -p "$CFG"; chmod 700 "$CFG"
printf 'LITELLM_MASTER_KEY=%s\n' "$KEY" > "$CFG/secrets.env"; chmod 600 "$CFG/secrets.env"
EMPTY="$T/empty-cfg"; mkdir -p "$EMPTY"

# A stub gateway: /v1/models and /model_group/info answer only the key in $T/gw-key
# (as LiteLLM does: an unknown key gets 400 "No connected db."); the last
# Authorization header is kept in $T/gw-hdr.
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
        hdr = self.headers.get("Authorization", "")
        open(os.path.join(T, "gw-hdr"), "w").write(hdr)
        want = "Bearer " + open(os.path.join(T, "gw-key")).read().strip()
        body = {"/model_group/info": INFO, "/v1/models": MODELS}.get(self.path)
        if body is None:
            self.send_response(404); self.end_headers(); return
        if hdr != want:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"No connected db."}'); return
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(os.path.join(T, "gw-port"), "w").write(str(s.server_port))
s.serve_forever()
PY
printf '%s\n' "$KEY" > "$T/gw-key"
python3 "$T/gw.py" "$T" & STUB_PID=$!
for _ in $(seq 1 50); do [ -s "$T/gw-port" ] && break; sleep 0.1; done
GW="http://127.0.0.1:$(cat "$T/gw-port")"

# A stub `hermes` with the three config verbs hermes-setup.sh uses; it keeps the
# config as JSON (valid YAML) in $HERMES_HOME/config.yaml.
mkdir -p "$T/bin"; cat > "$T/bin/hermes" <<'PY'
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
chmod +x "$T/bin/hermes"
export HERMES_BIN="$T/bin/hermes"

# ── 1. docker compose interpolates the stored key ────────────────────────────
echo "1. services/litellm/docker-compose.yml"
compose_key() {   # <env file> → LITELLM_MASTER_KEY as `docker compose config` renders it
  (cd "$ROOT/services/litellm" && env -u LITELLM_MASTER_KEY docker compose --env-file "$1" -f docker-compose.yml config --format json 2>/dev/null) \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["litellm"]["environment"]["LITELLM_MASTER_KEY"])' 2>/dev/null
}
if docker compose version >/dev/null 2>&1; then
  # the env file gpu-mode passes, written by the loader from the temp config
  envf="$(CLUB3090_CONFIG_DIR="$CFG" python3 "$ROOT/scripts/lib/club_config.py" compose-env-file --root "$T/no-repo")"
  [ "$(compose_key "$envf")" = "$KEY" ] && ok "a key stored only in secrets.env reaches the gateway's environment" \
    || bad "docker compose config did not interpolate the stored key into LITELLM_MASTER_KEY"
  rm -f "$envf"
  [ "$(compose_key /dev/null)" = "$DEFAULT" ] && ok "nothing stored → the public default (existing installs keep working)" \
    || bad "with nothing stored the gateway must fall back to $DEFAULT"
else
  echo "  SKIP (docker compose not available): interpolation legs"
fi
command grep -qE '^\s*-\s*LITELLM_MASTER_KEY=\$\{LITELLM_MASTER_KEY:-sk-litellm-master-key\}\s*$' "$ROOT/services/litellm/docker-compose.yml" \
  && ok "the compose line is \${LITELLM_MASTER_KEY:-sk-litellm-master-key}" \
  || bad "services/litellm/docker-compose.yml must read LITELLM_MASTER_KEY=\${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

# ── 2. the agent setup scripts use the stored key ────────────────────────────
echo "2. omp / pi / Hermes setup"
A="$T/agent"
omp_key() { sed -n 's/^[[:space:]]*apiKey:[[:space:]]*//p' "$A/models.yml" | head -1; }
CLUB3090_CONFIG_DIR="$CFG" PI_CODING_AGENT_DIR="$A" bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1 || bad "omp-setup.sh failed"
[ "$(omp_key)" = "$KEY" ] && ok "omp-setup.sh writes the key from secrets.env" || bad "omp-setup.sh wrote apiKey '$(omp_key)', not the stored key"
rm -rf "$A"
CLUB3090_CONFIG_DIR="$EMPTY" PI_CODING_AGENT_DIR="$A" bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1
[ "$(omp_key)" = "$DEFAULT" ] && ok "omp-setup.sh: nothing stored → the public default" || bad "omp-setup.sh with nothing stored wrote '$(omp_key)'"
rm -rf "$A"
CLUB3090_CONFIG_DIR="$CFG" PI_CODING_AGENT_DIR="$A" LITELLM_MASTER_KEY=key-from-shell bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1
[ "$(omp_key)" = "key-from-shell" ] && ok "omp-setup.sh: LITELLM_MASTER_KEY in the shell wins (a gateway on another box)" \
  || bad "omp-setup.sh must let a shell LITELLM_MASTER_KEY win (wrote '$(omp_key)')"
rm -rf "$A"

: > "$T/gw-hdr"
CLUB3090_CONFIG_DIR="$CFG" PI_CODING_AGENT_DIR="$A" bash "$ROOT/scripts/pi-setup.sh" --gateway "$GW/v1" >/dev/null 2>&1 || bad "pi-setup.sh failed"
pi_out="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["providers"]["club"]; print(c["apiKey"], ",".join(m["id"] for m in c["models"]))' "$A/models.json" 2>/dev/null)"
[ "$pi_out" = "$KEY qwen3.8-27b,gemma-4-31b" ] && ok "pi-setup.sh writes the stored key and reads the gateway with it (both routes listed)" \
  || bad "pi-setup.sh: expected the stored key and both gateway routes, got '${pi_out/$KEY/<key>}' (header sent: $(sed "s/$KEY/<key>/" "$T/gw-hdr"))"
rm -rf "$A"

mkdir -p "$T/hermes"
CLUB3090_CONFIG_DIR="$CFG" HERMES_HOME="$T/hermes" bash "$ROOT/scripts/hermes-setup.sh" --gateway "$GW/v1" >/dev/null 2>&1 || bad "hermes-setup.sh failed"
h_out="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["providers"]["club"]; print(c["api_key"], ",".join(sorted(c["models"])))' "$T/hermes/config.yaml" 2>/dev/null)"
[ "$h_out" = "$KEY gemma-4-31b,qwen3.8-27b" ] && ok "hermes-setup.sh writes the stored key and reads the gateway with it" \
  || bad "hermes-setup.sh: expected the stored key and both gateway routes, got '${h_out/$KEY/<key>}'"
rm -rf "$T/hermes"

# ── 3. gateway-key.sh ────────────────────────────────────────────────────────
echo "3. scripts/gateway-key.sh"
G="$T/gk"; mkdir -p "$G"; GK="$ROOT/scripts/gateway-key.sh"
gk() { CLUB3090_CONFIG_DIR="$G" CLUB3090_GATEWAY_URL="$GW" bash "$GK" "$@"; }
stored() { sed -n 's/^LITELLM_MASTER_KEY=//p' "$G/secrets.env" 2>/dev/null; }

out="$(gk status 2>&1)"; rc=$?
[[ $rc -eq 0 && "$out" == *"PUBLIC DEFAULT"* ]] && ok "status: nothing stored → says it is the public default" || bad "status on a fresh config: rc=$rc $out"

printf 'HF_TOKEN=hf_keepme\n' > "$G/secrets.env"; chmod 644 "$G/secrets.env"   # a looser file made by hand
out="$(gk rotate 2>&1)"; rc=$?
k1="$(stored)"
[[ $rc -eq 0 && "$k1" =~ ^sk-club-[0-9a-f]{32}$ ]] && ok "rotate stores a random sk-club-<32 hex> key" || bad "rotate: rc=$rc, stored key has the wrong shape"
[ "$(stat -c %a "$G/secrets.env")" = "600" ] && ok "secrets.env is 0600 after a rotate (a looser one is tightened)" || bad "secrets.env mode is $(stat -c %a "$G/secrets.env"), want 600"
[[ $rc -eq 0 ]] && command grep -qx 'HF_TOKEN=hf_keepme' "$G/secrets.env" && ok "rotate keeps the other secrets" || bad "rotate lost HF_TOKEN from secrets.env (rc=$rc)"
[ "$(command grep -c '^LITELLM_MASTER_KEY=' "$G/secrets.env")" = "1" ] || bad "secrets.env must hold exactly one LITELLM_MASTER_KEY line"
[[ -n "$k1" && "$out" != *"$k1"* ]] && ok "rotate never prints the key" || bad "rotate printed the key"
[[ "$out" == *" $ROOT/scripts/gpu-mode.sh gateway"* && "$out" == *"gateway-key.sh status"* && "$out" == *"rotate --apply"* ]] \
  && ok "rotate prints the command that recreates only the gateway (gpu-mode gateway), how to check it took, and --apply" \
  || bad "rotate must print gpu-mode gateway + the check + --apply: $out"

out="$(gk status 2>&1)"
[[ "$out" == *"your own (per install)"* && "$out" != *"$k1"* ]] && ok "status after a rotate: your own key, not printed" || bad "status after rotate: $out"
[[ "$out" == *"REFUSES it"* ]] && ok "status: a gateway still on another key is reported as refusing it" || bad "status must say the running gateway refuses the new key: $out"
stored > "$T/gw-key"
out="$(gk status 2>&1)"
[[ "$out" == *"accepts it"* ]] && ok "status: a gateway recreated on the key accepts it" || bad "status must say the gateway accepts the stored key: $out"

gk rotate >/dev/null 2>&1; k2="$(stored)"
[[ -n "$k2" && "$k2" != "$k1" ]] && ok "a second rotate stores a different key" || bad "a second rotate must change the key"

before="$(md5sum < "$G/secrets.env")"
out="$(LITELLM_MASTER_KEY=key-in-shell gk rotate 2>&1)"; rc=$?
[[ $rc -ne 0 && "$(md5sum < "$G/secrets.env")" == "$before" && "$out" == *"set in this shell"* ]] \
  && ok "rotate refuses while the shell sets LITELLM_MASTER_KEY (it would shadow the new key)" || bad "rotate with a shell key: rc=$rc $out"
printf 'LITELLM_MASTER_KEY=key-in-settings\n' > "$G/club3090.env"
out="$(gk rotate 2>&1)"; rc=$?
[[ $rc -ne 0 && "$(md5sum < "$G/secrets.env")" == "$before" && "$out" == *"club3090.env"* ]] \
  && ok "rotate refuses while club3090.env sets it (it wins over secrets.env)" || bad "rotate with a club3090.env key: rc=$rc $out"
rm -f "$G/club3090.env"

# agent setups: omp on this rig's gateway, pi pointed at another box, Hermes on this rig
mkdir -p "$HOME/.omp/agent" "$HOME/.pi/agent" "$HOME/.hermes"
CLUB3090_CONFIG_DIR="$G" PI_CODING_AGENT_DIR="$HOME/.omp/agent" bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1
CLUB3090_CONFIG_DIR="$G" PI_CODING_AGENT_DIR="$HOME/.pi/agent" bash "$ROOT/scripts/pi-setup.sh" --gateway http://gpu-box.invalid:4000/v1 >/dev/null 2>&1
CLUB3090_CONFIG_DIR="$G" HERMES_HOME="$HOME/.hermes" bash "$ROOT/scripts/hermes-setup.sh" --gateway "$GW/v1" >/dev/null 2>&1
out="$(gk rotate 2>&1)"
[[ "$out" == *"bash $ROOT/scripts/omp-setup.sh"$'\n'* ]] && ok "rotate lists the omp setup to re-run" || bad "rotate must list omp-setup.sh: $out"
[[ "$out" == *"bash $ROOT/scripts/hermes-setup.sh --gateway $GW/v1"* ]] && ok "rotate lists the Hermes setup with the gateway it points at" || bad "rotate must list hermes-setup.sh --gateway $GW/v1: $out"
[[ "$out" == *"pi points at http://gpu-box.invalid:4000/v1, not this rig's gateway"* && "$out" != *"scripts/pi-setup.sh"* ]] \
  && ok "rotate leaves out an agent pointed at another box's gateway" || bad "rotate must not tell pi (another box) to take this key: $out"

# ── 4. gpu-mode: the status probe and the start warning ──────────────────────
echo "4. scripts/gpu-mode.sh"
# A scratch copy, so its sibling lib/litellm-sync.sh is a stub: the real one would
# re-render the gateway config of whichever checkout this test runs in.
GM="$T/gm/scripts"; mkdir -p "$GM/lib"
cp "$ROOT/scripts/gpu-mode.sh" "$GM/"
cp "$ROOT/scripts/lib/club-config.sh" "$ROOT/scripts/lib/club_config.py" "$GM/lib/"
printf '#!/usr/bin/env bash\nexit 0\n' > "$GM/lib/litellm-sync.sh"
C="$T/club"; mkdir -p "$C/services/litellm"; echo "services: {}" > "$C/services/litellm/docker-compose.yml"
mkdir -p "$T/stub"
printf '{"name":"litellm","services":{"litellm":{"image":"ghcr.io/berriai/litellm:v1.100.3","container_name":"litellm"}}}\n' > "$T/stub/litellm.json"
cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == *=* ]]; do shift; done
exec "$@"
EOF
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
case "$1" in
  compose)
    if [[ " $* " == *" config "* ]]; then cat "$STUB_DIR/litellm.json"; fi
    if [[ " $* " == *" up -d"* ]]; then
      envf=""; prev=""
      for a in "$@"; do [[ "$prev" == --env-file ]] && envf="$a"; prev="$a"; done
      if [[ -n "$envf" && -n "${EXPECT_KEY:-}" ]] && command grep -qF "$EXPECT_KEY" "$envf"; then echo "UP-ENVFILE-HAS-KEY" >> "$STUB_LOG"; fi
    fi
    exit 0 ;;
  inspect)
    case "$4" in
      *Config.Image*)  echo "ghcr.io/berriai/litellm:v1.99.0" ;;    # drifted → upgrade starts it
      *State.Running*) echo true ;;
    esac ;;
  *) exit 0 ;;
esac
EOF
# curl: records argv, and the header when it comes on stdin (-H @-); answers the
# gateway probe only for the expected key.
cat > "$T/bin/curl" <<'EOF'
#!/usr/bin/env bash
hdr=""; url=""; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  a="${args[$i]}"
  if [[ "$a" == -H ]]; then v="${args[$((i + 1))]}"; if [[ "$v" == @- ]]; then hdr="$(cat)"; else hdr="$v"; fi; fi
  [[ "$a" == http* ]] && url="$a"
done
printf 'ARGV %s\n' "$*" >> "$CURL_LOG"
[[ "$url" == http://localhost:4000/* ]] && printf 'HDR %s\n' "$hdr" >> "$CURL_LOG"
if [[ "$url" == http://localhost:4000/v1/models && "$hdr" == "Authorization: Bearer $EXPECT_KEY" ]]; then
  echo '{"data":[{"id":"stub-model"}]}'; exit 0
fi
exit 7
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/du"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/nvidia-smi"
chmod +x "$T/bin/"*
for b in docker sudo curl; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done
export STUB_DIR="$T/stub" CLUB3090_DIR="$C"

export CURL_LOG="$T/curl1" STUB_LOG="$T/docker1" EXPECT_KEY="$KEY"; : > "$CURL_LOG"
out="$(CLUB3090_CONFIG_DIR="$CFG" PATH="$T/bin:$PATH" bash "$T/gm/scripts/gpu-mode.sh" status 2>&1)"
[[ "$out" == *"LiteLLM @ :4000"*"stub-model"* ]] && ok "status probes the gateway with the stored key" \
  || bad "status must reach the gateway with the stored key (sent: $(command grep '^HDR' "$CURL_LOG" | sed "s/$KEY/<key>/" | head -1))"
command grep -q '^ARGV .*Authorization' "$CURL_LOG" && bad "the gateway key must not be on curl's command line (visible in ps)" \
  || ok "no key on curl's command line (the header goes in on stdin)"

export CURL_LOG="$T/curl2" EXPECT_KEY="$DEFAULT"; : > "$CURL_LOG"
out="$(CLUB3090_CONFIG_DIR="$EMPTY" PATH="$T/bin:$PATH" bash "$T/gm/scripts/gpu-mode.sh" status 2>&1)"
[[ "$out" == *"LiteLLM @ :4000"*"stub-model"* ]] && ok "status: nothing stored → probes with the public default" || bad "status with nothing stored must probe with the default key"

export STUB_LOG="$T/docker3" EXPECT_KEY="$KEY"; : > "$STUB_LOG"
out="$(CLUB3090_CONFIG_DIR="$EMPTY" PATH="$T/bin:$PATH" bash "$T/gm/scripts/gpu-mode.sh" upgrade 2>&1)"
command grep -q 'docker compose .* up -d' "$STUB_LOG" || bad "fixture: upgrade did not start the (drifted) gateway"
[ "$(command grep -c 'public default key' <<<"$out")" = "1" ] && [[ "$out" == *"gateway-key.sh rotate"* ]] \
  && ok "starting the gateway on the public default prints one warning pointing at gateway-key.sh rotate" \
  || bad "gpu-mode must warn once when it starts the gateway on the default key: $out"
export STUB_LOG="$T/docker4"; : > "$STUB_LOG"
out="$(CLUB3090_CONFIG_DIR="$CFG" PATH="$T/bin:$PATH" bash "$T/gm/scripts/gpu-mode.sh" upgrade 2>&1)"
[[ "$out" != *"public default key"* ]] && ok "no warning when a key of its own is stored" || bad "gpu-mode must not warn with a key stored"
command grep -q 'UP-ENVFILE-HAS-KEY' "$STUB_LOG" && ok "gpu-mode's compose up gets the stored key in its --env-file" \
  || bad "gpu-mode's compose up must pass an --env-file carrying the stored key"
[[ "$out" != *"$KEY"* ]] || bad "gpu-mode printed the key"

[ "$fail" -eq 0 ] && echo "test-gateway-key: ok (compose interpolation + default, omp/pi/Hermes setups use the stored key, gateway-key.sh status/rotate: 0600, never printed, refuses a shadowed key, lists the setups to re-run; gpu-mode probes with it on stdin and warns on the default)"
exit "$fail"
