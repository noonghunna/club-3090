#!/usr/bin/env bash
# Register a running model endpoint as an Open WebUI OpenAI connection, so a
# catalog model launched via switch.sh --owui auto-appears in the OWUI chat picker.
#
#   owui-register.sh <port> [owui_container]
#
# OPT-IN + CONDITIONAL: it's a no-op (exit 0) if Open WebUI isn't running — most
# catalog users drive models from the API directly (aider/opencode/agents) and
# never run OWUI, so this must never block or fail a launch.
#
# OWUI stores connections in its DB (PersistentConfig, not env), so we drive its
# admin config API with a short-lived HS256 token forged from the container's
# secret (/app/backend/.webui_secret_key) + the admin user id. Idempotent: skips
# if the endpoint is already registered. host.docker.internal:<port> is the
# container→host path (works on the bundle's OWUI; on a setup without it, add the
# host IP instead via Admin → Settings → Connections).
set -uo pipefail
PORT="${1:?usage: owui-register.sh <port> [owui_container]}"
OWUI="${2:-${OWUI_CONTAINER:-open-webui}}"
log(){ echo "[owui-register] $*"; }

if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$OWUI"; then
  log "Open WebUI ('$OWUI') not running — skipping (it's optional; start OWUI + re-run with --owui to wire)."
  exit 0
fi
SECRET="$(docker exec "$OWUI" cat /app/backend/.webui_secret_key 2>/dev/null || true)"
if [[ -z "$SECRET" ]]; then log "couldn't read OWUI secret key — skipping."; exit 0; fi

docker exec -i "$OWUI" python3 - "$SECRET" "$PORT" <<'PY'
import sys, sqlite3, hmac, hashlib, base64, json, urllib.request
secret = sys.argv[1].encode(); port = sys.argv[2]
db = "/app/backend/data/webui.db"
try:
    admins = [r[0] for r in sqlite3.connect(db).execute(
        "select id from user where role='admin' order by created_at limit 1")]
except Exception as e:
    print("[owui-register] cannot read OWUI db (%s) — skipping." % e); sys.exit(0)
if not admins:
    print("[owui-register] no admin user yet — open the UI + create one, then re-run with --owui. Skipping.")
    sys.exit(0)
b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
hdr = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
pl  = b64(json.dumps({"id": admins[0]}).encode())
jwt = (hdr + b"." + pl + b"." + b64(hmac.new(secret, hdr + b"." + pl, hashlib.sha256).digest())).decode()
H = {"Authorization": "Bearer " + jwt, "Content-Type": "application/json"}
def call(path, data=None):
    req = urllib.request.Request("http://localhost:8080" + path, headers=H,
                                 data=(json.dumps(data).encode() if data is not None else None))
    return json.load(urllib.request.urlopen(req, timeout=30))
url = "http://host.docker.internal:%s/v1" % port
try:
    cfg = call("/openai/config")
except Exception as e:
    print("[owui-register] OWUI config API unreachable (%s) — skipping." % e); sys.exit(0)
urls = cfg.get("OPENAI_API_BASE_URLS") or []; keys = cfg.get("OPENAI_API_KEYS") or []
if url in urls:
    print("[owui-register] already registered: %s" % url); sys.exit(0)
urls.append(url); keys.append("sk-noauth")
new = dict(cfg); new.update({"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": urls, "OPENAI_API_KEYS": keys})
try:
    call("/openai/config/update", new)
    served = [m["id"] for m in call("/openai/models").get("data", [])]
    print("[owui-register] ✓ wired %s into Open WebUI; picker now lists: %s"
          % (url, ", ".join(served[-3:]) if served else "(endpoint up)"))
except Exception as e:
    print("[owui-register] update failed (%s) — add it manually in Admin → Settings → Connections." % e)
PY
