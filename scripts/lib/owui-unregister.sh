#!/usr/bin/env bash
# Remove an Open WebUI OpenAI connection (the inverse of owui-register.sh), so a
# retired/duplicate endpoint stops appearing in the OWUI chat picker.
#
#   owui-unregister.sh <port> [owui_container]
#
# OPT-IN + CONDITIONAL: it's a no-op (exit 0) if Open WebUI isn't running, has no
# admin user yet, or the connection isn't present — so it never blocks a launch or
# a setup re-run. Use it to drop a connection whose model list is misleading (e.g.
# the always-up LiteLLM :4000 gateway that lists every catalog model regardless of
# which gpu-mode scene is actually serving — see services/openwebui/docker-compose.yml).
#
# Mirrors owui-register.sh exactly: OWUI stores connections in its DB
# (PersistentConfig, not env), so we drive its admin config API with a short-lived
# HS256 token forged from the container's secret (/app/backend/.webui_secret_key) +
# the admin user id. Removes http://host.docker.internal:<port>/v1 and its paired key.
set -uo pipefail
PORT="${1:?usage: owui-unregister.sh <port> [owui_container]}"
OWUI="${2:-${OWUI_CONTAINER:-open-webui}}"
DOCKER="${DOCKER:-docker}"
log(){ echo "[owui-unregister] $*"; }

if ! $DOCKER ps --format '{{.Names}}' 2>/dev/null | grep -qx "$OWUI"; then
  log "Open WebUI ('$OWUI') not running — skipping (nothing to unregister)."
  exit 0
fi
SECRET="$($DOCKER exec "$OWUI" cat /app/backend/.webui_secret_key 2>/dev/null || true)"
if [[ -z "$SECRET" ]]; then log "couldn't read OWUI secret key — skipping."; exit 0; fi

$DOCKER exec -i "$OWUI" python3 - "$SECRET" "$PORT" <<'PY'
import sys, sqlite3, hmac, hashlib, base64, json, urllib.request
secret = sys.argv[1].encode(); port = sys.argv[2]
db = "/app/backend/data/webui.db"
try:
    admins = [r[0] for r in sqlite3.connect(db).execute(
        "select id from user where role='admin' order by created_at limit 1")]
except Exception as e:
    print("[owui-unregister] cannot read OWUI db (%s) — skipping." % e); sys.exit(0)
if not admins:
    print("[owui-unregister] no admin user yet — skipping."); sys.exit(0)
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
    print("[owui-unregister] OWUI config API unreachable (%s) — skipping." % e); sys.exit(0)
urls = cfg.get("OPENAI_API_BASE_URLS") or []; keys = cfg.get("OPENAI_API_KEYS") or []
if url not in urls:
    print("[owui-unregister] not registered (nothing to do): %s" % url); sys.exit(0)
# Drop every occurrence of url, keeping keys index-aligned.
new_urls, new_keys = [], []
for i, u in enumerate(urls):
    if u == url:
        continue
    new_urls.append(u); new_keys.append(keys[i] if i < len(keys) else "")
new = dict(cfg); new.update({"OPENAI_API_BASE_URLS": new_urls, "OPENAI_API_KEYS": new_keys})
try:
    call("/openai/config/update", new)
    print("[owui-unregister] ✓ removed %s from Open WebUI (%d connection(s) remain)." % (url, len(new_urls)))
except Exception as e:
    print("[owui-unregister] update failed (%s) — remove it manually in Admin → Settings → Connections." % e)
PY
