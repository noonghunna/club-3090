#!/usr/bin/env bash
# Regenerate studio_pipe.py and push it into the RUNNING Open WebUI function, then reload.
#
# Why this exists: OWUI stores the pipe CODE in its own DB (the `function` table), NOT read from
# the file on disk. So editing build_studio_pipe.py + regenerating studio_pipe.py is NOT enough —
# the installed function stays on the old code (the "stale function" trap). This script:
#   1. builds studio_pipe.py
#   2. copies it into the open-webui container + UPSERTs the function row's `content`
#      (INSTALLs the function if it doesn't exist yet, else UPDATEs it — no manual paste needed)
#   3. restarts open-webui so it reloads the function (also refreshes the lane picker)
#
# First-time install needs an OWUI admin account to exist (create it at the OWUI URL first); after
# that this both installs and updates. setup-ai-studio.sh calls this as its pipe-install step.
#
# Usage:
#   bash services/studio/push-pipe-to-owui.sh                 # build → install/update → reload
#   bash services/studio/push-pipe-to-owui.sh --no-reload     # build → install/update, but don't restart OWUI
#   bash services/studio/push-pipe-to-owui.sh <function_id> <owui_container>   # defaults: studio open-webui
# Env: OWUI_DB (default /app/backend/data/webui.db) · DOCKER (default "sudo docker") · OWUI_HEALTH_URL
#      OWUI_FN_NAME (display name for a first-time install, default "🎬 Studio")
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RELOAD=1
POS=()
for a in "$@"; do
    case "$a" in
        --no-reload) RELOAD=0 ;;
        *) POS+=("$a") ;;
    esac
done
FN_ID="${POS[0]:-studio}"
OWUI="${POS[1]:-open-webui}"
DB="${OWUI_DB:-/app/backend/data/webui.db}"
DOCKER="${DOCKER:-sudo docker}"
HEALTH_URL="${OWUI_HEALTH_URL:-http://localhost:${OWUI_PORT:-8080}/health}"

echo "[push] building studio_pipe.py …"
python3 "$HERE/build_studio_pipe.py"
PIPE="$HERE/studio_pipe.py"
[ -f "$PIPE" ] || { echo "[push] ERROR: $PIPE not found"; exit 1; }
echo "[push] pipe = $(wc -c < "$PIPE") bytes"

# #715 gap 6 — stamp the configured LAN IP into the pipe's browser_base default so
# gallery/media links open from OTHER machines without the manual valve step (the
# runtime cousin of the #703 LANIP class). LANIP: env wins, else repo .env; unset /
# localhost keeps the localhost default (single-machine rigs). Stamps a TEMP copy —
# the tracked studio_pipe.py is never mutated.
_lanip="${LANIP:-}"
if [ -z "$_lanip" ] && [ -f "$HERE/../../.env" ]; then
    _lanip="$(grep -E '^LANIP=' "$HERE/../../.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
fi
_lanip="${_lanip%\"}"; _lanip="${_lanip#\"}"
PIPE_PUSH="$PIPE"
if [ -n "$_lanip" ] && [ "$_lanip" != "localhost" ] && [ "$_lanip" != "127.0.0.1" ]; then
    PIPE_PUSH="$(mktemp /tmp/studio_pipe_push.XXXXXX.py)"
    sed "s|http://localhost:8189|http://${_lanip}:8189|g" "$PIPE" > "$PIPE_PUSH"
    echo "[push] browser_base default stamped → http://${_lanip}:8189 (#715)"
fi

$DOCKER ps --format '{{.Names}}' | grep -qx "$OWUI" || { echo "[push] ERROR: container '$OWUI' is not running"; exit 1; }

echo "[push] copy → $OWUI : install/update function '$FN_ID' in $DB …"
$DOCKER cp "$PIPE_PUSH" "$OWUI:/tmp/_studio_pipe_push.py"
$DOCKER exec -e FN_ID="$FN_ID" -e DB="$DB" -e FN_NAME="${OWUI_FN_NAME:-🎬 Studio}" "$OWUI" python -c '
import os, sqlite3, time, json, sys
fn_id, db, fn_name = os.environ["FN_ID"], os.environ["DB"], os.environ["FN_NAME"]
new = open("/tmp/_studio_pipe_push.py").read()
now = int(time.time())
c = sqlite3.connect(db); cur = c.cursor()
exists = cur.execute("SELECT 1 FROM function WHERE id=?", (fn_id,)).fetchone()
if exists:
    cur.execute("UPDATE function SET content=?, updated_at=? WHERE id=?", (new, now, fn_id))
    action = "updated"
else:
    admin = cur.execute("SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1", ("admin",)).fetchone()
    if not admin:
        sys.exit("[push] OWUI has no admin user yet — create your account at the OWUI URL first, then re-run.")
    meta = json.dumps({"description": "Studio: a casual idea -> a director LLM crafts it -> image / video / audio."})
    cur.execute(
        "INSERT INTO function (id,user_id,name,type,content,meta,created_at,updated_at,valves,is_active,is_global) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (fn_id, admin[0], fn_name, "pipe", new, meta, now, now, None, 1, 1))
    action = "installed"
c.commit()
print("[push] %s row:" % action, cur.execute("SELECT id,name,type,is_active,length(content) FROM function WHERE id=?", (fn_id,)).fetchone())
c.close()
'
$DOCKER exec "$OWUI" rm -f /tmp/_studio_pipe_push.py 2>/dev/null || true

if [ "$RELOAD" = 1 ]; then
    echo "[push] restarting $OWUI to reload the function …"
    $DOCKER restart "$OWUI" >/dev/null
    for i in $(seq 1 30); do
        curl -sf -m 3 "$HEALTH_URL" >/dev/null 2>&1 && { echo "[push] ✓ OWUI up — pipe is live"; exit 0; }
        sleep 3
    done
    echo "[push] WARN: OWUI didn't report healthy in time — check '$DOCKER logs $OWUI'"
else
    echo "[push] --no-reload: DB updated. Restart OWUI to make it live: $DOCKER restart $OWUI"
fi
