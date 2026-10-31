#!/usr/bin/env bash
#
# Guard: a server that answers /props is not llama.cpp until nothing names another engine.
#
# Why this exists: QEngine serves a llama.cpp-style GET /props from 1.3.1. verify-full.sh and
# verify-stress.sh took any /props answer as llama.cpp unless a container name or image said
# otherwise, so a host-run QEngine (CONTAINER=none) was classified llamacpp. And bench.sh's
# capture (cap_kv_type) inferred llama.cpp's default KV type, "f16", from a /props answer, so every
# QEngine bench from 1.3.1 on, container or not, reported f16 for an engine whose only KV format is
# FP8 e4m3 (seen in #1870's report). The server's own name, /v1/models owned_by, now decides first.
#
# Drives the real functions against a stub server: detect_engine() extracted from both scripts with
# CONTAINER=none, and cap_kv_type from scripts/lib/capture.sh.
# ⚠️ Must FAIL against the pre-fix tree.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; PIDS=()
trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; rm -rf "$TMP"' EXIT
FAIL=0
ok()  { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    expected=%s got=%s\n" "$2" "$3"; FAIL=1; }
check() { [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "$2" "$3"; }

# stub <var> <owned_by or ""> — start a server with a llama.cpp-style /props (QEngine 1.3.1's shape) and
# /v1/models, and put its port in <var>. Runs in this shell (never as "$(stub …)"), so the PID reaches
# the EXIT trap (test-tests-stub-lifecycle).
stub() {
  local port
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  python3 - "$port" "$2" <<'PY' >/dev/null 2>&1 &
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, owned = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/props":
            body = {"model_alias": "m", "total_slots": 32, "default_generation_settings": {"n_ctx": 262144}}
        elif self.path == "/v1/models":
            card = {"id": "m", "object": "model"}
            if owned:
                card["owned_by"] = owned
            body = {"object": "list", "data": [card]}
        else:
            self.send_response(404); self.end_headers(); return
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        self.send_response(404); self.end_headers()
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
  PIDS+=($!)
  for _ in $(seq 1 50); do curl -sf -m 1 "http://127.0.0.1:$port/props" >/dev/null 2>&1 && break; sleep 0.1; done
  printf -v "$1" '%s' "$port"
}

stub QE qengine; stub LC llamacpp; stub NONAME ""

echo "== 1. detect_engine, host-run (CONTAINER=none)"
for script in verify-full.sh verify-stress.sh; do
  for case in "qengine $QE qengine" "llamacpp $LC llamacpp" "no-owned_by $NONAME llamacpp"; do
    set -- $case
    got="$(bash -c 'source "$1/scripts/lib/engine-kind.sh"
      eval "$(sed -n "/^detect_engine() {/,/^}/p" "$1/scripts/$2")"
      URL="http://127.0.0.1:$3" CONTAINER=none MODEL=m detect_engine' _ "$ROOT" "$script" "$2" 2>/dev/null)"
    check "$script: /props + owned_by '$1' -> $3" "$3" "$got"
  done
done

echo "== 2. cap_kv_type (bench.sh's KV capture)"
kv() {  # <port> <argv> <kind>
  bash -c 'source "$1/scripts/lib/capture.sh"; URL="http://127.0.0.1:$2"; CAP_LOG=""
    cap_kv_type "$3" "$4" 2>/dev/null || echo UNAVAILABLE' _ "$ROOT" "$1" "$2" "$3"
}
check "QEngine, kind from owned_by -> fp8_e4m3" "fp8_e4m3 (QEngine's only KV format)" \
  "$(kv "$QE" "python3 server/app.py --model /app/models/Qwen3.8-27B-INT4" "")"
check "QEngine, kind from the caller -> fp8_e4m3" "fp8_e4m3 (QEngine's only KV format)" \
  "$(kv "$QE" "python3 server/app.py --model /app/models/Qwen3.8-27B-INT4" qengine)"
check "llama.cpp with no KV flag -> its f16 default (unchanged)" "f16 (engine default; no -ctk/-ctv in argv)" \
  "$(kv "$LC" "llama-server -m /m/x.gguf -ngl 99" "")"
check "a family that names another engine -> no f16 guess" "UNAVAILABLE" \
  "$(kv "$NONAME" "python3 -m some.server" exllamav3)"
check "an argv KV flag still wins" "q8_0 (source: argv)" \
  "$(kv "$QE" "llama-server -m x.gguf -ctk q8_0" "")"

if (( FAIL )); then echo "test-engine-detection-props: FAIL"; exit 1; fi
echo "test-engine-detection-props: ok"
