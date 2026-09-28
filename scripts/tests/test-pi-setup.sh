#!/usr/bin/env bash
# test-pi-setup — scripts/pi-setup.sh manages ONE `club` provider in pi's
# models.json, from a snapshot of the gateway, and never touches anything else.
#
# WHY THIS TEST EXISTS
# --------------------
# models.json is the user's file: it holds their other providers and their API
# keys. A setup script that dropped a provider, replaced a `club` the user wrote
# themselves, or loosened the file's permissions would destroy or expose real
# configuration. pi has no gateway discovery, so the script also has to carry the
# gateway's per-route context window over correctly and fall back safely when the
# gateway is down. Offline: a stub stands in for the gateway's /model_group/info,
# and PI_CODING_AGENT_DIR points pi's agent dir at a scratch directory; pi itself
# is only used, when installed, to confirm it accepts the file.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; STUB_PID=""
trap '[ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; rm -rf "$T"' EXIT
export PI_CODING_AGENT_DIR="$T/agent"
M="$PI_CODING_AGENT_DIR/models.json"
fail=0
bad() { echo "✗ $1" >&2; fail=1; }
jq_() { python3 - "$M" "$1" <<'PY'
import json, sys
cur = json.load(open(sys.argv[1], encoding="utf-8"))
for k in sys.argv[2].split("/"):       # "/" — model ids contain dots; a number indexes a list
    cur = (cur[int(k)] if isinstance(cur, list) and k.isdigit() and int(k) < len(cur)
           else cur.get(k) if isinstance(cur, dict) else None)
print("" if cur is None else json.dumps(cur) if isinstance(cur, (dict, list, bool)) else cur)
PY
}
DOWN="http://127.0.0.1:1/v1"             # nothing listens there: the gateway-down path

# a stub gateway: two engine routes with model_info, one cloud route without
cat > "$T/stub.py" <<'PY'
import http.server, json, sys
BODY = json.dumps({"data": [
    {"model_group": "qwen3.8-27b", "max_input_tokens": 163840.0, "max_output_tokens": 32768.0},
    {"model_group": "gemma-4-31b", "max_input_tokens": 131072.0, "max_output_tokens": 8192.0, "supports_reasoning": False},
    {"model_group": "my-cloud-model", "max_input_tokens": None},
]}).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/model_group/info" and self.headers.get("Authorization", "").startswith("Bearer ")
        self.send_response(200 if ok else 404); self.end_headers()
        if ok: self.wfile.write(BODY)
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(s.server_port))
s.serve_forever()
PY
python3 "$T/stub.py" "$T/port" & STUB_PID=$!
for _ in $(seq 1 50); do [ -s "$T/port" ] && break; sleep 0.1; done
STUB="http://127.0.0.1:$(cat "$T/port")/v1"

# 1. --print changes nothing
bash "$ROOT/scripts/pi-setup.sh" --print --gateway "$DOWN" >/dev/null 2>&1
[ ! -e "$M" ] || bad "--print must not write models.json"

# 2. gateway down: qwen3.8-27b with the fallback window, file created 0600
bash "$ROOT/scripts/pi-setup.sh" --gateway "$DOWN" >/dev/null 2>&1 || bad "a run with the gateway down must still succeed"
[ "$(jq_ providers/club/models/0/id)" = "qwen3.8-27b" ] || bad "qwen3.8-27b must always be listed"
[ "$(jq_ providers/club/models/0/contextWindow)" = "262144" ] || bad "gateway down: the fallback window is 262144 (got '$(jq_ providers/club/models/0/contextWindow)')"
[ "$(jq_ providers/club/models/1/id)" = "" ] || bad "gateway down: nothing but qwen3.8-27b may be listed"
[ "$(stat -c %a "$M")" = "600" ] || bad "a new models.json holds API keys and must be created 0600 (got $(stat -c %a "$M"))"

# 3. the Qwen3.8 thinking wiring: effort as a template kwarg, only the levels the template has
[ "$(jq_ providers/club/models/0/compat/thinkingFormat)" = "chat-template" ] || bad "qwen3.8-27b must use thinkingFormat: chat-template"
[ "$(jq_ providers/club/models/0/compat/chatTemplateKwargs/reasoning_effort)" = '{"$var": "thinking.effort"}' ] \
  || bad "the effort must ride chat_template_kwargs.reasoning_effort (got '$(jq_ providers/club/models/0/compat/chatTemplateKwargs/reasoning_effort)')"
[ "$(jq_ providers/club/models/0/compat/chatTemplateKwargs/enable_thinking)" = '{"$var": "thinking.enabled"}' ] \
  || bad "thinking on/off must ride chat_template_kwargs.enable_thinking"
[ "$(jq_ providers/club/models/0/thinkingLevelMap)" = '{"minimal": null, "low": "low", "medium": "medium", "high": null, "xhigh": "xhigh", "max": null}' ] \
  || bad "thinkingLevelMap must offer exactly low/medium/xhigh (got '$(jq_ providers/club/models/0/thinkingLevelMap)')"
[ "$(jq_ providers/club/models/0/maxTokens)" = "32768" ] || bad "qwen3.8-27b must pin maxTokens: 32768"
[ "$(jq_ providers/club/compat/supportsDeveloperRole)" = "false" ] || bad "the Qwen3.8 template has no developer role: supportsDeveloperRole must be false"

# 4. gateway up: its windows are carried over; a route without model_info is skipped
bash "$ROOT/scripts/pi-setup.sh" --gateway "$STUB" >/dev/null 2>&1 || bad "a run against the stub gateway failed"
[ "$(jq_ providers/club/models/0/contextWindow)" = "163840" ] || bad "the gateway's window must be used (got '$(jq_ providers/club/models/0/contextWindow)')"
[ "$(jq_ providers/club/models/1/id)" = "gemma-4-31b" ] || bad "every engine route the gateway serves must be listed"
[ "$(jq_ providers/club/models/1/maxTokens)" = "8192" ] || bad "a non-Qwen route takes the gateway's reply cap"
[ "$(jq_ providers/club/models/1/compat)" = "" ] || bad "a non-Qwen route must not get the Qwen3.8 template kwargs"
[ "$(jq_ providers/club/models/2/id)" = "" ] || bad "a route without model_info (not a local engine) must be skipped"

# 5. existing file with another provider: kept exactly, mode kept, backup taken
cat > "$M" <<'EOF'
{"providers": {"modelscope": {"baseUrl": "https://api-inference.example/v1", "api": "openai-completions", "apiKey": "MY_SECRET", "models": [{"id": "x"}]}}}
EOF
chmod 640 "$M"; rm -f "$M".bak-*
before="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["providers"]["modelscope"], sort_keys=True))' "$M")"
bash "$ROOT/scripts/pi-setup.sh" --gateway "$STUB" >/dev/null 2>&1 || bad "a run over an existing file failed"
after="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["providers"]["modelscope"], sort_keys=True))' "$M")"
[ "$before" = "$after" ] || bad "another provider must survive exactly"
[ "$(stat -c %a "$M")" = "640" ] || bad "an existing file's mode must be kept (got $(stat -c %a "$M"))"
ls "$M".bak-* >/dev/null 2>&1 || bad "the previous file must be backed up"

# 6. re-run refreshes in place: same result, one club
cp "$M" "$T/first.json"
bash "$ROOT/scripts/pi-setup.sh" --gateway "$STUB" >/dev/null 2>&1 || bad "re-run failed"
cmp -s "$M" "$T/first.json" || bad "a re-run against the same gateway must give the same file"

# 7. a hand-written `club` provider is refused, file unchanged, no backup
echo '{"providers": {"club": {"baseUrl": "http://my-own-thing/v1", "api": "openai-completions"}}}' > "$M"
rm -f "$M".bak-*; before="$(md5sum < "$M")"
if bash "$ROOT/scripts/pi-setup.sh" --gateway "$STUB" >/dev/null 2>&1; then bad "must refuse to overwrite a hand-written club provider"; fi
[ "$(md5sum < "$M")" = "$before" ] || bad "a refused run must leave models.json unchanged"
ls "$M".bak-* >/dev/null 2>&1 && bad "a refused run must not write a backup either"

# 8. the Qwen3.8 id list is omp-setup's (test-omp-setup checks that one against the composes)
python3 - "$ROOT" <<'PY' || bad "pi-setup.sh QWEN38_IDS must equal omp-setup.sh's"
import ast, io, re, sys
ids = lambda f: ast.literal_eval(re.search(r"^QWEN38_IDS = (\[.*\])$", io.open(f"{sys.argv[1]}/scripts/{f}", encoding="utf-8").read(), re.M).group(1))
sys.exit(0 if ids("pi-setup.sh") == ids("omp-setup.sh") else 1)
PY

# 10. the JSON block docs/CODING_AGENTS.md shows is what the script writes
#     (gateway down → the 262,144 fallback, which is also the dual-fast window)
python3 - "$ROOT" "$DOWN" <<'PY2' || bad "docs/CODING_AGENTS.md's pi models.json block and pi-setup.sh --print have drifted apart"
import io, json, re, subprocess, sys
root, down = sys.argv[1], sys.argv[2]
doc = io.open(f"{root}/docs/CODING_AGENTS.md", encoding="utf-8").read()
block = json.loads(re.search(r"```json\n(.*?)```", doc.split("## pi — setup", 1)[1], re.S).group(1))
out = json.loads(subprocess.run(["bash", f"{root}/scripts/pi-setup.sh", "--print", "--gateway", down],
                                capture_output=True, text=True, encoding="utf-8").stdout)
out["providers"]["club"]["baseUrl"] = "http://127.0.0.1:4000/v1"     # the default the doc shows
sys.exit(0 if block == out else 1)
PY2

# 9. pi itself accepts the file (when pi is installed)
rm -f "$M" "$M".bak-*; bash "$ROOT/scripts/pi-setup.sh" --gateway "$STUB" >/dev/null 2>&1
if command -v pi >/dev/null 2>&1; then
  out="$(timeout 60 pi --offline --list-models club 2>&1)"
  echo "$out" | command grep -q 'qwen3.8-27b' || bad "pi must list club/qwen3.8-27b from the generated file: $(echo "$out" | head -3)"
else
  echo "SKIP (pi acceptance): pi is not installed; the structural checks ran"
fi

[ "$fail" -eq 0 ] && echo "test-pi-setup: ok (print-only, gateway down → fallback + 0600, Qwen3.8 thinking wiring, gateway windows, other providers + mode kept, idempotent, refuses a hand-written club, ids == omp-setup's, doc block == script, pi accepts it)"
exit "$fail"
