#!/usr/bin/env bash
# test-hermes-setup — scripts/hermes-setup.sh manages ONE `club` provider in Hermes
# Agent's config.yaml, from a snapshot of the gateway, and touches nothing else.
#
# WHY THIS TEST EXISTS
# --------------------
# config.yaml is the user's Hermes configuration: their default model and provider,
# their other providers and keys, their comments. The script must add or refresh
# only `providers.club`, refuse a `club` it didn't write, back the file up first,
# drop a model that fell out of the gateway snapshot on a refresh, and fall back
# safely when the gateway is down. Real Hermes runs a heavy first-run install in a
# fresh HERMES_HOME, so a stub `hermes` stands in for it (`config get/set/unset`
# against the same config.yaml), and a stub serves the gateway's /model_group/info.
# The live write through real Hermes was checked by hand (docs/CODING_AGENTS.md).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; STUB_PID=""
trap '[ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; rm -rf "$T"' EXIT
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

if ! python3 -c 'import yaml' 2>/dev/null; then
  echo "SKIP: test-hermes-setup needs python3 with PyYAML (for the stub hermes)"
  exit 0
fi

export HERMES_HOME="$T/home"; mkdir -p "$HERMES_HOME"
C="$HERMES_HOME/config.yaml"
# the stub `hermes`: just the three config verbs the script uses
mkdir -p "$T/bin"; cat > "$T/bin/hermes" <<'PY'
#!/usr/bin/env python3
import os, sys, yaml
path = os.path.join(os.environ["HERMES_HOME"], "config.yaml")
doc = (yaml.safe_load(open(path)) if os.path.exists(path) else None) or {}
verb, key = sys.argv[2], sys.argv[3]
parts = key.split(".")
def walk(create=False):
    cur = doc
    for p in parts[:-1]:
        if not isinstance(cur, dict) or (p not in cur and not create): return None
        cur = cur.setdefault(p, {})
    return cur
if verb == "get":
    parent = walk()
    if not isinstance(parent, dict) or parts[-1] not in parent:
        print(f"Config key not set: {key}"); sys.exit(1)
    v = parent[parts[-1]]
    print(yaml.safe_dump(v, sort_keys=False).rstrip() if isinstance(v, (dict, list)) else v); sys.exit(0)
if verb == "set":
    val = sys.argv[4]
    walk(create=True)[parts[-1]] = yaml.safe_load(val) if val.strip()[:1] in "{[" else val
elif verb == "unset":
    parent = walk()
    if isinstance(parent, dict): parent.pop(parts[-1], None)
open(path, "w").write(yaml.safe_dump(doc, sort_keys=False))
PY
chmod +x "$T/bin/hermes"
export HERMES_BIN="$T/bin/hermes"
cfg() { python3 - "$C" "$1" <<'PY'
import json, sys, yaml
cur = yaml.safe_load(open(sys.argv[1])) or {}
for k in sys.argv[2].split("/"):
    cur = cur.get(k) if isinstance(cur, dict) else None
print("" if cur is None else json.dumps(cur, sort_keys=True) if isinstance(cur, (dict, list)) else cur)
PY
}
DOWN="http://127.0.0.1:1/v1"

# a stub gateway: two engine routes with model_info, one cloud route without
cat > "$T/gw.py" <<'PY'
import http.server, json, sys
BODY = json.dumps({"data": [
    {"model_group": "qwen3.8-27b", "max_input_tokens": 163840.0},
    {"model_group": "gemma-4-31b", "max_input_tokens": 131072.0},
    {"model_group": "my-cloud-model", "max_input_tokens": None},
]}).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/model_group/info" and self.headers.get("Authorization", "").startswith("Bearer ")
        self.send_response(200 if ok else 404); self.end_headers()
        if ok: self.wfile.write(BODY)
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(s.server_port)); s.serve_forever()
PY
python3 "$T/gw.py" "$T/port" & STUB_PID=$!
for _ in $(seq 1 50); do [ -s "$T/port" ] && break; sleep 0.1; done
UP="http://127.0.0.1:$(cat "$T/port")/v1"

# 1. --print needs no Hermes and changes nothing; gateway down → the fallback window
out="$(HERMES_BIN=/nonexistent bash "$ROOT/scripts/hermes-setup.sh" --print --gateway "$DOWN" 2>/dev/null)" || bad "--print must work without Hermes installed"
[ ! -e "$C" ] || bad "--print must not write config.yaml"
python3 - "$out" <<'PY' || bad "--print: the club provider is wrong (marker, transport, default model, fallback window)"
import json, sys
c = json.loads(sys.argv[1])["providers"]["club"]
ok = [c["name"] == "club-3090 local models (scripts/hermes-setup.sh)", c["transport"] == "chat_completions",
      c["default_model"] == "qwen3.8-27b", c["models"] == {"qwen3.8-27b": {"context_length": 262144}},
      c["api"] == "http://127.0.0.1:1/v1"]
sys.exit(0 if all(ok) else 1)
PY

# 2. no Hermes on PATH → refuses, writes nothing
if HERMES_BIN=/nonexistent bash "$ROOT/scripts/hermes-setup.sh" --gateway "$DOWN" >/dev/null 2>&1; then bad "must fail when Hermes isn't installed"; fi
[ ! -e "$C" ] || bad "a run without Hermes must not create config.yaml"

# 3. an existing config: club added, everything else kept, backup taken with the same mode
cat > "$C" <<'EOF'
model:
  provider: deepseek
  default: deepseek-v4-flash
agent:
  reasoning_effort: xhigh
providers:
  work:
    api: https://gpu.example/v1
    key_env: WORK_KEY
EOF
chmod 600 "$C"
bash "$ROOT/scripts/hermes-setup.sh" --gateway "$UP" >/dev/null 2>&1 || bad "a run over an existing config failed"
[ "$(cfg providers/club/name)" = "club-3090 local models (scripts/hermes-setup.sh)" ] || bad "the club provider must be written"
[ "$(cfg model/provider)" = "deepseek" ] && [ "$(cfg model/default)" = "deepseek-v4-flash" ] || bad "the user's default model must not change"
[ "$(cfg agent/reasoning_effort)" = "xhigh" ] || bad "the user's reasoning effort must not change"
[ "$(cfg providers/work/key_env)" = "WORK_KEY" ] || bad "another provider must survive"
ls "$C".bak-* >/dev/null 2>&1 || bad "the previous config must be backed up"
[ "$(stat -c %a "$(ls "$C".bak-* | head -1)")" = "600" ] || bad "the backup must keep the config's mode (it holds keys)"

# 4. the gateway's windows are carried over; a route without model_info is skipped
[ "$(cfg providers/club/models)" = '{"gemma-4-31b": {"context_length": 131072}, "qwen3.8-27b": {"context_length": 163840}}' ] \
  || bad "the snapshot must carry the gateway's windows and skip routes without model_info (got '$(cfg providers/club/models)')"

# 5. a refresh drops a model that left the snapshot (`config set` replaces the mapping —
#    the stub does what Hermes 0.21.5 does)
bash "$ROOT/scripts/hermes-setup.sh" --gateway "$DOWN" >/dev/null 2>&1 || bad "a refresh failed"
[ "$(cfg providers/club/models)" = '{"qwen3.8-27b": {"context_length": 262144}}' ] \
  || bad "a refresh must replace the model list, not merge onto it (got '$(cfg providers/club/models)')"
[ "$(cfg providers/work/key_env)" = "WORK_KEY" ] || bad "a refresh must keep the other provider"

# 6. a `club` the script didn't write is refused: file unchanged, no backup
python3 - "$C" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["providers"]["club"] = {"api": "http://my-own/v1"}
open(sys.argv[1], "w").write(yaml.safe_dump(d, sort_keys=False))
PY
rm -f "$C".bak-*; before="$(md5sum < "$C")"
if bash "$ROOT/scripts/hermes-setup.sh" --gateway "$UP" >/dev/null 2>&1; then bad "must refuse a club provider it didn't write"; fi
[ "$(md5sum < "$C")" = "$before" ] || bad "a refused run must leave config.yaml unchanged"
ls "$C".bak-* >/dev/null 2>&1 && bad "a refused run must not write a backup"

# 7. the block docs/CODING_AGENTS.md shows is what --print emits (gateway down = 262,144)
python3 - "$ROOT" "$DOWN" <<'PY' || bad "docs/CODING_AGENTS.md's Hermes config block and hermes-setup.sh --print have drifted apart"
import io, json, re, subprocess, sys, yaml
root, down = sys.argv[1], sys.argv[2]
doc = io.open(f"{root}/docs/CODING_AGENTS.md", encoding="utf-8").read()
block = yaml.safe_load(re.search(r"```yaml\n(.*?)```", doc.split("## Hermes Agent — setup", 1)[1], re.S).group(1))
out = json.loads(subprocess.run(["bash", f"{root}/scripts/hermes-setup.sh", "--print", "--gateway", down],
                                capture_output=True, text=True, encoding="utf-8").stdout)
out["providers"]["club"]["api"] = "http://127.0.0.1:4000/v1"     # the default the doc shows
sys.exit(0 if block == out else 1)
PY

[ "$fail" -eq 0 ] && echo "test-hermes-setup: ok (print without Hermes, no Hermes → refuse, add keeps model/effort/other providers + 0600 backup, gateway windows, refresh replaces the list, refuses a hand-written club, doc block == script)"
exit "$fail"
