#!/usr/bin/env bash
# test-dsh-setup — scripts/dsh-setup.sh manages ONE marked `club` block in each dsh
# profile's cordis.patch.yml, plus one key line in $DSH_HOME/.env, and never
# touches anything else.
#
# WHY THIS TEST EXISTS
# --------------------
# cordis.patch.yml and .env are the user's files: they hold other plugin config,
# their default model and their API keys. A setup script that dropped an entry,
# replaced an llm-pi-ai provider the user configured, overrode the default model
# they chose, printed the key or loosened a file's permissions would destroy or
# expose real configuration. And dsh's first run of a profile with no default
# model goes to DeepSeek's paid API, so the default has to be in place before any
# run. Offline: a stub stands in for the gateway's /model_group/info, a fake `dsh`
# creates profiles, and DSH_HOME points at a scratch directory; the real dsh is
# only used, when installed, to confirm it accepts the file (--dump-config, no
# model call).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; STUB_PID=""
trap '[ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null; rm -rf "$T"' EXIT
export DSH_HOME="$T/home"
export LITELLM_MASTER_KEY="test-gateway-key-4711"
H="$DSH_HOME/profiles/headless/cordis.patch.yml"
W="$DSH_HOME/profiles/web/cordis.patch.yml"
E="$DSH_HOME/.env"
fail=0
bad() { echo "✗ $1" >&2; fail=1; }
DOWN="http://127.0.0.1:1/v1"             # nothing listens there: the gateway-down path
REAL_PATH="$PATH"

# a fake dsh: `--profile NAME --dump-config` creates the profile like dsh does
mkdir -p "$T/bin"
cat > "$T/bin/dsh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$DSH_MOCK_LOG"
p=""; while [ $# -gt 0 ]; do case "$1" in --profile) p="$2"; shift 2 ;; *) shift ;; esac; done
mkdir -p "$DSH_HOME/profiles/$p"
printf '%s\n' "# Your patch layer for this dsh profile, applied after every bundle layer:" \
  "# a top-level YAML array of loader patch entries." "[]" > "$DSH_HOME/profiles/$p/cordis.patch.yml"
EOF
chmod +x "$T/bin/dsh"
export DSH_MOCK_LOG="$T/dsh-calls.log"
run() { PATH="$T/bin:$REAL_PATH" bash "$ROOT/scripts/dsh-setup.sh" "$@"; }

y() {  # y <file> <python expression on d, the parsed YAML list>
  python3 - "$1" "$2" <<'PY'
import sys
try:
    import yaml
except ImportError:
    print("NOYAML"); sys.exit(0)
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
e = {x["id"]: x for x in d if isinstance(x, dict) and "id" in x}
pr = e.get("llm-pi-ai", {}).get("config", {}).get("providers", {}).get("club", {})
models = {m["id"]: m for m in pr.get("models", [])}
print(eval(sys.argv[2]))
PY
}
HAVE_YAML=1; python3 -c "import yaml" 2>/dev/null || { HAVE_YAML=0; echo "SKIP (structural YAML checks): PyYAML not installed; text checks still run"; }
chk() { [ "$HAVE_YAML" = 0 ] && return 0; [ "$(y "$1" "$2")" = "$3" ] || bad "$4 (got '$(y "$1" "$2")')"; }

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

# 1. --print changes nothing and never shows the key
out="$(run --print --gateway "$DOWN" 2>&1)"
[ ! -e "$DSH_HOME" ] || bad "--print must not create or write anything"
[[ "$out" == *"$LITELLM_MASTER_KEY"* ]] && bad "--print must not show the gateway key"

# 2. fresh home, gateway down: both profiles created (no model run), block written, key stored 0600
out="$(run --gateway "$DOWN" 2>&1)" || bad "a fresh run with the gateway down must succeed: $out"
[ -f "$H" ] && [ -f "$W" ] || bad "the headless and web profiles must be created"
command grep -q -- "--profile headless --dump-config" "$DSH_MOCK_LOG" || bad "a missing profile must be created with --dump-config"
command grep -vq -- "--dump-config" "$DSH_MOCK_LOG" && bad "dsh may only be called with --dump-config (no model run)"
command grep -q '^\[\]$' "$H" && bad "the empty list '[]' must be replaced by the block"
[[ "$out" == *"$LITELLM_MASTER_KEY"* ]] && bad "the run must never print the gateway key"
[ "$(command grep -c "^CLUB3090_GATEWAY_KEY=$LITELLM_MASTER_KEY$" "$E" 2>/dev/null)" = "1" ] || bad "\$DSH_HOME/.env must hold the gateway key once"
[ "$(stat -c %a "$E")" = "600" ] || bad "a new .env holds the key and must be 0600 (got $(stat -c %a "$E"))"
chk "$H" 'type(d).__name__' "list" "the patch must stay a top-level YAML list"
chk "$H" 'models["qwen3.8-27b"]["contextWindow"]' "262144" "gateway down: qwen3.8-27b gets the 262144 fallback window"
chk "$H" 'sorted(models)' "['qwen3.8-27b']" "gateway down: nothing but qwen3.8-27b may be listed"
chk "$H" 'e["agent-default-model"]["config"]' "{'provider': 'club', 'model': 'qwen3.8-27b', 'reasoningEffort': 'low'}" \
  "a profile with no default model of its own gets club/qwen3.8-27b at low — before any run"

# 3. the Qwen3.8 wiring
chk "$H" 'pr["apiKeyEnv"]' "CLUB3090_GATEWAY_KEY" "the provider must name the key, never contain it"
command grep -q "$LITELLM_MASTER_KEY" "$H" && bad "the patch file must not contain the key"
chk "$H" 'pr["compat"]' "{'supportsDeveloperRole': False}" "the Qwen3.8 template has no developer role"
chk "$H" 'pr["api"]' "openai-completions" "the gateway route is chat completions"
chk "$H" 'models["qwen3.8-27b"]["reasoningEfforts"]' "{'off': None, 'low': 'low', 'medium': 'medium', 'xhigh': 'xhigh'}" \
  "efforts must be off/low/medium/xhigh with an EMPTY off"
chk "$H" 'models["qwen3.8-27b"]["compat"]["thinkingFormat"]' "chat-template" "thinking must ride chat_template_kwargs"
chk "$H" 'models["qwen3.8-27b"]["compat"]["chatTemplateKwargs"]' \
  "{'enable_thinking': {'\$var': 'thinking.enabled'}, 'reasoning_effort': {'\$var': 'thinking.effort'}}" "the template kwargs must carry the switch and the effort"
chk "$H" 'models["qwen3.8-27b"]["maxTokens"]' "32768" "qwen3.8-27b must pin maxTokens: 32768"

# 4. gateway up: its windows are carried over; a route without model_info is skipped
run --gateway "$STUB" >/dev/null 2>&1 || bad "a run against the stub gateway failed"
chk "$H" 'models["qwen3.8-27b"]["contextWindow"]' "163840" "the gateway's window must be used"
chk "$H" 'models["gemma-4-31b"]["maxTokens"]' "8192" "a non-Qwen route takes the gateway's reply cap"
chk "$H" '"reasoningEfforts" in models["gemma-4-31b"]' "False" "a non-Qwen route must not get the Qwen3.8 efforts"
chk "$H" '"my-cloud-model" in models' "False" "a route without model_info (not a local engine) must be skipped"

# 5. re-run refreshes in place: same file, no new backup
cp "$H" "$T/first.yml"; rm -f "$H".bak-*
run --gateway "$STUB" >/dev/null 2>&1 || bad "re-run failed"
cmp -s "$H" "$T/first.yml" || bad "a re-run against the same gateway must give the same file"
ls "$H".bak-* >/dev/null 2>&1 && bad "an unchanged re-run must not write a backup"

# 6. other entries are kept exactly, the block is appended, mode kept, backup taken
printf '%s\n' "# mine" "- id: some-plugin" "  config:" "    answer: 42" > "$H"; chmod 640 "$H"; rm -f "$H".bak-*
run --gateway "$STUB" --profile headless >/dev/null 2>&1 || bad "a run over a file with other entries failed"
head -4 "$H" | cmp -s - <(printf '%s\n' "# mine" "- id: some-plugin" "  config:" "    answer: 42") || bad "another entry must survive exactly, at the top"
chk "$H" 'e["some-plugin"]["config"]' "{'answer': 42}" "another entry must still parse"
[ "$(stat -c %a "$H")" = "640" ] || bad "an existing file's mode must be kept (got $(stat -c %a "$H"))"
ls "$H".bak-* >/dev/null 2>&1 || bad "the previous file must be backed up"

# 7. the user's own default model is kept, and ours is not added
printf '%s\n' "- id: agent-default-model" "  config:" "    provider: deepseek-official" "    model: deepseek-flash" > "$H"
run --gateway "$STUB" --profile headless >/dev/null 2>&1 || bad "a run beside the user's own default model failed"
chk "$H" 'e["agent-default-model"]["config"]["provider"]' "deepseek-official" "the user's default model must be kept"
[ "$(command grep -c '^- id: agent-default-model' "$H")" = "1" ] || bad "a second agent-default-model must not be added"
chk "$H" '"club" in str(pr)' "True" "the club provider must still be added"

# 8. an llm-pi-ai entry the user configured is refused: file unchanged, no backup
printf '%s\n' "- id: llm-pi-ai" "  config:" "    providers:" "      mine:" "        api: openai-completions" > "$H"
rm -f "$H".bak-*; before="$(md5sum < "$H")"
if run --gateway "$STUB" --profile headless >/dev/null 2>&1; then bad "must refuse a profile whose llm-pi-ai entry isn't ours"; fi
[ "$(md5sum < "$H")" = "$before" ] || bad "a refused run must leave the file unchanged"
ls "$H".bak-* >/dev/null 2>&1 && bad "a refused run must not write a backup either"

# 9. .env: a stale key is replaced in place, other lines and the mode kept
printf '%s\n' "DEEPSEEK_API_KEY=keep-me" "CLUB3090_GATEWAY_KEY=old-key" "OTHER=1" > "$E"; chmod 640 "$E"
run --gateway "$STUB" --profile web >/dev/null 2>&1 || bad "a run over an existing .env failed"
[ "$(cat "$E")" = "$(printf '%s\n' "DEEPSEEK_API_KEY=keep-me" "CLUB3090_GATEWAY_KEY=$LITELLM_MASTER_KEY" "OTHER=1")" ] \
  || bad ".env must keep its other lines and replace only the club key: $(cat "$E")"
[ "$(stat -c %a "$E")" = "640" ] || bad "an existing .env's mode must be kept"

# 10. a legacy settings.yaml that names a default model is warned about, not touched
printf '%s\n' "agent-default-model:" "  provider: deepseek-official" > "$DSH_HOME/settings.yaml"
before="$(md5sum < "$DSH_HOME/settings.yaml")"
out="$(run --gateway "$STUB" --profile web 2>&1)"
[[ "$out" == *"settings.yaml"* ]] || bad "a legacy settings.yaml with a default model must be warned about"
[ "$(md5sum < "$DSH_HOME/settings.yaml")" = "$before" ] || bad "settings.yaml must not be modified"

# 11. a profile name is a name, not a path
run --print --profile ../escape >/dev/null 2>&1 && bad "a profile name with a path in it must be rejected"

# 12. the Qwen3.8 id list is omp-setup's (test-omp-setup checks that one against the composes)
python3 - "$ROOT" <<'PY' || bad "dsh-setup.sh QWEN38_IDS must equal omp-setup.sh's"
import ast, io, re, sys
ids = lambda f: ast.literal_eval(re.search(r"^QWEN38_IDS = (\[.*\])$", io.open(f"{sys.argv[1]}/scripts/{f}", encoding="utf-8").read(), re.M).group(1))
sys.exit(0 if ids("dsh-setup.sh") == ids("omp-setup.sh") else 1)
PY

# 13. the block docs/CODING_AGENTS.md shows is what the script writes (gateway down,
#     default gateway URL → the 262,144 fallback, which is also the dual-fast window)
python3 - "$ROOT" <<'PY' || bad "docs/CODING_AGENTS.md's dsh block and dsh-setup.sh --print have drifted apart"
import io, re, subprocess, sys
root = sys.argv[1]
doc = io.open(f"{root}/docs/CODING_AGENTS.md", encoding="utf-8").read()
block = re.search(r"```yaml\n(.*?)```", doc.split("## dsh (DeepSeek Harness) — setup", 1)[1], re.S).group(1)
# the doc is written with the gateway down; the default gateway URL is what it shows
out_down = subprocess.run(["bash", f"{root}/scripts/dsh-setup.sh", "--print", "--gateway", "http://127.0.0.1:1/v1"],
                          capture_output=True, text=True, encoding="utf-8").stdout
out_down = out_down.replace('"http://127.0.0.1:1/v1"', '"http://127.0.0.1:4000/v1"')
sys.exit(0 if block == out_down else 1)
PY

# 14. dsh itself accepts the file (when dsh is installed): --dump-config composes it, no model call
if command -v dsh >/dev/null 2>&1; then
  export DSH_HOME="$T/real"; rm -rf "$DSH_HOME"
  PATH="$REAL_PATH" bash "$ROOT/scripts/dsh-setup.sh" --gateway "$STUB" --profile headless >/dev/null 2>&1 || bad "a run with the real dsh failed"
  dump="$(cd "$T" && timeout 120 dsh --profile headless --dump-config 2>&1 </dev/null)"
  echo "$dump" | command grep -q "club" && echo "$dump" | command grep -q "qwen3.8-27b" \
    || bad "dsh must compose the club provider from the generated file: $(echo "$dump" | tail -3)"
else
  echo "SKIP (dsh acceptance): dsh is not installed; the structural checks ran"
fi

[ "$fail" -eq 0 ] && echo "test-dsh-setup: ok (print-only, profiles created without a model run, block + 0600 key, Qwen3.8 wiring, gateway windows, idempotent, other entries + mode kept, user's default kept, foreign llm-pi-ai refused, .env merged, legacy settings warned, name not a path, ids == omp-setup's, doc block == script, dsh accepts it)"
exit "$fail"
