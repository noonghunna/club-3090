#!/usr/bin/env bash
# test-omp-setup — scripts/omp-setup.sh manages ONE marked `club` provider block in
# omp's models.yml and never touches anything else in that file.
#
# WHY THIS TEST EXISTS
# --------------------
# models.yml is the user's file: it holds their other providers (cloud keys,
# hand-tuned models). A setup script that rewrote it wholesale, duplicated its
# block on a re-run, or silently replaced a provider the user named `club`
# themselves would destroy real configuration. Offline: PI_CODING_AGENT_DIR points
# omp's agent dir at a scratch directory; omp itself is not needed.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export PI_CODING_AGENT_DIR="$T/agent"
fail=0
bad() { echo "✗ $1" >&2; fail=1; }
yq() { python3 - "$PI_CODING_AGENT_DIR/models.yml" "$1" <<'PY'
import io, sys, yaml
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
cur = d
for k in sys.argv[2].split("/"):   # "/" — model ids contain dots
    cur = cur.get(k) if isinstance(cur, dict) else None
print("" if cur is None else cur)
PY
}

# 1. --print changes nothing
bash "$ROOT/scripts/omp-setup.sh" --print >/dev/null
[ ! -e "$PI_CODING_AGENT_DIR/models.yml" ] || bad "--print must not write models.yml"

# 2. fresh: creates a valid file with the club provider on LiteLLM discovery
bash "$ROOT/scripts/omp-setup.sh" >/dev/null || bad "fresh run failed"
[ "$(yq providers/club/discovery/type)" = "litellm" ] || bad "club provider must use discovery: litellm"
[ "$(yq providers/club/baseUrl)" = "http://127.0.0.1:4000/v1" ] || bad "club baseUrl must be the gateway (:4000/v1)"
[ "$(yq providers/club/modelOverrides/qwen3.8-27b/compat/qwenTemplateReasoningEffort)" = "True" ] \
  || bad "qwen3.8-27b override must set qwenTemplateReasoningEffort"
# The reply cap must not depend on the gateway reporting it: without model_info omp
# falls back to its catalog's 65,536, the whole window of the 65K single-card slug.
[ "$(yq providers/club/modelOverrides/qwen3.8-27b/maxTokens)" = "32768" ] \
  || bad "qwen3.8-27b override must pin maxTokens: 32768 (got '$(yq providers/club/modelOverrides/qwen3.8-27b/maxTokens)')"
# `qwen` would send a top-level enable_thinking that vLLM and SGLang ignore, so omp's
# "off" level would keep thinking on; chat_template_kwargs is what the engines read.
[ "$(yq providers/club/compat/thinkingFormat)" = "qwen-chat-template" ] \
  || bad "club provider must use thinkingFormat: qwen-chat-template (got '$(yq providers/club/compat/thinkingFormat)')"

# 3. existing file with another provider: appended, other provider untouched, backup taken
cat > "$PI_CODING_AGENT_DIR/models.yml" <<'EOF'
providers:
  modelscope:
    baseUrl: https://api-inference.example/v1
    api: openai-completions
    apiKey: MY_SECRET
EOF
bash "$ROOT/scripts/omp-setup.sh" >/dev/null || bad "run over an existing file failed"
[ "$(yq providers/modelscope/apiKey)" = "MY_SECRET" ] || bad "another provider must survive untouched"
[ "$(yq providers/club/discovery/type)" = "litellm" ] || bad "club provider must be added beside the existing one"
ls "$PI_CODING_AGENT_DIR"/models.yml.bak-* >/dev/null 2>&1 || bad "the previous file must be backed up"

# 4. re-run is idempotent: still exactly one managed block
bash "$ROOT/scripts/omp-setup.sh" --gateway http://10.0.0.5:4000/v1 >/dev/null || bad "re-run failed"
n=$(command grep -c '>>> club-3090 local models' "$PI_CODING_AGENT_DIR/models.yml")
[ "$n" = "1" ] || bad "re-run must refresh, not duplicate (found $n managed blocks)"
[ "$(yq providers/club/baseUrl)" = "http://10.0.0.5:4000/v1" ] || bad "re-run must refresh the block (new --gateway)"
[ "$(yq providers/modelscope/apiKey)" = "MY_SECRET" ] || bad "re-run must keep the other provider"

# 5. a hand-written `club` provider (no markers) is refused, file unchanged
cat > "$PI_CODING_AGENT_DIR/models.yml" <<'EOF'
providers:
  club:
    baseUrl: http://my-own-thing/v1
    api: openai-completions
EOF
before=$(md5sum < "$PI_CODING_AGENT_DIR/models.yml")
if bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1; then bad "must refuse to overwrite a hand-written club provider"; fi
[ "$(md5sum < "$PI_CODING_AGENT_DIR/models.yml")" = "$before" ] || bad "a refused run must leave models.yml unchanged"
rm -f "$PI_CODING_AGENT_DIR"/models.yml.bak-*; bash "$ROOT/scripts/omp-setup.sh" >/dev/null 2>&1
ls "$PI_CODING_AGENT_DIR"/models.yml.bak-* >/dev/null 2>&1 && bad "a refused run must not write a backup either"

# 6. the overlay parses and carries the settings the doc promises
python3 - "$ROOT/services/omp/omp-club.yml" <<'PY' || bad "services/omp/omp-club.yml is missing a promised setting"
import io, sys, yaml
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
need = [d["provider"]["appendOnlyContext"] == "on", d["compaction"]["thresholdPercent"] > 0,
        d["tools"]["artifactSpillThreshold"] <= 16, d["task"]["maxConcurrency"] >= 1,
        d["providers"]["streamFirstEventTimeoutSeconds"] >= 600,
        all(":" in v for v in d["modelRoles"].values()),   # every role names an effort
        # no enabledModels: it narrows omp's model scope for every session, and
        # omp must not be limited to club models (maintainer, 2026-09-27). The
        # startup risk it covered is documented instead ("Start the slug before omp").
        "enabledModels" not in d,
        # one model per role: an ordered list (`a, b`) does not prefer its first
        # entry at startup — omp started on the OpenRouter entry with the club
        # model served, cold and warm discovery cache alike
        all("," not in v for v in d["modelRoles"].values())]
sys.exit(0 if all(need) else 1)
PY

# 8. the override keys are EXACTLY the ids the Qwen3.8-family composes serve —
#    the registry's model name is not what omp sees; the engine's /v1/models (and
#    so the gateway) says --served-model-name / --alias. A key nothing serves is
#    dead; a served id with no key gets no effort (#1444: ThinkingCap's key was
#    `thinkingcap-qwen3.8-27b` while every ThinkingCap compose served thinkingcap38-27b).
python3 - "$ROOT" <<'PY' || bad "omp-setup.sh QWEN38_IDS must equal the ids the Qwen3.8-family composes serve"
import ast, glob, io, re, sys
root = sys.argv[1]
src = io.open(f"{root}/scripts/omp-setup.sh", encoding="utf-8").read()
keys = set(ast.literal_eval(re.search(r"^QWEN38_IDS = (\[.*\])$", src, re.M).group(1)))
served = set()
def add(tok):
    m = re.fullmatch(r"\$\{\w+:-([^}]+)\}", tok)          # ${SERVED_NAME:-qwen3.8-27b} -> its default
    if m: tok = m.group(1)
    if re.fullmatch(r"[A-Za-z0-9][\w.\-]*", tok): served.add(tok)
for d in ("qwen3.8-27b", "thinkingcap-qwen3.8-27b"):
    for f in glob.glob(f"{root}/models/{d}/*/compose/*/*/*.yml"):
        lines = [l for l in io.open(f, encoding="utf-8").read().splitlines() if not l.lstrip().startswith("#")]
        for i, l in enumerate(lines):
            m = re.search(r"--(?:served-model-name|alias)\b(.*)$", l)
            if not m: continue
            rest = m.group(1).strip().rstrip("\\").strip().strip('"')
            if rest and not rest.startswith("-"):                    # one-line form
                for tok in rest.split():
                    add(tok.strip('"'))
                continue
            for nxt in lines[i + 1:]:                                # YAML list form
                v = nxt.strip()
                if not v.startswith("- ") or v[2:].lstrip().startswith("-"): break
                add(v[2:].strip().strip("\"'"))
if not served:
    sys.exit("found no served names — the scan itself is broken")
if keys != served:
    print(f"  served but no override: {sorted(served - keys)}   override nothing serves: {sorted(keys - served)}", file=sys.stderr)
    sys.exit(1)
PY

# 7. the config.yml block the doc tells readers to paste == the overlay we ship
python3 - "$ROOT/docs/CODING_AGENTS.md" "$ROOT/services/omp/omp-club.yml" <<'PY' || bad "docs/CODING_AGENTS.md's config.yml block and services/omp/omp-club.yml have drifted apart"
import io, re, sys, yaml
doc = io.open(sys.argv[1], encoding="utf-8").read()
sec = doc.split("### Settings for `~/.omp/agent/config.yml`", 1)[1]
block = yaml.safe_load(re.search(r"```yaml\n(.*?)```", sec, re.S).group(1))
overlay = yaml.safe_load(io.open(sys.argv[2], encoding="utf-8"))
if block != overlay:
    for k in sorted(set(block) | set(overlay)):
        if block.get(k) != overlay.get(k):
            print(f"  {k}: doc={block.get(k)!r} overlay={overlay.get(k)!r}", file=sys.stderr)
    sys.exit(1)
PY

[ "$fail" -eq 0 ] && echo "test-omp-setup: ok (print-only, create, append beside other providers, idempotent refresh, refuses a hand-written club, overlay, doc block == overlay, override keys == served ids)"
exit "$fail"
