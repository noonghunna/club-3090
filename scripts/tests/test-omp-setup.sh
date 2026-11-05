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
            m = re.search(r"--(?:served-model-name|served-name|alias)\b(.*)$", l)   # served-name: QEngine's
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

# 9. the same for Qwen3.8-Flash-Next and GLM-5.3-Flash (Discussion #1690): without a
#    key omp treated them as non-reasoning, sent NO effort, and every request ran at
#    the server's slowest default. Two engines name the model themselves, so their ids
#    are derived the way they derive them: Strata as "<family name>-<quant>" (its
#    setup; the family is the argument after strata-entrypoint.sh in the compose's
#    entrypoint, the quant the launcher's MODEL=), Project Maya from its launcher's model_name. A compose whose
#    id this can't tell fails, rather than going unchecked.
python3 - "$ROOT" <<'PY' || bad "omp-setup.sh FLASH_NEXT_IDS / GLM53_IDS must equal the ids the Flash-Next / GLM composes serve"
import ast, glob, io, os, re, sys
root = sys.argv[1]
src = io.open(f"{root}/scripts/omp-setup.sh", encoding="utf-8").read()
def listed(name):
    return set(ast.literal_eval(re.search(rf"^{name} = (\[.*?\])$", src, re.M | re.S).group(1)))
STRATA_FAMILY = {"qwen": "qwen3.8-flash-next", "swift": "swift-1.5"}   # Strata setup.py FAMILIES[...]["name"]
def read(p): return io.open(p, encoding="utf-8").read()
def strata_quant(engine_dir):
    m = re.search(r"^(?:export )?MODEL=(\S+)", read(f"{engine_dir}/launcher/strata-entrypoint.sh"), re.M)
    return m.group(1).lower() if m else None
def maya_name(engine_dir):
    m = re.search(r'"model_name":\s*"([^"]+)"', read(f"{engine_dir}/launcher/maya_prepare.py"))
    return m.group(1) if m else None
def compose_ids(f):
    out = set()
    def add(tok):
        tok = tok.strip().strip("\"'")
        m = re.fullmatch(r"\$\{\w+:-([^}]+)\}", tok)
        if m: tok = m.group(1)
        if re.fullmatch(r"[A-Za-z0-9][\w.\-]*", tok): out.add(tok)
    lines = [l for l in read(f).splitlines() if not l.lstrip().startswith("#")]
    for i, l in enumerate(lines):
        m = re.search(r"SERVED_MODEL_NAME:\s*\"?(\$\{SERVED_MODEL_NAME:-[^}]+\})", l)
        if m: add(m.group(1))
        m = re.search(r"--(?:served-model-name|served-name|alias|model-name)\b['\"]?(.*)$", l)   # served-name: QEngine's
        if not m: continue
        rest = m.group(1).strip().rstrip("\\").strip().strip("\"'")
        if rest and not rest.startswith("-"):
            for tok in rest.split(): add(tok)
            continue
        for nxt in lines[i + 1:]:                                    # YAML list form
            v = nxt.strip()
            val = v[2:].strip().strip("\"'") if v.startswith("- ") else ""
            if not val or val.startswith("-"): break                 # the next flag ('--x' quoted too)
            add(val)
    engine_dir = f.split("/compose/")[0]
    engine = os.path.basename(engine_dir)
    if engine == "strata":
        m = re.search(r"entrypoint:\s*\[([^\]]*)\]", read(f))
        args = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        at = next((i for i, a in enumerate(args) if a.endswith("strata-entrypoint.sh")), None)   # the family follows it
        fam = args[at + 1] if at is not None and len(args) > at + 1 else "qwen"
        q = strata_quant(engine_dir)
        if fam in STRATA_FAMILY and q: out.add(f"{STRATA_FAMILY[fam]}-{q}")
    elif engine == "strata-maya":
        n = maya_name(engine_dir)
        if n: out.add(n)
    return out
ok = True
for model_dir, name in (("qwen3.8-flash-next", "FLASH_NEXT_IDS"), ("glm-5.3-flash", "GLM53_IDS")):
    keys, served, unknown = listed(name), set(), []
    for f in sorted(glob.glob(f"{root}/models/{model_dir}/*/compose/*/*/*.yml")):
        got = compose_ids(f)
        if not got: unknown.append(os.path.relpath(f, root))
        served |= got
    if unknown:
        print(f"  {name}: can't tell what these serve — teach test-omp-setup: {unknown}", file=sys.stderr); ok = False
    if keys != served:
        print(f"  {name}: served but no override: {sorted(served - keys)}   override nothing serves: {sorted(keys - served)}", file=sys.stderr); ok = False
sys.exit(0 if ok else 1)
PY

# 10. each family's ladder: Flash-Next takes the Qwen3.8 template's levels, GLM its
#     own (it has no `medium`, and Project Maya 400s an effort it doesn't know). The
#     new overrides send no reasoning summary — LiteLLM's Responses -> chat bridge
#     (tabbyAPI and Maya routes) otherwise forwards the whole {effort, summary} object
#     as reasoning_effort — while the Qwen3.8-27B ones stay as they were measured.
bash "$ROOT/scripts/omp-setup.sh" --print | python3 -c '
import sys, yaml
mo = yaml.safe_load(sys.stdin.read())["providers"]["club"]["modelOverrides"]
bad = []
for mid, v in mo.items():
    t, c = v.get("thinking") or {}, v.get("compat") or {}
    if mid == "glm-5.3-flash":
        if t.get("efforts") != ["low", "high", "max"] or t.get("defaultLevel") != "max": bad.append(f"{mid}: ladder {t}")
        if c.get("supportsReasoningSummary") is not False: bad.append(f"{mid}: must set supportsReasoningSummary: false")
    else:
        if t.get("efforts") != ["low", "medium", "xhigh"]: bad.append(f"{mid}: ladder {t}")
        flash = mid.startswith("qwen3.8-flash-next") or mid.startswith("swift-1.5")
        if flash and c.get("supportsReasoningSummary") is not False: bad.append(f"{mid}: must set supportsReasoningSummary: false")
        if not flash and "supportsReasoningSummary" in c: bad.append(f"{mid}: the 27B overrides keep their measured shape")
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
' || bad "omp-setup.sh overrides: wrong effort ladder or summary setting (see above)"

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

# 8. --auto-switching: installs the extension AND writes its club-live.json
#    settings beside it — and never touches config.yml, which stays the user's.
#    The settings file must carry the gateway (minus /v1), a key, the rendered
#    view path, and families that MATCH the modelOverrides written into
#    models.yml: one source, two renderings; this is the drift tripwire.
export PI_CODING_AGENT_DIR="$T/agent-auto"
bash "$ROOT/scripts/omp-setup.sh" --auto-switching --gateway http://127.0.0.1:4100/v1 >/dev/null \
  || bad "--auto-switching run failed"
[ -f "$PI_CODING_AGENT_DIR/extensions/club-live.ts" ] \
  || bad "--auto-switching must install the club-live extension into the agent dir"
[ -f "$PI_CODING_AGENT_DIR/extensions/club-live.json" ] \
  || bad "--auto-switching must write the club-live settings file"
[ ! -e "$PI_CODING_AGENT_DIR/config.yml" ] \
  || bad "--auto-switching wrote config.yml — the roles stay the user's"
[[ "$(stat -c '%a' "$PI_CODING_AGENT_DIR/extensions/club-live.json")" == "600" ]] \
  || bad "club-live.json holds the gateway key and must be mode 600"
bash "$ROOT/scripts/omp-setup.sh" --auto-switching --gateway http://127.0.0.1:4100/v1 2>&1 \
  | grep -q 'club-live/live:medium' || bad "--auto-switching must print the modelRoles block to paste"
python3 - "$PI_CODING_AGENT_DIR/extensions/club-live.json" "$PI_CODING_AGENT_DIR/models.yml" <<'PY' || bad "--auto-switching settings drifted from models.yml or are malformed"
import io, json, sys, yaml
j = json.load(io.open(sys.argv[1], encoding="utf-8"))
ov = yaml.safe_load(io.open(sys.argv[2], encoding="utf-8"))["providers"]["club"]["modelOverrides"]
if j["gateway"] != "http://127.0.0.1:4100":
    print(f"  gateway {j['gateway']!r} != the --gateway arg minus /v1", file=sys.stderr); sys.exit(1)
if not j["key"]:
    print("  settings file carries no gateway key", file=sys.stderr); sys.exit(1)
if not j["renderedCfg"].endswith("config.runtime.yaml"):
    print(f"  renderedCfg {j['renderedCfg']!r} is not the rendered view", file=sys.stderr); sys.exit(1)
seen = set()
for fam in j["families"]:
    ladder, cap = fam["ladder"], fam["maxTokens"]
    for mid in fam["ids"]:
        seen.add(mid)
        if mid not in ov:
            print(f"  {mid}: in the settings families but not in models.yml", file=sys.stderr); sys.exit(1)
        o, t = ov[mid], ov[mid]["thinking"]
        if t["efforts"] != ladder["efforts"] or t["defaultLevel"] != ladder["defaultLevel"]:
            print(f"  {mid}: ladder {t} != settings {ladder}", file=sys.stderr); sys.exit(1)
        if o["maxTokens"] != cap:
            print(f"  {mid}: maxTokens {o['maxTokens']} != settings {cap}", file=sys.stderr); sys.exit(1)
        for k, want in fam["compat"].items():
            if o["compat"].get(k) != want:
                print(f"  {mid}: compat {k}={o['compat'].get(k)!r} != settings {want!r}", file=sys.stderr); sys.exit(1)
if set(ov) - seen:
    print(f"  models.yml overrides with no settings family: {sorted(set(ov) - seen)}", file=sys.stderr); sys.exit(1)
PY
bash "$ROOT/scripts/omp-setup.sh" --auto-switching >/dev/null || bad "--auto-switching re-run failed"
[ "$(grep -c 'club-3090 local models' "$PI_CODING_AGENT_DIR/models.yml")" = 2 ] \
  || bad "--auto-switching re-run must keep exactly one marked provider block"
python3 -c "import io, json, sys; json.load(io.open(sys.argv[1], encoding='utf-8'))" \
  "$PI_CODING_AGENT_DIR/extensions/club-live.json" \
  || bad "club-live.json must stay valid JSON across re-runs"

# 9. --auto-switching with the gateway on ANOTHER machine stops before writing
#    anything: the alias reads the rendered config from this machine's disk, so
#    it could only follow a stale local file or this machine's own gateway.
#    192.0.2.1 is TEST-NET-1 (RFC 5737): never an address of this machine, and a
#    literal, so no DNS. Positive control: this machine's own LAN address (not
#    loopback) is allowed, so the check isn't a loopback-only rule.
export PI_CODING_AGENT_DIR="$T/agent-remote"
if out="$(bash "$ROOT/scripts/omp-setup.sh" --auto-switching --gateway http://192.0.2.1:4000/v1 2>&1)"; then
  bad "--auto-switching must refuse a gateway on another machine"
fi
printf '%s\n' "$out" | grep -q 'not on this machine' \
  || bad "--auto-switching must say why it refused a gateway on another machine"
[ ! -e "$PI_CODING_AGENT_DIR/models.yml" ] && [ ! -e "$PI_CODING_AGENT_DIR/extensions" ] \
  || bad "--auto-switching with a gateway on another machine must write nothing"
lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -n "$lan_ip" && "$lan_ip" != 127.* ]]; then
  export PI_CODING_AGENT_DIR="$T/agent-lan"
  bash "$ROOT/scripts/omp-setup.sh" --auto-switching --gateway "http://$lan_ip:4000/v1" >/dev/null 2>&1 \
    || bad "--auto-switching must accept this machine's own LAN address ($lan_ip)"
  [ -f "$PI_CODING_AGENT_DIR/extensions/club-live.json" ] \
    || bad "--auto-switching with this machine's LAN address must write club-live.json"
else
  echo "  (no LAN address on this machine: the LAN-address control is skipped)"
fi

# 10. No litellm container on this machine (a docker stub that finds none): the
#     run still installs, falls back to this checkout's rendered view, and WARNS
#     that the alias only follows a gateway whose container runs here (a loopback
#     tunnel to another machine passes check 9). Hermetic: the stub, not the rig.
export PI_CODING_AGENT_DIR="$T/agent-nocontainer"
mkdir -p "$T/stub"
printf '#!/usr/bin/env bash\nexit 1\n' > "$T/stub/docker"; chmod +x "$T/stub/docker"
out="$(PATH="$T/stub:$PATH" bash "$ROOT/scripts/omp-setup.sh" --auto-switching 2>&1)" \
  || bad "--auto-switching must still install when no litellm container runs here"
printf '%s\n' "$out" | grep -q 'no litellm container runs on this machine' \
  || bad "--auto-switching must warn when no litellm container runs here"
python3 - "$PI_CODING_AGENT_DIR/extensions/club-live.json" "$ROOT/services/litellm/config.runtime.yaml" <<'PY' \
  || bad "without a litellm container, renderedCfg must be this checkout's rendered view"
import io, json, sys
sys.exit(0 if json.load(io.open(sys.argv[1], encoding="utf-8"))["renderedCfg"] == sys.argv[2] else 1)
PY


[ "$fail" -eq 0 ] && echo "test-omp-setup: ok (print-only, create, append beside other providers, idempotent refresh, refuses a hand-written club, overlay, doc block == overlay, override keys == served ids, Flash-Next + GLM keys and ladders, auto-switching installs extension + settings and leaves config.yml alone, refuses a gateway on another machine, warns with no local litellm container)"
exit "$fail"
