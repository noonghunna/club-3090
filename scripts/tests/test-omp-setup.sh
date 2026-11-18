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

# 8. the override keys are EXACTLY the served names the gateway serves for the models
#    the `club` provider covers. They are generated from the registry (the same source
#    the gateway's model_info renders), not written down: the registry's served_name
#    (else the model id) is what the engine's own /v1/models — and so the gateway —
#    reports, so a key nothing serves is impossible and a served name with no key can't
#    slip through. The hand-maintained id lists this used to pin are gone.
bash "$ROOT/scripts/omp-setup.sh" --print | python3 -c '
import sys, yaml
mo = yaml.safe_load(sys.stdin.read())["providers"]["club"]["modelOverrides"]
expected = {"qwen3.8-27b", "thinkingcap38-27b", "qwen3.8-flash-next", "glm-5.3-flash"}
if set(mo) != expected:
    print("  modelOverrides keys %s != expected %s" % (sorted(mo), sorted(expected)), file=sys.stderr)
    sys.exit(1)
' || bad "omp-setup.sh modelOverrides keys must be exactly the served names the gateway serves"


# 9. each family's ladder and default are the ones the registry's per-variant effort
#    facts derive, not written down: the rungs every route serving the name accepts
#    (the model's levels minus any rung a route refuses, plus the fold sources only when
#    every route folds the same src->dst), and the LOWEST of the routes' compose defaults
#    (an empty REASONING_EFFORT falling back to that engine's pair default). This is the
#    rendering test the #1951 drift tripwire becomes: the override and the routes it
#    names share one source, so they cannot drift.
bash "$ROOT/scripts/omp-setup.sh" --print | python3 -c '
import sys, yaml
mo = yaml.safe_load(sys.stdin.read())["providers"]["club"]["modelOverrides"]
expected = {
    "qwen3.8-27b": (["low", "medium", "xhigh"], "low"),
    "thinkingcap38-27b": (["low", "medium", "xhigh", "high", "max", "minimal"], "low"),
    "qwen3.8-flash-next": (["low", "medium", "xhigh"], "low"),
    "glm-5.3-flash": (["low", "high", "max"], "max"),
}
bad = []
for mid, v in mo.items():
    t = v.get("thinking") or {}
    if mid not in expected:
        bad.append("%s: unexpected family" % mid); continue
    ladder, default = expected[mid]
    if t.get("efforts") != ladder: bad.append("%s: ladder %r != %r" % (mid, t.get("efforts"), ladder))
    if t.get("defaultLevel") != default: bad.append("%s: default %r != %r" % (mid, t.get("defaultLevel"), default))
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
' || bad "omp-setup.sh overrides: a family's ladder or default drifted from the registry (see above)"


# 10. the compat pins ride the right families: supportsReasoningSummary: false on
#     Flash-Next and GLM (omp adds `summary: "auto"` to the Responses reasoning object,
#     and LiteLLM's Responses -> chat-completions bridge — tabbyAPI, Project Maya —
#     would otherwise forward the whole {effort, summary} object as reasoning_effort,
#     which both reject), and the 8192 thinking budget on the Qwen line (vLLM enforces
#     it on the THINKING field, not the reply; SGLang ignores it).
bash "$ROOT/scripts/omp-setup.sh" --print | python3 -c '
import sys, yaml
mo = yaml.safe_load(sys.stdin.read())["providers"]["club"]["modelOverrides"]
bad = []
for mid, v in mo.items():
    c = v.get("compat") or {}
    if mid in ("qwen3.8-flash-next", "glm-5.3-flash"):
        if c.get("supportsReasoningSummary") is not False: bad.append("%s: must set supportsReasoningSummary: false" % mid)
    elif "supportsReasoningSummary" in c:
        bad.append("%s: the Qwen line keeps its measured shape (no summary pin)" % mid)
    if mid in ("qwen3.8-27b", "thinkingcap38-27b", "qwen3.8-flash-next"):
        if c.get("extraBody", {}).get("thinking_token_budget") != 8192: bad.append("%s: must pin the 8192 thinking budget" % mid)
    elif "extraBody" in c:
        bad.append("%s: no thinking budget on this family" % mid)
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
' || bad "omp-setup.sh overrides: wrong compat pins (see above)"


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


# 11. the generator's edge shapes, on synthetic registry rows. The live catalog reaches
#     none of these today — each is a state a future profile, a new served name, or a
#     hand-edited MODEL_FAMILIES entry CAN reach, and the render must never emit a value
#     omp would put on the wire. (a) a name whose routes reach no nameable default gets
#     NO family plus a warning naming the profile to fix: a rendered None would go out as
#     the effort name "None", and an omitted defaultLevel makes omp fall back to the
#     ladder TOP, a silent xhigh — so the honest answer is no override at all. (b) a
#     served name claimed by two models is refused, never resolved by row order. (c) the
#     block shapes the emit legitimately produces for an unprofiled pair skip instead of
#     crashing the setup.
python3 - "$ROOT" <<'PY' || bad "effort_families edge shapes (see above)"
import contextlib, io, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "scripts", "lib"))
from effort_families import compute_families

MF = [{"models": ["m"], "maxTokens": 1024,
       "compat": {"supportsReasoningEffort": True, "qwenTemplateReasoningEffort": True}}]

def gen(*rows):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        fams = compute_families(list(rows), MF)
    return fams, err.getvalue()

def row(**eff):
    return {"model": "m", "served_name": "x", "effort": eff}

# (a) the only default is no rung and no fold reaches it -> no family, with a Fix: hint
fams, err = gen(row(levels=["low", "medium"], compose_default="xhigh"))
if fams:
    print("  a name with no nameable default must generate no family, got:", fams)
    sys.exit(1)
if "no route serving this name" not in err or "Fix:" not in err:
    print("  the skip must warn and name the fix, got:", err)
    sys.exit(1)
# a fold source that DOES reach it keeps the family, rendered as the SOURCE
fams, _ = gen(row(levels=["low", "medium"], compose_default="xhigh", maps={"high": "xhigh"}))
if [f["ladder"]["defaultLevel"] for f in fams] != ["high"]:
    print("  a fold-target default must render as the source that reaches it:", fams)
    sys.exit(1)

# (b) two models under one served name -> refuse, never rows[0]
try:
    compute_families([{"model": "m", "served_name": "x",
                       "effort": {"levels": ["low"], "compose_default": "low"}},
                      {"model": "other", "served_name": "x",
                       "effort": {"levels": ["low", "high"], "compose_default": "low"}}], MF)
except ValueError as e:
    if "one served name must map" not in str(e):
        print("  the refusal must name the fix, got:", e)
        sys.exit(1)
else:
    print("  a served name claimed by two models must raise, not pick rows[0]")
    sys.exit(1)
# the same refusal when rows[0] is the UNCOVERED model: the check must not sit behind
# the coverage skip, or the guard depends on registry row order
try:
    compute_families([{"model": "other", "served_name": "x",
                       "effort": {"levels": ["low"], "compose_default": "low"}},
                      {"model": "m", "served_name": "x",
                       "effort": {"levels": ["low"], "compose_default": "low"}}], MF)
except ValueError as e:
    if "one served name must map" not in str(e):
        print("  the reversed-order refusal must name the fix, got:", e)
        sys.exit(1)
else:
    print("  a collision whose first row is an uncovered model must still raise")
    sys.exit(1)

# (c) the shapes the emit really produces for an unprofiled pair: skip, never crash
for label, v in (("no effort key at all", {"model": "m", "served_name": "x"}),
                 ("budget-only block", row()),
                 ("levels null with a compose default",
                  row(levels=None, compose_default="low"))):
    fams, _ = gen(v)
    if fams:
        print(f"  {label}: must generate no family, got:", fams)
        sys.exit(1)
# a maps that is not a dict is not a fold table: the base ladder survives, the fold does not
fams, _ = gen(row(levels=["low"], maps="high:xhigh", compose_default="low"))
if [f["ladder"] for f in fams] != [{"efforts": ["low"], "defaultLevel": "low"}]:
    print("  a non-dict maps must be ignored, not crash or invent a fold:", fams)
    sys.exit(1)
# (d) the rules the live catalog's fold tables exercise, on synthetic rows: a refusal by
#     ANY route drops the rung for all; a route with no fold table means NO fold sources;
#     the default is the lowest of the routes' compose defaults; and two fold sources
#     reaching one target resolve deterministically (ladder order), not by set iteration.
fams, _ = gen(row(levels=["low", "medium"], compose_default="low"),
              row(levels=["low", "medium"], refuses=["medium"], compose_default="low"))
if [f["ladder"] for f in fams] != [{"efforts": ["low"], "defaultLevel": "low"}]:
    print("  a refusal by any route must drop the rung for all:", fams)
    sys.exit(1)
fams, _ = gen(row(levels=["low", "medium", "xhigh"], maps={"high": "xhigh"},
                  compose_default="low"),
              row(levels=["low", "medium", "xhigh"], compose_default="low"))
if [f["ladder"]["efforts"] for f in fams] != [["low", "medium", "xhigh"]]:
    print("  a route with no fold table means no fold sources:", fams)
    sys.exit(1)
fams, _ = gen(row(levels=["low", "high"], compose_default="high"),
              row(levels=["low", "high"], compose_default="low"))
if [f["ladder"]["defaultLevel"] for f in fams] != ["low"]:
    print("  the default must be the lowest of the routes' compose defaults:", fams)
    sys.exit(1)
# two fold sources reach one target: the earliest in ladder order wins, in every process
import subprocess
snippet = ("import sys; sys.path.insert(0, %r);"
           "from effort_families import compute_families;"
           "MF = [{'models': ['m'], 'maxTokens': 1024,"
           " 'compat': {'supportsReasoningEffort': True,"
           "            'qwenTemplateReasoningEffort': True}}];"
           "case = [{'model': 'm', 'served_name': 'x', 'effort':"
           " {'levels': ['low', 'high'], 'refuses': ['xhigh'],"
           "  'maps': {'high': 'xhigh', 'max': 'xhigh'},"
           "  'compose_default': 'xhigh'}}];"
           "print(compute_families(case, MF)[0]['ladder']['defaultLevel'])"
           % os.path.join(sys.argv[1], "scripts", "lib"))
seen = set()
for seed in ("0", "1", "2", "3"):
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONHASHSEED=seed))
    seen.add(r.stdout.strip())
if seen != {"high"}:
    print("  two fold sources to one target must resolve deterministically to the "
          "earliest ladder source (high), got:", sorted(seen))
    sys.exit(1)
# (e) a MODEL_FAMILIES entry the renderer would index into a crash or a silent drop is
#     refused with the entry and the field named
from effort_families import validate_model_families
good = {"models": ["m"], "maxTokens": 1024,
        "compat": {"supportsReasoningEffort": True, "qwenTemplateReasoningEffort": True}}
for label, bad_fam in (
        ("missing compat key", {"models": ["m"], "maxTokens": 1024, "compat": {}}),
        ("maxTokens not an int", {"models": ["m"], "maxTokens": "1024",
                                  "compat": {"supportsReasoningEffort": True,
                                             "qwenTemplateReasoningEffort": True}}),
        ("compat not a dict", {"models": ["m"], "maxTokens": 1024, "compat": []}),
        ("extraBody budget not an int",
         {"models": ["m"], "maxTokens": 1024,
          "compat": {"supportsReasoningEffort": True, "qwenTemplateReasoningEffort": True,
                     "extraBody": {"thinking_token_budget": "8192"}}}),
        ("models not a list", {"models": "m", "maxTokens": 1024,
                               "compat": {"supportsReasoningEffort": True,
                                          "qwenTemplateReasoningEffort": True}})):
    try:
        validate_model_families([bad_fam])
    except ValueError:
        continue
    print(f"  {label}: validate_model_families must refuse it")
    sys.exit(1)
try:
    validate_model_families([good, dict(good)])
except ValueError:
    pass
else:
    print("  a model listed in two entries must be refused (the last one silently wins)")
    sys.exit(1)
PY


[ "$fail" -eq 0 ] && echo "test-omp-setup: ok (print-only, create, append beside other providers, idempotent refresh, refuses a hand-written club, overlay, doc block == overlay, override keys == served names, per-family ladders and defaults, compat pins, auto-switching installs extension + settings and leaves config.yml alone, refuses a gateway on another machine, warns with no local litellm container, generator edge shapes: refusals, fold sources, lowest default, deterministic fold, MODEL_FAMILIES types)"
exit "$fail"
