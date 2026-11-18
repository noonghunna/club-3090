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

# The expected render, written down on purpose. Check 8 derives the NAME SET from the
# composes independently (a YAML walk of command/entrypoint, not the generator's regex
# scan); this table pins what each name must RENDER and which alias belongs to which
# family. A new alias in the catalog fails check 8 first — add it here when you add one.
cat > "$T/omp_expected.py" <<'PY'
CANON = {
    "qwen3.8-27b": (["low", "medium", "xhigh"], "low"),
    "thinkingcap38-27b": (["low", "medium", "xhigh", "high", "max", "minimal"], "low"),
    "qwen3.8-flash-next": (["low", "medium", "xhigh"], "low"),
    "glm-5.3-flash": (["low", "high", "max"], "max"),
}
ALIASES = {
    "qwen3.8-27b-fp8": "qwen3.8-27b",
    "qwen3.8-27b-prism-ternary": "qwen3.8-27b",
    "qwen3.8-27b-prism-ternary-mtp": "qwen3.8-27b",
    "qwen3.8-27b-swift15": "qwen3.8-27b",
    "thinkingcap38-27b-fp8": "thinkingcap38-27b",
    "qwen3.8-flash-next-exl3-3.05bpw": "qwen3.8-flash-next",
    "qwen3.8-flash-next-exl3-4.05bpw": "qwen3.8-flash-next",
    "qwen3.8-flash-next-iq3_s": "qwen3.8-flash-next",
    "qwen3.8-flash-next-ple": "qwen3.8-flash-next",
    "qwen3.8-flash-next-swift15": "qwen3.8-flash-next",
    "qwen3.8-flash-next-unsloth-ud-iq3_xxs": "qwen3.8-flash-next",
    "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31": "qwen3.8-flash-next",
    "swift-1.5-iq3_s": "qwen3.8-flash-next",
}
# The families whose engines serve /v1/responses themselves; the others sit behind
# LiteLLM's Responses -> chat bridge, which forwards the whole {effort, summary} object
# as reasoning_effort, so those must pin the summary off.
SUMMARY_FALSE = {"qwen3.8-flash-next", "glm-5.3-flash"}
BUDGET = {"qwen3.8-27b", "thinkingcap38-27b", "qwen3.8-flash-next"}

def group(name):
    return ALIASES.get(name, name)

def ladder(name):
    return CANON[group(name)]
PY

# 8. the override keys are EXACTLY the names the covered models' routes answer to. The
#    gateway names its routes from the engine's own /v1/models (litellm_sync.probe), NOT
#    from the registry, so every alias a compose boots under — --served-model-name,
#    --alias (one-line or YAML list), --served-name, SERVED_MODEL_NAME, Strata's
#    "<family>-<quant>" name, Project Maya's model_name — is a route that needs its own
#    override. A route without one reads as a non-reasoning model: omp sends no effort and
#    the engine runs its own slowest default (#1690). Derived here INDEPENDENTLY of the
#    generator: this walks the compose YAML and reads command/entrypoint tokens, where
#    effort_families scans the same files as text with regexes.
PRINTED="$T/omp-print.yml"
bash "$ROOT/scripts/omp-setup.sh" --print > "$PRINTED" || bad "omp-setup --print failed"
python3 - "$ROOT" "$PRINTED" <<'PY' || bad "omp-setup.sh modelOverrides keys must be exactly the names the covered models' routes serve"
import glob, os, re, sys, yaml
root, printed = sys.argv[1], sys.argv[2]
FLAGS = {"--served-model-name", "--served-name", "--alias", "--model-name"}
STRATA = {"qwen": "qwen3.8-flash-next", "swift": "swift-1.5",
          "unsloth": "qwen3.8-flash-next-unsloth"}
NAME = re.compile(r"[A-Za-z0-9][\w.\-]*")
def resolved(tok):
    m = re.fullmatch(r"\$\{\w+:-([^}]+)\}", tok)      # ${SERVED_NAME:-x} -> its default
    return m.group(1) if m else tok
def seq(v):
    if v is None:
        return []
    return v.split() if isinstance(v, str) else [str(x) for x in v]
def after_flags(tokens):
    out, i = set(), 0
    while i < len(tokens):
        if tokens[i] in FLAGS:
            i += 1
            while i < len(tokens) and not tokens[i].startswith("-"):
                n = resolved(tokens[i]).strip().strip("\"'")
                if NAME.fullmatch(n):
                    out.add(n)
                i += 1
            continue
        i += 1
    return out
served = set()
for model_dir in ("qwen3.8-27b", "thinkingcap-qwen3.8-27b", "qwen3.8-flash-next",
                  "glm-5.3-flash"):
    for f in sorted(glob.glob(os.path.join(root, "models", model_dir, "*", "compose",
                                           "*", "*", "*.yml"))):
        doc = yaml.safe_load(open(f, encoding="utf-8")) or {}
        services = doc.get("services") or {}
        got = set()
        for svc in services.values():
            got |= after_flags(seq(svc.get("entrypoint")) + seq(svc.get("command")))
            env = svc.get("environment") or {}
            items = (env.items() if isinstance(env, dict) else
                     [(str(e).split("=", 1)[0], str(e).split("=", 1)[-1]) for e in env])
            for k, v in items:
                if v is None or "SERVED" not in str(k).upper():
                    continue
                n = resolved(str(v)).strip().strip("\"'")
                if NAME.fullmatch(n):
                    got.add(n)
        engine = f.split("/compose/")[0].split("/")[-1]
        args = seq(next(iter(services.values()), {}).get("entrypoint"))
        if engine == "strata":
            at = next((j for j, a in enumerate(args) if a.endswith("strata-entrypoint.sh")),
                      None)
            fam = args[at + 1] if at is not None and len(args) > at + 1 else "qwen"
            quant = args[at + 2].lower() if at is not None and len(args) > at + 2 else None
            if quant is None:
                p = os.path.join(f.split("/compose/")[0], "launcher",
                                 "strata-entrypoint.sh")
                m = re.search(r'^SET="\$\{2:-([^}"]+)\}"',
                              open(p, encoding="utf-8").read(), re.M) if os.path.exists(p) \
                    else None
                quant = m.group(1).lower() if m else None
            if fam in STRATA and quant:
                got.add("%s-%s" % (STRATA[fam], quant))
        elif engine == "strata-maya":
            p = os.path.join(f.split("/compose/")[0], "launcher", "maya_prepare.py")
            m = re.search(r'"model_name":\s*"([^"]+)"',
                          open(p, encoding="utf-8").read()) if os.path.exists(p) else None
            if m:
                got.add(m.group(1))
        served |= got
if not served:
    sys.exit("the compose scan found no served names — the scan itself is broken")
mo = yaml.safe_load(open(printed, encoding="utf-8"))["providers"]["club"]["modelOverrides"]
if set(mo) != served:
    print("  served but no override: %s\n  override nothing serves: %s"
          % (sorted(served - set(mo)), sorted(set(mo) - served)), file=sys.stderr)
    sys.exit(1)
PY


# 9. each family's ladder and default are the ones the registry's per-variant effort
#    facts derive, not written down: the rungs every route serving the name accepts
#    (the model's levels minus any rung a route refuses, plus the fold sources only when
#    every route folds the same src->dst), and the LOWEST of the routes' compose defaults
#    (an empty REASONING_EFFORT falling back to that engine's pair default). And every
#    alias renders the SAME block as its canonical name — an alias is the same model on
#    another route. This is the rendering test the #1951 drift tripwire becomes.
PYTHONPATH="$T" python3 - "$PRINTED" <<'PY' || bad "omp-setup.sh overrides: a family's ladder or default drifted from the registry (see above)"
import sys, yaml
from omp_expected import ALIASES, CANON, ladder
mo = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["providers"]["club"]["modelOverrides"]
bad = []
for mid, v in sorted(mo.items()):
    t = v.get("thinking") or {}
    exp_ladder, exp_default = ladder(mid)
    if t.get("efforts") != exp_ladder:
        bad.append("%s: ladder %r != %r" % (mid, t.get("efforts"), exp_ladder))
    if t.get("defaultLevel") != exp_default:
        bad.append("%s: default %r != %r" % (mid, t.get("defaultLevel"), exp_default))
    if v.get("maxTokens") != 32768:
        bad.append("%s: maxTokens %r != 32768" % (mid, v.get("maxTokens")))
for name in sorted(CANON):
    if name not in mo:
        bad.append("%s: the canonical family is missing from the render" % name)
for alias, canon in sorted(ALIASES.items()):
    if alias not in mo or canon not in mo:
        bad.append("%s: the alias or its canonical %s is missing" % (alias, canon))
    elif mo[alias] != mo[canon]:
        bad.append("%s: renders differently from its canonical %s" % (alias, canon))
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
PY


# 10. the compat pins ride the right families — aliases included, since an alias is the
#     same model on another route: supportsReasoningSummary: false on Flash-Next and GLM
#     (omp adds `summary: "auto"` to the Responses reasoning object, and LiteLLM's
#     Responses -> chat-completions bridge — tabbyAPI, Project Maya — would otherwise
#     forward the whole {effort, summary} object as reasoning_effort, which both reject),
#     and the 8192 thinking budget on the Qwen line (vLLM enforces it on the THINKING
#     field, not the reply; SGLang ignores it).
PYTHONPATH="$T" python3 - "$PRINTED" <<'PY' || bad "omp-setup.sh overrides: wrong compat pins (see above)"
import sys, yaml
from omp_expected import BUDGET, SUMMARY_FALSE, group
mo = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["providers"]["club"]["modelOverrides"]
bad = []
for mid, v in sorted(mo.items()):
    c = v.get("compat") or {}
    g = group(mid)
    if g in SUMMARY_FALSE:
        if c.get("supportsReasoningSummary") is not False:
            bad.append("%s: must set supportsReasoningSummary: false" % mid)
    elif "supportsReasoningSummary" in c:
        bad.append("%s: the Qwen line keeps its measured shape (no summary pin)" % mid)
    if g in BUDGET:
        if c.get("extraBody", {}).get("thinking_token_budget") != 8192:
            bad.append("%s: must pin the 8192 thinking budget" % mid)
    elif "extraBody" in c:
        bad.append("%s: no thinking budget on this family" % mid)
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
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
# (h) a route whose default is a fold SOURCE its own family does not carry still counts,
#     resolved through that route's own fold table: the route RUNS the target, so the
#     family must not round up past it. And a fold source the ladder DOES carry is sent as
#     the compose wrote it — rewriting it through maps would move the default UP.
fams, _ = gen(row(levels=["low", "medium", "xhigh"], maps={"minimal": "low"},
                  compose_default="minimal"),
              row(levels=["low", "medium", "xhigh"], compose_default="xhigh"))
if [f["ladder"]["defaultLevel"] for f in fams] != ["low"]:
    print("  a fold-source default must resolve through that route's own maps (the route "
          "runs low), not round up to the other route's xhigh:", fams)
    sys.exit(1)
fams, _ = gen(row(levels=["low", "high", "xhigh"], maps={"high": "xhigh"},
                  compose_default="high"))
if [f["ladder"]["defaultLevel"] for f in fams] != ["high"]:
    print("  a fold source the ladder carries is sent as the compose wrote it:", fams)
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

# (f) an alias two models' composes both boot under is REFUSED, not rendered twice: one
#     route cannot carry two overrides, and a duplicated YAML key silently keeps the last.
# (g) a compose's --alias entry becomes an id of its family, and a name another family
#     owns canonically belongs to that family, not to whoever aliases it.
alias_root = os.path.join(sys.argv[1], "aliascase")
MF2 = [dict(MF[0], models=["ma"]), dict(MF[0], models=["mb"])]
def write_compose(model, *aliases):
    d = os.path.join(alias_root, "models", model, "engine", "compose", "single", "q")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "c.yml"), "w", encoding="utf-8") as fh:
        fh.write("services:\n  s:\n    command:\n")
        for a in aliases:
            fh.write("      - '--alias'\n      - '%s'\n" % a)
def arow(model, name):
    return {"model": model, "served_name": name,
            "effort": {"levels": ["low"], "compose_default": "low"},
            "compose_path": "models/%s/engine/compose/single/q/c.yml" % model}
def gen2(mf, *rows):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        fams = compute_families(list(rows), mf, root=alias_root)
    return fams, err.getvalue()
write_compose("ma", "sb", "shared-extra")
write_compose("mb", "sb")
fams, _ = gen2(MF2, arow("ma", "sa"), arow("mb", "sb"))
if [f["ids"] for f in fams] != [["sa", "shared-extra"], ["sb"]]:
    print("  a name another family owns belongs to that family, a free alias becomes an id "
          "of its own:", [f["ids"] for f in fams])
    sys.exit(1)
write_compose("ma", "shared-name")
write_compose("mb", "shared-name")
try:
    gen2(MF2, arow("ma", "sa"), arow("mb", "sb"))
except ValueError as e:
    if "two families" not in str(e):
        print("  the alias-collision refusal must name the fix, got:", e)
        sys.exit(1)
else:
    print("  an alias two models' composes both boot under must raise, not render twice")
    sys.exit(1)
PY


# 12. an unreadable registry FAILS CLOSED. registry_variants() turns any failure into [],
#     and an empty family list would otherwise write a `club` provider with no
#     modelOverrides — every model then reads as non-reasoning, omp sends no effort, and
#     the engine runs its own slowest default: #1690, silently. A bare community VM with no
#     PyYAML is the real case; a PYTHONPATH shim reproduces it without uninstalling
#     anything from this rig.
mkdir -p "$T/noyaml"
printf 'raise ImportError("no PyYAML on this VM")\n' > "$T/noyaml/yaml.py"
if PYTHONPATH="$T/noyaml" bash "$ROOT/scripts/omp-setup.sh" --print > "$T/fc.out" 2> "$T/fc.err"; then
    bad "an empty registry must fail, not render a provider with no modelOverrides"
elif ! grep -q "registry-emit:" "$T/fc.err"; then
    bad "the empty-registry failure must carry the emit's own diagnostic, so the operator "
        "sees WHY (no PyYAML? a broken emit?) and not just that it failed"
    sed 's/^/  /' "$T/fc.err" >&2
fi


[ "$fail" -eq 0 ] && echo "test-omp-setup: ok (print-only, create, append beside other providers, idempotent refresh, refuses a hand-written club, overlay, doc block == overlay, override keys == every served alias, per-family ladders and defaults, aliases render as their canonical, compat pins, auto-switching installs extension + settings and leaves config.yml alone, refuses a gateway on another machine, warns with no local litellm container, generator edge shapes: refusals, fold sources, lowest default, deterministic fold, MODEL_FAMILIES types, alias ownership and collisions, empty registry fails closed)"
exit "$fail"
