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

# The expected render, written down on purpose: one row per ROUTE ID, with the ladder and
# the default that route's own engines give it. Check 8 derives the NAME SET from the
# composes independently (a YAML walk of command/entrypoint, not the generator's regex
# scan); this table pins what each name must RENDER. A new alias in the catalog fails
# check 8 first — add it here when you add one. The third field is the CLIENT family (the
# MODEL_FAMILIES entry) that supplies maxTokens and the compat pins: those ARE keyed by
# model, unlike the ladder. ("family" in effort_families.STRATA_FAMILY_NAMES is a different
# thing — Strata's own weights family, which names a served id.)
cat > "$T/omp_expected.py" <<'PY'
QWEN = ["low", "medium", "xhigh"]
WIDE = ["low", "medium", "xhigh", "high", "max", "minimal"]
STRATA = ["low", "medium", "xhigh", "high", "max"]
PER_ID = {
    # qwen3.8-27b: 42 routes over 6 engines; the aliases are single-engine routes, so
    # each carries what its own engine accepts and what its own compose defaults to.
    "qwen3.8-27b": (QWEN, "low", "qwen38"),
    "qwen3.8-27b-fp8": (WIDE, "low", "qwen38"),               # vLLM only
    # llama-cpp-prism: the profile records only the evidenced `unknown_name: xhigh` fact,
    # no `low` refusal, so the ladder keeps low — while docs/engines/LLAMA_CPP_PRISM.md
    # reports upstream treats low as xhigh. That evidence question belongs to the engine
    # profile (phase 1/2); this row pins what the EMIT says, and test-pi-setup /
    # test-dsh-setup name the same aliases as a KNOWN GAP for their own hand-written map.
    "qwen3.8-27b-prism-ternary": (QWEN, "medium", "qwen38"),
    "qwen3.8-27b-prism-ternary-mtp": (QWEN, "medium", "qwen38"),
    "qwen3.8-27b-swift15": (QWEN, "medium", "qwen38"),        # llama.cpp
    "thinkingcap38-27b": (WIDE, "low", "qwen38"),
    "thinkingcap38-27b-fp8": (WIDE, "low", "qwen38"),
    # qwen3.8-flash-next: the shared id is served by engines that refuse high/max and
    # default low; Strata's and bucko's own ids keep high/max and their xhigh default,
    # which is the #1951 per-id decision this table exists to hold.
    "qwen3.8-flash-next": (QWEN, "low", "flashnext"),
    "qwen3.8-flash-next-exl3-3.05bpw": (QWEN, "xhigh", "flashnext"),   # ExLlamaV3
    "qwen3.8-flash-next-exl3-4.05bpw": (QWEN, "xhigh", "flashnext"),
    "qwen3.8-flash-next-iq3_s": (STRATA, "xhigh", "flashnext"),        # Strata
    "qwen3.8-flash-next-unsloth-ud-iq3_xxs": (STRATA, "xhigh", "flashnext"),
    "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31": (STRATA, "xhigh", "flashnext"),
    "qwen3.8-flash-next-ple": (QWEN, "xhigh", "flashnext"),            # bucko-vllm
    "qwen3.8-flash-next-swift15": (QWEN, "xhigh", "flashnext"),
    "swift-1.5-iq3_s": (STRATA, "xhigh", "flashnext"),                 # Strata, swift
    "glm-5.3-flash": (["low", "high", "max"], "max", "glm"),
}
# The client families, not the routes: which sit behind LiteLLM's Responses -> chat
# bridge (omp adds `summary: "auto"` to the reasoning object and the bridge forwards the
# whole {effort, summary} as reasoning_effort, which those engines reject), and which
# take the chat-completions thinking budget.
SUMMARY_FALSE = {"flashnext", "glm"}
BUDGET = {"qwen38", "flashnext"}

# Which engines' rows answer to each route id. The ladder and the default can stay equal
# while the SET behind them changes — that is exactly what the union rule did (Prism's rows
# counted under qwen3.8-27b, Strata's, bucko's and ExLlamaV3's under qwen3.8-flash-next),
# and one profile fact (a refusal) would then move a shared id no such engine serves. Row
# COUNTS are not pinned: a new slug on an existing engine is not a fact change.
PER_ID_ENGINES = {
    "qwen3.8-27b": {"llama-cpp-local", "qengine", "sglang-stable", "vllm-hyperqwen",
                    "vllm-stable"},
    "qwen3.8-27b-fp8": {"vllm-stable"},
    "qwen3.8-27b-prism-ternary": {"llama-cpp-prism"},
    "qwen3.8-27b-prism-ternary-mtp": {"llama-cpp-prism"},
    "qwen3.8-27b-swift15": {"llama-cpp-local"},
    "thinkingcap38-27b": {"sglang-stable", "vllm-stable"},
    "thinkingcap38-27b-fp8": {"vllm-stable"},
    "qwen3.8-flash-next": {"llamacpp-club3090-v1.6", "qengine", "vllm-stable"},
    "qwen3.8-flash-next-exl3-3.05bpw": {"exllamav3"},
    "qwen3.8-flash-next-exl3-4.05bpw": {"exllamav3"},
    "qwen3.8-flash-next-iq3_s": {"strata"},
    "qwen3.8-flash-next-unsloth-ud-iq3_xxs": {"strata"},
    "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31": {"strata"},
    "qwen3.8-flash-next-ple": {"bucko-vllm"},
    "qwen3.8-flash-next-swift15": {"bucko-vllm"},
    "swift-1.5-iq3_s": {"strata"},
    "glm-5.3-flash": {"llamacpp-club3090-v1.6", "strata-maya"},
}

def group(name):
    return PER_ID[name][2]

def ladder(name):
    return PER_ID[name][:2]
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
by_file = {}
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
        by_file[f] = got
        served |= got
# A compose that declares NO name is served under the name the registry declares for it —
# the one place a route takes its name from the registry (HyperQwen today; the gateway's
# booting placeholder is the only other user, and the ready render renames it). The
# generator's rule is "the compose's names, or the registry name when it declares none",
# so the expected set here needs the same fallback.
COVERED = {"qwen3.8-27b", "thinkingcap-qwen3.8-27b", "qwen3.8-flash-next", "glm-5.3-flash"}
sys.path.insert(0, os.path.join(root, "scripts", "lib"))     # the registry reader only
from litellm_sync import registry_variants
for _v in registry_variants(root):
    if _v.get("model") not in COVERED or not _v.get("compose_path"):
        continue
    _cp = os.path.join(root, _v["compose_path"])
    if _cp in by_file and not by_file[_cp]:
        _sid = _v.get("served_name") or _v.get("model")
        if _sid:
            served.add(_sid)
if not served:
    sys.exit("the compose scan found no served names — the scan itself is broken")
mo = yaml.safe_load(open(printed, encoding="utf-8"))["providers"]["club"]["modelOverrides"]
if set(mo) != served:
    print("  served but no override: %s\n  override nothing serves: %s"
          % (sorted(served - set(mo)), sorted(set(mo) - served)), file=sys.stderr)
    sys.exit(1)
PY


# 9. each route id's ladder and default are the ones the registry's per-variant effort
#    facts derive for THAT ROUTE, not written down: the rungs every route serving the id
#    accepts (its engines' levels minus any rung one refuses, plus the fold sources only
#    when every route folds the same src->dst), and the LOWEST of those routes' compose
#    defaults (an empty REASONING_EFFORT falling back to that engine's pair default).
#    This is the per-id rule #1951 settled, and the shape guard: a family that lumps two
#    routes together shows up here as a wrong ladder or a wrong default, not as a passing
#    "alias == canonical" assertion. This is the rendering test the #1949 drift tripwire
#    becomes.
PYTHONPATH="$T:$ROOT/scripts/lib" python3 - "$ROOT" "$PRINTED" <<'PY' || bad "omp-setup.sh overrides: a route's ladder, engine set or default drifted from the registry (see above)"
import sys, yaml
from omp_expected import PER_ID, PER_ID_ENGINES
from litellm_sync import registry_variants
from effort_families import route_names
root, printed = sys.argv[1], sys.argv[2]
# Which engines answer to each route, from the registry rows through the SAME route_names
# the generator uses. The independent guard for the NAME set is check 8's own YAML walk;
# this one pins the SET behind a name, which is the column that moved while every ladder
# and default stayed equal.
engines = {}
for _v in registry_variants(root):
    for _n in route_names(root, _v)[0]:
        engines.setdefault(_n, set()).add(_v.get("engine"))
mo = yaml.safe_load(open(printed, encoding="utf-8"))["providers"]["club"]["modelOverrides"]
bad = []
for name in sorted(set(PER_ID) | set(mo)):
    if name not in PER_ID:
        bad.append("%s: rendered but not in the expected per-id table" % name)
        continue
    if name not in mo:
        bad.append("%s: expected from the table but not rendered" % name)
        continue
    exp_ladder, exp_default = PER_ID[name][:2]
    v = mo[name]
    t = v.get("thinking") or {}
    if t.get("efforts") != exp_ladder:
        bad.append("%s: ladder %r != %r" % (name, t.get("efforts"), exp_ladder))
    if t.get("defaultLevel") != exp_default:
        bad.append("%s: default %r != %r" % (name, t.get("defaultLevel"), exp_default))
    if v.get("maxTokens") != 32768:
        bad.append("%s: maxTokens %r != 32768" % (name, v.get("maxTokens")))
    exp_engines = PER_ID_ENGINES.get(name)
    if exp_engines is not None and engines.get(name) != exp_engines:
        bad.append("%s: served by %s, expected %s — the ladder can stay equal while the "
                   "engines behind it change, and one refusal fact would then move this id "
                   "for engines that never serve it"
                   % (name, sorted(engines.get(name) or []), sorted(exp_engines)))
if bad:
    print("  " + "\n  ".join(bad), file=sys.stderr); sys.exit(1)
PY


# 10. the compat pins ride the right client family — keyed by MODEL, so the aliases of a
#     model carry its pins too: supportsReasoningSummary: false on Flash-Next and GLM
#     (omp adds `summary: "auto"` to the Responses reasoning object, and LiteLLM's
#     Responses -> chat-completions bridge — tabbyAPI, Project Maya — would otherwise
#     forward the whole {effort, summary} object as reasoning_effort, which both reject),
#     and the 8192 thinking budget on the Qwen line (vLLM enforces it on the THINKING
#     field, not the reply; SGLang ignores it).
PYTHONPATH="$T" python3 - "$PRINTED" <<'PY' || bad "omp-setup.sh overrides: wrong compat pins (see above)"
import sys, yaml
from omp_expected import BUDGET, PER_ID, SUMMARY_FALSE, group
mo = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["providers"]["club"]["modelOverrides"]
bad = []
for mid, v in sorted(mo.items()):
    c = v.get("compat") or {}
    g = group(mid) if mid in PER_ID else "?"
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
python3 - "$ROOT" "$T" <<'PY' || bad "effort_families edge shapes (see above)"
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
    if "one route name must map" not in str(e):
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
    if "one route name must map" not in str(e):
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

# (f) an alias two models' composes both boot under is REFUSED: one route cannot carry two
#     overrides, and its ladder and compat would depend on which row was read first. A
#     route NO client family covers must not trip it — an unrelated catalog entry must not
#     break the render.
# (g) a row counts under the names its OWN compose boots under: the alias route gets its
#     own ladder and default, and the shared route is built only from the rows that answer
#     to it. The per-id rule #1951 settled, which grouping by the registry name breaks —
#     and the registry-name-AND-alias union breaks it the other way, letting an engine that
#     serves only its own alias move the shared id.
alias_root = os.path.join(sys.argv[2], "aliascase")
MF2 = [dict(MF[0], models=["ma"]), dict(MF[0], models=["mb"])]
Q = "models/%s/engine/compose/single/q/%s"
def write_compose(model, name, text):
    d = os.path.join(alias_root, "models", model, "engine", "compose", "single", "q")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        fh.write(text)
def alias_list(*names):
    return "services:\n  s:\n    command:\n" + "".join(
        "      - '--alias'\n      - '%s'\n" % n for n in names)
def arow(model, compose, served=None, **eff):
    r = {"model": model,
         "effort": dict({"levels": ["low", "high"], "compose_default": "low"}, **eff),
         "compose_path": Q % (model, compose)}
    if served:
        r["served_name"] = served
    return r
def gen2(mf, *rows):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        fams = compute_families(list(rows), mf, root=alias_root)
    return fams, err.getvalue()

write_compose("ma", "shared.yml", alias_list("a-shared"))
write_compose("ma", "alias.yml", alias_list("a-only"))
fams, _ = gen2(MF2, arow("ma", "shared.yml", levels=["low"]),
               arow("ma", "alias.yml", levels=["low", "high"], compose_default="high"))
got = {f["name"]: (f["ids"], f["ladder"]) for f in fams}
if got.get("a-only") != (["a-only"], {"efforts": ["low", "high"], "defaultLevel": "high"}):
    print("  an alias is its own family, with its own ladder and its own default:", got)
    sys.exit(1)
if got.get("a-shared") != (["a-shared"], {"efforts": ["low"], "defaultLevel": "low"}):
    print("  the shared route keeps the ladder of the rows that answer to it — the alias "
          "must not widen it or steal its default:", got)
    sys.exit(1)
if "ma" in got:
    print("  a compose that declares its own name is not a route for the registry name — "
          "nothing here answers to `ma`:", sorted(got))
    sys.exit(1)

# (g2) the refusal half, which is what the union actually broke. The catalog shape exactly:
#     the shared route is the registry served_name, one compose declares it, the other
#     declares ONLY its own alias — and under the union its row still counts under the
#     registry name. A Prism `refuses: [low]` then rounded the main qwen3.8-27b id up, and a
#     Strata `refuses: [xhigh]` took xhigh away from llama.cpp, QEngine and vLLM, which
#     never refuse it.
write_compose("ma", "main.yml", alias_list("ma"))
write_compose("ma", "ref.yml", alias_list("a-ref"))
fams, _ = gen2(MF2, arow("ma", "main.yml"), arow("ma", "ref.yml", refuses=["high"]))
got = {f["name"]: f["ladder"]["efforts"] for f in fams}
if got.get("ma") != ["low", "high"] or got.get("a-ref") != ["low"]:
    print("  a row whose compose declares only an alias must not drop a rung from the "
          "shared route:", got)
    sys.exit(1)

write_compose("ma", "c1.yml", alias_list("shared-name"))
write_compose("mb", "c2.yml", alias_list("shared-name"))
try:
    gen2(MF2, arow("ma", "c1.yml"), arow("mb", "c2.yml"))
except ValueError as e:
    if "served by 2 models" not in str(e):
        print("  the alias-collision refusal must name the fix, got:", e)
        sys.exit(1)
else:
    print("  an alias two models' composes both boot under must raise, not render twice")
    sys.exit(1)
write_compose("mc", "c3.yml", alias_list("shared2"))
write_compose("md", "c4.yml", alias_list("shared2"))
fams, _ = gen2(MF2, arow("mc", "c3.yml"), arow("md", "c4.yml"))
if fams:
    print("  a route no client family covers must generate nothing, got:", fams)
    sys.exit(1)

# (i) the one-line scan stops at the next flag and reads the `=` form. A value list that
#     ran past the next option turns "8080" into an alias — and two such composes in two
#     families would then collide and refuse the whole render.
write_compose("ma", "flags.yml",
              "services:\n  s:\n    command: llama-server --model /m.gguf --alias my-alias "
              "--port 8080 -c 262144 --jinja\n")
write_compose("mb", "eq.yml",
              "services:\n  s:\n    command: tabby --served-model-name=eq-alias "
              "--port 8080\n")
fams, _ = gen2(MF2, arow("ma", "flags.yml"), arow("mb", "eq.yml"))
if sorted(f["name"] for f in fams) != ["eq-alias", "my-alias"]:
    print("  the one-line form must stop at the next flag and read --flag=value:",
          sorted(f["name"] for f in fams))
    sys.exit(1)

# (j) a route that records no default of its own falls back to its MODEL's — but only when
#     its OWN ladder carries it. An off-ladder fallback is omp's silent xhigh, so that id
#     is skipped with the Fix: hint while its model's own route keeps rendering.
write_compose("ma", "nd.yml", alias_list("a-nd"))
fams, _ = gen2(MF2, arow("ma", "shared.yml"),
               arow("ma", "nd.yml", served="a-nd", compose_default=None))
if [f["ladder"]["defaultLevel"] for f in fams if f["name"] == "a-nd"] != ["low"]:
    print("  a route with no default of its own must fall back to its model's:", fams)
    sys.exit(1)
write_compose("ma", "nd2.yml", alias_list("a-nd2"))
fams, err = gen2(MF2, arow("ma", "shared.yml"),
                 arow("ma", "nd2.yml", served="a-nd2", refuses=["low"],
                      compose_default=None))
if sorted(f["name"] for f in fams) != ["a-shared"]:
    print("  a fallback the route's own ladder does not carry must skip that id, not the "
          "rest of its model:", sorted(f["name"] for f in fams))
    sys.exit(1)
if "no route serving this name" not in err or "Fix:" not in err:
    print("  the skipped fallback must warn and name the fix, got:", err)
    sys.exit(1)

# (k) an inline comment ends the one-line value list, and SERVED_MODEL_NAME is read in the
#     `KEY: value` and `KEY=value` forms as well as the `${…:-…}` one. A word absorbed from
#     a comment is a bogus route that can collide with another compose and refuse the whole
#     render; a missed env form is a route with no override (#1690).
write_compose("ma", "cmt.yml",
              "services:\n  s:\n    command: llama-server --alias cmt-alias # the alias\n")
fams, _ = gen2(MF2, arow("ma", "cmt.yml"))
if sorted(f["name"] for f in fams) != ["cmt-alias"]:
    print("  an inline comment must not become an alias:", sorted(f["name"] for f in fams))
    sys.exit(1)
write_compose("ma", "env1.yml", "services:\n  s:\n    environment:\n"
              "      - SERVED_MODEL_NAME=env-eq\n")
write_compose("ma", "env2.yml", "services:\n  s:\n    environment:\n"
              "      SERVED_MODEL_NAME: env-colon\n")
fams, _ = gen2(MF2, arow("ma", "env1.yml"), arow("ma", "env2.yml"))
if sorted(f["name"] for f in fams) != ["env-colon", "env-eq"]:
    print("  SERVED_MODEL_NAME must be read as `KEY: value` and `KEY=value` too:",
          sorted(f["name"] for f in fams))
    sys.exit(1)

# (l) a covered route whose engines declare no ladder at all is skipped WITH a warning.
#     omp-setup's completeness refusal tells the operator to read the warning above it and
#     names a default fix, so a silent skip here would send them looking for a problem that
#     is not theirs — the fix is the profile's effort block.
write_compose("ml", "nol.yml", alias_list("a-nol"))
fams, err = gen2([dict(MF[0], models=["ml"])],
                 {"model": "ml", "effort": {"compose_default": "low"},
                  "compose_path": Q % ("ml", "nol.yml")})
if fams:
    print("  a route with no ladder must generate nothing, got:", fams)
    sys.exit(1)
if "declares an effort ladder" not in err or "Fix:" not in err:
    print("  the no-ladder skip must warn and name the profile to fix, got:", err)
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


[ "$fail" -eq 0 ] && echo "test-omp-setup: ok (print-only, create, append beside other providers, idempotent refresh, refuses a hand-written club, overlay, doc block == overlay, override keys == every route the composes serve, per-route ladders and defaults, compat pins by model, auto-switching installs extension + settings and leaves config.yml alone, refuses a gateway on another machine, warns with no local litellm container, generator edge shapes: refusals, fold sources, lowest default, deterministic fold, MODEL_FAMILIES types, per-alias ladders, collisions, one-line scan, default fallback, empty registry fails closed)"
exit "$fail"
