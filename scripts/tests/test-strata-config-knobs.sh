#!/usr/bin/env bash
# Guard: the Strata recipe's strata_config.py, which writes club-3090's launch settings into
# Strata's run config at every boot (models/qwen3.8-flash-next/strata/launcher/).
#   1. no settings: the run config is setup's, unchanged;
#   2. each setting lands on the key Strata reads (reasoning_effort, reasoning_budget_tokens,
#      sampling.*, --spec in args, --rope-scaling/--rope-scale in args, aliases; never max_tokens, never
#      user_args, never setup's model_name);
#      xhigh / max become Strata's "high"; ENABLE_THINKING=false wins over REASONING_EFFORT;
#   3. the input is never written; a removed setting is gone at the next apply, also after Strata 0.2's
#      setup has kept our keys and folded our --spec into user_args / args_written (setup's own --spec
#      comes back: the recorded one, else the documented 4; the rope pair goes back to setup's own for the
#      context -- none inside the trained 262144, its yarn past it -- also after an apply with the settings
#      still set has consumed the folded user_args, the sequence that kept a removed ROPE_SCALING in force);
#      an unchanged in-place apply leaves the bytes;
#   4. effective: what the server runs of our settings -- user_args over args, as it merges them -- blind to
#      args_written;
#   5. a bad value exits 2 with the reason, for each setting (the launch-knob catalogue's
#      domain checks run the same `validate`); a short API_KEY is a warning, not a refusal;
#   6. every key a setting writes has evidence that the pinned Strata reads it (engine_config_reads.py check:
#      a new setting needs an entry in engine-config-reads.json);
#   7. forget: our flags leave the launch config's user_args (Strata 0.2.11 keeps them from launch to
#      launch), what it starts is unchanged, the person's own user_args and the file's mode stay, and the
#      next launch -- emulated from strata_control/arg_layers.py -- takes --spec from the run config again.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
H="models/qwen3.8-flash-next/strata/launcher/strata_config.py"

rc=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; rc=1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# setup's run config as Strata 0.2.1 writes it for the ISTA family (/data/install/strata-iq3_s.json, trimmed)
cat > "$work/setup.json" <<'JSON'
{"exe": "component:strata-engine",
 "args": ["--pack", "/data/Strata-data/packs/iq3_s", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
          "--mtp", "/data/Strata-data/mtp/rt", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
 "args_written": ["--pack", "/data/Strata-data/packs/iq3_s", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
          "--mtp", "/data/Strata-data/mtp/rt", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
 "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080, "gpu": [0, 1], "gpus_asked": true, "layer_split": "auto"}
JSON
cp "$work/setup.json" "$work/setup.orig"
SETTINGS="REASONING_EFFORT ENABLE_THINKING THINKING_BUDGET SPEC_N ROPE_SCALING ROPE_SCALE MAX_TOKENS TEMP TOP_P TOP_K MIN_P PRESENCE_PENALTY REPEAT_PENALTY SERVED_NAME CONTEXT KV API_KEY"
export CLUB3090_STRATA_SETUP_SPEC_FILE="$work/setup-spec"
run() { env $(for v in $SETTINGS; do printf -- '-u %s ' "$v"; done) "$@" python3 "$H" "${CMD[@]}"; }
get() { python3 -c "import json,sys; c=json.load(open('$work/cfg.json')); print(eval(sys.argv[1]))" "$1"; }
same() { cmp -s <(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),sort_keys=True))" "$1") \
                <(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),sort_keys=True))" "$2"); }

echo "== 1. no settings"
CMD=(apply "$work/setup.json" "$work/cfg.json")
out="$(run)" || bad "apply with no settings failed: $out"
same "$work/setup.json" "$work/cfg.json" && ok "run config == setup's" || bad "no settings changed the config"
command grep -q "settings: none" <<<"$out" && ok "says it applied none" || bad "summary: $out"

echo "== 2. each setting reaches Strata's key"
out="$(run REASONING_EFFORT=xhigh THINKING_BUDGET=2048 MAX_TOKENS=4096 SPEC_N=3 TEMP=0.6 TOP_P=0.95 TOP_K=20 \
          MIN_P=0 PRESENCE_PENALTY=1.5 REPEAT_PENALTY=1.05)" || bad "apply failed: $out"
[[ "$(get 'c["reasoning_effort"]')" == high ]] && ok "REASONING_EFFORT=xhigh -> reasoning_effort high" || bad "effort: $(get 'c.get("reasoning_effort")')"
[[ "$(get 'c["reasoning_budget_tokens"]')" == 2048 ]] && ok "THINKING_BUDGET -> reasoning_budget_tokens" || bad "budget"
# MAX_TOKENS is not a Strata setting (#1723 wrote it; Strata never read it): nothing may write max_tokens
[[ "$(get '"max_tokens" in c')" == False ]] && ok "MAX_TOKENS ignored (Strata reads no output cap from its config)" \
  || bad "max_tokens written into the config"
[[ "$(get 'c["args"][c["args"].index("--spec")+1]')" == 3 && "$(get 'c["args"].count("--spec")')" == 1 ]] \
  && ok "SPEC_N=3 replaces --spec 4 (once)" || bad "spec: $(get 'c["args"]')"
[[ "$(get '"user_args" in c or c["args_written"] != json.load(open("'"$work"'/setup.orig"))["args_written"]')" == False ]] \
  && ok "SPEC_N leaves setup's bookkeeping alone (no user_args; args_written as setup wrote it)" || bad "bookkeeping written"
[[ "$(get 'sorted(c["sampling"].items())')" == "[('min_p', 0.0), ('presence_penalty', 1.5), ('repetition_penalty', 1.05), ('temperature', 0.6), ('top_k', 20), ('top_p', 0.95)]" ]] \
  && ok "sampler settings -> sampling block" || bad "sampling: $(get 'c.get("sampling")')"
out="$(run SERVED_NAME=llm)" || bad "apply failed: $out"
[[ "$(get 'c["aliases"], c["model_name"]')" == "(['llm'], 'qwen3.8-flash-next-iq3_s')" ]] \
  && ok "SERVED_NAME=llm -> aliases [llm]; setup's model_name kept" || bad "served name: $(get 'c.get("aliases"), c.get("model_name")')"
out="$(run ENABLE_THINKING=false REASONING_EFFORT=low)" || bad "apply failed: $out"
[[ "$(get 'c["reasoning_effort"]')" == none ]] && command grep -q "ENABLE_THINKING=false wins" <<<"$out" \
  && ok "ENABLE_THINKING=false -> none, and says it overrode REASONING_EFFORT" || bad "thinking off: $out"
out="$(run ROPE_SCALING=yarn ROPE_SCALE=2 CONTEXT=524288)" || bad "apply failed: $out"
[[ "$(get 'c["args"][c["args"].index("--rope-scaling")+1] + "/" + c["args"][c["args"].index("--rope-scale")+1]')" == yarn/2 \
   && "$(get 'c["args"].count("--rope-scaling") + c["args"].count("--rope-scale")')" == 2 ]] \
  && ok "ROPE_SCALING/ROPE_SCALE -> --rope-scaling yarn --rope-scale 2 (once each)" || bad "rope: $(get 'c["args"]')"
[[ "$(get 'c["args_written"] != json.load(open("'"$work"'/setup.orig"))["args_written"]')" == False ]] \
  && ok "rope settings leave setup's args_written alone" || bad "rope wrote args_written"
out="$(run ROPE_SCALING=yarn CONTEXT=524288)" || bad "apply failed: $out"
[[ "$(get 'c["args"][c["args"].index("--rope-scale")+1]')" == 2 ]] && ok "ROPE_SCALE derived from CONTEXT=524288 (2)" \
  || bad "derived scale: $(get 'c["args"]')"

echo "== 3. the input is never written; a removed setting is gone"
cmp -s "$work/setup.json" "$work/setup.orig" && ok "input untouched (apply <in> <out>)" || bad "apply wrote its input"
cp "$work/setup.orig" "$work/cfg.json"; rm -f "$work/setup-spec"
CMD=(apply "$work/cfg.json" "$work/cfg.json")
run >/dev/null
[[ "$(cat "$work/setup-spec" 2>/dev/null)" == 4 ]] && ok "setup's own --spec recorded (4) from a run config we hadn't changed" || bad "setup's --spec not recorded"
run REASONING_EFFORT=low SPEC_N=3 TEMP=0.5 SERVED_NAME=llm >/dev/null
run >/dev/null
same "$work/cfg.json" "$work/setup.orig" && ok "in place: set, then nothing set -> back to setup's config" || bad "a setting lingered: $(cat "$work/cfg.json")"
# setup ran again (a new CONTEXT): it kept our keys and folded our --spec into user_args, args and args_written
run REASONING_EFFORT=low SPEC_N=3 TEMP=0.5 >/dev/null
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
a.remove("--spec"); a.remove("3"); a += ["--spec", "3"]
c["args_written"] = list(a); c["user_args"] = ["--spec", "3"]
json.dump(c, open(sys.argv[1], "w"))
PY
run >/dev/null
[[ "$(get '"reasoning_effort" in c or "sampling" in c or "user_args" in c')" == False \
   && "$(get 'c["args"][c["args"].index("--spec")+1] + "/" + c["args_written"][c["args_written"].index("--spec")+1]')" == 4/4 ]] \
  && ok "after setup folded --spec 3 into its bookkeeping: removed settings gone, --spec back to setup's 4" \
  || bad "folded: $(get '[c.get(k) for k in ("reasoning_effort","user_args")] + [c["args"][-2:]]')"
# the rope pair, same fold: our flags in user_args -> ours -> gone at the next apply
cp "$work/setup.orig" "$work/cfg.json"
run ROPE_SCALING=yarn ROPE_SCALE=2 >/dev/null
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["args_written"] = list(c["args"]); c["user_args"] = ["--rope-scaling", "yarn", "--rope-scale", "2"]
json.dump(c, open(sys.argv[1], "w"))
PY
run >/dev/null
[[ "$(get '"user_args" in c or "--rope-scaling" in c["args"] or "--rope-scale" in c["args"]')" == False ]] \
  && ok "after setup folded our rope flags into user_args: a removed setting is gone" \
  || bad "rope folded: $(get 'c.get("user_args")')"
# and the other direction: setup's OWN auto-yarn past the trained length (a 524288 context: yarn, factor 2) survives strip
cp "$work/setup.orig" "$work/cfg.json"
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
a[a.index("--max-context") + 1] = "524288"
c["args"] += ["--rope-scaling", "yarn", "--rope-scale", "2"]; c["args_written"] = list(c["args"])
json.dump(c, open(sys.argv[1], "w"))
PY
out="$(run 2>&1)"
[[ "$(get 'c["args"][c["args"].index("--rope-scaling")+1] + "/" + c["args"][c["args"].index("--rope-scale")+1]')" == yarn/2 ]] \
  && ok "setup's own auto-yarn (in args_written, no user_args) survives strip" || bad "stripped setup's rope: $(get 'c["args"]')"
# the third state: our override OF setup's resolution (args linear/4, args_written yarn/2, no user_args) ->
# strip restores setup's value; leaving ours is a stale override, deleting would unscale the boot
cp "$work/setup.orig" "$work/cfg.json"
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
a[a.index("--max-context") + 1] = "524288"
c["args"] += ["--rope-scaling", "yarn", "--rope-scale", "2"]; c["args_written"] = list(c["args"])
json.dump(c, open(sys.argv[1], "w"))
PY
run ROPE_SCALING=linear ROPE_SCALE=4 >/dev/null          # apply rewrites args only (args_written stays yarn/2)
[[ "$(get 'c["args"][c["args"].index("--rope-scaling")+1] + "/" + c["args"][c["args"].index("--rope-scale")+1]')" == linear/4 ]] \
  && ok "override of setup's resolution lands in args" || bad "override apply: $(get 'c["args"]')"
run >/dev/null                                            # settings removed: strip must restore setup's pair
[[ "$(get 'c["args"][c["args"].index("--rope-scaling")+1] + "/" + c["args"][c["args"].index("--rope-scale")+1]')" == yarn/2 \
   && "$(get 'c["args"].count("--rope-scaling") + c["args"].count("--rope-scale")')" == 2 ]] \
  && ok "a removed override reverts to setup's yarn/2 (not left stale, not deleted)" \
  || bad "override strip: $(get 'c["args"]')"
# the rig's sequence (#1958 follow-up), inside the trained length: a boot with SPEC_N=2 ROPE_SCALING=yarn ROPE_SCALE=2
# whose setup ran (it folds all three into user_args and stamps them into args_written), that boot's second apply with
# the settings still set (it takes the user_args entries out), then a boot with nothing set. The rope pair sat in
# args_written with no user_args entry, was read as setup's own and stayed: the engine kept yarn x2.
cp "$work/setup.orig" "$work/cfg.json"; echo 4 > "$work/setup-spec"
run SPEC_N=2 ROPE_SCALING=yarn ROPE_SCALE=2 >/dev/null
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["user_args"] = ["--spec", "2", "--rope-scaling", "yarn", "--rope-scale", "2"]; c["args_written"] = list(c["args"])
json.dump(c, open(sys.argv[1], "w"))
PY
run SPEC_N=2 ROPE_SCALING=yarn ROPE_SCALE=2 >/dev/null
run >/dev/null
[[ "$(get '[f in c[k] for k in ("args", "args_written") for f in ("--rope-scaling", "--rope-scale")].count(True)')" == 0 \
   && "$(get 'c["args"][c["args"].index("--spec")+1] + "/" + c["args_written"][c["args_written"].index("--spec")+1]')" == 4/4 ]] \
  && ok "set, folded by setup, applied again, then removed: no rope flag in args or args_written, --spec 4" \
  || bad "after the fold: args $(get 'c["args"][-6:]') args_written $(get 'c["args_written"][-6:]')"
# a run config already left in that state (the rope pair in args and args_written inside 262144, no user_args)
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["args"] += ["--rope-scaling", "yarn", "--rope-scale", "2"]; c["args_written"] = list(c["args"])
json.dump(c, open(sys.argv[1], "w"))
PY
run >/dev/null
[[ "$(get '[f in c[k] for k in ("args", "args_written") for f in ("--rope-scaling", "--rope-scale")].count(True)')" == 0 ]] \
  && ok "a run config left in that state is cleaned at the next apply (setup writes no rope inside 262144)" \
  || bad "left state: $(get 'c["args"][-6:]')"
rm -f "$work/setup-spec"
run SPEC_N=3 >/dev/null
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["args_written"] = list(c["args"]); c["user_args"] = ["--spec", "3"]; json.dump(c, open(sys.argv[1], "w"))
PY
run >/dev/null
[[ "$(get 'c["args"][c["args"].index("--spec")+1]')" == 4 ]] && ok "with no record of setup's --spec: the documented 4" || bad "no record: $(get 'c["args"]')"
cp "$work/setup.orig" "$work/cfg.json"; run REASONING_EFFORT=low >/dev/null; h="$(sha256sum < "$work/cfg.json")"
run REASONING_EFFORT=low >/dev/null
[[ "$h" == "$(sha256sum < "$work/cfg.json")" ]] && ok "an unchanged in-place apply leaves the bytes alone" || bad "an unchanged apply rewrote the file"

echo "== 4. effective: what the server runs of our settings"
CMD=(effective "$work/cfg.json")
e1="$(run)"
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["args_written"] = ["--spec", "9"]; c["user_args"] = ["--x", "1"]; json.dump(c, open(sys.argv[1], "w"))
PY
[[ "$(run)" == "$e1" ]] && ok "blind to args_written and to the person's other user_args" || bad "effective changed with setup's bookkeeping"
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["user_args"] = ["--x", "1", "--spec", "9"]; json.dump(c, open(sys.argv[1], "w"))
PY
[[ "$(run)" == *'"--spec": "9"'* ]] && ok "an --spec in user_args wins over args, as the server merges them" || bad "user_args --spec: $(run)"
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c.pop("user_args"); json.dump(c, open(sys.argv[1], "w"))
PY
python3 - "$work/cfg.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["reasoning_effort"] = "high"; json.dump(c, open(sys.argv[1], "w"))
PY
[[ "$(run)" != "$e1" ]] && ok "sees a managed key change" || bad "effective missed reasoning_effort"

echo "== 5. a bad value exits 2 with the reason"
CMD=(validate)
for probe in REASONING_EFFORT=minimal ENABLE_THINKING=off THINKING_BUDGET=-1 SPEC_N=0 SPEC_N=9 \
             ROPE_SCALING=ntk ROPE_SCALE=0.5 \
             TEMP=x TOP_P=0 TOP_K=65 MIN_P=2 PRESENCE_PENALTY=-1 REPEAT_PENALTY=0 KV=q8 CONTEXT=100; do
  err="$(run "$probe" 2>&1)"; r=$?
  [[ $r -eq 2 && "$err" == *"${probe%%=*}"* ]] && ok "$probe refused" || bad "$probe: rc=$r $err"
done
err="$(run ROPE_SCALING=none CONTEXT=524288 2>&1)"; r=$?
[[ $r -eq 2 && "$err" == *ROPE_SCALING* ]] && ok "ROPE_SCALING=none past the trained 262144 refused (setup refuses it too)" \
  || bad "ROPE_SCALING=none CONTEXT=524288: rc=$r $err"
err="$(run "SERVED_NAME=my model" 2>&1)"; r=$?
[[ $r -eq 2 && "$err" == *SERVED_NAME* ]] && ok "SERVED_NAME with a space refused" || bad "SERVED_NAME=my model: rc=$r $err"
run REASONING_EFFORT=max SPEC_N=8 KV=k8v4 CONTEXT=262144 ROPE_SCALING=yarn ROPE_SCALE=2 >/dev/null && ok "valid values pass" || bad "valid values refused"
out="$(run API_KEY=short 2>&1)"; r=$?
[[ $r -eq 0 && "$out" == *"API_KEY is 5 characters"* ]] && ok "a short API_KEY: a warning, not a refusal" || bad "short key: rc=$r $out"

echo "== 6. the pinned engine reads every key a setting writes (engine-config-reads.json; no docker)"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/lib/profiles/engine_config_reads.py check --engine strata || rc=1

echo "== 7. forget: our flags leave the launch config Strata 0.2.11 keeps from launch to launch"
# the launch config after the rig's frozen boot: the launch copied setup's folded user_args (ours, plus one of the
# person's) and stamped args_written / args_derived; the run config is back to setup's --spec 4, no rope. The rope
# pair appears twice in args and args_written: Strata compares a flag's whole list of groups, so one copy left in
# args_written reads as a hand edit (a live boot showed it)
python3 - "$work" <<'PY'
import json, sys
w = sys.argv[1]
run = json.load(open(w + "/setup.orig"))
a = list(run["args"]); a[a.index("--spec") + 1] = "2"
a += ["--rope-scaling", "yarn", "--rope-scale", "2", "--vram-reserve-mib", "0", "--rope-scaling", "yarn", "--rope-scale", "2"]
launch = {"args": a, "args_written": list(a), "args_derived": list(a), "api_key": "k" * 24,
          "user_args": ["--spec", "2", "--vram-reserve-mib", "0", "--rope-scaling", "yarn", "--rope-scale", "2"]}
json.dump(launch, open(w + "/launch.json", "w")); json.dump(run, open(w + "/run.json", "w"))
PY
chmod 600 "$work/launch.json"
CMD=(effective "$work/launch.json"); e_before="$(run)"
CMD=(forget "$work/launch.json"); run >/dev/null || bad "forget failed"
CMD=(effective "$work/launch.json")
[[ "$(run)" == "$e_before" ]] && ok "what the launch config starts is unchanged" || bad "forget changed what it starts: $(run)"
[[ "$(stat -c %a "$work/launch.json")" == 600 ]] && ok "its mode stays 600 (it holds the API key)" || bad "mode: $(stat -c %a "$work/launch.json")"
python3 - "$work" <<'PY' && ok "user_args keeps only the person's own; the next launch takes --spec 4 and no rope from the run config" || rc=1
import json, sys
w = sys.argv[1]
prev, run = json.load(open(w + "/launch.json")), json.load(open(w + "/run.json"))
# strata_control/user_args.py + arg_layers.py (Strata 0.2.11), reduced to what decides an option's value
def groups(a):
    out = []
    for t in a or []:
        (out.append([t]) if t.startswith("--") or not out else out[-1].append(t))
    return out
flat = lambda gs: [t for g in gs for t in g]
def merge(args, user):
    ug = groups(user); mine = {g[0] for g in ug}
    return flat([g for g in groups(args) if g[0] not in mine] + ug)
def by_flag(gs):                                      # a flag's whole list of groups, as Strata compares them
    d = {}
    for g in gs:
        d.setdefault(g[0], []).append(g)
    return d
def capture(cfg):
    mine = by_flag(groups(cfg.get("user_args")))
    cur, was = by_flag(groups(cfg["args"])), by_flag(groups(cfg["args_written"]))
    for f, gs in cur.items():
        if was.get(f) != gs:
            mine[f] = gs
    for f in list(mine):
        if f in was and f not in cur:
            del mine[f]
    return flat([g for gs in mine.values() for g in gs])
hand = capture(prev)                                  # _hand_edits(prev): prev has args_derived and no launch_set
args = merge(run["args"], merge(run.get("user_args"), hand))
val = lambda a, f: a[a.index(f) + 1] if f in a else None
bad = []
if prev.get("user_args") != ["--vram-reserve-mib", "0"]:
    bad.append(f"user_args {prev.get('user_args')}")
if val(args, "--spec") != "4" or "--rope-scaling" in args or "--rope-scale" in args:
    bad.append(f"the next launch runs --spec {val(args, '--spec')}, rope {val(args, '--rope-scaling')}/{val(args, '--rope-scale')}")
if val(args, "--vram-reserve-mib") != "0":
    bad.append("the person's --vram-reserve-mib 0 was lost")
for b in bad:
    print("  ✗ " + b)
sys.exit(1 if bad else 0)
PY
h="$(sha256sum < "$work/launch.json")"; CMD=(forget "$work/launch.json"); run >/dev/null
[[ "$h" == "$(sha256sum < "$work/launch.json")" ]] && ok "forget on a clean launch config leaves the bytes alone" || bad "forget rewrote a clean file"

if [[ "$rc" -eq 0 ]]; then echo "test-strata-config-knobs: ok"; else echo "test-strata-config-knobs: FAILED" >&2; fi
exit "$rc"
