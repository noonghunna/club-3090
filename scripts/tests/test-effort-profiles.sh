#!/usr/bin/env bash
#
# test-effort-profiles — guards the #1951 single-source effort metadata: the
# `effort:` blocks in the model profiles and the engine profiles, and the
# cross-file invariants the gateway render (phase 3) and the omp overrides will
# be built on. The loader (compat.py _effort/_engine_effort) already pins the
# SHAPE of every block; this test pins the SEMANTICS across files:
#
#   1. GATE — every curated model that declares a `thinking` sampler profile,
#      or whose composes/patches carry REASONING_EFFORT / reasoning_effort,
#      declares an `effort:` block (levels once probed, or an honest status).
#   2. BUDGET MAP — budget_map: true only for a ladder exactly equal (same
#      ORDER) to effort_budget.py's LEVELS (the map only knows the Qwen
#      ladder; a GLM ladder must not suggest budget support that isn't wired
#      — #1951 risk 4).
#   3. COMPOSE DEFAULTS — every `REASONING_EFFORT:-X` compose default is one of
#      its model's declared levels (principle 1: the compose holds the default,
#      so the default must be a rung the model accepts). `max` defaults are
#      ratcheted to an allowlist: litellm's default_reasoning_effort Literal
#      (v1.100.3) EXCLUDES max, so phase 2's render must OMIT a max default,
#      never downgrade it — the allowlist forces a conscious re-verification.
#   4. LITELM VOCABULARY — every rung the catalog declares is a member of the
#      verified LiteLLM's REASONING_EFFORT Literal, or /model_group/info
#      silently drops it (declared_reasoning_efforts filters through get_args).
#      A tripwire fails when the gateway image pin moves: re-read the Literal
#      from the new image before trusting the set.
#   5. ENGINE FOLDS — for every (model, engine) pair the registry actually
#      serves: the engine's maps VALUES and default_when_unset land on rungs
#      the model's template accepts (identity folds src->src are legal), and
#      an engine serving a budget_map model folds only inside effort_budget.py's
#      ALIASES. Also pins the off_switch round-trip: "none" survives the YAML
#      load as a string, never null (#1951 review r1-1).
#   6. OMP OVERRIDES — FAMILIES and the *_IDS constants are parsed from
#      omp-setup.sh source (no hand-kept family->id map): every family
#      renders, every rendered ladder equals its model profile's levels,
#      every family id is present with that ladder, and every defaultLevel
#      is a rung of its own ladder (ABSENT is legal — omp falls back to the
#      ladder TOP; a PRESENT off-ladder default would be a silent xhigh).
#
# Unknown stays unknown: a `status: unprobed` / `no-string-ladder` model
# renders nothing, so checks 3/5 skip it rather than inventing a ladder.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

echo "--- #1951 effort metadata ---"

T_HOME="$T" python3 - <<'PY' || fail "effort-profiles checks failed"
import io, os, re, subprocess, sys, yaml

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "lib"))
from profiles.compat import load_profiles

# LiteLLM's REASONING_EFFORT Literal, VERIFIED against the pinned gateway
# image v1.100.3 on 2026-11-15 (litellm/types/llms/openai.py:1867):
#   REASONING_EFFORT = Literal["none","minimal","low","medium","high","xhigh","max"]
# litellm's declared_reasoning_efforts() DROPS any declared level outside that
# Literal without a word — a ladder rung off this set silently vanishes from
# /model_group/info. Re-read the pinned image on every gateway pin bump (the
# tripwire in check 4 forces it).
LITELLM_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

# The family -> models mapping of omp-setup.sh's hand-written FAMILIES list —
# the drift surface check 6 pins; it disappears when phase 2 generates the
# list from the profiles. EVERY catalog model a family's omp ids serve is
# listed, and every listed model's ladder must equal the family's — so a
# newly probed model with no family fails loudly instead of silently missing
# from the omp surface (#1951 review r7).
FAMILY_MODELS = {"qwen38": ["qwen3.8-27b", "thinkingcap-qwen3.8-27b"],
                 "flash-next": ["qwen3.8-flash-next"],
                 "glm53": ["glm-5.3-flash"]}

failures = []
def bad(msg):
    failures.append(msg)
    print(f"  ✗ {msg}", file=sys.stderr)

profiles = load_profiles()
reg = yaml.safe_load(io.open("scripts/lib/profiles/registry.yaml", encoding="utf-8"))
entries = reg["entries"]

# effort_budget.py's own vocabulary — read it, never restate it.
eb_src = io.open("scripts/lib/effort_budget.py", encoding="utf-8").read()
eb_levels = tuple(x.strip().strip('"\'') for x in
                  re.search(r"^LEVELS = \((.*?)\)", eb_src, re.M).group(1).rstrip(",").split(","))
eb_aliases = {k: v for k, v in re.findall(r'"(\w+)": "(\w+)"',
              re.search(r"^ALIASES = \{(.*?)\}", eb_src, re.M).group(1))}

def model_levels(model_id):
    """(levels, status) of a model's effort block; ([], None) when it has none."""
    eff = (profiles.models.get(model_id) or type("x", (), {"effort": None})()).effort
    if not eff:
        return [], None
    return list(eff.get("levels") or []), eff.get("status")

# ---- 1. GATE: effort-aware models declare an effort block -------------------
effort_aware = {}   # model id -> reason the gate fired
n0 = len(failures)
for tag, e in entries.items():
    m = e.get("model")
    if not m:
        continue
    sp = e.get("sampler_profiles")
    if isinstance(sp, dict) and any("thinking" in str(k).lower() for k in sp):
        effort_aware.setdefault(m, f"thinking sampler profile on {tag}")
mdir = "models"
for model_dir in sorted(os.listdir(mdir)):
    yml = f"scripts/lib/profiles/models/{model_dir}.yml"
    if not os.path.exists(yml):
        continue
    for dirpath, _dirs, files in os.walk(os.path.join(mdir, model_dir)):
        if "_archive" in dirpath:
            continue
        for fn in files:
            if not fn.endswith((".yml", ".jinja", ".py")):
                continue
            text = io.open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
            if fn.endswith((".yml", ".py")):
                # Prose in a comment ("this template doesn't read
                # reasoning_effort") is not an effort knob; only code counts —
                # full-line OR inline, or a comment mention fires the gate on a
                # model whose only mention is prose (#1951 review r6).
                text = "\n".join(l.split("#", 1)[0] for l in text.splitlines()
                                 if not l.lstrip().startswith("#"))
            if "REASONING_EFFORT" in text or "reasoning_effort" in text:
                effort_aware.setdefault(model_dir, f"effort knob/template under {dirpath}")
for model_id, reason in sorted(effort_aware.items()):
    levels, status = model_levels(model_id)
    if not levels and not status:
        bad(f"{model_id}: effort-aware ({reason}) but no `effort:` block — "
            "declare the probed ladder or an honest status (#1951)")
if len(failures) == n0:
    print(f"  ✓ gate: {len(effort_aware)} effort-aware models all declare an effort block")

# ---- 2. BUDGET MAP: budget_map only over effort_budget.py's ladder ----------
n0 = len(failures)
for model_id, m in sorted(profiles.models.items()):
    eff = m.effort
    if not eff or not eff.get("levels"):
        if eff and eff.get("budget_map"):
            bad(f"{model_id}: budget_map without levels")
        continue
    if eff.get("budget_map"):
        # ORDERED equality: the map indexes rungs positionally, so a same-set
        # ladder in a different order would silently shift every budget.
        if list(eff["levels"]) != list(eb_levels):
            bad(f"{model_id}: budget_map true but levels {eff['levels']} != "
                f"effort_budget.py LEVELS {list(eb_levels)} (ordered) — the map "
                "only knows the Qwen ladder (#1951 risk 4)")
if len(failures) == n0:
    print(f"  ✓ budget_map agrees with effort_budget.py LEVELS {list(eb_levels)}")

# ---- 3. COMPOSE DEFAULTS inside the model's ladder --------------------------
# Three forms a compose can set THE knob in: the env-default (`:-`), the YAML
# mapping (`REASONING_EFFORT: low`), and the equals list form (`= low`) — plus
# a variable reference WITH a fallback (`REASONING_EFFORT: ${VAR:-low}`, the
# normal compose `environment:` form), whose fallback names the compose's own
# unset-case default. The lookbehind/lookahead keep a LONGER variable
# (QE_REASONING_EFFORT, LLAMA_ARG_REASONING_EFFORT_ALLOW) from being mistaken
# for it — those are other knobs, not defaults for this one.
default_re = re.compile(
    r'(?<![A-Za-z_])REASONING_EFFORT(?:["\']?:-["\']?|["\']?[:=][ \t]*["\']?\$\{[A-Za-z_]+:-|["\']?:\s*["\']?|=)["\']?([a-z]+)')
# The knob handed to a launcher with no value of its own (the strata-maya
# composes' bare `REASONING_EFFORT:`, the knob-list lines) is NOT a default —
# the launcher's default is what the pair fact records. A value that is a
# VARIABLE reference is a hop the launch-time override owns, not a catalogue
# default — UNLESS it carries a fallback, which default_re reads as the
# compose's own default. A literal the emit cannot read is what stays flagged.
bare_knob_re = re.compile(r'^[ \t\-]*["\']?REASONING_EFFORT["\']?:?[ \t]*$')
knob_name_re = re.compile(r'(?<![A-Za-z_])REASONING_EFFORT(?![A-Za-z_])')
var_ref_re = re.compile(
    r'(?<![A-Za-z_])REASONING_EFFORT(?![A-Za-z_])["\']?[:=][ \t]*["\']?\$')
# `max` defaults are ratcheted: litellm's default_reasoning_effort Literal
# (v1.100.3) EXCLUDES max, so phase 2's gateway render must OMIT a max
# default, never downgrade it. Today only GLM-5.3-Flash legitimately defaults
# to max (it IS a rung of its ladder); adding a model here is a conscious
# re-verification of the render rule, not a rubber stamp.
MAX_DEFAULT_MODELS = {"glm-5.3-flash"}
n0 = len(failures)
checked = 0
for model_dir in sorted(os.listdir(mdir)):
    levels, status = model_levels(model_dir)
    if not levels:
        continue                      # unprobed / no ladder: nothing to check against
    for dirpath, _dirs, files in os.walk(os.path.join(mdir, model_dir)):
        if "_archive" in dirpath or "compose" not in dirpath.split(os.sep):
            continue
        for fn in files:
            if not fn.endswith(".yml"):
                continue
            path = os.path.join(dirpath, fn)
            text = io.open(path, encoding="utf-8", errors="replace").read()
            # A commented-out default is not a default — full-line OR inline.
            text = "\n".join(l.split("#", 1)[0] for l in text.splitlines()
                             if not l.lstrip().startswith("#"))
            for dflt in sorted(set(default_re.findall(text))):
                checked += 1
                if dflt not in levels:
                    bad(f"{path}: REASONING_EFFORT default {dflt!r} is not one "
                        f"of {model_dir}'s levels {levels}")
                if dflt == "max" and model_dir not in MAX_DEFAULT_MODELS:
                    bad(f"{path}: REASONING_EFFORT:-max default for {model_dir} — "
                        "litellm's default_reasoning_effort Literal excludes max, "
                        "so the render must OMIT it, never downgrade; add the model "
                        "to MAX_DEFAULT_MODELS only after re-verifying that rule")
            for line in text.splitlines():
                if not knob_name_re.search(line):
                    continue
                if (default_re.search(line) or bare_knob_re.match(line.strip())
                        or var_ref_re.search(line)):
                    continue
                bad(f"{path}: unparseable REASONING_EFFORT line — a form the "
                    f"emit's _compose_effort_default does not read, so the pair "
                    f"default would fill it: {line.strip()[:120]}")
if len(failures) == n0:
    print(f"  ✓ {checked} compose REASONING_EFFORT defaults all inside their "
          "model's ladder (max ratcheted to GLM-5.3-Flash)")

# ---- 4. every declared rung is litellm vocabulary ---------------------------
n0 = len(failures)
runks = set()
for model_id, m in sorted(profiles.models.items()):
    runks |= set((m.effort or {}).get("levels") or [])
# The PAIR facts are wire vocabulary too (#2055 review it4-2): a pair default
# is what the route applies when the request names nothing, and a fold's
# SOURCE is a name the client can actually send — both reach
# /model_group/info through the render, so both must be litellm Literal
# members. (The engine-level sweep this replaces was dead: the loader refuses
# engine-level default_when_unset/maps, so those were always empty.)
for eid, en in sorted(profiles.engines.items()):
    eff = en.effort or {}
    for model_id, facts in sorted((eff.get("by_model") or {}).items()):
        if facts.get("default_when_unset"):
            runks.add(facts["default_when_unset"])
        runks |= set(facts.get("maps") or {}) | set((facts.get("maps") or {}).values())
    # unknown_name is the rung an engine applies to names it does NOT
    # recognise — a wire value in principle, but the emit carries no
    # unknown_name and the render never advertises it, so it stays out here.
off_vocab = runks - LITELLM_EFFORTS
if off_vocab:
    bad(f"effort names outside litellm's REASONING_EFFORT Literal (they would be "
        f"silently dropped from /model_group/info): {sorted(off_vocab)}")
pin_src = io.open("services/litellm/docker-compose.yml", encoding="utf-8").read()
# The ACTUAL image line — a comment or an alternate occurrence must not satisfy
# the tripwire — and the WHOLE tag: v1.100.3-rc1 is a different build whose
# Literal this test has not read.
pin_tag = None
for _line in pin_src.splitlines():
    if _line.lstrip().startswith("#") or "image:" not in _line:
        continue
    _m = re.search(r"image:\s*[\"']?\S*litellm:(\S+?)[\"']?\s*$", _line)
    if _m:
        pin_tag = _m.group(1)
        break
if pin_tag is None:
    bad("could not parse the litellm image pin from services/litellm/"
        "docker-compose.yml — the tripwire needs the tag")
elif pin_tag != "v1.100.3":
    bad(f"gateway pin moved to {pin_tag} — re-read the REASONING_EFFORT "
        "Literal from the new image (litellm/types/llms/openai.py) and update "
        "LITELLM_EFFORTS before trusting this set")
# The llama.cpp engines' none/off-switch claims are tied to a build the same
# way (#2055 review it4-3): the disclosure was measured on v0.7.0 (b11669) and
# its STALE note accounts for the CURRENT pin. A bump invalidates the note —
# fail and force the re-measure, exactly like the litellm tag above.
LLAMA_PIN_BUILDS = {
    "llama-cpp-local": "b12359",           # lineage.base_version (upstream pin)
    "llamacpp-club3090-v1.5": "7104d5f4eaf19870d50af015d326957347549ccd",   # fork commit
    "llamacpp-club3090-v1.6": "d9cc8af98559dbd54c26c18aed59a2e6f834e95d",   # fork commit
}
for eid, want in sorted(LLAMA_PIN_BUILDS.items()):
    eng = profiles.engines.get(eid)
    lin = (getattr(eng, "lineage", None) or {}) if eng else {}
    got = lin.get("base_version") or lin.get("source_ref")
    if got != want:
        bad(f"{eid}: llama.cpp pin moved (build/commit {got!r} != {want!r}) — "
            f"the effort disclosure's STALE note no longer names the running "
            f"binary; re-measure `none`'s rendering and the off-switch kwargs "
            f"on the new binary and update the disclosure")
if len(failures) == n0:
    print(f"  ✓ llama.cpp effort disclosures tied to the current pins "
          f"(llama-cpp-local b12359, both forks by commit)")
if len(failures) == n0:
    print(f"  ✓ all {len(runks)} declared effort names are litellm Literal members "
          f"(pin {pin_tag or '?'})")

# ---- 5. PAIR FOLD FACTS land on rungs the served model accepts --------------
# `default_when_unset` and `maps` are per (model, engine) facts, not engine-wide
# ones (#1951 review r5): strata-maya's three launchers default to max / high /
# auto and two of them fold differently, and qwen3.8-flash-next folds differently
# on strata than on vLLM. So this walks the engine's `by_model` entries — a pair
# that is not listed simply has no fold facts.
budget_models = {mid for mid, m in profiles.models.items()
                 if (m.effort or {}).get("budget_map")}
pairs = {(e.get("model"), e.get("engine")) for e in entries.values()
         if e.get("model") and e.get("engine")}
n0 = len(failures)
fold_checked = 0
for engine_id, eng in sorted(profiles.engines.items()):
    for model_id, facts in sorted(((eng.effort or {}).get("by_model") or {}).items()):
        if model_id not in profiles.models:
            bad(f"{engine_id}: effort.by_model.{model_id} is not a known model id")
            continue
        if (model_id, engine_id) not in pairs:
            bad(f"{engine_id}: effort.by_model.{model_id} names a pair no slug "
                f"routes — the fact is unreachable, drop it or fix the id")
        levels, _status = model_levels(model_id)
        if not levels:
            bad(f"{engine_id}: effort.by_model.{model_id} records fold facts for a "
                f"model with no probed ladder — a fold has nothing to land on")
            continue
        d = facts.get("default_when_unset")
        if d and d not in levels:
            bad(f"{model_id} on {engine_id}: default_when_unset {d!r} is not a "
                f"rung of the model's ladder {levels}")
        if d == "max" and model_id not in MAX_DEFAULT_MODELS:
            bad(f"{model_id} on {engine_id}: pair default_when_unset 'max' — the "
                f"render must OMIT it (litellm's Literal excludes max), so treat it "
                f"like a compose default: add the model to MAX_DEFAULT_MODELS only "
                f"after re-verifying that rule")
        maps = facts.get("maps") or {}
        for src, dst in maps.items():
            if dst not in levels:
                bad(f"{model_id} on {engine_id}: fold {src}->{dst} lands outside "
                    f"the model's ladder {levels}")
        refuses = facts.get("refuses") or []
        for r in refuses:
            if r not in levels:
                bad(f"{model_id} on {engine_id}: refused rung {r!r} is not a "
                    f"rung of the model's ladder {levels}")
            if r in maps:
                bad(f"{model_id} on {engine_id}: refused rung {r!r} is also a "
                    f"fold source — the pair cannot 400 a name and accept it "
                    f"via a fold at the same time")
        # Identity folds (src->src) are legal — the engine accepts the name as-is.
        if maps and model_id in budget_models and not all(
                dst == src or (src, dst) in eb_aliases.items()
                for src, dst in maps.items()):
            bad(f"{engine_id}: by_model.{model_id} maps {maps} fold outside "
                f"effort_budget.py ALIASES {eb_aliases} for a budget_map model")
        fold_checked += 1
# off_switch round-trip: the LOADED value must be the STRING "none". YAML 1.1
# nulls only an empty value, `null` and `~` — never the word `none` (measured
# on this rig 2026-11-15: `off_switch: none` loads as 'none'), so what this pins
# is the present-but-null case the loader rejects: the silent downgrade to
# UNKNOWN cannot come back (#1951 review r1-1).
for eid in ("qengine", "sglang-stable", "vllm-stable", "strata", "strata-maya"):
    off = (profiles.engines[eid].effort or {}).get("off_switch")
    if off != "none":
        bad(f"{eid}: off_switch must load as the string 'none', got {off!r} — "
            'an empty value, `null` or `~` parses to null; give it a value '
            '(off_switch: "none") or drop the key')
# A wholesale deletion of the pair facts would pass everything above: the loader
# negatives need no data, and an empty by_model is legal on an engine that has no
# pair facts yet. Pin the floor — 10 pair facts across 6 engines today; bump on
# purpose when the catalogue legitimately changes (#1951 review r6).
PAIR_FACT_FLOOR = 10
if fold_checked < PAIR_FACT_FLOOR:
    bad(f"only {fold_checked} pair fold facts loaded, floor is {PAIR_FACT_FLOOR} — "
        f"bump the floor on purpose if the catalogue legitimately changed")
if len(failures) == n0:
    print(f"  ✓ {fold_checked} pair fold facts land on accepted rungs (floor "
          f"{PAIR_FACT_FLOOR} pinned); refusals stay on the ladder and off "
          "fold sources; off_switch round-trips as 'none'")

# ---- 5c. ONE PORT, ONE MODEL ------------------------------------------------
# A port hosting two models advertises one model's facts for both: port_facts
# agree-or-drops across the rows, but a no-thinking model beside a knob-less
# thinking model would advertise supports_reasoning false for a thinker, and
# two ladders on one port advertise nothing at all. The render already refuses
# to name such a port (booting-<port>); the catalogue must not create one
# (#1951 review r6).
n0 = len(failures)
port_models: "dict[int, set]" = {}
for e in entries.values():
    if e.get("port") and e.get("model"):
        port_models.setdefault(e["port"], set()).add(e["model"])
for port, models in sorted(port_models.items()):
    if len(models) > 1:
        bad(f"port {port} hosts {len(models)} models ({sorted(models)}) — split "
            f"the ports: the render advertises one model's effort facts for both")
if len(failures) == n0:
    print("  ✓ every registry port serves exactly one model")
# ---- 5d. supports_none: true is a per-pair claim -----------------------------
# An engine-level `supports_none: true` says the engine maps `none` to
# enable_thinking=false before the template runs; whether THIS model's template
# then closes the thinking block is the model's contract. A ladder model on
# such an engine advertises `supports_none: true` the moment it probes a
# ladder, so its pair entry must exist AND carry the pair's own none claim —
# `supports_none` or `off_switch` with its citation — before the advertise is
# trusted: an entry holding only fold facts would leave the route advertising
# a wire none claim resting on nothing pair-specific (#1951 review r7 + r8).
# `false` engines need no entry: the server ignoring `none` is engine-level
# behaviour, and the advertise direction is conservative.
n0 = len(failures)
for e in entries.values():
    m, eng_id = e.get("model"), e.get("engine")
    if not m or not eng_id:
        continue
    eng = profiles.engines.get(eng_id)
    eff = (eng.effort or {}) if eng else {}
    if eff.get("supports_none") is not True:
        continue
    levels, _status = model_levels(m)
    if not levels:
        continue
    pair = (eff.get("by_model") or {}).get(m)
    if not pair:
        bad(f"{m} on {eng_id}: probed ladder {levels} but no by_model entry — the "
            f"route advertises the engine's supports_none: true without a pair "
            f"fact citing THIS model's template (verify it honors `none`, or "
            f"record the pair's own measurement in its by_model entry)")
    elif "supports_none" not in pair and "off_switch" not in pair:
        bad(f"{m} on {eng_id}: by_model entry holds fold facts but no none claim "
            f"— the route advertises the engine's supports_none: true resting on "
            f"nothing pair-specific; declare the pair's own supports_none (or "
            f"off_switch) with its citation (#1951 review r8)")
if len(failures) == n0:
    print("  ✓ every supports_none:true engine's ladder pairs carry a by_model "
          "entry with the pair's own none claim")
# ---- 5b. the loader REFUSES an engine-wide fold fact (#1951 review r5) -------
# An engine-level `default_when_unset`/`maps` is the silent-inheritance bug the
# review named: a model never measured on it would be folded and defaulted by a
# fact from its sibling's launcher. The loader must refuse it and name by_model.
from pathlib import Path
from profiles.compat import _engine_effort, ProfileError

def expect_refusal(block, needle, what):
    try:
        _engine_effort(block, Path("negative-test.yml"))
    except ProfileError as exc:
        if needle not in str(exc):
            bad(f"{what}: refused, but the message does not name {needle!r}: {exc}")
        return
    bad(f"{what}: the loader ACCEPTED it — silent inheritance is back")

n0 = len(failures)
expect_refusal({"evidence": "x", "default_when_unset": "xhigh"},
               "by_model", "engine-level effort.default_when_unset")
expect_refusal({"evidence": "x", "maps": {"high": "xhigh"}},
               "by_model", "engine-level effort.maps")
expect_refusal({"evidence": "x", "by_model": {}}, "non-empty", "empty by_model")
expect_refusal({"evidence": "x", "by_model": {"m": {"levels": ["low"]}}},
               "allows only", "by_model entry holding a model-level key")
expect_refusal({"evidence": "x", "by_model": {"m": {"off_switch": "bogus"}}},
               "off_switch must be one of", "bad off_switch inside by_model")
expect_refusal({"evidence": "x", "by_model": {"m": {"supports_none": "false"}}},
               "supports_none must be a bool", "string supports_none inside by_model")
expect_refusal({"evidence": "x", "by_model": {"m": {}}}, "is empty", "empty by_model entry")
expect_refusal({"evidence": "x", "by_model": {"m": {"maps": {}}}},
               "non-empty", "empty maps inside by_model")
expect_refusal({"evidence": "x",
                "by_model": {"m": {"default_when_unset": None}}},
               "explicitly null", "null pair default inside by_model")
expect_refusal({"evidence": "x",
                "by_model": {"m": {"default_when_unset": 5}}},
               "non-empty string", "non-string pair default inside by_model")
expect_refusal({"evidence": "x", "by_model": {"m": {"maps": {"a": 5}}}},
               "non-empty", "non-string maps value inside by_model")
expect_refusal({"evidence": "x",
                "by_model": {"m": {"refuses": "xhigh"}}},
               "non-empty list", "non-list pair refusal inside by_model")
expect_refusal({"evidence": "x",
                "by_model": {"m": {"default_when_unset": "xhigh"}}},
               "evidence is REQUIRED", "pair entry with no own evidence")
try:
    _engine_effort({"off_switch": "none", "supports_none": True,
                    "by_model": {"m": {"default_when_unset": "xhigh",
                                       "maps": {"high": "xhigh"},
                                       "refuses": ["xhigh"],
                                       "evidence": "x"}},
                    "evidence": "x"}, Path("negative-test.yml"))
except ProfileError as exc:
    bad(f"valid pair-scoped block was refused: {exc}")
try:
    # The r7 escape hatch: a pair's OWN none measurement overrides the
    # engine-level claim (the emit reads the pair value first).
    _engine_effort({"off_switch": "none", "supports_none": True,
                    "by_model": {"m": {"default_when_unset": "xhigh",
                                       "supports_none": False,
                                       "off_switch": "enable-thinking-kwarg",
                                       "evidence": "the pair's own probe"}},
                    "evidence": "x"}, Path("negative-test.yml"))
except ProfileError as exc:
    bad(f"a pair's own none measurement was refused: {exc}")
if len(failures) == n0:
    print("  ✓ engine-level fold facts refused (13 shapes), pair-scoped block "
          "and its none-override load")

# ---- 6. omp-setup FAMILIES agree with the profiles ---------------------------
env = dict(os.environ, HOME=os.environ["T_HOME"],
           PI_CODING_AGENT_DIR=os.path.join(os.environ["T_HOME"], "agent"))
out = subprocess.run(["bash", "scripts/omp-setup.sh", "--print"],
                     capture_output=True, text=True, env=env)
n0 = len(failures)
if out.returncode != 0:
    bad(f"omp-setup.sh --print failed: {out.stderr.strip()[:400]}")
else:
    doc = yaml.safe_load(out.stdout)
    ovr = (doc.get("providers") or {}).get("club", {}).get("modelOverrides") or {}
    if not ovr:
        bad("omp-setup --print rendered no modelOverrides")
    # Parse FAMILIES and the *_IDS constants from omp-setup.sh SOURCE — the
    # ids bind each family to its rendered overrides, so the check survives
    # id-list edits without a hand-kept map (the family->MODEL binding below
    # stays hand-kept until phase 2 generates the list from the profiles).
    omp_src = io.open("scripts/omp-setup.sh", encoding="utf-8").read()
    ids_bodies = {m2.group(1): m2.group(2)
                  for m2 in re.finditer(r"^(\w+_IDS) = \[(.*?)\]", omp_src, re.M | re.S)}
    ids_const = {k: re.findall(r'"([^"]+)"', b) for k, b in ids_bodies.items()}
    # `[a-z0-9-]` not `w+`: the family name "flash-next" has a hyphen, and a
    # word-only name group silently skips that whole entry (the completeness
    # scan below is what makes that class of miss fail loudly).
    fam_re = re.compile(r'\{"name": "([\w-]+)", "ids": (\w+),\s*'
                        r'"ladder": \{"efforts": \[([^\]]*)\], "defaultLevel": "(\w+)"\}')
    families = {m2.group(1): {"ids": ids_const.get(m2.group(2), []),
                              "ids_const": m2.group(2),
                              "ladder": [x.strip().strip("\"'") for x in m2.group(3).split(",") if x.strip()],
                              "dflt": m2.group(4)}
                for m2 in fam_re.finditer(omp_src)}
    if not families:
        bad("could not parse FAMILIES from omp-setup.sh — the literal changed "
            "shape; fix check 6's fam_re")
    # Completeness (r3-1): an entry fam_re silently SKIPS would under-check —
    # two families share the [low,medium,xhigh] ladder, so the set-membership
    # test below cannot see a missing one. Scan the region tolerantly (name and
    # ids-constant only, any key order/shape) and require the strict parse to
    # have found exactly that set. Same for ids: count quoted items in each
    # constant's source body, so a partially parsed id list fails loudly.
    fam_region = re.search(r"^FAMILIES = \[(.*?)^\]", omp_src, re.M | re.S)
    if not fam_region:
        bad("could not locate the FAMILIES literal in omp-setup.sh — check 6's "
            "region regex needs the new shape")
    else:
        tol_names = re.findall(r'"name":\s*"([^"]+)"', fam_region.group(1))
        tol_consts = re.findall(r'"ids":\s*(\w+)', fam_region.group(1))
        if len(tol_names) != len(tol_consts):
            bad("FAMILIES: name/ids fields do not pair up in the literal — "
                "the entry shape drifted")
        if sorted(tol_names) != sorted(families):
            bad(f"FAMILIES: the tolerant scan sees {sorted(tol_names)} but "
                f"fam_re parsed {sorted(families)} — an entry's shape drifted "
                "or a name is duplicated, and check 6 would silently skip it; fix fam_re")
        # A duplicate family name collapses into the dict below, so the two scans
        # above would still agree while one entry goes unchecked.
        if len(tol_names) != len(set(tol_names)):
            bad(f"FAMILIES has a duplicate family name: {tol_names}")
    for fam, f in sorted(families.items()):
        body = ids_bodies.get(f["ids_const"])
        if body is None:
            bad(f"FAMILIES {fam}: ids constant {f['ids_const']} not found in "
                "omp-setup.sh")
        else:
            # Count ITEMS (comma-separated, these lists are flat), not quoted
            # items: a single-quoted id is an item the "..." regex would miss,
            # and counting only quoted ones would then agree with itself.
            n_items = len([x for x in body.split(",") if x.strip()])
            if n_items != len(f["ids"]):
                bad(f"{f['ids_const']} ({fam}): {n_items} ids in the source but "
                    f"{len(f['ids'])} parsed — ids_const's regex missed one "
                    "(a non-double-quoted id?)")
    id_family = {mid: fam for fam, f in families.items() for mid in f["ids"]}
    # An id in two families would silently bind to whichever came last, checking
    # it against an arbitrary ladder — fail on the ambiguity instead.
    owner: dict = {}
    shared = []
    for fam, f in sorted(families.items()):
        for mid in f["ids"]:
            if mid in owner:
                shared.append(f"{mid} ({owner[mid]} and {fam})")
            owner[mid] = fam
    if shared:
        bad("a model id in more than one FAMILIES entry, so its override is "
            f"bound to an arbitrary ladder: {', '.join(shared)}")
    for fam, f in sorted(families.items()):
        mdls = FAMILY_MODELS.get(fam)
        if not mdls:
            bad(f"FAMILIES entry {fam!r} is not in FAMILY_MODELS — extend the map "
                "(the family->models binding stays hand-kept until phase 2)")
            continue
        for mdl in mdls:
            if f["ladder"] != model_levels(mdl)[0]:
                bad(f"FAMILIES {fam}: ladder {f['ladder']} != model {mdl}'s levels "
                    f"{model_levels(mdl)[0]} — drifted from the profiles (#1951)")
        if not f["ids"]:
            bad(f"FAMILIES {fam}: no ids parsed — check the {fam.upper()}_IDS "
                "constant name in fam_re/ids_const")
        for mid in f["ids"]:
            o = ovr.get(mid) or {}
            t = o.get("thinking") or {}
            if not t.get("efforts"):
                bad(f"models.yml {mid} (family {fam}): no thinking.efforts ladder rendered")
            elif list(t["efforts"]) != f["ladder"]:
                bad(f"models.yml {mid} (family {fam}): ladder {t['efforts']} != "
                    f"family ladder {f['ladder']}")
    # The inverse (r7): a probed ladder no family covers renders nowhere on the
    # omp surface, and every check above still passes. Every catalog model with
    # a ladder must be listed in some family's models.
    covered = {m for ms in FAMILY_MODELS.values() for m in ms}
    for model_dir in sorted(os.listdir("models")):
        levels, _status = model_levels(model_dir)
        if levels and model_dir not in covered:
            bad(f"{model_dir}: probed ladder {levels} but no FAMILIES family covers "
                f"it — the omp surface never renders its ladder; add it to a "
                f"family's models list (the family serving its omp ids)")
    for mid, o in ovr.items():
        t = (o.get("thinking") or {})
        ladder, dflt = list(t.get("efforts") or []), t.get("defaultLevel")
        if not ladder:
            continue
        # Bind each override to ITS OWN family's ladder: two families share
        # [low, medium, xhigh], so set membership cannot see one family's
        # ladder drifting onto the other's ids.
        fam = id_family.get(mid)
        if fam is None:
            bad(f"models.yml {mid}: rendered with a ladder but bound to no "
                "FAMILIES ids list — the render and FAMILIES drifted (#1951)")
            continue
        if tuple(ladder) != tuple(families[fam]["ladder"]):
            bad(f"models.yml {mid}: ladder {ladder} != its family {fam}'s "
                f"ladder {families[fam]['ladder']} — drifted from omp-setup.sh")
            continue
        # ABSENT defaultLevel is legal (omp falls back to the ladder TOP);
        # a PRESENT one must be a rung.
        if dflt is not None and dflt not in ladder:
            bad(f"models.yml {mid}: defaultLevel {dflt!r} is not a rung of its "
                f"ladder {ladder} (omp's fallback for a missing defaultLevel is "
                "the ladder TOP — an off-ladder default is a silent xhigh)")
    if len(failures) == n0:
        print(f"  ✓ omp-setup renders {len(ovr)} overrides: every FAMILIES ladder "
              "a profile's, every family id bound, every defaultLevel a rung")

sys.exit(1 if failures else 0)
PY

echo "test-effort-profiles: ok"
