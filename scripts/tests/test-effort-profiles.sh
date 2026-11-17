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
#      is a rung of its own ladder AND nameable on every engine serving the
#      family's models — a pair's `refuses` are rungs the server 400s as
#      request names, so a refused defaultLevel would 400 omp's default
#      (#2055 review it6-4). ABSENT is NOT silently legal: fam_re requires
#      the field, so an absent one fails the shape parse; omp's fallback
#      there is the ladder TOP, exactly the rung the refuses check clears.
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
            elif fn.endswith(".jinja"):
                # Same rule for Jinja comments: {# ... #} is prose, not code
                # (#2055 review it5-6).
                text = re.sub(r"\{#.*?#\}", " ", text, flags=re.S)
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
# for it — those are other knobs, not defaults for this one. All four effort
# regexes are CASE-INSENSITIVE (#2055 review it9-6): the chat_template_kwargs
# key form is lowercase (`"reasoning_effort": "high"`), the same shape check 1
# and catalog_rules._model_dir_knob already treat as a knob, and the emit's
# reader must see every default those scans see.
default_re = re.compile(
    r'(?i)(?<![A-Za-z_])REASONING_EFFORT(?:["\']?:-["\']?|["\']?[:=][ \t]*["\']?\$\{[A-Za-z_]+:-|["\']?:[ \t]*["\']?|=)["\']?([a-z]+)')
# The knob handed to a launcher with no value of its own (the strata-maya
# composes' bare `REASONING_EFFORT:`, the knob-list lines) is NOT a default —
# the launcher's default is what the pair fact records. A value that is a
# VARIABLE reference is a hop the launch-time override owns, not a catalogue
# default — UNLESS it carries a fallback, which default_re reads as the
# compose's own default. A literal the emit cannot read is what stays flagged.
bare_knob_re = re.compile(r'(?i)^[ \t\-]*["\']?REASONING_EFFORT["\']?:?[ \t]*$')
knob_name_re = re.compile(r'(?i)(?<![A-Za-z_])REASONING_EFFORT(?![A-Za-z_])')
var_ref_re = re.compile(
    r'(?i)(?<![A-Za-z_])REASONING_EFFORT(?![A-Za-z_])\\?["\']?[:=][ \t]*\\?["\']?\$')
# The thinking-budget knob gets the same drift guard as the effort default
# (#2055 review it6-7c): a compose writing CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET
# with a differently-named inner variable is silently unread by the emit's
# _compose_thinking_budget — conservative (the honest 512 retry runs) but
# invisible. budget_knob_re is a byte-copy of the emit's _THINKING_BUDGET_RX;
# the 16b-style parity assertion in test-litellm-sync.sh pins the two copies.
budget_knob_re = re.compile(
    r'CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET:\s*(?:"?\$\{THINKING_TOKEN_BUDGET:-(\d+)\}"?|(\d+)\s*)')
# `max` defaults are ratcheted: litellm's default_reasoning_effort Literal
# (v1.100.3) EXCLUDES max, so phase 2's gateway render must OMIT a max
# default, never downgrade it. Today only GLM-5.3-Flash legitimately defaults
# to max (it IS a rung of its ladder); adding a model here is a conscious
# re-verification of the render rule, not a rubber stamp.
MAX_DEFAULT_MODELS = {"glm-5.3-flash"}
n0 = len(failures)
checked = 0
# The composes the emit can read: the walk covers everything under a compose/
# directory; the registry's own compose_paths cover the rest — a relocated
# compose or an entry outside a compose/ dir escapes the walk but still
# reaches _compose_effort_default (#2055 review it5-4). mdir is CWD-relative
# ("models"), the same shape as a registry compose_path, so seen_scan dedups
# the overlap by plain string equality.
scan = []                     # (path, text, levels, model_id)
seen_scan = set()
for model_dir in sorted(os.listdir(mdir)):
    levels, status = model_levels(model_dir)
    # No levels still scans: the thinking-budget knob sweep below runs on
    # every compose regardless of the model's ladder (#2055 review it7-4) —
    # only the REASONING_EFFORT default check needs the ladder, and it is
    # gated on `levels` in the loop below.
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
            scan.append((path, text, levels, model_dir))
            seen_scan.add(path)
for tag, e in sorted(entries.items()):
    cp, mid = e.get("compose_path"), e.get("model")
    if not cp or not mid or "_archive" in cp or cp in seen_scan:
        continue
    levels, _status = model_levels(mid)
    # No levels still scans — same reason as the walk above (it7-4): the knob
    # sweep runs regardless; only the default check needs the ladder.
    try:
        text = io.open(cp, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    text = "\n".join(l.split("#", 1)[0] for l in text.splitlines()
                     if not l.lstrip().startswith("#"))
    scan.append((cp, text, levels, mid))
    seen_scan.add(cp)
for path, text, levels, mid in scan:
    # The default check needs the ladder; the knob sweeps below run on every
    # compose regardless (#2055 review it7-4 — a knob line in a no-ladder
    # model's compose was invisible drift before).
    if levels:
        for dflt in sorted(set(default_re.findall(text))):
            checked += 1
            if dflt not in levels:
                bad(f"{path}: REASONING_EFFORT default {dflt!r} is not one "
                    f"of {mid}'s levels {levels}")
            if dflt == "max" and mid not in MAX_DEFAULT_MODELS:
                bad(f"{path}: REASONING_EFFORT:-max default for {mid} — "
                    "litellm's default_reasoning_effort Literal excludes max, "
                    "so the render must OMIT it, never downgrade; add the model "
                    "to MAX_DEFAULT_MODELS only after re-verifying that rule")
    # The unparseable-line tripwire runs on EVERY compose regardless of the
    # ladder (#2055 review it9-2): a knob form the emit cannot read makes it
    # return compose_default None and the pair default fill the advertise,
    # ladder or not — the comment above promised exactly this coverage.
    for line in text.splitlines():
        if not knob_name_re.search(line):
            continue
        if (default_re.search(line) or bare_knob_re.match(line.strip())
                or var_ref_re.search(line)):
            continue
        bad(f"{path}: unparseable REASONING_EFFORT line — a form the "
            f"emit's _compose_effort_default does not read, so the pair "
            f"default would fill it: {line.strip()[:120]}")
    # The thinking-budget knob's own drift guard (#2055 review it6-7c): a
    # compose line naming the knob that budget_knob_re cannot read means the
    # emit silently returns None for it — the route loses its declared
    # budget and the probe falls back to the honest 512 retry without
    # anyone knowing why.
    for line in text.splitlines():
        if "CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET" not in line:
            continue
        if budget_knob_re.search(line):
            continue
        bad(f"{path}: unreadable CLUB3090_DEFAULT_THINKING_TOKEN_BUDGET line — "
            f"a form the emit's _compose_thinking_budget does not read, so "
            f"the route's declared budget is silently lost: {line.strip()[:120]}")
if len(failures) == n0:
    print(f"  ✓ {checked} compose REASONING_EFFORT defaults all inside their "
          "model's ladder, registry compose_paths included (max ratcheted to "
          "GLM-5.3-Flash)")

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
    # The pin field follows the lineage's relation: a stock engine pins the
    # upstream build (base_version), a fork pins its own commit (source_ref) —
    # a fork that later records a non-null base_version (the upstream version
    # it forked from, the natural thing to note on a rebase) must not shadow
    # its commit, or this check fails permanently and tempts a loosening
    # (#2055 review it7-6).
    if (lin.get("relation") or "fork") == "stock":
        got = lin.get("base_version") or lin.get("source_ref")
    else:
        got = lin.get("source_ref") or lin.get("base_version")
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
fold_facts = 0
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
        # The floor counts FACTS, not entries: ten evidence-only shells (or
        # none-claims whose fold facts were stripped) must not hold it
        # (#2055 review it5-2).
        fold_facts += ((1 if facts.get("default_when_unset") else 0)
                       + len(facts.get("maps") or {})
                       + len(facts.get("refuses") or []))
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
# pair facts yet. The floor counts FACTS, not entries — an evidence-only shell
# (now refused by the loader) or a stripped none-claim must not hold it (#2055
# review it5-2). 35 fold facts across 10 pair entries today; bump on purpose
# when the catalogue legitimately changes (#1951 review r6).
PAIR_FACT_FLOOR = 35
if fold_facts < PAIR_FACT_FLOOR:
    bad(f"only {fold_facts} pair fold facts loaded across {fold_checked} entries, "
        f"floor is {PAIR_FACT_FLOOR} — bump the floor on purpose if the catalogue "
        f"legitimately changed")
if len(failures) == n0:
    print(f"  ✓ {fold_facts} pair fold facts across {fold_checked} entries land "
          f"on accepted rungs (floor {PAIR_FACT_FLOOR} pinned); refusals stay on "
          "the ladder and off fold sources; off_switch round-trips as 'none'")

# ---- 5c. ONE PORT, ONE MODEL ------------------------------------------------
# A port hosting two models advertises one model's facts for both: port_facts
# agree-or-drops across the rows, but a no-thinking model beside a knob-less
# thinking model would advertise supports_reasoning false for a thinker, and
# two ladders on one port advertise nothing at all. The render already refuses
# to name such a port (booting-<port>); the catalogue must not create one
# (#1951 review r6). The registry entry's field is default_port — reading
# `port` made this check vacuous (it saw no ports at all while :8030/:8031
# hosted two models each behind deprecated+experimental pairings; measured
# 2026-11-16, #2055 review it5-5 follow-up). Deprecated slugs are excluded:
# they are historical, need --force to boot, and the render's agree-or-drop
# rule (port_facts/_port_effort) already advertises nothing when a co-hosted
# slug's ladder differs — the booting-<port> placeholder rule covers only the
# booting port (#2055 review it9-3).
n0 = len(failures)
port_models: "dict[int, set]" = {}
for e in entries.values():
    port = e.get("port") or e.get("default_port")
    if port and e.get("model") and e.get("status") != "deprecated":
        port_models.setdefault(int(port), set()).add(e["model"])
for port, models in sorted(port_models.items()):
    if len(models) > 1:
        bad(f"port {port} hosts {len(models)} live models ({sorted(models)}) — "
            f"split the ports: the render advertises one model's effort facts "
            f"for both")
if len(failures) == n0:
    print("  ✓ every registry port serves exactly one live model")
# ---- 5d. supports_none: true is a per-pair claim -----------------------------
# An engine-level `supports_none: true` says the engine maps `none` to
# enable_thinking=false before the template runs; whether THIS model's template
# then closes the thinking block is the model's contract. A ladder model on
# such an engine advertises `supports_none: true` the moment it probes a
# ladder, so its pair entry must exist AND carry the pair's own
# `supports_none` with its citation — before the advertise is trusted: an
# entry holding only fold facts would leave the route advertising a wire none
# claim resting on nothing pair-specific (#1951 review r7 + r8). `off_switch`
# does NOT satisfy this: it records how thinking turns OFF (the kwargs form),
# not whether `none` is honored, and the emit reads only the pair's
# `supports_none` — an off_switch-only entry would leave the advertise on the
# engine-level claim (#2055 review it7-2). `false` engines need no entry: the
# server ignoring `none` is engine-level behaviour, and the advertise
# direction is conservative.
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
    elif "supports_none" not in pair:
        bad(f"{m} on {eng_id}: by_model entry holds fold facts but no pair "
            f"supports_none claim — the route advertises the engine's "
            f"supports_none: true resting on nothing pair-specific; declare "
            f"the pair's own supports_none with its citation (#1951 review r8, "
            f"#2055 review it7-2: off_switch is not a none claim and the emit "
            f"reads only the pair's supports_none)")
if len(failures) == n0:
    print("  ✓ every supports_none:true engine's ladder pairs carry a by_model "
          "entry with the pair's own supports_none claim")
# ---- 5e. a refuses-recording engine's other ladder pairs declare theirs -----
# `refuses` is pair-scoped by design (the intersection of the engine's 400
# behaviour and THIS model's ladder — the loader refuses an engine-level one,
# #2055 review it7-5). Once one pair on an engine records a non-empty
# `refuses`, the engine's server 400s request names outside its accepted set —
# a fact that does not depend on the template — so every OTHER ladder model on
# that engine must probe and record its own outcome too: absent the key the
# render names the full ladder (litellm_sync.py:681-683), advertising a rung
# the server 400s (#2055 review it8-4). The EMPTY list is the expressible
# "probed, nothing refused" record (#2091 GLM review f1): a ladder disjoint
# from the server's 400-set honestly refuses nothing, and while the loader
# rejected empty lists there was NO recordable shape — the sweep forced a
# false refusal (inventing a rung would feed the render's union a lie).
# Omission still means unprobed. Engines with no refuses recorded anywhere
# stay untriggered: the guard has no structured 400 fact to key on, and
# forcing a probe it cannot cite would be a false positive.
def _pair_records_refusal(pair):
    # The pair recorded its probe outcome iff the refuses key is present:
    # non-empty = rungs the server 400s, empty = probed, nothing refused.
    # Absent = unprobed (#2091 GLM review f1).
    return bool(pair) and pair.get("refuses") is not None

n0 = len(failures)
refusing_engines = {}
for eng_id, eng in profiles.engines.items():
    eff = (eng.effort or {}) if eng else {}
    for mid, pair in (eff.get("by_model") or {}).items():
        if pair.get("refuses"):
            refusing_engines.setdefault(eng_id, []).append(mid)
for e in entries.values():
    m, eng_id = e.get("model"), e.get("engine")
    if not m or not eng_id or eng_id not in refusing_engines:
        continue
    if m in refusing_engines[eng_id]:
        continue
    levels, _status = model_levels(m)
    if not levels:
        continue
    pair = ((profiles.engines.get(eng_id).effort or {}).get("by_model") or {}).get(m)
    if not _pair_records_refusal(pair):
        bad(f"{m} on {eng_id}: probed ladder {levels} but no pair refuses while "
            f"sibling {refusing_engines[eng_id]} records {[(profiles.engines[eng_id].effort or {}).get('by_model', {}).get(s, {}).get('refuses') for s in refusing_engines[eng_id]]} — "
            f"the render names the full ladder when the key is absent, so a "
            f"rung the server 400s would be advertised; probe this pair's "
            f"request names and record its refuses — a non-empty list of the "
            f"rungs it 400s, or an empty list recording probed-nothing-refused "
            f"(#2055 review it8-4, #2091 GLM review f1)")
if len(failures) == n0:
    print("  ✓ every refuses-recording engine's other ladder pairs declare "
          "their own refuses")

for _shape, _want in (({"evidence": "probed, nothing refused", "refuses": []}, False),
                      ({"evidence": "never probed"}, True)):
    if (not _pair_records_refusal(_shape)) != _want:
        bad(f"5e predicate: {_shape} must flag={_want}")
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
               "must be a list of rung names", "non-list pair refusal inside by_model")
expect_refusal({"evidence": "x",
                "by_model": {"m": {"default_when_unset": "xhigh"}}},
               "evidence is REQUIRED", "pair entry with no own evidence")
expect_refusal({"evidence": "x", "by_model": {"m": {"evidence": "citation only"}}},
               "record at least one fact", "evidence-only pair entry inside by_model")
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
try:
    # The empty list is the "probed, nothing refused" record (#2091 GLM
    # review f1): a ladder disjoint from the server's 400-set must be
    # expressible, not forced into a false refusal.
    _engine_effort({"off_switch": "none", "supports_none": True,
                    "by_model": {"m": {"refuses": [],
                                       "evidence": "probed the ladder, nothing 400'd"}},
                    "evidence": "x"}, Path("negative-test.yml"))
except ProfileError as exc:
    bad(f"an empty refuses (probed, nothing refused) was refused: {exc}")
if len(failures) == n0:
    print("  ✓ engine-level fold facts refused (14 shapes), pair-scoped block "
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
            # The omp ladder is the rungs EVERY engine serving the family's
            # models accepts as a request name — the model's levels minus the
            # union of the engines' pair `refuses` (#2055 review it7-1: the omp
            # override is rendered at setup time and cannot know which engine
            # will boot, so a rung ANY engine refuses is offered by NONE).
            # No pair records a refusal today — strata's flash-next pair once
            # refused xhigh on a mis-read 400, corrected 2026-11-17 (see
            # engines/strata.yml). An engine with no by_model entry records
            # no refuses — unknown stays unknown there.
            refused: set = set()
            for eid, eng in sorted(profiles.engines.items()):
                pair = ((eng.effort or {}).get("by_model") or {}).get(mdl) or {}
                refused |= set(pair.get("refuses") or [])
            expected = [r for r in model_levels(mdl)[0] if r not in refused]
            if f["ladder"] != expected:
                bad(f"FAMILIES {fam}: ladder {f['ladder']} != model {mdl}'s "
                    f"nameable rungs {expected} (levels {model_levels(mdl)[0]} "
                    f"minus refused {sorted(refused)}) — drifted from the "
                    "profiles (#1951)")
            # The defaultLevel is what omp requests when the agent names no
            # effort, and the request lands on whichever slug of the family's
            # models is booted — so it must be a name every engine serving
            # those models accepts (#2055 review it6-4). The ladder pin above
            # already drops refused rungs from the ladder itself; this sweep
            # catches the defaultLevel, which is not a ladder rung. An engine
            # with no by_model entry records no refuses — unknown stays
            # unknown there.
            for eid, eng in sorted(profiles.engines.items()):
                pair = ((eng.effort or {}).get("by_model") or {}).get(mdl) or {}
                if f["dflt"] in (pair.get("refuses") or []):
                    bad(f"FAMILIES {fam}: defaultLevel {f['dflt']!r} is refused on "
                        f"{eid}'s {mdl} pair — omp would default to a rung that "
                        "engine 400s as a request name")
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
    # A model listed under a family must also be REACHABLE from that family's
    # ids constant (#2055 review it8-2): the overrides render only the ids in
    # the *_IDS constants, so a FAMILY_MODELS entry whose omp names (the model
    # id itself, or a non-deprecated slug's --served-model-name for it) appear
    # in no constant is pinned, covered, and never offered. Quant-variant ids
    # (qwen3.8-27b-fp8) are omp-side aliases, not catalog models — the check
    # asks only that each listed model has SOME id in its family's constant.
    served_names: "dict[str, set]" = {}
    for path, text, _levels, mdl in scan:
        for line in text.splitlines():
            m2 = re.search(r"^\s*-?\s*--served-model-name[=:\s]+\"?([\w.-]+)", line)
            if m2:
                served_names.setdefault(mdl, set()).add(m2.group(1))
    for fam, f in sorted(families.items()):
        for mdl in FAMILY_MODELS.get(fam, []):
            names = {mdl} | served_names.get(mdl, set())
            if not (names & set(f["ids"])):
                bad(f"FAMILY_MODELS {fam}: model {mdl} has no omp id in "
                    f"{f['ids_const']} {f['ids']} (model id or a slug's "
                    f"--served-model-name) — the omp surface never offers it")
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
