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
#   6. OMP OVERRIDES — the FAMILIES are generated from the registry (one per served
#      name), so check 6 is a rendering test: it re-derives each family's ladder
#      (the model's levels minus any rung a route serving the name refuses, plus the
#      fold sources only when every route folds the same src->dst) and its default
#      (the lowest of the routes' compose defaults, mapped to a rung) from the
#      registry's per-variant effort facts, and requires the render to agree. Every
#      defaultLevel is a rung of its own ladder and not refused by any engine serving
#      the model — a pair's `refuses` are rungs the server 400s as request names, so
#      a refused default would 400 omp's default (#2055 review it6-4). A probed model
#      no family covers renders nowhere on the omp surface, so every one with a
#      ladder must be covered.

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

# The family -> model binding is no longer hand-kept: check 6 derives it from the
# registry (each family is a served name; its model is the name's variants' model),
# so a newly probed model with no family fails loudly instead of silently missing
# from the omp surface (#1951 review r7).

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

# ---- 3b. the EMIT CARRIES every compose default the scan above reads ----------
# Phase 3's defaults fold reads `compose_default` per variant: a variant whose compose
# sets the knob but whose emitted effort block drops it — the `_effort_facts` early
# return for a pair with no effort block on either side — silently rounds its family's
# default UP instead of down, breaking "rounds down, per served name" with an ABSENT
# fact rather than a wrong one, which no other check would notice. Zero instances today
# (every compose that sets the knob belongs to a profiled pair); this fires the first
# time one does not.
n0 = len(failures)
from litellm_sync import registry_variants
by_cp = {}
for v in registry_variants("."):
    if v.get("compose_path"):
        by_cp.setdefault(v["compose_path"], []).append(v)
carried = 0
for path, text, _levels, _mid in scan:
    m = default_re.search(text)
    if not m:
        continue
    for v in by_cp.get(path, ()):
        carried += 1
        got = (v.get("effort") or {}).get("compose_default")
        if got != m.group(1):
            bad(f"{v.get('slug')}: its compose sets REASONING_EFFORT {m.group(1)!r} but the "
                f"emit carries compose_default {got!r} — the family default for that served "
                "name would round up; _effort_facts must emit the fact it read")
if len(failures) == n0:
    print(f"  ✓ the emit carries the compose default for all {carried} variants whose "
          "compose sets REASONING_EFFORT")

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
# such an engine does NOT inherit the engine's claim: the emit carries a pair's
# `supports_none` only from the pair itself — inheriting True for an unprobed
# pair is the r6/r8 failure #2055 review it10-3 corrects — so an entry holding
# only fold facts leaves the route advertising NO none claim: safe, but
# unverified. Check 5d forces the pair to record its own `supports_none` with
# its citation, so the advertise, when it appears at all, rests on THIS model's
# template (#1951 review r7 + r8). `off_switch` does NOT satisfy this: it
# records how thinking turns OFF (the kwargs form), not whether `none` is
# honored, and the emit reads only the pair's `supports_none` — an
# off_switch-only entry also leaves the route advertising nothing (#2055
# review it7-2). `false` engines need no entry: the server ignoring `none` is
# engine-level behaviour, and the advertise direction is conservative.
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
            f"emit drops the engine-level supports_none (it never inherits True, "
            f"#2055 review it10-3), so the route advertises NO none claim today; "
            f"verify THIS model's template honors `none` and record the pair's "
            f"own measurement to make the advertise appear")
    elif "supports_none" not in pair:
        bad(f"{m} on {eng_id}: by_model entry holds fold facts but no pair "
            f"supports_none claim — the route advertises no none claim (the emit "
            f"drops the engine-level one rather than inherit it); declare "
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
    # The FAMILIES are generated from the registry (the same source the gateway's
    # model_info renders), not written down: one family per registry served name, each
    # carrying every alias its routes also answer to. Check 9 in test-omp-setup.sh pins
    # each rendered ladder/default against a hardcoded table, and its check 8 pins the
    # rendered NAME SET against an independent scan of the composes — those are the golden
    # tripwires, independent of the generator's own logic. This check adds what neither
    # can see: a rendered name that is a registry name must be one of its model's names
    # ALL (a family silently missing for one served name of a model is a silent
    # no-effort), at least one alias must render (the gateway names routes from the
    # engine's /v1/models, so a compose's --alias route with no override is #1690), and a
    # rendered default must be a rung of its own ladder (omp's fallback for a missing
    # default is the ladder TOP, so an off-ladder default is a silent xhigh). It
    # deliberately does NOT re-derive the ladder: a copy of the generator's logic here
    # would pass a bug present in both copies.
    from litellm_sync import registry_variants
    variants = registry_variants(os.getcwd())
    served: "dict[str, list]" = {}
    for v in variants:
        sid = v.get("served_name") or v.get("model")
        if sid:
            served.setdefault(sid, []).append(v)

    families = {}
    for mid, o in ovr.items():
        t = o.get("thinking") or {}
        rows = served.get(mid, [])
        families[mid] = {"ids": [mid], "ladder": list(t.get("efforts") or []),
                         "dflt": t.get("defaultLevel"),
                         "model": rows[0].get("model") if rows else None}

    rendered_models = {f["model"] for f in families.values() if f["model"]}
    for fam, f in sorted(families.items()):
        if f["dflt"] is not None and f["dflt"] not in f["ladder"]:
            bad(f"family {fam}: defaultLevel {f['dflt']!r} is not a rung of its "
                f"ladder {f['ladder']} (omp's fallback for a missing default is the "
                f"ladder TOP — an off-ladder default is a silent xhigh)")
    for name, rows in sorted(served.items()):
        if rows[0].get("model") in rendered_models and name not in ovr:
            bad(f"{name}: its model has overrides rendered but this served name has none "
                f"— the gateway serves it as its own route, so omp sends it no effort")
    if not [n for n in ovr if n not in served]:
        bad("no alias override rendered at all: the gateway names routes from the "
            "engine's /v1/models, so every compose --alias / --served-model-name route "
            "would get no effort (#1690)")
    # The inverse (r7): a probed ladder no family covers renders nowhere on the omp
    # surface, and every check above still passes. The covered set is the models the
    # generated FAMILIES bind to; every catalog model with a probed ladder must be in
    # it (a model you want on the omp surface but no live route serves is a gap).
    covered = rendered_models
    for model_dir in sorted(os.listdir("models")):
        levels, _status = model_levels(model_dir)
        if levels and model_dir not in covered:
            bad(f"{model_dir}: probed ladder {levels} but no generated family covers "
                f"it — the omp surface never renders its ladder; add it to a "
                f"MODEL_FAMILIES entry in omp-setup.sh")
    if len(failures) == n0:
        print(f"  ✓ omp-setup renders {len(ovr)} overrides: every served name of a "
              f"covered model, aliases included, every default on its own ladder, every "
              f"probed model covered")


sys.exit(1 if failures else 0)
PY

echo "test-effort-profiles: ok"
