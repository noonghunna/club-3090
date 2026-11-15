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

# The family -> model mapping of omp-setup.sh's hand-written FAMILIES list —
# the drift surface check 6 pins; it disappears when phase 2 generates the
# list from the profiles.
FAMILY_MODEL = {"qwen38": "qwen3.8-27b", "flash-next": "qwen3.8-flash-next",
                "glm53": "glm-5.3-flash"}

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
                # reasoning_effort") is not an effort knob; only code counts.
                text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
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
default_re = re.compile(r'REASONING_EFFORT["\']?:-["\']?([a-z]+)')
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
            # A commented-out default is not a default.
            text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
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
                if "REASONING_EFFORT:-" in line and not default_re.search(line):
                    bad(f"{path}: unparseable REASONING_EFFORT default line: "
                        f"{line.strip()[:120]}")
if len(failures) == n0:
    print(f"  ✓ {checked} compose REASONING_EFFORT defaults all inside their "
          "model's ladder (max ratcheted to GLM-5.3-Flash)")

# ---- 4. every declared rung is litellm vocabulary ---------------------------
n0 = len(failures)
runks = set()
for model_id, m in sorted(profiles.models.items()):
    runks |= set((m.effort or {}).get("levels") or [])
for eid, en in sorted(profiles.engines.items()):
    eff = en.effort or {}
    if eff.get("default_when_unset"):
        runks.add(eff["default_when_unset"])
    runks |= set(eff.get("maps") or {}) | set((eff.get("maps") or {}).values())
    # unknown_name is engine-internal (the request field an engine maps FROM),
    # never a wire value — not part of litellm's vocabulary.
off_vocab = runks - LITELLM_EFFORTS
if off_vocab:
    bad(f"effort names outside litellm's REASONING_EFFORT Literal (they would be "
        f"silently dropped from /model_group/info): {sorted(off_vocab)}")
pin_src = io.open("services/litellm/docker-compose.yml", encoding="utf-8").read()
pin_m = re.search(r"litellm:(v[\d.]+)", pin_src)
if not pin_m:
    bad("could not parse the litellm image pin from services/litellm/"
        "docker-compose.yml — the tripwire needs the tag")
elif pin_m.group(1) != "v1.100.3":
    bad(f"gateway pin moved to {pin_m.group(1)} — re-read the REASONING_EFFORT "
        "Literal from the new image (litellm/types/llms/openai.py) and update "
        "LITELLM_EFFORTS before trusting this set")
if len(failures) == n0:
    print(f"  ✓ all {len(runks)} declared effort names are litellm Literal members "
          f"(pin {pin_m.group(1) if pin_m else '?'})")

# ---- 5. ENGINE FOLDS land on rungs the served model accepts -----------------
budget_models = {mid for mid, m in profiles.models.items()
                 if (m.effort or {}).get("budget_map")}
pairs = {(e.get("model"), e.get("engine")) for e in entries.values()
         if e.get("model") and e.get("engine")}
n0 = len(failures)
fold_checked = 0
for model_id, engine_id in sorted(pairs):
    eff = (profiles.engines.get(engine_id) or type("x", (), {"effort": None})()).effort
    if not eff:
        continue
    levels, status = model_levels(model_id)
    if not levels:
        continue                      # unknown model: the render emits nothing anyway
    d = eff.get("default_when_unset")
    if d and d not in levels:
        bad(f"{model_id} on {engine_id}: engine default_when_unset {d!r} is not a "
            f"rung of the model's ladder {levels}")
    maps = eff.get("maps") or {}
    for src, dst in maps.items():
        if dst not in levels:
            bad(f"{model_id} on {engine_id}: fold {src}->{dst} lands outside the "
                f"model's ladder {levels}")
    # Identity folds (src->src) are legal — the engine accepts the name as-is.
    if maps and model_id in budget_models and not all(
            dst == src or (src, dst) in eb_aliases.items() for src, dst in maps.items()):
        bad(f"{engine_id}: maps {maps} fold outside effort_budget.py ALIASES "
            f"{eb_aliases} while serving budget_map model {model_id}")
    fold_checked += 1
# off_switch round-trip: the YAML must carry the STRING "none" — a bare
# `none` parses to null (the loader now rejects it), and this pins the LOADED
# value so the silent downgrade can't come back (#1951 review r1-1).
for eid in ("qengine", "sglang-stable", "vllm-stable"):
    off = (profiles.engines[eid].effort or {}).get("off_switch")
    if off != "none":
        bad(f"{eid}: off_switch must load as the string 'none', got {off!r} — "
            'quote it in the YAML (off_switch: "none")')
if len(failures) == n0:
    print(f"  ✓ {fold_checked} (model, engine) pairs: folds and defaults land on "
          "accepted rungs; off_switch round-trips as 'none'")

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
    ids_const = {m2.group(1): re.findall(r'"([^"]+)"', m2.group(2))
                 for m2 in re.finditer(r"^(\w+_IDS) = \[(.*?)\]", omp_src, re.M | re.S)}
    fam_re = re.compile(r'\{"name": "(\w+)", "ids": (\w+),\s*'
                        r'"ladder": \{"efforts": \[([^\]]*)\], "defaultLevel": "(\w+)"\}')
    families = {m2.group(1): {"ids": ids_const.get(m2.group(2), []),
                              "ladder": [x.strip().strip("\"'") for x in m2.group(3).split(",") if x.strip()],
                              "dflt": m2.group(4)}
                for m2 in fam_re.finditer(omp_src)}
    if not families:
        bad("could not parse FAMILIES from omp-setup.sh — the literal changed "
            "shape; fix check 6's fam_re")
    ladders = {tuple(f["ladder"]) for f in families.values()}
    for fam, f in sorted(families.items()):
        mdl = FAMILY_MODEL.get(fam)
        if not mdl:
            bad(f"FAMILIES entry {fam!r} is not in FAMILY_MODEL — extend the map "
                "(the family->model binding stays hand-kept until phase 2)")
            continue
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
    for mid, o in ovr.items():
        t = (o.get("thinking") or {})
        ladder, dflt = list(t.get("efforts") or []), t.get("defaultLevel")
        if not ladder:
            continue
        if tuple(ladder) not in ladders:
            bad(f"models.yml {mid}: ladder {ladder} matches no FAMILIES entry — "
                "drifted from omp-setup.sh (#1951)")
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
