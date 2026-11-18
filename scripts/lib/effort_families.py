"""Generate the omp FAMILIES (models.yml modelOverrides + club-live.json) from the
registry's per-variant effort facts — the single source #1951 established.

The hand-written FAMILIES list goes away. Instead, one family is generated per
ROUTE ID — every name a route actually answers to — and each family's ladder and
default are derived, not written down:

  * its ladder is the set of rungs EVERY route serving that id accepts — the model's
    levels minus any rung a serving engine refuses, plus the fold sources only when
    every one of them folds the same src->dst (an engine with no fold table means the
    id advertises no fold sources). A rung is offered only if no route 400s it, so the
    family and the route it names cannot drift;
  * its default is the LOWEST of the compose defaults across the routes serving the id,
    an empty REASONING_EFFORT falling back to that engine's pair default — the #1951
    "rounds down, per id" decision. A default that is a fold target (e.g. strata's
    xhigh) is expressed by the fold source that reaches it; an id whose routes reach no
    nameable default falls back to its model's own default, and if that is unreachable
    too it gets NO family — see compute_families.

Why per route ID and not per registry served name: the gateway names its routes from the
engine's own /v1/models (litellm_sync.probe), NOT from the registry, so a compose's
--served-model-name / --alias (one-line or YAML list), Strata's "<family>-<quant>" name
and Project Maya's model_name are each a route of their own. omp matches modelOverrides
by that route name, so an alias with no override reads as a non-reasoning model, is sent
no effort, and runs the engine's own (slowest) default — the #1690 regression. But two
names of one model are also served by DIFFERENT engines: Strata's own id accepts high/max
and defaults to xhigh, while the shared qwen3.8-flash-next id is served by engines that
refuse those rungs and default to low. Grouping by the registry name would hand every
alias the shared id's ladder and default, which is the per-id rule #1951 settled. So the
route id IS the family, `ids` is [that id], and only maxTokens and compat are keyed by
model.

The only thing still written down is the client-side wire knowledge that has no
home in the profiles: which models the family covers, and each model's reply cap
and compat pins (the chat_template_kwargs form, the summary-bridge hazard, the
thinking budget). That is the `model_families` map this module takes.
"""
import os
import re
import sys


def served_id(v):
    """The name the gateway serves this variant under: its registry served_name,
    else the model id (a slug that declares no served_name is served by its model).
    Public: omp-setup matches its covered-model check against this same rule."""
    return v.get("served_name") or v.get("model")


def _id_facts(variants):
    """(efforts, defaultLevel) for one served name, or None when it has no ladder.

    `variants` are the registry rows served under that name; each carries the merged
    (model, engine) effort facts — levels, refuses, maps, compose_default and
    default_when_unset — that Phase 1's profiles and the compose defaults produce.
    """
    blocks = [v["effort"] for v in variants if isinstance(v.get("effort"), dict)]
    if not blocks:
        return None
    # The ladder anchors on the model's levels. One served name is one model —
    # compute_families refuses a name claimed by two — so every block here carries that
    # model's levels and the first block with levels IS the model's ladder. A name whose
    # blocks carry no levels at all (an unprofiled model) advertises nothing.
    levels = None
    for b in blocks:
        if b.get("levels"):
            levels = list(b["levels"])
            break
    if not levels:
        return None
    # Refusal is a NEGATIVE fact: a rung ANY route serving the name refuses is not
    # nameable, so it is dropped from every route's ladder at once.
    refuses = set()
    for b in blocks:
        refuses |= set(b.get("refuses") or [])
    # A fold source is nameable only if EVERY route serving the name folds the same
    # src->dst; a route with no fold table means the name advertises no fold sources.
    common = None
    for b in blocks:
        m = b.get("maps")
        pairs = {(s, d) for s, d in m.items()} if isinstance(m, dict) and m else set()
        common = pairs if common is None else (common & pairs)
        if not common:
            break
    # The base rungs that survive, in the model's order, then the fold sources (kept
    # in the first engine's maps order so the rendered order is stable).
    efforts = [l for l in levels if l not in refuses]
    for b in blocks:
        m = b.get("maps")
        if isinstance(m, dict) and m:
            for s, d in m.items():
                if (s, d) in (common or set()) and s not in refuses and s not in efforts:
                    efforts.append(s)
            break
    # The default: the lowest (in ladder order) of the routes' effective defaults,
    # mapped to a nameable rung — a base rung is itself, a fold target is the source
    # that reaches it. A default no rung reaches (refused, no fold) is not expressible
    # and is skipped, not guessed.
    rung = {l: l for l in efforts}
    # Two fold sources can reach the same target (vLLM, SGLang, QEngine and Strata fold
    # both high and max onto xhigh), so which source expresses a fold-target default is
    # decided here, not by a set's hash-randomized iteration order: the earliest source
    # in ladder order wins — the "rounds down" rule — and the render is deterministic
    # across processes.
    order = {l: i for i, l in enumerate(efforts)}
    for s, d in sorted(common or set(),
                       key=lambda p: (order.get(p[0], len(order)), p[0], p[1])):
        if s in rung and d not in rung:
            rung[d] = s
    defaults = []
    for b in blocks:
        eff = b.get("compose_default") or b.get("default_when_unset")
        # A compose can set the knob to a fold SOURCE name (REASONING_EFFORT=high on an
        # engine that folds high->xhigh) that this family's ladder does not carry,
        # because another route serving the name has no such fold. That route still RUNS
        # the target, so fall back through its own fold table — otherwise the route drops
        # out of the default decision and the family rounds UP. Only when the rung lookup
        # misses: a nameable rung is sent as the compose wrote it, fold or no fold.
        if eff is not None and eff not in rung:
            m = b.get("maps")
            if isinstance(m, dict):
                eff = m.get(eff, eff)
        if eff in rung:
            defaults.append(rung[eff])
    if not defaults:
        return {"efforts": efforts, "defaultLevel": None}
    return {"efforts": efforts, "defaultLevel": min(defaults, key=efforts.index)}


# The names a compose's route answers to. The gateway names its routes from the engine's
# own /v1/models (litellm_sync.probe: "Names come from /v1/models, NOT from the
# registry's served_name"), so a compose that boots under an alias is a ROUTE the
# registry never names — and omp matches modelOverrides by that route name. An alias with
# no override is treated as a non-reasoning model: omp sends no effort and the engine runs
# its own default, which is the #1690 slowest-rung regression. These are the ways a
# compose names itself; test-omp-setup scans them independently and pins the render.
STRATA_FAMILY_NAMES = {"qwen": "qwen3.8-flash-next", "swift": "swift-1.5",
                       "unsloth": "qwen3.8-flash-next-unsloth"}  # Strata setup.py FAMILIES
_SERVED_FLAGS = r"--(?:served-model-name|served-name|alias|model-name)"


def _served_token(tok):
    """A served name out of a compose token, or None when it is not one.
    `${SERVED_MODEL_NAME:-qwen3.8-27b}` stands for its default; a value that is another
    flag, a path or an expression is not a name."""
    tok = tok.strip().strip("\"'")
    m = re.fullmatch(r"\$\{\w+:-([^}]+)\}", tok)
    if m:
        tok = m.group(1).strip()
    return tok if re.fullmatch(r"[A-Za-z0-9][\w.\-]*", tok) else None


def _read(root, *parts):
    try:
        with open(os.path.join(root, *parts), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def compose_served_names(root, compose_path):
    """Every name one compose's route answers to (see _SERVED_FLAGS).

    The compose is read as text: these are launch arguments, not settings the loader
    resolves, and the point is the alias a compose DECLARES even where the loader would
    default it away. A compose that cannot be read contributes nothing — family_ids
    warns once per family rather than per row.
    """
    text = _read(root, *compose_path.split("/")) if not os.path.isabs(compose_path) \
        else None
    if text is None and os.path.isabs(compose_path):
        try:
            with open(compose_path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            text = None
    if text is None:
        return set()
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    names = set()
    for i, line in enumerate(lines):
        m = re.search(r'SERVED_MODEL_NAME:\s*"?(\$\{SERVED_MODEL_NAME:-[^}]+\})', line)
        if m:
            tok = _served_token(m.group(1))
            if tok:
                names.add(tok)
        m = re.search(_SERVED_FLAGS + r"\b(?:=['\"]?)?(.*)$", line)
        if not m:
            continue
        rest = m.group(1).strip().rstrip("\\").strip().strip("\"'")
        if rest and not rest.startswith("-"):            # one-line: --alias a b / --alias=a
            for tok in rest.split():
                if tok.startswith("-"):                  # the next flag ends the values
                    break
                name = _served_token(tok)
                if name:
                    names.add(name)
            continue
        for nxt in lines[i + 1:]:                        # YAML list: - name
            v = nxt.strip()
            val = v[2:].strip().strip("\"'") if v.startswith("- ") else ""
            if not val or val.startswith("-"):           # the next flag ends the list
                break
            name = _served_token(val)
            if name:
                names.add(name)
    # Two engines name the model themselves, so their ids are derived the way they derive
    # them: Strata as "<family name>-<quant>" (its setup; the family is the argument after
    # strata-entrypoint.sh in the compose's entrypoint, the quant the one after it, else
    # the launcher's default set), Project Maya from its launcher's model_name.
    engine_dir = compose_path.split("/compose/")[0]
    engine = os.path.basename(engine_dir)
    if engine == "strata":
        m = re.search(r"entrypoint:\s*\[([^\]]*)\]", text)
        args = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        at = next((j for j, a in enumerate(args) if a.endswith("strata-entrypoint.sh")),
                  None)
        fam = args[at + 1] if at is not None and len(args) > at + 1 else "qwen"
        quant = args[at + 2].lower() if at is not None and len(args) > at + 2 else None
        if quant is None:
            m = re.search(r'^SET="\$\{2:-([^}"]+)\}"',
                          _read(root, *engine_dir.split("/"), "launcher",
                                "strata-entrypoint.sh") or "", re.M)
            quant = m.group(1).lower() if m else None
        if fam in STRATA_FAMILY_NAMES and quant:
            names.add(f"{STRATA_FAMILY_NAMES[fam]}-{quant}")
    elif engine == "strata-maya":
        m = re.search(r'"model_name":\s*"([^"]+)"',
                      _read(root, *engine_dir.split("/"), "launcher", "maya_prepare.py")
                      or "")
        if m:
            names.add(m.group(1))
    return names


def route_names(root, row):
    """(every name this registry row's route answers to, its compose unreadable?).

    The name the registry declares plus every alias the serving compose boots under.
    Each of those is a route the gateway names on its own (see the module docstring), so
    each needs its own override — and its own ladder, because the engines behind two
    names of one model are not the same set.
    """
    sid = served_id(row)
    names = {sid} if sid else set()
    cp = row.get("compose_path")
    if not cp or root is None:
        return names, False
    if not os.path.exists(cp if os.path.isabs(cp) else os.path.join(root, cp)):
        return names, True
    return names | compose_served_names(root, cp), False


def compute_families(variants, model_families, root=None):
    """model_families: [{models: [...], maxTokens: int, compat: {...}}, ...] — the
    client-side wire knowledge (which models, their reply cap and compat pins).

    Returns the FAMILIES list — one entry per ROUTE ID (see the module docstring), each
    {name, ids: [name], ladder: {efforts, defaultLevel}, maxTokens, compat}, the last two
    from the model_families entry covering the model behind that route. `root` is the
    checkout the composes are read from; without it a route is only the name the registry
    declares, which is what the unit tests exercise.

    An id whose routes reach NO nameable default falls back to its model's own (the
    default its canonical id renders) when that is a rung this route accepts. If neither
    is reachable the id gets no family: rendering the None would put the literal string
    "None" on the wire as an effort name, and leaving the default out makes omp fall back
    to the ladder TOP — a silent xhigh, the slowest rung. Neither is honest, so the id is
    skipped with a warning naming the profile to fix.
    """
    model_to_fam = {}
    fam_index = {}
    for i, fam in enumerate(model_families):
        fam_index[id(fam)] = i
        for m in fam["models"]:
            model_to_fam[m] = fam
    by_route = {}
    unreadable = []
    for v in variants:
        names, missing = route_names(root, v)
        if missing:
            unreadable.append(v.get("compose_path"))
        for n in names:
            by_route.setdefault(n, []).append(v)
    if unreadable:
        print(f"effort_families: {len(unreadable)} route compose(s) unreadable from root "
              f"{root!r}, so the aliases they declare are missing from the family list and "
              "those routes get no effort. Fix: run this from the checkout the registry "
              f"points at; first missing: {unreadable[0]}.", file=sys.stderr)
    facts = {rid: _id_facts(rows) for rid, rows in by_route.items()}
    model_rows = {}
    for v in variants:
        if v.get("model") is not None:
            model_rows.setdefault(v["model"], []).append(v)
    model_facts = {m: _id_facts(rows) for m, rows in model_rows.items()}
    out = []
    for rid in sorted(by_route):
        rows = by_route[rid]
        # One route id is ONE model — but only for a route this client actually serves:
        # the catalog has uncovered models that share a route name, and refusing those
        # would make an unrelated catalog entry break the render. Both sets are computed
        # from ALL the rows, so neither decision depends on registry row order (a name
        # claimed by two models whose first row is an uncovered model must still be
        # refused, not dropped silently). The ladder, the compat pins and the coverage set
        # all key off the rows, so a collision is a catalog mistake, not a multi-model
        # route.
        models = {r.get("model") for r in rows}
        entries = {fam_index[id(model_to_fam[m])] for m in models if m in model_to_fam}
        if not entries:
            continue                       # a route no client family covers
        if len(models) > 1:
            raise ValueError(
                f"route {rid!r} is served by {len(models)} models "
                f"({', '.join(sorted(str(m) for m in models))}): one route name must map "
                "to one model, or its ladder, compat pins and coverage depend on registry "
                "row order. Fix: give one of them its own --served-model-name / alias.")
        if len(entries) > 1:
            raise ValueError(
                f"route {rid!r} resolves to {len(entries)} MODEL_FAMILIES entries: one "
                "route cannot carry two overrides. Fix: one entry per model in "
                "MODEL_FAMILIES.")
        fam = model_families[next(iter(entries))]
        f = facts[rid]
        if f is None:
            continue
        dflt = f["defaultLevel"]
        if dflt is None:
            # No route serving this id records a default that reaches a nameable rung
            # (ExLlamaV3 recorded none at all before #2101). The MODEL's own default is
            # the honest fallback — but only if this route's ladder carries it.
            fb = (model_facts.get(rows[0].get("model")) or {}).get("defaultLevel")
            if fb in f["efforts"]:
                dflt = fb
            else:
                print(f"effort_families: {rid}: no route serving this name has a default "
                      f"that reaches a nameable rung of {f['efforts']} — no override "
                      "generated, so omp will send no effort and the server default runs. "
                      "Fix: set REASONING_EFFORT in the serving compose, or the pair's "
                      "default_when_unset in the engine profile.", file=sys.stderr)
                continue
        out.append({
            "name": rid,
            "ids": [rid],
            "ladder": {"efforts": f["efforts"], "defaultLevel": dflt},
            "maxTokens": fam["maxTokens"],
            "compat": fam["compat"],
        })
    return out


def validate_model_families(model_families):
    """Refuse a MODEL_FAMILIES entry the renderer would index into a crash or a silent
    drop. The renderer reads models/maxTokens/compat and the compat pins; club-live.ts
    re-reads maxTokens and drops a family whose value is not a number, so a wrong type
    here registers a model with NO thinking config. Raise ValueError naming the entry
    and the field — the caller turns it into an actionable SystemExit.
    """
    seen = {}
    for fam in model_families:
        models = fam.get("models")
        if not isinstance(models, list) or not models or not all(
                isinstance(m, str) and m for m in models):
            raise ValueError(f"the entry {fam!r} needs models: a non-empty list of model ids")
        for m in models:
            if m in seen:
                raise ValueError(
                    f"model {m!r} is listed in two entries ({seen[m]!r} and {models!r}): "
                    "the last one silently wins, so its compat pins depend on list order. "
                    "Fix: list each model in exactly one entry.")
            seen[m] = models
        mt = fam.get("maxTokens")
        if not isinstance(mt, int) or isinstance(mt, bool):
            raise ValueError(f"the entry for {models!r} needs maxTokens: an int (got {mt!r}); "
                             "club-live.ts drops a family whose maxTokens is not a number, "
                             "so the model would register with no thinking config")
        c = fam.get("compat")
        if not isinstance(c, dict):
            raise ValueError(f"the entry for {models!r} needs compat: a dict (got {c!r})")
        for k in ("supportsReasoningEffort", "qwenTemplateReasoningEffort"):
            if k not in c:
                raise ValueError(f"the entry for {models!r} is missing compat.{k} — the "
                                 "override renderer indexes it. Fix: fill it in, or drop the entry.")
        if "extraBody" in c:
            eb = c["extraBody"]
            if not isinstance(eb, dict):
                raise ValueError(f"the entry for {models!r} needs compat.extraBody: a dict "
                                 f"(got {eb!r})")
            tb = eb.get("thinking_token_budget")
            if not isinstance(tb, int) or isinstance(tb, bool):
                raise ValueError(f"the entry for {models!r} needs "
                                 "compat.extraBody.thinking_token_budget: an int "
                                 f"(got {tb!r}); the renderer formats it with %d")

