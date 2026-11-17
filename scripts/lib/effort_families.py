"""Generate the omp FAMILIES (models.yml modelOverrides + club-live.json) from the
registry's per-variant effort facts — the single source #1951 established.

The hand-written FAMILIES list goes away. Instead, one family is generated per
SERVED NAME the gateway actually serves (the registry's served_name, else the model
id — the same name litellm_sync puts on the route), and each family's ladder and
default are derived, not written down:

  * its ladder is the set of rungs EVERY route serving that name accepts — the model's
    levels minus any rung a serving engine refuses, plus the fold sources only when
    every one of them folds the same src->dst (an engine with no fold table means the
    name advertises no fold sources). A rung is offered only if no route 400s it, so
    the family and the routes it names cannot drift;
  * its default is the LOWEST of the compose defaults across the routes serving the
    name, an empty REASONING_EFFORT falling back to that engine's pair default — the
    #1951 "club/<id> rounds down, per id" decision. A default that is a fold target
    (e.g. strata's xhigh) is expressed by the fold source that reaches it.

The only thing still written down is the client-side wire knowledge that has no
home in the profiles: which models the family covers, and each model's reply cap
and compat pins (the chat_template_kwargs form, the summary-bridge hazard, the
thinking budget). That is the `model_families` map this module takes.
"""

import json


def _served_id(v):
    """The name the gateway serves this variant under: its registry served_name,
    else the model id (a slug that declares no served_name is served by its model)."""
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
    # The ladder anchors on the model's levels; every other effort fact names a rung,
    # so a set of variants that cannot agree on the rungs advertises nothing.
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
    for s, d in (common or set()):
        if s in rung and d not in rung:
            rung[d] = s
    defaults = []
    for b in blocks:
        eff = b.get("compose_default") or b.get("default_when_unset")
        if eff in rung:
            defaults.append(rung[eff])
    if not defaults:
        return {"efforts": efforts, "defaultLevel": None}
    return {"efforts": efforts, "defaultLevel": min(defaults, key=efforts.index)}


def compute_families(variants, model_families):
    """model_families: [{models: [...], maxTokens: int, compat: {...}}, ...] — the
    client-side wire knowledge (which models, their reply cap and compat pins).

    Returns the FAMILIES list — one entry per served name whose model a family covers,
    each {name, ids: [name], ladder: {efforts, defaultLevel}, maxTokens, compat}.
    """
    model_to_fam = {}
    for fam in model_families:
        for m in fam["models"]:
            model_to_fam[m] = fam
    by_id = {}
    for v in variants:
        sid = _served_id(v)
        if sid:
            by_id.setdefault(sid, []).append(v)
    out = []
    for sid, rows in by_id.items():
        fam = model_to_fam.get(rows[0].get("model"))
        if fam is None:
            continue
        facts = _id_facts(rows)
        if facts is None:
            continue
        out.append({
            "name": sid,
            "ids": [sid],
            "ladder": {"efforts": facts["efforts"], "defaultLevel": facts["defaultLevel"]},
            "maxTokens": fam["maxTokens"],
            "compat": fam["compat"],
        })
    return out


def families_json(families):
    """The club-live.json `families` payload (the same list, JSON-serialisable)."""
    return json.loads(json.dumps(families))
