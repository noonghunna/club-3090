#!/usr/bin/env python3
"""setup_next_steps — the slug setup.sh's "Next — launch it" block names.

  setup_next_steps.py <model> <repo-root> <registry-emit.json> [<slug>]

Prints SAMPLE_* and NEXT_STEPS_NOTE assignments (shell-quoted) for setup.sh to eval, or
nothing when the model has no slug at all. <registry-emit.json> is the output of
`scripts/lib/registry-emit.sh --json`.

Slug pick (deterministic): the slug setup.sh was given, when it was given one (`setup.sh
<slug>` downloads that slug's weights, so the hint names it, #1741); else the model's curated
default slug when it is FUNCTIONAL (status production/caveats — mirror of
compose_registry.FUNCTIONAL_STATUSES), else the highest-ranked functional slug, else the
highest-ranked slug overall. Non-functional pick ⇒ the launch hint carries --force, exactly
like switch.sh's gate.
"""
import json, os, re, shlex, sys

model, root, reg_path = sys.argv[1], sys.argv[2], sys.argv[3]
asked = sys.argv[4] if len(sys.argv) > 4 else ""
with open(reg_path, encoding="utf-8") as _fh:
    data = json.load(_fh)

RANK = {
    "production": 5,
    "caveats": 4,
    "preview": 3,
    "experimental": 2,
    "incubating": 1,
    "upstream-gated": 1,
    "deprecated": 0,
}
FUNCTIONAL = {"production", "caveats"}

variants = [v for v in data.get("variants", []) if v.get("model") == model]
defaults = [d["slug"] for d in data.get("defaults", []) if d.get("model") == model]


def q(v):
    return shlex.quote(str(v if v is not None else ""))


pick = next((v for v in variants if asked and v.get("slug") == asked), None)
if pick is None:
    for slug in defaults:
        c = [v for v in variants if v.get("slug") == slug]
        if c and c[0].get("status") in FUNCTIONAL:
            pick = c[0]
            break
if pick is None:
    pool = [v for v in variants if v.get("status") in FUNCTIONAL] or variants
    if pool:
        # sorted() is stable: ties keep registry order.
        pick = sorted(pool, key=lambda v: RANK.get(v.get("status", ""), 0), reverse=True)[0]
if pick is None:
    raise SystemExit(0)

slug = pick.get("slug", "")
status = pick.get("status", "")
print(f"SAMPLE_SLUG={q(slug)}")
print(f"SAMPLE_STATUS={q(status)}")
print(f"SAMPLE_CONTAINER={q(pick.get('container'))}")
print(f"SAMPLE_PORT={q(pick.get('port'))}")
print(f"SAMPLE_COMPOSE_PATH={q(pick.get('compose_path'))}")

# served_name is a first-class registry fact now (registry-emit --json); the
# direct compose grep below stays only as the fallback for a null field.
served = pick.get("served_name") or ""
if not served:
    try:
        with open(os.path.join(root, pick.get("compose_path") or ""), encoding="utf-8") as fh:
            m = re.search(r"--served-model-name\s+(?:-\s+)?(\S+)", fh.read())
        if m:
            served = m.group(1)
    except OSError:
        pass
print(f"SAMPLE_MODEL_NAME={q(served)}")

force = status not in FUNCTIONAL
print(f"SAMPLE_LAUNCH_HINT={q('bash scripts/switch.sh ' + ('--force ' if force else '') + slug)}")
if force:
    note = f"status '{status}' — launch needs --force (non-functional by default)."
    if status == "incubating":
        note += " Incubating slugs are hidden from 'switch.sh --list'; reveal with --list --all."
else:
    sibs = sorted(
        {
            v["slug"]
            for v in variants
            if v.get("status") in FUNCTIONAL and v.get("slug") != slug
        }
    )
    if sibs:
        lines = ["Functional variants:"]
        lines += [f"  bash scripts/switch.sh {s}" for s in sibs[:6]]
        if len(sibs) > 6:
            lines.append(f"  ... and {len(sibs) - 6} more: bash scripts/switch.sh --list")
        note = "\n".join(lines)
    else:
        note = ""
print(f"NEXT_STEPS_NOTE={q(note)}")
