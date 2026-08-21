#!/usr/bin/env bash
# Generate the LOCAL block of services/litellm/config.yaml from the registry
# (#1078) — kills the hand-typed "where is the model" drift class.
#
#   bash scripts/lib/litellm-emit.sh [--check] [ROOT]
#
#     (no flag)   rewrite the generated region of config.yaml in place.
#     --check     fail (exit 1, unified diff) when the checked-in generated
#                 region != freshly generated output. Gate-shaped, like the
#                 other registry-parity checks.
#
# Data source: `scripts/lib/registry-emit.sh --json` (the emitted per-variant
# facts, incl. served_name) + scripts.lib.profiles.compose_registry for the
# gateway/serve_aliases flags and the curated-default walk. Never re-derived
# here.
#
# Design answers implemented (#1078):
#   Eligibility   explicit, opt-in: compose_registry._entry(gateway=True).
#                 Nothing is inferred — a model is on the LAN gateway only if
#                 a PR flagged its scene. Today: vllm/dual (qwen3.6-27b) and
#                 vllm/qwen38-27b-dual-max (qwen3.8-27b, the #1062 fix).
#   Canonical port  the model's curated default via curated_default_target()
#                 (the model_default_target/DEFAULTS walk), accepted only when
#                 it resolves to a gateway=True entry; topologies tried in the
#                 order dual → single → multi4 → multiN because the gateway's
#                 canonical scene is the multi-card primary, not a single-card
#                 debug baseline. If the walk resolves nothing (e.g. qwen3.8-27b
#                 deliberately has NO DEFAULTS rows), a model with EXACTLY ONE
#                 gateway entry uses it; anything ambiguous fails loudly rather
#                 than guessing a port.
#   model_name    the emitted served-name fact(s) of the canonical slug — ONE
#                 ROUTE PER SERVED-NAME, not per slug. A compose serving
#                 several --served-model-name values (dual-max serves both
#                 qwen3.8-27b and qwen3.8-27b-fp8 on :8091) yields one route
#                 per name on the same api_base. An _entry served_name=
#                 override wins as the primary name.
#
# #1073 mechanism: _entry(serve_aliases=(...)) emits each extra name as an
# additional model_name on the SAME route (same upstream port). No core entry
# sets aliases yet — mechanism + synthetic-fixture test only
# (scripts/tests/test-litellm-generate.sh).
#
# Region contract: everything between the BEGIN/END GENERATED markers is
# machine-owned and rewritten wholesale. Everything else in config.yaml —
# hand-maintained local routes whose served-name is not yet a registry fact
# (llama.cpp default-id servers, back-compat alias routes) AND the cloud
# block — is untouched by this script.
#
# Env overrides (test seams):
#   LITELLM_CONFIG             path to the config (relative to ROOT or absolute).
#   LITELLM_EMIT_REGISTRY_JSON path to pre-emitted registry JSON (skips the
#                              registry-emit.sh subprocess; synthetic fixtures).
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

CHECK=0
ROOT_DIR=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    *) ROOT_DIR="$arg" ;;
  esac
done
ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_EMIT_CHECK="$CHECK"

python3 - "$ROOT_DIR" <<'PY'
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

from scripts.lib.profiles.compose_registry import (
    curated_default_target,
    get_registry,
)

CFG = Path(os.environ.get("LITELLM_CONFIG", "services/litellm/config.yaml"))
if not CFG.is_absolute():
    CFG = root / CFG

BEGIN_RX = re.compile(r"^\s*#\s*===\s*BEGIN GENERATED LOCAL BLOCK\b.*$")
END_RX = re.compile(r"^\s*#\s*===\s*END GENERATED LOCAL BLOCK\b.*$")
BEGIN_LINE = "  # === BEGIN GENERATED LOCAL BLOCK — scripts/lib/litellm-emit.sh (#1078); DO NOT hand-edit ==="
END_LINE = "  # === END GENERATED LOCAL BLOCK ==="

# --- data source: the emitted registry facts ---------------------------------

json_src = os.environ.get("LITELLM_EMIT_REGISTRY_JSON")
if json_src:
    facts = json.loads(Path(json_src).read_text(encoding="utf-8"))
else:
    proc = subprocess.run(
        ["bash", str(root / "scripts/lib/registry-emit.sh"), "--json", str(root)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"litellm-emit: registry-emit.sh --json failed:\n{proc.stderr}")
    facts = json.loads(proc.stdout)
variant_by_slug = {v["slug"]: v for v in facts.get("variants", [])}

# --- eligibility: explicit gateway=True entries only -------------------------

gw_by_model: dict = {}
for slug, entry in get_registry(root).items():
    if entry.get("gateway"):
        gw_by_model.setdefault(entry["model"], {})[slug] = entry
if not gw_by_model:
    sys.exit("litellm-emit: no gateway=True entries in the registry — nothing to emit")

_TOPOLOGY_ORDER = ("dual", "single", "multi4", "multiN")


def canonical_slug(model, entries):
    """Curated-default walk restricted to this model's gateway entries."""
    for topo in _TOPOLOGY_ORDER:
        slug = curated_default_target(model, topo)
        if slug and slug in entries:
            return slug
    if len(entries) == 1:
        return next(iter(entries))  # unambiguous fallback (e.g. qwen3.8-27b)
    sys.exit(
        f"litellm-emit: no canonical gateway scene for model {model!r} — "
        f"candidates {sorted(entries)}; curate one via the DEFAULTS walk or "
        f"flag exactly one slug gateway=True"
    )


def compose_served_names(compose_path):
    """ALL --served-model-name values in the compose, in order (the dual-max
    dual-name case). Plain-text parse, same shape as registry-emit's
    _compose_served_name but collecting the whole list; ${VAR:-default}
    unwrapped. Empty when the compose sets no override."""
    try:
        lines = (root / compose_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for i, ln in enumerate(lines):
        if "--served-model-name" not in ln:
            continue
        rest = ln.split("--served-model-name", 1)[1].split("#", 1)[0].strip()
        names = rest.split() if rest else []
        j = i + 1
        while not rest and j < len(lines):
            m = re.match(r"\s*-\s*(\S+)\s*$", lines[j])
            if not m:
                break
            tok = m.group(1)
            if tok.startswith("-"):
                break  # next flag → end of the name list
            unwrapped = re.fullmatch(r"\$\{[A-Z_0-9]+:-(.+)\}", tok)
            names.append(unwrapped.group(1) if unwrapped else tok)
            j += 1
        return names
    return []


routes = []  # (model, port, name) — sorted + deduped before emission
for model in sorted(gw_by_model):
    entries = gw_by_model[model]
    slug = canonical_slug(model, entries)
    entry = entries[slug]
    port = entry["default_port"]
    if not port:
        sys.exit(f"litellm-emit: gateway entry {slug!r} has no default_port")
    if len(entries) > 1:
        others = ", ".join(sorted(set(entries) - {slug}))
        print(f"litellm-emit: note: {model}: gateway scene {slug} (: {port}); "
              f"non-canonical gateway flag(s) ignored: {others}", file=sys.stderr)
    names = []
    primary = entry.get("served_name") or (variant_by_slug.get(slug) or {}).get("served_name")
    if primary:
        names.append(primary)
    for n in compose_served_names(entry["compose_path"]):
        if n not in names:
            names.append(n)
    for n in entry.get("serve_aliases") or ():  # #1073 mechanism
        if n not in names:
            names.append(n)
    if not names:
        sys.exit(
            f"litellm-emit: {slug} is gateway=True but no served-name is "
            f"derivable (no _entry served_name override, no --served-model-name "
            f"in {entry['compose_path']}) — a gateway route needs a name"
        )
    for n in names:
        routes.append((model, port, n))

seen = set()
deduped = []
for r in sorted(routes, key=lambda t: (t[0], t[1], t[2])):
    if (r[1], r[2]) in seen:
        continue
    seen.add((r[1], r[2]))
    deduped.append(r)

chunks = []
for _model, port, name in deduped:
    chunks.append("\n".join([
        f"  - model_name: {name}",
        "    litellm_params:",
        f"      model: openai/{name}",
        f"      api_base: http://host.docker.internal:{port}/v1",
        "      api_key: EMPTY",
    ]))
generated = "\n\n".join([BEGIN_LINE, *chunks, END_LINE])

# --- splice / check -----------------------------------------------------------

try:
    cfg_text = CFG.read_text(encoding="utf-8")
except OSError as exc:
    sys.exit(f"litellm-emit: cannot read {CFG}: {exc}")

lines = cfg_text.splitlines()
begin_idx = next((i for i, ln in enumerate(lines) if BEGIN_RX.match(ln)), None)
end_idx = next((i for i, ln in enumerate(lines) if END_RX.match(ln)), None)
if begin_idx is None or end_idx is None:
    sys.exit(
        f"litellm-emit: {CFG} has no BEGIN/END GENERATED LOCAL BLOCK markers — "
        f"add them around the registry-derived routes once, then rerun"
    )
if end_idx <= begin_idx:
    sys.exit(f"litellm-emit: {CFG}: END GENERATED marker precedes BEGIN")

existing = "\n".join(lines[begin_idx:end_idx + 1])
check_mode = os.environ.get("LITELLM_EMIT_CHECK") == "1"
if existing == generated:
    print(f"OK: {CFG} generated local block matches the registry "
          f"({len(deduped)} routes from {len(gw_by_model)} gateway models).")
    sys.exit(0)

diff = "".join(difflib.unified_diff(
    existing.splitlines(True), generated.splitlines(True),
    fromfile=f"{CFG} (checked-in)", tofile="registry-generated",
))
if check_mode:
    print(f"FAIL: {CFG} local block drifted from the registry:", file=sys.stderr)
    sys.stderr.write(diff)
    if not diff.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.write("Fix: run `bash scripts/lib/litellm-emit.sh` to regenerate.\n")
    sys.exit(1)

new_text = "\n".join(lines[:begin_idx]) + "\n" + generated + "\n" + "\n".join(lines[end_idx + 1:])
if not new_text.endswith("\n"):
    new_text += "\n"  # never eat the file's trailing newline
CFG.write_text(new_text, encoding="utf-8")
print(f"Wrote {CFG}: regenerated local block ({len(deduped)} routes from "
      f"{len(gw_by_model)} gateway models); all other content untouched.")
PY
