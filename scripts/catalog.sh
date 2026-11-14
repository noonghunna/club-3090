#!/usr/bin/env bash
# catalog.sh — register / unregister models in YOUR local catalog layer (#1202 P4).
#
# The local layer had an executor (promote.py / demote.py) but no front door:
# both live under scripts/lib/, which reads as internal, and registering meant
# hand-authoring a spec JSON. Meanwhile the compose->spec derivation lived only
# inside the cockpit, so the layer was write-only from the UI (#1153). This is
# the CLI half; c3 and this script now share one derivation.
#
#   catalog.sh register   --compose <path> [--engine ID] [--model ID]
#                         [--workload W] [--port N] [--weights PATH]
#                         [--engine-type T] [--min-sm N] [--hf-repo ID]
#                         [--weights-variant V] [--slug S] [--dry-run] [-y]
#       --model naming a CURATED model attaches the slug to it (extends.d/<id>.yml)
#       instead of creating a model: --weights-variant reuses one of its variants,
#       or names a new one (which then needs --weights).
#   catalog.sh register   --spec-file <path> [--dry-run] [-y]
#   catalog.sh unregister --slug <engine>/<name> [--dry-run] [-y]
#   catalog.sh rename     --slug <old> --to <new>  [--dry-run]
#   catalog.sh update     --slug <slug> --set K=V  [--dry-run]   (repeatable)
#   catalog.sh register-engine --id ID --type T --image REF [--image-env VAR]
#                         [--kv-formats a,b] [--families a,b] [--weight-formats a,b]
#                         [--relation R --base B --base-version V --source URL
#                          --source-ref REF] [--display-name S] [--min-sm N]
#                         [--dry-run] [-y]      an engine on its own, before its slugs
#   catalog.sh unregister-engine --id ID [--dry-run] [-y]
#                         a local engine no local slug runs on
#   catalog.sh update-weights --model M --variant V --set K=V [--dry-run] [-y]
#                         a weights variant in models.d/ or extends.d/ (files:, hf_repo, …)
#   catalog.sh update-engine --id ID [--set K=V] [--add K=a,b] [--remove K=a,b]
#                         a local engine: KV formats, families, lineage, image, …
#   catalog.sh list [--json]   everything in the local layer, and what is broken or unused
#   catalog.sh prune [--dry-run] [-y]   remove what `list` reports unused (not broken/shadowed)
#   (`KEY=` with no value removes the key; every command writes only the local layer)
#
#   --root <dir>  operate on a throwaway tree instead of this checkout (tests).
#
# NOTHING here can touch the curated catalog: promote.py defaults to the local
# layer (core needs C3_ALLOW_CORE_PROMOTE=1) and demote.py refuses any slug that
# is not in registry.local.json.
set -uo pipefail
# Non-UTF-8 locales break python3 reads/writes on this rig (#599/#584).
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

die() { printf '[catalog] %s\n' "$*" >&2; exit 2; }

usage() {
  sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

[[ $# -gt 0 ]] || usage 2
SUB="$1"; shift

# An engine on its own: its flags are its own, so it skips the option loop below
# (which refuses anything it doesn't know).
if [[ "$SUB" == "register-engine" ]]; then
  exec python3 scripts/lib/profiles/register_engine.py "$@"
fi
if [[ "$SUB" == "unregister-engine" ]]; then
  exec python3 scripts/lib/profiles/unregister_engine.py "$@"
fi
if [[ "$SUB" == "update-weights" ]]; then
  exec python3 scripts/lib/profiles/catalog_update.py weights "$@"
fi
if [[ "$SUB" == "update-engine" ]]; then
  exec python3 scripts/lib/profiles/catalog_update.py engine "$@"
fi
if [[ "$SUB" == "list" ]]; then
  exec python3 scripts/lib/profiles/catalog_list.py "$@"
fi
if [[ "$SUB" == "prune" ]]; then
  exec python3 scripts/lib/profiles/catalog_prune.py "$@"
fi

COMPOSE="" SPEC_FILE="" ENGINE="" MODEL="" WORKLOAD="" PORT="" SLUG="" DRY="" YES="" RROOT="" WEIGHTS="" HFREPO="" ETYPE="" MINSM="" TO="" WVARIANT=""; declare -a SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose)   COMPOSE="${2:-}"; shift 2 ;;
    --spec-file) SPEC_FILE="${2:-}"; shift 2 ;;
    --engine)    ENGINE="${2:-}"; shift 2 ;;
    --model)     MODEL="${2:-}"; shift 2 ;;
    --workload)  WORKLOAD="${2:-}"; shift 2 ;;
    --port)      PORT="${2:-}"; shift 2 ;;
    --weights)   WEIGHTS="${2:-}"; shift 2 ;;
    --weights-variant) WVARIANT="${2:-}"; shift 2 ;;
    --hf-repo)   HFREPO="${2:-}"; shift 2 ;;
    --engine-type) ETYPE="${2:-}"; shift 2 ;;
    --min-sm)    MINSM="${2:-}"; shift 2 ;;
    --slug)      SLUG="${2:-}"; shift 2 ;;
    --to)        TO="${2:-}"; shift 2 ;;
    --set)       SETS+=("${2:-}"); shift 2 ;;
    --root)      RROOT="${2:-}"; shift 2 ;;
    --dry-run)   DRY="--dry-run"; shift ;;
    -y|--yes)    YES="--yes"; shift ;;
    -h|--help)   usage 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$SUB" in
  unregister)
    [[ -n "$SLUG" ]] || die "unregister needs --slug <engine>/<name>"
    # The slug's ENGINE, read from its entry before demote.py removes it. The slug
    # prefix is not the engine: since #1642 a slug on a curated engine takes core's
    # prefix (`vllm/…` on vllm-stable), and --slug can name it anything.
    _eng="$(python3 - "${RROOT:-$ROOT}" "$SLUG" <<'PY_PRE' 2>/dev/null || true
import json, sys
from pathlib import Path
reg = Path(sys.argv[1], "scripts/lib/profiles-local/registry.local.json")
try:
    print((json.loads(reg.read_text(encoding="utf-8")).get(sys.argv[2]) or {}).get("engine") or "")
except Exception:
    print("")
PY_PRE
)"
    python3 scripts/lib/profiles/demote.py --slug "$SLUG" ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+-y} || exit $?
    # `register` may have written a LOCAL engine profile for this slug's engine as a
    # side effect. demote.py knows nothing about it (engines are not its layer), so
    # removing the model alone orphans it — and orphans accumulate silently. Drop it
    # only when NO remaining local row uses that engine, only from the local layer,
    # and only if `register` wrote it: an engine registered on purpose
    # (`register-engine`) stays until you remove it.
    [[ -n "$DRY" || -z "$_eng" ]] && exit 0
    python3 - "${RROOT:-$ROOT}" "$_eng" <<'PY_ENG' || true
import json, re, sys
from pathlib import Path

root, engine = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / "scripts/lib/profiles"))
from register_engine import REGISTER_SIDE_EFFECT_MARK  # noqa: E402
prof = root / "scripts/lib/profiles-local/engines.d" / f"{engine}.yml"
if not prof.is_file():
    raise SystemExit(0)                       # curated engine, or none written
reg = root / "scripts/lib/profiles-local/registry.local.json"
still = {}
if reg.is_file():
    try:
        still = json.loads(reg.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(0)                   # unreadable: leave it alone, loudly nothing
if any(isinstance(e, dict) and e.get("engine") == engine for e in still.values()):
    raise SystemExit(0)                       # another local slug still needs it
m = re.search(r"(?m)^display_name:\s*(.*)$", prof.read_text(encoding="utf-8"))
if REGISTER_SIDE_EFFECT_MARK not in (m.group(1) if m else ""):
    print(f"[catalog] kept engines.d/{engine}.yml: it was registered on its own "
          f"(register-engine); remove it with catalog.sh unregister-engine --id {engine}")
    raise SystemExit(0)
prof.unlink()
print(f"[catalog] removed the now-unused local engine profile: engines.d/{engine}.yml")
PY_ENG
    exit 0
    ;;

  rename)
    # The remedy #1202 promised for a SHADOWED slug and never shipped: a curated
    # entry appearing under your name leaves your row loaded but unreachable, and
    # "rename it" was the advice with no way to do it.
    [[ -n "$SLUG" && -n "$TO" ]] || die "rename needs --slug <old> --to <new>"
    exec python3 scripts/lib/profiles/amend.py --slug "$SLUG" --to "$TO" \
         ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+-y}
    ;;

  update)
    [[ -n "$SLUG" ]] || die "update needs --slug <slug>"
    [[ ${#SETS[@]} -gt 0 ]] || die "update needs at least one --set KEY=VALUE"
    _args=(); for _kv in "${SETS[@]}"; do _args+=(--set "$_kv"); done
    exec python3 scripts/lib/profiles/amend.py --slug "$SLUG" "${_args[@]}" \
         ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+-y}
    ;;

  register)
    if [[ -n "$SPEC_FILE" ]]; then
      # Escape hatch, retained: BRING_YOUR_OWN.md documents this and removing it
      # would break anyone following those docs today.
      [[ -f "$SPEC_FILE" ]] || die "no such spec file: $SPEC_FILE"
      exec python3 scripts/lib/profiles/promote.py --spec-file "$SPEC_FILE" \
           --layer local ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+--yes}
    fi
    [[ -n "$COMPOSE" ]] || die "register needs --compose <path> (or --spec-file)"
    [[ -f "$COMPOSE" ]] || die "no such compose: $COMPOSE"

    # min_sm is a capability CLAIM. The honest value is the card the engine is
    # demonstrably running on, not a low number that claims old hardware works.
    if [[ -z "$MINSM" ]]; then
      MINSM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    fi
    SPEC="$(mktemp)"; trap 'rm -f "$SPEC"' EXIT
    # Derivation + spec assembly. Everything derived is PRINTED before the write:
    # a compose is read mechanically and cannot know whether `-ts 1,1` means a
    # layer split or tensor parallelism in the catalog's sense, so a wrong value
    # nobody saw is worse than a prompt.
    python3 - "$COMPOSE" "$SPEC" "$ENGINE" "$MODEL" "$WORKLOAD" "$PORT" "$WEIGHTS" \
             "${RROOT:-$ROOT}" "$ETYPE" "$MINSM" "$HFREPO" "$WVARIANT" "$SLUG" <<'PY' || exit $?
import json, os, re, sys, zlib
from pathlib import Path

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_facts import (derive_compose_facts, compose_layout,
                                                topology_cards, proposed_workload)

(src, out, engine_in, model_in, workload_in, port_in, weights_in, wroot,
 etype_in, minsm_in, hf_repo_in, variant_in, slug_in) = sys.argv[1:14]
text = Path(src).read_text(encoding="utf-8")
f = derive_compose_facts(text, src)
if not f.ok:
    print(f"[catalog] cannot read {src}: {f.error or 'unparseable'}", file=sys.stderr)
    raise SystemExit(2)

def _compose_default(v):
    """`${VAR:-x}` (nested too) -> x, the value a compose runs with when nothing is
    set; a bare `${VAR}` -> "" (the compose states none). Curated composes write
    the KV format as `${KV_CACHE_DTYPE:-fp8_e4m3}`, and that literal string is in
    no engine's or hardware profile's supported_kv_formats."""
    v = str(v or "")
    while (m := re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^${}]*)\}", v)):
        v = v[:m.start()] + m.group(1) + v[m.end():]
    return "" if "${" in v else v

kv = _compose_default(f.kv_dtype)

# ENGINE — the one field that cannot be inferred for an image we do not ship.
# Writing engine="unknown" into the catalog would be worse than refusing, and
# this is exactly the user the local layer exists for (their own build).
engine = engine_in or (f.engine if f.engine and f.engine != "unknown" else "")
if not engine:
    print(f"[catalog] the engine could not be inferred from image "
          f"{f.image or '(none)'!r}. Pass --engine <id> (e.g. --engine my-llamacpp).",
          file=sys.stderr)
    raise SystemExit(2)

mid = model_in or f.served_name
if not mid:
    print("[catalog] no model id: the compose has no --served-model-name/-a. "
          "Pass --model <id>.", file=sys.stderr)
    raise SystemExit(2)

# ATTACH. A model the catalog already ships gets another slug, not a second
# model: the slug attaches through extends.d/<id>.yml (local_extensions.py),
# which holds only the weights variants it adds. Only on an explicit --model: a
# served name that merely equals a curated id stays a refusal that names the fix.
_core_models = Path(wroot, "scripts/lib/profiles/models")
attach = bool(model_in) and (_core_models / f"{model_in}.yml").is_file()
if not model_in and (_core_models / f"{mid}.yml").is_file():
    print(f"[catalog] the compose serves {mid!r}, which is a curated model. To add a "
          f"slug of it, attach: --model {mid} --weights-variant <one of its variants, "
          f"or a new name>.", file=sys.stderr)
    raise SystemExit(2)
if variant_in and not attach:
    print("[catalog] --weights-variant applies when --model names a curated model.",
          file=sys.stderr)
    raise SystemExit(2)

max_ctx = int(f.max_ctx or 0) or 4096
# workload is pure taxonomy — nothing in a compose states it. Propose from ctx
# (the same rule `catalog.sh update --set max_ctx=…` re-applies).
workload = workload_in or proposed_workload(max_ctx)

# LOCAL models live in the 202xx band so a `git pull` of new curated slugs cannot
# collide with them. promote.py suggests exactly this value on refusal; compute it
# up front so registration succeeds the first time instead of failing with advice.
port = int(port_in) if port_in else 20200 + (zlib.crc32(mid.encode("utf-8")) % 100)

# ⛔ ARCH IS REQUIRED, AND MUST BE CHECKED BEFORE THE WRITE.
# promote.py reads spec["arch"] with .get, so a missing arch passes its own
# validation — and then the post-write re-check dies with KeyError: 'hidden_size'
# in compat.py, leaving the layer WRITTEN AND BROKEN ("NO ROLLBACK"). A compose
# cannot carry arch dims by construction, so read them from the weights.
arch = {}
cfg_extra = {}          # provenance/shape facts read off the checkpoint, not defaulted
if weights_in:
    from scripts.lib.profiles.deriver import gguf_facts_from_file
    # A DIRECTORY is what anyone naturally passes (it is what they downloaded),
    # so resolve it to the file we actually read instead of refusing with advice
    # that looks like it was already followed.
    wpath = Path(weights_in)
    if wpath.is_dir():
        if (wpath / "config.json").is_file():
            weights_in = str(wpath / "config.json")
        else:
            ggufs = sorted(g for g in wpath.glob("*.gguf")
                           if "mmproj" not in g.name.lower())
            if len(ggufs) == 1:
                weights_in = str(ggufs[0])
            elif ggufs:
                print(f"[catalog] {wpath} holds {len(ggufs)} .gguf files — "
                      f"name the one to read with --weights <file>.", file=sys.stderr)
                raise SystemExit(2)
            else:
                print(f"[catalog] {wpath} has no config.json and no .gguf — "
                      f"arch dims cannot be read from it.", file=sys.stderr)
                raise SystemExit(2)
    facts = gguf_facts_from_file(weights_in) if weights_in.endswith(".gguf") else None
    if facts is None and weights_in.endswith(".json"):
        cfg = json.loads(Path(weights_in).read_text(encoding="utf-8"))
        # A MULTIMODAL wrapper config (…ForConditionalGeneration) states the text
        # dims in a NESTED section and leaves the top level holding only the
        # wrapper's own fields, so a top-level-only read sees None for every dim
        # and refuses a model that is perfectly describable. Prefer the top level
        # when it has the dims; otherwise take the first nested section that does.
        # Per-key fallback to the top level, because some configs split
        # max_position_embeddings out of the text section.
        _NESTED = ("text_config", "language_model_config", "thinker_config",
                   "llm_config")
        sect = cfg if cfg.get("hidden_size") is not None else next(
            (cfg[k] for k in _NESTED
             if isinstance(cfg.get(k), dict)
             and cfg[k].get("hidden_size") is not None),
            cfg,
        )
        def _cfg(key):
            v = sect.get(key)
            return cfg.get(key) if v is None else v
        # Provenance + shape the model profile needs, taken from the SAME config
        # rather than defaulted. `family` was hardcoded "dense", which is not even
        # in this field's vocabulary (it names the model FAMILY — qwen/gemma/glm),
        # and vision_capable was hardcoded False for checkpoints that ship a
        # vision_config.
        _mt = str(cfg.get("model_type") or sect.get("model_type") or "")
        for _fam in ("qwen", "gemma", "glm", "deepseek", "llama", "mistral",
                     "phi", "nemotron"):
            if _mt.startswith(_fam):
                cfg_extra["family"] = _fam
                break
        if isinstance(cfg.get("vision_config"), dict):
            cfg_extra["vision_capable"] = True
        facts = {"hidden_size": _cfg("hidden_size"),
                 "num_hidden_layers": _cfg("num_hidden_layers"),
                 "num_attn_heads": _cfg("num_attention_heads"),
                 "num_kv_heads": _cfg("num_key_value_heads"),
                 "max_ctx_supported": _cfg("max_position_embeddings"),
                 # A KV-math hint. The GGUF path derives it; an HF config does not
                 # state it, and False is the ordinary case (K and V differ). Said
                 # out loud because it is the one value here that is assumed.
                 "attention_k_eq_v": False}
    for k in ("hidden_size", "num_hidden_layers", "num_attn_heads", "num_kv_heads",
              "head_dim_attn", "max_ctx_supported", "attention_k_eq_v"):
        if facts and facts.get(k) is not None:
            arch[k] = facts[k]

# The profile factory bracket-accesses SIX arch fields; a missing one is a
# KeyError in compat.py AFTER promote has written the layer. Enumerated from the
# factory rather than guessed — twice now a shorter list let a broken write through.
# NOT attention_k_eq_v: promote.py fills it with a documented conservative
# default ("compat.ModelProfile REQUIRES attention_k_eq_v — emit the conservative
# default when the spec doesn't carry the fact"), and the post-write diagnose +
# kv-calc gates flag it for correction. Requiring it here refused every
# --weights <gguf> registration, because gguf_facts_from_file does not return it
# — so the GGUF path, the likelier one for this rig, was broken while the
# config.json path (which sets it explicitly) passed the test.
# ⛔ THE WORKLOAD MUST EXIST, AND MUST ALSO BE CHECKED BEFORE THE WRITE.
# compat's cross-reference validation runs AFTER promote has written all three
# artifacts, and a dangling workload ref fails it — which does not just break the
# new entry, it takes the WHOLE profile system down: every launcher calls
# load_profiles(), so one typo in one local registration makes every CORE slug
# unloadable until the user hand-edits a gitignored JSON. `--workload max-context`
# (a plausible-sounding name that does not exist) did exactly that here.
_known_wl = sorted(x.stem for x in Path("scripts/lib/profiles/workloads").glob("*.yml"))
if workload and _known_wl and workload not in _known_wl:
    print(f"[catalog] unknown workload {workload!r}. Known: {', '.join(_known_wl)}.",
          file=sys.stderr)
    print(f"[catalog] Refusing BEFORE writing — a dangling workload reference fails "
          f"compat's cross-reference check AFTER the write, and that failure takes "
          f"the ENTIRE profile system down, core slugs included.", file=sys.stderr)
    raise SystemExit(2)

_ARCH_REQUIRED = ("hidden_size", "num_hidden_layers", "num_attn_heads",
                  "num_kv_heads", "max_ctx_supported")
if "max_ctx_supported" not in arch and max_ctx:
    arch["max_ctx_supported"] = max_ctx      # the compose proves at least this much
# An attached slug writes no model profile: the curated one states the arch.
need = [] if attach else [k for k in _ARCH_REQUIRED if arch.get(k) is None]
if need:
    print(f"[catalog] the model profile needs arch dims a compose cannot state: "
          f"{', '.join(need)}.", file=sys.stderr)
    print(f"[catalog] pass --weights <path-to.gguf> (header read only) or "
          f"<config.json>. Refusing BEFORE writing — promote would otherwise write "
          f"the layer and then fail its own re-check, leaving it broken.",
          file=sys.stderr)
    raise SystemExit(2)

# ── the engine profile ────────────────────────────────────────────────────────
# cross-reference validation requires `engine` to resolve, and engines used to be
# core-only — so a user on their OWN build could not register at all. Write a
# minimal profile for an engine the catalog does not know, from EVIDENCE only:
# what the compose states, and what the rig demonstrably runs. Nothing is assumed
# on the user's behalf; unverified capability blocks are left empty so the stack
# makes no promises it cannot keep.
sys.path.insert(0, wroot)
from scripts.lib.profiles.compat import load_profiles  # noqa: E402
try:
    _engines = load_profiles().engines
except Exception:
    _engines = {}
known = set(_engines)

# ⚠️ TWO VOCABULARIES. compose_facts yields `llama-cpp` (hyphen); engine profiles
# use `llama.cpp` (dot), and `type` has no enum validation — so passing a value
# through would write something that matches nothing, silently costing the fork
# its drafter compatibility. register_engine.normalize_type folds the aliases,
# shared with `catalog.sh register-engine` so the two write the same profile.
from scripts.lib.profiles.register_engine import normalize_type, render_engine_yaml, ENGINE_TYPES
engine_yaml = ""
if engine not in known:
    etype = normalize_type(etype_in or f.engine or "")
    if not etype:
        if etype_in:
            print(f"[catalog] --engine-type {etype_in!r}: one of {', '.join(ENGINE_TYPES)}.",
                  file=sys.stderr)
        else:
            print(f"[catalog] {engine!r} is not a known engine and its lineage cannot be "
                  f"inferred from image {f.image or '(none)'!r}. Pass --engine-type "
                  f"(llama.cpp | vllm | …) so drafter/feature logic knows what it "
                  f"behaves like.", file=sys.stderr)
        raise SystemExit(2)
    drafted = ("--model-draft" in text) or (" -md " in text)
    # The profile is the source of truth for the image (docs/ADDING_ENGINES.md →
    # CONTRACT — image provenance): `method: docker_image`, and the spec as one
    # unquoted token, since engine-pin-bump.sh and resolved_config.py parse it that
    # way. A `${VAR:-image}` compose default is the image; a bare `${VAR}` names none.
    _img = f.image or ""
    if _img.startswith("${") and ":-" in _img and _img.endswith("}"):
        _img = _img[_img.index(":-") + 2:-1]
    if "${" in _img or any(c.isspace() for c in _img):
        _img = ""
    engine_yaml = render_engine_yaml(
        eid=engine, display_name=f"{engine} (local, registered from a compose)",
        etype=etype, min_sm=minsm_in or "7.5", image=_img,
        kv_formats=[kv] if kv else [],
        notes=[f"Registered by catalog.sh from {Path(src).name}. Capabilities are",
               "EVIDENCED, not assumed: only what the compose demonstrably uses is",
               "declared. Widen supported_kv_formats / supported_drafters / features",
               "once measured." + ("" if not drafted else " A drafter flag was seen in the compose.")])

fmt = "gguf" if (f.model_path or "").endswith(".gguf") else "safetensors"
# The emitter splits compose_path on "/compose/" and reads <topology>/<quant>/<file>
# from the tail — so the segment after /compose/ IS the topology in every listing.
# Writing ".../compose/local/base.yml" made "local" render as the topology.
# A compose filed in that layout already says its topology and its serving name,
# so the copy keeps both (#2028): `multi4/…/mtp.yml` stays `multi4/…/mtp.yml`.
# Only a compose outside the layout gets the topology from its TP and `base.yml`.
_layout = compose_layout(os.path.abspath(src))
_tp = int(f.tp or 1)
_tp_src = "derived" if f.tp else "defaulted"
if _layout:
    _topo = _layout[0]
    if not f.tp:
        # QEngine, Strata and Maya set TP at boot, so the compose states none; the
        # directory it is filed in names the card count.
        _tp, _tp_src = topology_cards(_topo), "from its directory"
    elif topology_cards(_topo) != _tp:
        print(f"[catalog] note: the compose states TP {_tp} but is filed under {_topo}/; "
              f"the entry keeps both. Check which one it serves with.", file=sys.stderr)
else:
    _topo = "single" if _tp <= 1 else ("dual" if _tp == 2 else f"multi{_tp}")
_cfile = _layout[2] if _layout else "base.yml"
# Where core keeps this engine's composes, and the slug prefix it uses there
# (core_layout.py, shared with export_pr.py), so a slug registered here exports
# under the name it already has. `vllm-stable` slugs are `vllm/…`, filed under
# `vllm/`; an engine core doesn't have keeps its id for both. This named every
# slug after the engine ID, and export_pr.py refuses `vllm-stable/…`.
sys.path.insert(0, str(Path(wroot, "scripts/lib/profiles")))
import core_layout
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY as _CORE_REG
_by_engine, _ = core_layout.engine_layout(_CORE_REG)
_pair, _pairs = core_layout.pick_pair(_by_engine, engine, f.engine or "")
if _pairs and _pair is None:
    print(f"[catalog] core keeps {engine} composes in "
          f"{', '.join(d + '/' for d, _ in _pairs)}, and this compose reads as "
          f"{f.engine!r}, which picks none of them.", file=sys.stderr)
    raise SystemExit(2)
_edir, _eprefix = _pair if _pair else (engine, engine)

cpath = (f"scripts/lib/profiles-local/composes/{mid}/{_edir}"
         f"/compose/{_topo}/{fmt}/{_cfile}")
slug = slug_in or f"{_eprefix}/{mid}"


# Weights FACTS, not placeholders. size_gb was hardcoded 1.0 and hf_repo "",
# so the catalog reported a 120 GiB checkpoint as 1 GB from an unnamed provider
# — the c3 "provider" column is blank for exactly this reason.
_wdir = None
if weights_in:
    _wp = Path(weights_in)
    _wdir = _wp if _wp.is_dir() else _wp.parent
_size_gb, _size_src = 1.0, "defaulted"
if _wdir and _wdir.is_dir():
    try:
        _b = sum(x.stat().st_size for x in _wdir.rglob("*")
                 if x.is_file() and ".cache" not in x.parts)
        if _b:
            _size_gb, _size_src = round(_b / 2**30, 1), "derived"
    except OSError:
        pass
# ⚠️ RELATIVE TO MODEL_DIR, never absolute. The launcher's preflight joins
# MODEL_DIR onto `path`, so an absolute value yields
# "/mnt/models/huggingface//mnt/models/huggingface/<dir>" and the boot refuses on a
# path that plainly exists. Core profiles are relative
# ("glm-5.3-flash-gguf/UD-IQ4_XS"), with local_subdir the FIRST segment. And the
# subdir is the real DIRECTORY name — it was the model id, which is only the same
# string by coincidence.
_md = os.environ.get("MODEL_DIR") or "/mnt/models/huggingface"
_rel_path = ""
if _wdir:
    try:
        _rel_path = str(_wdir.resolve().relative_to(Path(_md).resolve()))
    except ValueError:
        _rel_path = _wdir.name
        print(f"[catalog] note: {_wdir} is not under MODEL_DIR ({_md}); recording "
              f"{_rel_path!r} as the subdir. Launch with MODEL_DIR pointing at its "
              f"parent, or the preflight will not find it.", file=sys.stderr)
_rel_subdir = _rel_path.split("/")[0] if _rel_path else mid

_hf_repo = hf_repo_in or ""
spec_family = cfg_extra.get("family", "dense")

# OFFLOAD. The vocabulary is a closed set whose values mean different mechanisms
# ("uva" = vLLM demand-paged experts, "prefetch" = bulk layer prefetch,
# "residency"/"tensor-override" = llama.cpp -ot). A wrong value here has already
# caused a real misdiagnosis, so name it only when the compose NAMES it; when the
# compose only proves offload is ON, say so and let the user state which.
_offload = None
_ob = (f.offload_backend or "").strip().lower()
if _ob in ("uva", "prefetch", "residency", "tensor-override"):
    _offload = _ob
elif f.cpu_offload_gb and int(f.cpu_offload_gb) > 0:
    print(f"[catalog] note: the compose offloads to host RAM "
          f"(cpu-offload-gb={f.cpu_offload_gb}) but does not name the BACKEND, and "
          f"the value decides the mechanism. Left unset; state it with: "
          f"catalog.sh update --slug <slug> --set offload=uva", file=sys.stderr)

# ⚠️ DETECTABILITY. A compose can be perfectly valid and still be INVISIBLE to
# c3: club3090_tui_core.detect classifies engine containers by NAME PREFIX and by
# the CONTAINER-SIDE port. Miss either and no ServingTarget is built, health.sh
# falls back to the curated default port, and the estate reads "not reachable"
# while the GPU bars plainly show the model loaded. That is a heuristic we own,
# and it predates the local layer — a user's own engine cannot be expected to
# satisfy it. WARN, never refuse: the registration is correct, our detector is
# the narrow part. Tracked in club-3090-todo.md (registry-first detection).
#
# ⚠️ These two constants MIRROR club3090_tui_core.detect. They are duplicated
# because catalog.sh must run on the launcher's no-extra-deps path;
# test-catalog-detectability.sh asserts they still match, so a change there
# cannot silently strand this warning.
_DETECT_PREFIXES = ("vllm-", "llama-cpp-", "ik-llama-", "sglang-", "beellama-", "tabbyapi-", "exl3-", "exllamav3-", "strata-", "qengine-")
_DETECT_PORTS = ("8000", "8080", "30000", "5000")
_undetectable = []
if f.container_name and not f.container_name.startswith(_DETECT_PREFIXES):
    _undetectable.append(
        f"container_name {f.container_name!r} starts with none of "
        f"{', '.join(_DETECT_PREFIXES)}"
    )
if f.internal_port and f.internal_port not in _DETECT_PORTS:
    _undetectable.append(
        f"container-side port {f.internal_port} is not one of "
        f"{', '.join(_DETECT_PORTS)}"
    )
if _undetectable:
    print("[catalog] ⚠️  REGISTERED, BUT c3 WILL NOT SEE IT SERVING:", file=sys.stderr)
    for _u in _undetectable:
        print(f"[catalog]     - {_u}", file=sys.stderr)
    print("[catalog]   The slug, weights and compose are fine — this is our estate "
          "detector,", file=sys.stderr)
    print("[catalog]   which infers engines from naming conventions that predate the "
          "local layer.", file=sys.stderr)
    print("[catalog]   Until that is registry-driven, rename the container to a "
          "listed prefix", file=sys.stderr)
    print("[catalog]   and publish on a listed container-side port to be detected.",
          file=sys.stderr)

# mem_util is the compose's --gpu-memory-utilization / --mem-fraction-static, so only a
# vLLM- or SGLang-type engine has one; every core llama.cpp / exl3 entry carries None,
# and export_pr.py requires it only for those two types.
_reg_type = (getattr(_engines.get(engine), "type", "") if engine in known
             else normalize_type(etype_in or f.engine or "") or "")
_mem_util = 0.9 if _reg_type in ("vllm", "sglang") else None

spec = {
    "model_id": mid,
    "display_name": mid,
    "family": spec_family,
    "weights": {"local": {"path": _rel_path, "local_subdir": _rel_subdir,
                          "size_gb": _size_gb,
                          "format": fmt, "status": "incubating", "hf_repo": _hf_repo,
                          "engine": _edir, "kind": "main",
                          "verify_glob": "*.gguf" if fmt == "gguf" else "*.safetensors"}},
    "arch": arch,
    "default_weight_variant": "local",
    "vision_capable": bool(cfg_extra.get("vision_capable", False)),
    "compose": {"path": cpath, "content": text},
    "registry_entry": {"slug": slug, "kwargs": {
        "model": mid, "weights_variant": "local", "workload": workload,
        "engine": engine, "drafter": None,
        "spec_method": f.spec_method or None,
        "offload": _offload,
        "kv_format": kv or "f16", "tp": _tp,
        "max_ctx": max_ctx, "max_num_seqs": 1, "mem_util": _mem_util,
        "compose_path": cpath, "default_port": port, "kvcalc_key": "SKIP"}},
}

# ── ATTACH: a slug of a curated model (extends.d/<id>.yml) ───────────────────
variant, reuse, export_notes = "local", "", []
if attach:
    import yaml as _yaml
    _core = _yaml.safe_load((_core_models / f"{mid}.yml").read_text(encoding="utf-8")) or {}
    _core_w = _core.get("weights") or {}
    _ext_p = Path(wroot, "scripts/lib/profiles-local/extends.d", f"{mid}.yml")
    _ext_w = {}
    if _ext_p.is_file():
        _ext_w = (_yaml.safe_load(_ext_p.read_text(encoding="utf-8")) or {}).get("weights") or {}
    if not variant_in:
        _local = f"; local ones: {', '.join(sorted(_ext_w))}" if _ext_w else ""
        print(f"[catalog] attaching to the curated model {mid!r} needs --weights-variant: "
              f"one of its variants to reuse its weights ({', '.join(sorted(_core_w)) or 'none'}"
              f"{_local}), or a new name for weights of your own (with --weights).",
              file=sys.stderr)
        raise SystemExit(2)
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", variant_in):
        print(f"[catalog] --weights-variant {variant_in!r}: expected [a-z0-9][a-z0-9._-]*",
              file=sys.stderr)
        raise SystemExit(2)
    variant = variant_in
    _attach_w = {}
    if variant in _core_w:
        reuse = "curated, reused"
    elif variant in _ext_w and not weights_in:
        reuse = "local, reused"
    else:
        if not weights_in:
            print(f"[catalog] {variant!r} is a new weights variant of {mid}: pass --weights "
                  f"<dir or file> so the catalog knows where the weights are.", file=sys.stderr)
            raise SystemExit(2)
        reuse = "local, reused" if variant in _ext_w else "new"
        _meta = {"path": _rel_path, "local_subdir": _rel_subdir, "size_gb": _size_gb,
                 "format": fmt, "status": "incubating", "kind": "main",
                 "verify_glob": "*.gguf" if fmt == "gguf" else "*.safetensors"}
        if _hf_repo:
            _meta["hf_repo"] = _hf_repo
        _attach_w = {variant: _meta}
        if reuse == "new":
            _uw = f"catalog.sh update-weights --model {mid} --variant {variant}"
            if not _hf_repo:
                export_notes.append(f"weights.{variant} has no hf_repo: {_uw} --set hf_repo=<org/repo>")
            export_notes.append(f"weights.{variant} has no files: list (the globs setup.sh "
                                f"fetches): {_uw} --set files=<glob,glob>")
    # A weights variant names its engine DIRECTORY (`vllm`, `llama-cpp`), as every
    # curated profile does (ADDING_MODELS, case 2) — not the engine id.
    for _m in _attach_w.values():
        _m["engine"] = _edir
    slug = slug_in or f"{_eprefix}/{mid}-{_topo}-{variant}"
    cpath = (f"scripts/lib/profiles-local/composes/{mid}/{_edir}"
             f"/compose/{_topo}/{variant}/{_cfile}")
    if Path(wroot, cpath).exists():
        cpath = cpath.rsplit("/", 1)[0] + f"/{slug.split('/', 1)[-1]}.yml"
    _tp_add = [_tp] if _tp not in (_core.get("valid_tp") or []) else []
    _etype = (getattr(_engines.get(engine), "type", "") if engine in _engines
              else (normalize_type(etype_in or f.engine or "") or ""))
    if _etype == "vllm":
        export_notes.append(f"vLLM-type entries need a kvcalc_key: catalog.sh update --slug "
                            f"{slug} --set kvcalc_key={mid}:<profile>")
    _kw = dict(spec["registry_entry"]["kwargs"], weights_variant=variant, compose_path=cpath)
    spec = {
        "model_id": mid,
        "attach": {"weights": _attach_w, "valid_tp_add": _tp_add},
        "compose": {"path": cpath, "content": text},
        "registry_entry": {"slug": slug, "kwargs": _kw},
    }
# The engine profile rides in the SPEC and promote.py writes it with the other
# local artifacts — AFTER its plan, its --dry-run exit and its confirm. It used to
# be written HERE, before promote ran, so `register --dry-run` (and a declined
# confirm) left engines.d/<engine>.yml behind while promote printed "nothing written".
if engine_yaml:
    spec["local_engine_profile"] = {"id": engine, "yaml": engine_yaml}
    print(f"[catalog] new local ENGINE profile for {engine!r} (type={etype}, capabilities "
          f"evidenced from the compose) — written by promote with the rest of the plan")
Path(out).write_text(json.dumps(spec, indent=1), encoding="utf-8")

def mark(v, given):  # show the user which values THEY chose vs which we guessed
    if isinstance(given, str):
        return given
    return "given" if given else "derived"

# "(defaulted)" is a placeholder the compose does not state — said as such, because a
# default that reads "(derived)" turns into a false record once written (#2028).
_defaulted = [k for k, d in (("max_ctx", not f.max_ctx), ("kv_format", not kv),
                             ("tp", _tp_src == "defaulted")) if d]
print("[catalog] resolved from the compose — check these before it writes:")
_rows = [
    ("slug",      slug,              bool(slug_in or (engine_in and model_in))),
    ("model",     mid + (" (curated: the slug attaches to it)" if attach else ""), bool(model_in)),
    ("engine",    engine,            bool(engine_in)),
    ("workload",  workload,          "given" if workload_in
                                     else ("derived" if f.max_ctx else "defaulted")),
    ("max_ctx",   max_ctx,           "derived" if f.max_ctx else "defaulted"),
    ("kv_format", kv or "f16",       "derived" if kv else "defaulted"),
    ("tp",        _tp,               _tp_src),
    ("port",      port,              bool(port_in)),
    ("weights",   f.model_path,      False),
    # These four decide c3 catalog COLUMNS (Spec Dec · Offload · provider ·
    # size). They were hardcoded to None/""/1.0 and so rendered as blanks, which
    # reads as "this recipe does not do that" rather than "nobody looked".
    ("spec",      (f"{f.spec_method} n={f.spec_depth}" if f.spec_method and f.spec_depth
                   else (f.spec_method or "(none found)")), False),
    ("offload",   _offload or (f"on, backend unnamed (cpu-offload-gb={f.cpu_offload_gb})"
                               if f.cpu_offload_gb else "(none found)"), False),
    ("provider",  _hf_repo or "(none — pass --hf-repo)", bool(hf_repo_in)),
    ("size_gb",   _size_gb,          _size_src),
]
_rows += [("compose", cpath.split("/composes/", 1)[-1], "given" if _layout else "derived")]
if attach:
    _rows += [("variant", f"{variant} ({reuse})", True)]
else:
    _rows += [
        ("family",    spec_family,       False),
        ("vision",    bool(cfg_extra.get("vision_capable", False)), False),
        ("arch",      f"hidden={arch['hidden_size']} layers={arch['num_hidden_layers']} "
                      f"heads={arch['num_attn_heads']}/{arch['num_kv_heads']}", True),
    ]
for label, val, given in _rows:
    print(f"[catalog]   {label:10s} {str(val):46s} ({mark(val, given)})")
if _defaulted:
    _sets = " ".join(f"--set {k}=<served value>" for k in _defaulted)
    print(f"[catalog]   defaulted: {', '.join(_defaulted)} — the compose states "
          f"{'none of these' if len(_defaulted) > 1 else 'no value'}, so the entry holds "
          f"a placeholder. Once it serves, record what it runs with: "
          f"catalog.sh update --slug {slug} {_sets}")
for _n in export_notes:
    print(f"[catalog]   before export_pr.py: {_n}")
if f.port and str(port) != str(f.port):
    print(f"[catalog]   note: the compose serves on {f.port}; the catalog entry uses "
          f"{port} (LOCAL 202xx band). Your compose is unchanged.")
PY

    python3 scripts/lib/profiles/promote.py --spec-file "$SPEC" \
         --layer local ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+--yes}
    rc=$?
    # Would this pass the central registry? Advisory only: the local layer may run
    # things the catalog would never ship. export_pr.py --check enforces the same
    # rules before a PR (docs/ADDING_ENGINES.md, docs/ADDING_MODELS.md).
    if [[ $rc -eq 0 && -z "$DRY" ]]; then
      _slug="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["registry_entry"]["slug"])' "$SPEC" 2>/dev/null || true)"
      if [[ -n "$_slug" ]]; then
        echo "[catalog] registered. Would it pass the central registry? (advisory)"
        python3 scripts/lib/profiles/catalog_rules.py --slug "$_slug" ${RROOT:+--root "$RROOT"} || true
      fi
    fi
    exit "$rc"
    ;;

  -h|--help) usage 0 ;;
  *) die "unknown subcommand: $SUB (expected register | register-engine | unregister | unregister-engine | rename | update | update-weights | update-engine | list | prune)" ;;
esac
