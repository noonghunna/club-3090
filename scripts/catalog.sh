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
#                         [--dry-run] [-y]
#   catalog.sh register   --spec-file <path> [--dry-run] [-y]
#   catalog.sh unregister --slug <engine>/<name> [--dry-run] [-y]
#   catalog.sh rename     --slug <old> --to <new>  [--dry-run]
#   catalog.sh update     --slug <slug> --set K=V  [--dry-run]   (repeatable)
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
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

[[ $# -gt 0 ]] || usage 2
SUB="$1"; shift

COMPOSE="" SPEC_FILE="" ENGINE="" MODEL="" WORKLOAD="" PORT="" SLUG="" DRY="" YES="" RROOT="" WEIGHTS="" HFREPO="" ETYPE="" MINSM="" TO=""; declare -a SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose)   COMPOSE="${2:-}"; shift 2 ;;
    --spec-file) SPEC_FILE="${2:-}"; shift 2 ;;
    --engine)    ENGINE="${2:-}"; shift 2 ;;
    --model)     MODEL="${2:-}"; shift 2 ;;
    --workload)  WORKLOAD="${2:-}"; shift 2 ;;
    --port)      PORT="${2:-}"; shift 2 ;;
    --weights)   WEIGHTS="${2:-}"; shift 2 ;;
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
    python3 scripts/lib/profiles/demote.py --slug "$SLUG" ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+-y} || exit $?
    # `register` may have written a LOCAL engine profile for this slug's engine.
    # demote.py knows nothing about it (engines are not its layer), so removing
    # the model alone orphans it — and orphans accumulate silently. Drop it only
    # when NO remaining local row uses that engine, and only from the local layer.
    [[ -n "$DRY" ]] && exit 0
    python3 - "${RROOT:-$ROOT}" "$SLUG" <<'PY_ENG' || true
import json, sys
from pathlib import Path

root, slug = Path(sys.argv[1]), sys.argv[2]
engine = slug.split("/", 1)[0]
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
if any(str(s).split("/", 1)[0] == engine for s in still):
    raise SystemExit(0)                       # another local model still needs it
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
             "${RROOT:-$ROOT}" "$ETYPE" "$MINSM" "$HFREPO" <<'PY' || exit $?
import json, sys, zlib
from pathlib import Path

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_facts import derive_compose_facts

(src, out, engine_in, model_in, workload_in, port_in, weights_in, wroot,
 etype_in, minsm_in, hf_repo_in) = sys.argv[1:12]
text = Path(src).read_text(encoding="utf-8")
f = derive_compose_facts(text, src)
if not f.ok:
    print(f"[catalog] cannot read {src}: {f.error or 'unparseable'}", file=sys.stderr)
    raise SystemExit(2)

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

max_ctx = int(f.max_ctx or 0) or 4096
# workload is pure taxonomy — nothing in a compose states it. Propose from ctx.
workload = workload_in or ("long-ctx-single" if max_ctx >= 65536 else "fast-chat")

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
need = [k for k in _ARCH_REQUIRED if arch.get(k) is None]
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
    known = set(load_profiles().engines)
except Exception:
    known = set()

# ⚠️ TWO VOCABULARIES. compose_facts yields `llama-cpp` (hyphen); engine profiles
# use `llama.cpp` (dot), and `type` has no enum validation — so passing the
# derived value through would write something that matches nothing, silently
# costing the fork its drafter compatibility. Map, never pass through.
_TYPE_BY_FACT = {"llama-cpp": "llama.cpp", "vllm": "vllm", "ik-llama": "ik-llama"}
engine_yaml = ""
if engine not in known:
    etype = etype_in or _TYPE_BY_FACT.get(f.engine, "")
    if not etype:
        print(f"[catalog] {engine!r} is not a known engine and its lineage cannot be "
              f"inferred from image {f.image or '(none)'!r}. Pass --engine-type "
              f"(llama.cpp | vllm | …) so drafter/feature logic knows what it "
              f"behaves like.", file=sys.stderr)
        raise SystemExit(2)
    kvs = [f.kv_dtype] if f.kv_dtype else []
    drafted = ("--model-draft" in text) or (" -md " in text)
    lines = [
        "schema_version: 1",
        f"id: {engine}",
        f"display_name: {engine} (local, registered from a compose)",
        f"type: {etype}",
        "stability: experimental",
        f"min_sm: {minsm_in or '7.5'}",
    ]
    if f.image:
        lines += ["install:", "  method: docker", f'  spec: "{f.image}"']
    if kvs:
        lines += ["supported_kv_formats:"] + [f"  - {k}" for k in kvs]
    lines += ["notes: >",
              f"  Registered by catalog.sh from {Path(src).name}. Capabilities are",
              "  EVIDENCED, not assumed: only what the compose demonstrably uses is",
              "  declared. Widen supported_kv_formats / supported_drafters / features",
              "  once measured." + ("" if not drafted else " A drafter flag was seen in the compose.")]
    engine_yaml = "\n".join(lines) + "\n"

fmt = "gguf" if (f.model_path or "").endswith(".gguf") else "safetensors"
# The emitter splits compose_path on "/compose/" and reads <topology>/<quant>/<file>
# from the tail — so the segment after /compose/ IS the topology in every listing.
# Writing ".../compose/local/base.yml" made "local" render as the topology.
_tp = int(f.tp or 1)
_topo = "single" if _tp <= 1 else ("dual" if _tp == 2 else f"multi{_tp}")
cpath = (f"scripts/lib/profiles-local/composes/{mid}/{engine}"
         f"/compose/{_topo}/{fmt}/base.yml")
slug = f"{engine}/{mid}"


# Weights FACTS, not placeholders. size_gb was hardcoded 1.0 and hf_repo "",
# so the catalog reported a 120 GiB checkpoint as 1 GB from an unnamed provider
# — the c3 "provider" column is blank for exactly this reason.
_wdir = None
if weights_in:
    _wp = Path(weights_in)
    _wdir = _wp if _wp.is_dir() else _wp.parent
_size_gb = 1.0
if _wdir and _wdir.is_dir():
    try:
        _b = sum(x.stat().st_size for x in _wdir.rglob("*")
                 if x.is_file() and ".cache" not in x.parts)
        if _b:
            _size_gb = round(_b / 2**30, 1)
    except OSError:
        pass
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

spec = {
    "model_id": mid,
    "display_name": mid,
    "family": spec_family,
    "weights": {"local": {"path": str(_wdir) if _wdir else f.model_path,
                          "local_subdir": mid, "size_gb": _size_gb,
                          "format": fmt, "status": "incubating", "hf_repo": _hf_repo,
                          "engine": engine, "kind": "main",
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
        "kv_format": f.kv_dtype or "f16", "tp": int(f.tp or 1),
        "max_ctx": max_ctx, "max_num_seqs": 1, "mem_util": 0.9,
        "compose_path": cpath, "default_port": port, "kvcalc_key": "SKIP"}},
}
Path(out).write_text(json.dumps(spec, indent=1), encoding="utf-8")
if engine_yaml:
    d = Path(wroot) / "scripts/lib/profiles-local/engines.d"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{engine}.yml").write_text(engine_yaml, encoding="utf-8")
    print(f"[catalog] wrote a local ENGINE profile: profiles-local/engines.d/{engine}.yml "
          f"(type={etype}, capabilities evidenced from the compose)")

def mark(v, given):  # show the user which values THEY chose vs which we guessed
    return "given" if given else "derived"

print("[catalog] resolved from the compose — check these before it writes:")
for label, val, given in (
    ("slug",      slug,              bool(engine_in and model_in)),
    ("model",     mid,               bool(model_in)),
    ("engine",    engine,            bool(engine_in)),
    ("workload",  workload,          bool(workload_in)),
    ("max_ctx",   max_ctx,           False),
    ("kv_format", f.kv_dtype or "f16", False),
    ("tp",        f.tp or 1,         False),
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
    ("size_gb",   _size_gb,          False),
    ("family",    spec_family,       False),
    ("vision",    bool(cfg_extra.get("vision_capable", False)), False),
    ("arch",      f"hidden={arch['hidden_size']} layers={arch['num_hidden_layers']} "
                  f"heads={arch['num_attn_heads']}/{arch['num_kv_heads']}", True),
):
    print(f"[catalog]   {label:10s} {str(val):46s} ({mark(val, given)})")
if f.port and str(port) != str(f.port):
    print(f"[catalog]   note: the compose serves on {f.port}; the catalog entry uses "
          f"{port} (LOCAL 202xx band). Your compose is unchanged.")
PY

    exec python3 scripts/lib/profiles/promote.py --spec-file "$SPEC" \
         --layer local ${RROOT:+--root "$RROOT"} ${DRY:+--dry-run} ${YES:+--yes}
    ;;

  -h|--help) usage 0 ;;
  *) die "unknown subcommand: $SUB (expected register | unregister | rename | update)" ;;
esac
