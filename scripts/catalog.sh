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
#                         [--engine-type T] [--min-sm N] [--dry-run] [-y]
#   catalog.sh register   --spec-file <path> [--dry-run] [-y]
#   catalog.sh unregister --slug <engine>/<name> [--dry-run] [-y]
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

COMPOSE="" SPEC_FILE="" ENGINE="" MODEL="" WORKLOAD="" PORT="" SLUG="" DRY="" YES="" RROOT="" WEIGHTS="" ETYPE="" MINSM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose)   COMPOSE="${2:-}"; shift 2 ;;
    --spec-file) SPEC_FILE="${2:-}"; shift 2 ;;
    --engine)    ENGINE="${2:-}"; shift 2 ;;
    --model)     MODEL="${2:-}"; shift 2 ;;
    --workload)  WORKLOAD="${2:-}"; shift 2 ;;
    --port)      PORT="${2:-}"; shift 2 ;;
    --weights)   WEIGHTS="${2:-}"; shift 2 ;;
    --engine-type) ETYPE="${2:-}"; shift 2 ;;
    --min-sm)    MINSM="${2:-}"; shift 2 ;;
    --slug)      SLUG="${2:-}"; shift 2 ;;
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
             "${RROOT:-$ROOT}" "$ETYPE" "$MINSM" <<'PY' || exit $?
import json, sys, zlib
from pathlib import Path

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_facts import derive_compose_facts

src, out, engine_in, model_in, workload_in, port_in, weights_in, wroot, etype_in, minsm_in = sys.argv[1:11]
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
if weights_in:
    from scripts.lib.profiles.deriver import gguf_facts_from_file
    facts = gguf_facts_from_file(weights_in) if weights_in.endswith(".gguf") else None
    if facts is None and weights_in.endswith(".json"):
        cfg = json.loads(Path(weights_in).read_text(encoding="utf-8"))
        facts = {"hidden_size": cfg.get("hidden_size"),
                 "num_hidden_layers": cfg.get("num_hidden_layers"),
                 "num_attn_heads": cfg.get("num_attention_heads"),
                 "num_kv_heads": cfg.get("num_key_value_heads"),
                 "max_ctx_supported": cfg.get("max_position_embeddings"),
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
_ARCH_REQUIRED = ("hidden_size", "num_hidden_layers", "num_attn_heads",
                  "num_kv_heads", "max_ctx_supported", "attention_k_eq_v")
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


spec = {
    "model_id": mid,
    "display_name": mid,
    "family": "dense",
    "weights": {"local": {"path": f.model_path, "local_subdir": mid, "size_gb": 1.0,
                          "format": fmt, "status": "incubating", "hf_repo": "",
                          "engine": engine, "kind": "main",
                          "verify_glob": "*.gguf" if fmt == "gguf" else "*.safetensors"}},
    "arch": arch,
    "default_weight_variant": "local",
    "vision_capable": False,
    "compose": {"path": cpath, "content": text},
    "registry_entry": {"slug": slug, "kwargs": {
        "model": mid, "weights_variant": "local", "workload": workload,
        "engine": engine, "drafter": None,
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
  *) die "unknown subcommand: $SUB (expected register | unregister)" ;;
esac
