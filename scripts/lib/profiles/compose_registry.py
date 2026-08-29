"""Loader + API surface for the compose registry (DATA lives in registry.yaml).

The curated catalog is data, not code: the ~100 slug entries and the policy
maps (defaults / engine preference / recommended models) live in
scripts/lib/profiles/registry.yaml, next to this file. This module stays the
single import surface — consumers keep importing COMPOSE_REGISTRY, DEFAULTS,
ENGINE_PREFERENCE, RECOMMENDED_DEFAULT_MODELS, get_registry() and
load_local_registry() exactly as they always did.

The registry intentionally mirrors the shipped compose files. It is not a
generator and it does not attempt to normalize away historical variants.
"""

import json
from pathlib import Path

# Slug lifecycle / availability statuses — the canonical health flag.
#
# These are the registry-side equivalent of the compose `Status:` header enum
# (see the repo CLAUDE.md "Status enum" table). The compose-header emoji maps
# to one of these words; the drift-guard test asserts the two never diverge.
#
#   functional → launches normally (production) or with a one-line notice
#                (caveats).
#   (NA)       → surfaced in --list but not reliable: launch warns and requires
#                --force so a user can't *unknowingly* boot a broken slug.
STATUS_VALUES = (
    "production",      # ✅ Production — recommended, fully validated.
    "caveats",         # ⚠️ Production w/ caveats — works under documented limits.
    "experimental",    # 🧪 Experimental — under active validation; may not boot.
    "incubating",      # 🐣 Incubating — pre-experimental: works but not ready for the
                       #    actionable list (niche / fails the standard gate by design).
                       #    HIDDEN from `switch.sh --list` by default; revealed by `--all`.
    "preview",         # 👁️ Preview — known quality issues; tracked, not for prod.
    "upstream-gated",  # ⏸️ Upstream-gated — blocked by external action (pin/PR/HW).
    "deprecated",      # 🗑️ Deprecated — kept for reference; flagged for removal.
)

# Statuses that launch without --force. Everything else is "(NA)".
FUNCTIONAL_STATUSES = frozenset({"production", "caveats"})

# Compose `Status:` header emoji → registry status word. The header may carry
# trailing prose after the canonical token (e.g. "✅ Production (NEW — ...)");
# matching is by the leading emoji, so prose is tolerated.
COMPOSE_STATUS_EMOJI = {
    "✅": "production",
    "⚠️": "caveats",
    "🧪": "experimental",
    "🐣": "incubating",
    "👁️": "preview",
    "⏸️": "upstream-gated",
    "🗑️": "deprecated",
}


def _entry(
    *,
    model,
    weights_variant,
    workload,
    engine,
    drafter,
    kv_format,
    # Activation compute format (the A in W4A16/W4A8/W8A8). Default "16bit" —
    # engine-native half precision (fp16/bf16 per the compose --dtype); a slug
    # that dynamic-quantizes activations sets "int8" (W4A8/W8A8 class) or "fp8".
    # Surfaced as the c3 catalog "act" column (#723).
    act_format="16bit",
    # True when the slug's compose is wired for the W4A8 int8-activation knob AND
    # its weights are positive-symmetric int4 (the c3 serve-confirm checkbox reads
    # this to offer VLLM_MARLIN_INPUT_DTYPE=int8 per-launch). #609.
    act8_capable=False,
    # Weight-offload backend when the slug serves a model too large to fit VRAM
    # by paging expert/layer weights to host RAM. None = fully resident (the
    # default — every existing slug). "uva" = vLLM zero-copy demand-paged expert
    # offload (GPU computes, weights stream over PCIe); "n-cpu-moe" = llama.cpp
    # CPU-computed expert offload (weights stay in RAM, only activations cross
    # PCIe); "prefetch" = vLLM bulk layer prefetch. Surfaced as the c3 catalog
    # "offload" column. First used by the Laguna 118B-MoE offload slugs.
    offload=None,
    # True when the compose runs an adaptive expert cache (llama.cpp `--moe-cache`):
    # hot CPU-resident experts are held in spare VRAM and served from there instead
    # of over PCIe. Only meaningful alongside an `offload` backend — the cache exists
    # to soften the miss path that offload creates.
    #
    # Load-bearing, not documentation: the launch-compat layer gates its
    # MOE_RESERVE_MB injection on this flag (`_moe_cache_env`), so a compose that
    # runs the cache without declaring it silently keeps a reserve derived on a
    # 24 GB card no matter what the detected hardware is. `--moe-cache auto` grants
    # free-minus-reserve, and on the reference rig a reserve 512 MiB too small cost
    # ~11% throughput while REPORTING a bigger pool and a higher hit rate.
    moe_cache=False,
    # Minimum HOST RAM in GB for a weight-offload slug — the worst case (all experts
    # on CPU). This is a HARD GATE, not a recommendation: below it the box thrashes or
    # OOMs, and preflight_cpu_offload_ram() REFUSES. Surfaced as the c3 catalog
    # "host RAM" column so a user sees it BEFORE selecting a slug, rather than
    # discovering it at launch refusal. None = fully VRAM-resident, nothing to warn about.
    host_ram_gb=None,
    chat_template="native",
    tp,
    max_ctx,
    max_num_seqs,
    mem_util,
    compose_path,
    default_port,
    kvcalc_key=None,
    requires_nvlink=False,
    # True when the slug has an ARCH-GATED kernel path that torch.compile (or a
    # hardcoded quant-method kernel) emits PER RANK, so a mixed-compute-capability
    # TP group must be REFUSED rather than warned. The canonical case is a
    # quantization method that bypasses the shared kernel selector and therefore
    # has no per-rank fallback to reach — nvidia modelopt NVFP4 hardcodes
    # FlashInferFP8ScaledMMLinearKernel for its FP8 attention layers, so a
    # sub-sm_90 rank in the group dies even though `fallback_sm` says that card
    # is individually fine (fallback_sm reasons per card; the weight-only
    # fallback is a property of the whole TP GROUP).
    #
    # ⚠️ NOT every NVFP4 slug: compressed-tensors exports (unsloth `nvfp4-fast`,
    # migtissera tess) route through init_fp8_linear_kernel() -> shared selector,
    # where Marlin IS reachable — mixed-arch is solvable there and gating them
    # would foreclose it (validated on DiffusionGemma, disc #768 Test 6).
    # The axis is the QUANT RUNTIME, not the weights_variant string.
    #
    # Mirrored by `# Requires-homogeneous-arch: true` in the compose header,
    # which is what preflight.sh actually reads. `test-homogeneous-arch-drift`
    # asserts the two agree in BOTH directions — #762 shipped the guard on only
    # the reported slug and the 35B-A3B sibling silently kept crash-looping
    # (#783). TP=1 slugs never need this: there is no TP group to disagree.
    requires_homogeneous_arch=False,
    required_engine_features=None,
    recommended_engine_features=None,
    required_sm=None,
    fallback_sm=None,
    default_arch_allow=None,
    status="production",
    status_note=None,
    # Data-first override for the emitted served_name fact: when set, the
    # --json emit reports THIS as the slug's OpenAI-API --served-model-name
    # instead of parsing the compose file (parse stays the fallback). Leave
    # None for existing entries — the compose remains the source of truth.
    served_name=None,
    # True when this slug is the model's LAN-gateway scene: scripts/lib/
    # litellm-emit.sh derives its route in services/litellm/config.yaml from
    # THIS entry (#1078). Eligibility is EXPLICIT, never inferred — a slug is
    # gateway-flagged by PR only when its port is the one gateway clients
    # should hit (the curated-default walk among gateway=True entries picks
    # the canonical one per model; see litellm-emit.sh). Default False: an
    # unflagged catalog never leaks onto the gateway.
    gateway=False,
    # Extra OpenAI served-names this slug answers to BEYOND its primary
    # served_name — each is emitted as an additional gateway model_name on the
    # SAME route (same upstream port), not as a separate backend (#1073).
    # Mechanism only today: no core entry sets aliases yet; the synthetic
    # fixture in test-litellm-generate.sh covers the emission. Order matters
    # (it is the emitted route order after the primary name).
    serve_aliases=(),
    # Per-reasoning-mode sampler rows from the MODEL CARD (#984/#1014), for
    # models that publish one row per mode: {"instruct": {...}, "thinking":
    # {...}} with temperature/top_p/top_k/min_p/presence_penalty/
    # repetition_penalty. None (the default) = single-row model, the compose's
    # static sampler is the only truth. When set, the compose entrypoint derives
    # its shipped default from the INSTRUCT row and flips to THINKING on
    # ENABLE_THINKING=true; test-compose-sampler-profiles.sh asserts the compose
    # can never drift from this data. Adding a key here is the allowlist gate:
    # local-layer rows go through _entry(**kwargs) too, so an unknown sampler
    # kwarg fails loudly (LocalRegistryError), never silently.
    sampler_profiles=None,
    # Speculation that needs NO drafter GGUF (the ngram-* runtime spec-types).
    # Those slugs keep `drafter=None` by design, so consumers deriving a label
    # from `drafter` alone would report 'no speculation' while it is running.
    spec_method=None,
    category=None,
    weights_companions=None,
):
    if status not in STATUS_VALUES:
        raise ValueError(
            f"{compose_path}: status={status!r} not in {STATUS_VALUES}"
        )
    entry = {
        "model": model,
        "weights_variant": weights_variant,
        "workload": workload,
        "engine": engine,
        "drafter": drafter,
        "spec_method": spec_method,
        "kv_format": kv_format,
        "act_format": act_format,
        "act8_capable": act8_capable,
        "offload": offload,
        "moe_cache": bool(moe_cache),
        "host_ram_gb": host_ram_gb,
        "chat_template": chat_template,
        "tp": tp,
        "pp": 1,
        "max_ctx": max_ctx,
        "max_num_seqs": max_num_seqs,
        "mem_util": mem_util,
        "compose_path": compose_path,
        "requires_nvlink": requires_nvlink,
        "requires_homogeneous_arch": requires_homogeneous_arch,
        "required_engine_features": list(required_engine_features or []),
        "default_port": default_port,
        "gpu_assignment_mode": "contiguous",
        "kvcalc_key": kvcalc_key,
        "status": status,
        "status_note": status_note,
        # Extra weight-variant keys (a DFlash draft / mmproj projector) this slug's
        # compose mounts from a separate subdir, BEYOND the core weights_variant.
        # The serve-cockpit Download action fetches these alongside the core so the
        # slug actually serves.  Bare keys, scoped to this entry's model.
        "weights_companions": list(weights_companions or []),
        # LAN-gateway eligibility (#1078) + extra served-names on the same
        # route (#1073). Always present (like moe_cache) so consumers can
        # .get() without shape forks between core and local entries.
        "gateway": bool(gateway),
        "serve_aliases": list(serve_aliases),
    }
    if recommended_engine_features:
        entry["recommended_engine_features"] = list(recommended_engine_features)
    if required_sm is not None:
        entry["required_sm"] = required_sm
    if fallback_sm is not None:
        # Weight-only fallback floor. required_sm = the NATIVE-kernel SM;
        # fallback_sm = the lowest SM where the format still RUNS via a
        # weight-only fallback kernel (e.g. NVFP4 -> Marlin W4A16, floor
        # sm 7.5 per vLLM marlin_utils_fp4). In the band
        # [fallback_sm, required_sm) the gates ALLOW with a fallback
        # annotation instead of refusing (kv-calc emits `hw_fallback`;
        # c3 shows the slug with a ⚑ badge instead of hiding it).
        # Live-confirmed on 2x3090 sm_86 2026-07-11: NVFP4-27B boots,
        # 69.7/85.5 TPS, 8-pack 110/150 (ties the fp8 tier's 109).
        entry["fallback_sm"] = fallback_sm
    if default_arch_allow is not None:
        # GPU arches (compute-cap strings, e.g. "8.6") on which this slug is
        # validated enough to be an AUTO-DEFAULT. When set, the curated-default
        # walk skips it on any OTHER detected arch (see default_arch_gated).
        # This gates only the *default* — an explicit selection / user pin still
        # launches it (with a warning). #693: beellama DFlash returns gibberish
        # on Ada/sm_8.9; we only ever validated it on sm_8.6.
        entry["default_arch_allow"] = list(default_arch_allow)
    if category is not None:
        entry["category"] = category
    if served_name is not None:
        entry["served_name"] = served_name
    if sampler_profiles is not None:
        entry["sampler_profiles"] = {
            mode: dict(row) for mode, row in sampler_profiles.items()
        }
    return entry


def compose_header_status(text):
    """Map a compose file's profile-schema `Status:` header to a status word.

    Reads ONLY the `Status:` line inside the leading `# Profile (at-a-glance):`
    comment block (the structured schema), stopping at the `# ---` separator so
    a free-form `# Status: ...` prose line further down can't be mistaken for it.
    Returns the status word (one of STATUS_VALUES) or None if no canonical
    emoji is found. Matching is by the leading enum emoji, so trailing prose
    after the canonical token (e.g. "✅ Production (NEW — ...)") is tolerated.
    """
    in_schema = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Profile (at-a-glance):"):
            in_schema = True
            continue
        if not in_schema:
            continue
        # The schema block ends at the dashed separator line.
        if stripped.startswith("# --") or stripped.startswith("#--"):
            break
        # Match "#   Status:    <emoji> ..." within the schema block.
        body = stripped.lstrip("#").strip()
        if body.startswith("Status:"):
            value = body[len("Status:"):].strip()
            for emoji, word in COMPOSE_STATUS_EMOJI.items():
                if value.startswith(emoji):
                    return word
            return None
    return None

# --- Core catalog DATA: scripts/lib/profiles/registry.yaml -------------------
#
# The curated catalog is DATA, not code. The entries and the policy maps live
# in registry.yaml next to this file; everything below wires that file into
# the module-level names every consumer imports. The old dict-literal source
# was migrated mechanically (scripts/lib/profiles/migrate_registry_to_yaml.py)
# with a proven round-trip — no entry was retyped by hand.
#
# ⚠️ registry.yaml is NOT a profile YAML: the strict profile-schema loader
# (compat.load_profiles, UnknownProfileKeyError) governs models/*.yml,
# engines/*.yml and hardware/*.yml. This file has its OWN tiny schema and its
# own reader, and PyYAML is deliberately NEVER imported here — the launcher
# table path must stay python-STDLIB-ONLY (#584: community VMs ship bare
# python3; guarded by test-registry-emit-no-yaml + test-registry-yaml-roundtrip).
#
# Schema (schema: 1):
#   entries: {slug: {_entry kwargs}}               — wrapped through _entry()
#   defaults: {model: {engine: {topology: slug}}}   — flattened to tuple keys
#   engine_preference: {topology-family: [engine, …]}
#   recommended_default_models: [model-id, …]
#
# Edit by PR. promote.py --layer core appends entries mechanically (whole-file
# canonical rewrite, no source anchoring). migrate_registry_to_yaml.py --check
# re-verifies any hand edit still round-trips.

_REGISTRY_YAML_REL = "scripts/lib/profiles/registry.yaml"
_REGISTRY_YAML_SCHEMA = 1

# Built-entry keys that _entry() DERIVES (never kwargs): stripped before entry
# kwargs are written back to YAML — the migrator and promote.py share this.
_DERIVED_ENTRY_KEYS = frozenset({"pp", "gpu_assignment_mode"})

# Top-level keys the schema allows — anything else is a loud error, not a
# silently ignored section.
_REGISTRY_TOP_KEYS = frozenset(
    {"schema", "entries", "defaults", "engine_preference", "recommended_default_models"}
)


class RegistryDataError(Exception):
    """registry.yaml is present but unusable (bad YAML / schema / entry kwargs).

    Raised LOUDLY at import time — a broken core catalog must never silently
    shrink or shadow the catalog (same philosophy as LocalRegistryError for
    the gitignored local layer)."""


# The emitter quotes anything on this list (or number/bool/null-shaped, or
# containing YAML structure characters) as a JSON-style double-quoted scalar
# — JSON string escapes are a valid YAML double-quoted scalar, so the reader
# below can hand those tokens straight to json.loads.
_YAML_UNSAFE_FIRST = "-?:,[]{}#&*!|>'\"%@`"
_YAML_BOOLISH = {"null", "~", "true", "false", "yes", "no", "on", "off"}


def _yaml_needs_quote(s):
    """True when the string must be quoted to survive the round-trip."""
    if s == "" or s != s.strip() or "\n" in s:
        return True
    if s[0] in _YAML_UNSAFE_FIRST:
        return True
    if s.lower() in _YAML_BOOLISH:
        return True
    try:
        float(s)
        return True
    except ValueError:
        pass
    return any(c in s for c in ":#{}[],\t")


def _yaml_scalar(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return json.dumps(s, ensure_ascii=False) if _yaml_needs_quote(s) else s


def _yaml_key(k):
    s = str(k)
    return json.dumps(s, ensure_ascii=False) if _yaml_needs_quote(s) else s


_REGISTRY_YAML_HEADER = """\
# ===========================================================================
# club-3090 compose registry — CORE CATALOG DATA
#
# Canonical, generated form: the catalog lives HERE, not in Python source.
# compose_registry.py is the loader + API surface (COMPOSE_REGISTRY, DEFAULTS,
# ENGINE_PREFERENCE, RECOMMENDED_DEFAULT_MODELS, get_registry()).
#
#   * Edit by PR. `promote.py --layer core` appends entries mechanically.
#     After ANY hand edit run:
#       python3 scripts/lib/profiles/migrate_registry_to_yaml.py --check
#     to prove the file still round-trips through the loader.
#   * NOT a profile YAML — the strict profile-schema loader
#     (compat.load_profiles / UnknownProfileKeyError) does not apply to this
#     file; it has its own stdlib-only reader (#584: no PyYAML dependency).
#   * Entry rows are `_entry(**kwargs)` argument maps — the same schema as
#     profiles-local/registry.local.json. `pp` / `gpu_assignment_mode` are
#     derived by _entry() and MUST NOT appear here.
# ===========================================================================
"""


def nest_defaults(defaults):
    """{(model, engine, topology): slug} → {model: {engine: {topology: slug}}}.

    YAML has no tuple keys; the DEFAULTS map is stored nested and flattened
    back at load. Insertion order is preserved both ways."""
    nested = {}
    for (model, engine, topology), slug in defaults.items():
        nested.setdefault(model, {}).setdefault(engine, {})[topology] = slug
    return nested


def _dump_node(lines, key, value, indent):
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            lines.append(f"{pad}{_yaml_key(key)}: {{}}")
            return
        lines.append(f"{pad}{_yaml_key(key)}:")
        for k, v in value.items():
            _dump_node(lines, k, v, indent + 2)
    elif isinstance(value, list):
        if not value:
            lines.append(f"{pad}{_yaml_key(key)}: []")
            return
        lines.append(f"{pad}{_yaml_key(key)}:")
        for item in value:
            if isinstance(item, (dict, list)):
                raise TypeError(
                    f"registry YAML schema: lists of {type(item).__name__} "
                    f"are not supported (field {key!r})"
                )
            lines.append(f"{pad}  - {_yaml_scalar(item)}")
    else:
        lines.append(f"{pad}{_yaml_key(key)}: {_yaml_scalar(value)}")


def dump_registry_yaml(data):
    """Registry DATA dict → canonical YAML text (deterministic, insertion order).

    The ONE writer for registry.yaml, shared by the one-shot migrator and
    promote.py's core path — so the stdlib-only reader can never drift from
    what gets written. Stdlib-only by design (#584)."""
    lines = [_REGISTRY_YAML_HEADER, f"schema: {_REGISTRY_YAML_SCHEMA}", "", "entries:"]
    for slug, kwargs in data["entries"].items():
        lines.append(f"  {_yaml_key(slug)}:")
        for k, v in kwargs.items():
            _dump_node(lines, k, v, 4)
    lines += ["", "defaults:"]
    for model, engines in data["defaults"].items():
        lines.append(f"  {_yaml_key(model)}:")
        for engine, topos in engines.items():
            lines.append(f"    {_yaml_key(engine)}:")
            for topo, slug in topos.items():
                lines.append(f"      {_yaml_key(topo)}: {_yaml_scalar(slug)}")
    lines += ["", "engine_preference:"]
    for fam, ranked in data["engine_preference"].items():
        lines.append(
            f"  {_yaml_key(fam)}: [{', '.join(_yaml_scalar(e) for e in ranked)}]"
        )
    lines.append(
        "recommended_default_models: ["
        + ", ".join(_yaml_scalar(m) for m in data["recommended_default_models"])
        + "]"
    )
    return "\n".join(lines) + "\n"


def _yaml_parse_scalar(tok, source, lineno):
    t = tok.strip()
    if t.startswith('"'):
        try:
            return json.loads(t)
        except ValueError as exc:
            raise RegistryDataError(
                f"{source}:{lineno}: bad quoted scalar {tok!r}: {exc}"
            ) from exc
    if t.startswith("'"):
        if len(t) < 2 or not t.endswith("'"):
            raise RegistryDataError(f"{source}:{lineno}: unterminated quote {tok!r}")
        return t[1:-1].replace("''", "'")
    if " #" in t:  # trailing comment on an unquoted scalar
        t = t.split(" #", 1)[0].rstrip()
    if t in ("", "~", "null", "Null", "NULL"):
        return None
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _yaml_split_flow(inner):
    """Split a flow-sequence body on top-level commas (quote-aware)."""
    parts, buf, in_q, esc = [], [], False, False
    for ch in inner:
        if in_q:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_q = False
        elif ch == '"':
            in_q = True
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _yaml_parse_flow_seq(tok, source, lineno):
    t = tok.strip()
    if t == "[]":
        return []
    if not (t.startswith("[") and t.endswith("]")):
        raise RegistryDataError(
            f"{source}:{lineno}: unsupported value {tok!r} "
            "(expected a scalar, [] or a single-line [a, b] list)"
        )
    inner = t[1:-1].strip()
    if not inner:
        return []
    return [_yaml_parse_scalar(p, source, lineno) for p in _yaml_split_flow(inner)]


def parse_registry_text(text, source="registry.yaml"):
    """Parse registry-DATA YAML (exactly the subset dump_registry_yaml emits).

    A deliberate, tiny YAML subset — nested block maps, scalar block
    sequences, single-line flow lists, quoted/plain scalars — parsed with a
    ~100-line recursive reader instead of PyYAML so the launcher table path
    stays dependency-free (#584). Anything outside the subset raises
    RegistryDataError loudly; this parser NEVER guesses."""
    toks = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent_part = line[: len(line) - len(line.lstrip())]
        if "\t" in indent_part:
            raise RegistryDataError(f"{source}:{lineno}: tab indentation is not allowed")
        toks.append((lineno, len(indent_part), stripped))

    pos = 0

    def parse_node(indent):
        lineno, ind, content = toks[pos]
        if content.startswith("- ") or content == "-":
            return parse_seq(ind)
        return parse_map(ind)

    def parse_seq(indent):
        nonlocal pos
        items = []
        while pos < len(toks):
            lineno, ind, content = toks[pos]
            if not (content.startswith("- ") or content == "-") or ind < indent:
                break
            if ind > indent:
                raise RegistryDataError(f"{source}:{lineno}: bad sequence indent")
            item = "" if content == "-" else content[2:]
            items.append(_yaml_parse_scalar(item, source, lineno))
            pos += 1
        return items

    def parse_kv(content, lineno):
        if content.startswith('"'):
            i, n = 1, len(content)
            while i < n and content[i] != '"':
                i += 2 if content[i] == "\\" else 1
            if i >= n:
                raise RegistryDataError(
                    f"{source}:{lineno}: unterminated quoted key {content!r}"
                )
            key = json.loads(content[: i + 1])
            rest = content[i + 1:]
            if not rest.startswith(":"):
                raise RegistryDataError(
                    f"{source}:{lineno}: expected 'key: value', got {content!r}"
                )
            return key, rest[1:].strip()
        ci = content.find(":")
        if ci <= 0:
            raise RegistryDataError(
                f"{source}:{lineno}: expected 'key: value', got {content!r}"
            )
        return content[:ci].rstrip(), content[ci + 1:].strip()

    def parse_map(indent):
        nonlocal pos
        out = {}
        while pos < len(toks):
            lineno, ind, content = toks[pos]
            if ind < indent or content.startswith("- ") or content == "-":
                break
            if ind > indent:
                raise RegistryDataError(f"{source}:{lineno}: bad map indent")
            key, rest = parse_kv(content, lineno)
            pos += 1
            if rest == "":
                if pos < len(toks) and toks[pos][1] > indent:
                    out[key] = parse_node(toks[pos][1])
                else:
                    out[key] = None
            elif rest.startswith("["):
                out[key] = _yaml_parse_flow_seq(rest, source, lineno)
            elif rest == "{}":
                out[key] = {}
            else:
                out[key] = _yaml_parse_scalar(rest, source, lineno)
        return out

    if not toks:
        raise RegistryDataError(f"{source}: file is empty")
    data = parse_map(0)
    if pos != len(toks):
        raise RegistryDataError(
            f"{source}:{toks[pos][0]}: unexpected content {toks[pos][2]!r}"
        )
    if not isinstance(data, dict):
        raise RegistryDataError(f"{source}: top level must be a mapping")
    return data


def load_registry_data(path=None):
    """Read a registry DATA file → validated raw dict (entries as _entry kwargs).

    Used by this module at import AND by promote.py --layer core, which passes
    an explicit path under its --root. Raises RegistryDataError on anything
    unusable — a half-working catalog is worse than a loud failure."""
    path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent / "registry.yaml"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryDataError(f"{path}: unreadable: {exc}") from exc
    data = parse_registry_text(text, source=str(path))
    if set(data) != _REGISTRY_TOP_KEYS:
        raise RegistryDataError(
            f"{path}: top-level keys must be exactly {sorted(_REGISTRY_TOP_KEYS)}, "
            f"got {sorted(data)}"
        )
    if data["schema"] != _REGISTRY_YAML_SCHEMA:
        raise RegistryDataError(
            f"{path}: unsupported schema {data['schema']!r} "
            f"(expected {_REGISTRY_YAML_SCHEMA})"
        )
    entries = data["entries"]
    if not isinstance(entries, dict) or not all(
        isinstance(v, dict) for v in entries.values()
    ):
        raise RegistryDataError(f"{path}: 'entries' must be {{slug: {{kwargs}}}}")
    for slug, kwargs in entries.items():
        derived = _DERIVED_ENTRY_KEYS.intersection(kwargs)
        if derived:
            raise RegistryDataError(
                f"{path}: entry {slug!r}: {sorted(derived)} are derived by "
                "_entry() and must not appear in the YAML"
            )
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)) and k != "sampler_profiles" and not (
                isinstance(v, list)
            ):
                raise RegistryDataError(
                    f"{path}: entry {slug!r}: field {k!r} has an unsupported type "
                    f"{type(v).__name__}"
                )
    defaults = data["defaults"]
    if not isinstance(defaults, dict):
        raise RegistryDataError(f"{path}: 'defaults' must be a nested mapping")
    for model, engines in defaults.items():
        if not isinstance(engines, dict):
            raise RegistryDataError(f"{path}: defaults[{model!r}] must be a mapping")
        for engine, topos in engines.items():
            if not isinstance(topos, dict):
                raise RegistryDataError(
                    f"{path}: defaults[{model!r}][{engine!r}] must be a mapping"
                )
    pref = data["engine_preference"]
    if not isinstance(pref, dict) or not all(isinstance(v, list) for v in pref.values()):
        raise RegistryDataError(
            f"{path}: 'engine_preference' must be {{family: [engine, …]}}"
        )
    if not isinstance(data["recommended_default_models"], list):
        raise RegistryDataError(f"{path}: 'recommended_default_models' must be a list")
    return data


def _build_core_catalog(data, source):
    """Validated raw data → (COMPOSE_REGISTRY, DEFAULTS).

    Every row is wrapped through _entry(**kwargs) so a YAML row carries
    EXACTLY the shape (and validation) the pre-migration Python rows had —
    an unknown kwarg or bad status fails the import loudly, never silently."""
    entries = {}
    for slug, kwargs in data["entries"].items():
        try:
            entries[slug] = _entry(**kwargs)
        except TypeError as exc:
            raise RegistryDataError(
                f"{source}: entry {slug!r}: bad _entry kwargs: {exc}"
            ) from exc
        except ValueError as exc:
            raise RegistryDataError(f"{source}: entry {slug!r}: {exc}") from exc
    defaults = {}
    for model, engines in data["defaults"].items():
        for engine, topology_rows in engines.items():
            for topology, slug in topology_rows.items():
                defaults[(model, engine, topology)] = slug
    return entries, defaults


def _load_core_registry(path=None):
    """registry.yaml → (entries, defaults, engine_preference, recommended)."""
    path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent / "registry.yaml"
    )
    data = load_registry_data(path)
    entries, defaults = _build_core_catalog(data, str(path))
    return entries, defaults, data["engine_preference"], data["recommended_default_models"]

_CORE_ENTRIES, _CORE_DEFAULTS, _CORE_ENGINE_PREFERENCE, _CORE_RECOMMENDED_MODELS = (
    _load_core_registry()
)
COMPOSE_REGISTRY = _CORE_ENTRIES
DEFAULTS = _CORE_DEFAULTS
ENGINE_PREFERENCE = _CORE_ENGINE_PREFERENCE
RECOMMENDED_DEFAULT_MODELS = _CORE_RECOMMENDED_MODELS

# The per-mode sampler rows that used to be the QWEN38_27B_SAMPLER_PROFILES
# module constant now live in registry.yaml alongside the entries that
# reference them (same data, one home).



# --- C4-rev: the gitignored LOCAL layer (community-added models) --------------
#
# Community users add models WITHOUT touching any core catalog file. A local
# model lives under scripts/lib/profiles-local/ (gitignored except its README):
#   models.d/<id>.yml        ModelProfile YAML (same schema as models/)
#   composes/<id>/...        the compose files
#   registry.local.json      plain-dict registry entries ({slug: _entry-kwargs})
#
# get_registry() below is the SINGLE merged view every runtime consumer reads
# (launch / switch / diagnose / emit / estate / cockpit). The local layer can
# NEVER leak into DEFAULTS / ENGINE_PREFERENCE / RECOMMENDED_DEFAULT_MODELS —
# those stay core-only, so a local entry can never become a curated default.

LOCAL_LAYER_DIR_REL = "scripts/lib/profiles-local"
LOCAL_REGISTRY_REL = "scripts/lib/profiles-local/registry.local.json"
# The slug namespace every LOCAL entry lives under (promote.py --layer local
# enforces it at write time; the loader refuses anything else).
LOCAL_SLUG_PREFIX = "local/"


class LocalRegistryError(Exception):
    """The local layer exists but is unusable (bad JSON / shape / collision).

    Raised LOUDLY on purpose: a broken local layer must never silently shrink
    or shadow the curated catalog — launchers surface the error and exit."""


def local_layer_root():
    """The repo root this registry module was imported FROM.

    Resolved from __file__ so a THROWAWAY copy of the repo (promote.py --root
    / tests / a user checkout) sees ITS OWN local layer, not another's."""
    return Path(__file__).resolve().parents[3]


def load_local_registry(root=None):
    """Read registry.local.json → {slug: entry-dict} ({} when no local layer).

    Plain-dict entries (the JSON form of _entry kwargs) are wrapped through
    _entry(**kwargs) so a local row carries EXACTLY the same shape + defaults
    as a core row. ANY problem — unreadable JSON, a non-object, an unknown
    kwarg, a bad status, a slug outside the `local/` namespace, or a slug /
    model-id colliding with a core row (or another local row) — raises
    LocalRegistryError. Loud, never a silent partial catalog."""
    root = Path(root) if root is not None else local_layer_root()
    path = root / LOCAL_REGISTRY_REL
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalRegistryError(
            f"{path}: unreadable registry.local.json: {exc}"
        ) from exc
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise LocalRegistryError(
            f"{path}: expected a JSON object of {{slug: {{_entry kwargs}}}}"
        )
    core_slugs = set(COMPOSE_REGISTRY)
    core_models = {e.get("model") for e in COMPOSE_REGISTRY.values()}
    local: dict = {}
    local_models: set = set()
    for slug, kwargs in raw.items():
        if not slug.startswith(LOCAL_SLUG_PREFIX):
            raise LocalRegistryError(
                f"{path}: local slug {slug!r} must live under the "
                f"{LOCAL_SLUG_PREFIX!r} namespace"
            )
        if slug in core_slugs or slug in local:
            raise LocalRegistryError(f"{path}: local slug collides: {slug!r}")
        model = kwargs.get("model")
        if model in core_models:
            raise LocalRegistryError(
                f"{path}: local model id {model!r} collides with a core model"
            )
        if model in local_models:
            raise LocalRegistryError(f"{path}: duplicate local model id {model!r}")
        try:
            entry = _entry(**kwargs)
        except TypeError as exc:
            raise LocalRegistryError(
                f"{path}: local entry {slug!r} has bad _entry kwargs: {exc}"
            ) from exc
        except ValueError as exc:
            raise LocalRegistryError(f"{path}: local entry {slug!r}: {exc}") from exc
        local[slug] = entry
        local_models.add(model)
    return local


def get_registry(root=None):
    """The SINGLE merged catalog view: core COMPOSE_REGISTRY + the local layer.

    With no local layer present this IS COMPOSE_REGISTRY (the identical object
    — zero behavior change on a pristine checkout). Slug lookups at runtime
    read THIS; DEFAULTS / ENGINE_PREFERENCE / RECOMMENDED_DEFAULT_MODELS stay
    core-only. Raises LocalRegistryError when a present local layer is broken —
    a half-working catalog is worse than a loud failure."""
    local = load_local_registry(root)
    if not local:
        return COMPOSE_REGISTRY
    merged = dict(COMPOSE_REGISTRY)
    merged.update(local)
    return merged


# --- PR-B: model-default resolver knobs (maintainer-owned, design §13.3) ----
#
# RECOMMENDED_DEFAULT_MODELS and ENGINE_PREFERENCE are DATA now — they live in
# scripts/lib/profiles/registry.yaml (`recommended_default_models` /
# `engine_preference`) and are wired to the module names above. Maintainer
# knobs, edited by PR, never auto-grown; see docs/model-default-resolver
# design + the repo AGENTS.md "Default rule" note. History that used to sit on
# these literals: qwen3.6-27b left the recommended list 2026-08-12 (all its
# llama.cpp/ik single-card slugs deprecated → its curated default degraded to
# vllm/minimal); `beellama` left every ENGINE_PREFERENCE walk 2026-07-27
# (engine retired) and `ik-llama` followed 2026-08-12 (no functional slug).
#
# DEFAULTS is flattened back from the YAML's nested {model: {engine:
# {topology: slug}}} map at import; tuple-key lookups work exactly as before.


def _topology_family(topology):
    """Map a concrete topology to its ENGINE_PREFERENCE family.

    Concrete topologies are `single` · `dual` · `multi4` · `multiN`; the
    preference table keys on the family `single` · `dual` · `multi`.
    """
    if topology == "single":
        return "single"
    if topology == "dual":
        return "dual"
    if topology.startswith("multi"):
        return "multi"
    return topology


def _nearest_lower_topology(topology):
    """Degradation order (design §6): notice + nearest-lower topology.

    multiN → dual → single → None. Returns the next topology to try, or None
    when there is nowhere lower to fall.
    """
    if topology.startswith("multi"):
        return "dual"
    if topology == "dual":
        return "single"
    return None


def engine_set():
    """The closed set of engine namespace-prefixes (DEFAULTS keys + ranked).

    `X/default` dispatch (design §13.1): `X ∈ engine_set` → engine
    recommendation; else `X ∈ model_set` → model default; else error.
    Engines and model-ids are disjoint by construction.
    """
    engines = set()
    for _model, engine, _topology in DEFAULTS:
        engines.add(engine)
    for ranked in ENGINE_PREFERENCE.values():
        engines.update(ranked)
    return engines


def model_set():
    """The set of model-ids that appear in DEFAULTS (the runnable catalog)."""
    return {model for (model, _engine, _topology) in DEFAULTS}


def default_arch_gated(slug, detected_sm):
    """True when `slug` may NOT auto-default on the detected GPU arch.

    A slug with a `default_arch_allow` list (compute-cap strings) is validated
    as a default only on those arches. FAIL-OPEN: no allow-list, or an
    unknown/empty `detected_sm`, never gates — so CI / headless / an
    undetectable GPU keep today's behavior. #693: beellama DFlash is validated
    only on sm_8.6 and returns gibberish on sm_8.9 (Ada), so its default slug
    carries default_arch_allow=["8.6"].
    """
    if not detected_sm:
        return False
    entry = get_registry().get(slug)
    if not entry:
        return False
    allow = entry.get("default_arch_allow")
    if not allow:
        return False
    return detected_sm not in allow


def _functional_default(model, engine, topology, detected_sm=None):
    """A DEFAULTS slug for (model, engine, topology) whose status is functional
    AND (when detected_sm is known) not arch-gated off the detected arch.

    Returns the slug only when an entry exists, its registry status is NOT in
    the (NA) set (experimental/preview/upstream-gated/deprecated), and it is not
    default_arch_gated for detected_sm — a broken/preview/off-arch config must
    never become someone's auto-default (§12.5, #693). Returns None otherwise.
    """
    slug = DEFAULTS.get((model, engine, topology))
    if not slug:
        return None
    entry = get_registry().get(slug)
    if entry is None:
        return None
    if entry.get("status", "production") not in FUNCTIONAL_STATUSES:
        return None
    if default_arch_gated(slug, detected_sm):
        return None
    return slug


def curated_default_target(model, topology, detected_sm=None):
    """Curated fallback (§4): walk ENGINE_PREFERENCE[family], first functional
    (and, for detected_sm, not-arch-gated) DEFAULTS slug wins. Returns the slug,
    or None if no functional curated default exists for (model, topology[, sm]).
    """
    family = _topology_family(topology)
    for engine in ENGINE_PREFERENCE.get(family, []):
        slug = _functional_default(model, engine, topology, detected_sm)
        if slug:
            return slug
    return None


def community_default_target(model, topology, hw_class=None):  # noqa: ARG001
    """Community-ranked best config — the FUTURE middle precedence rung (§13.4).

    Contract: returns a ranked slug when the submissions/ranking app exists;
    returns None today (always skipped). The resolver inserts a non-None result
    BETWEEN the user pin and the curated fallback. v1 ships this stub returning
    None so the ladder rung is real, not aspirational; a test asserts it is
    skipped.
    """
    return None


def _model_pin_suffix(model):
    """Shared `<MODELID uppercased, non-alnum→_>` suffix for the per-model .env
    pin keys, e.g. qwen3.6-27b → QWEN3_6_27B."""
    return "".join(c if c.isalnum() else "_" for c in model).upper()


def model_default_pin_key(model):
    """The .env key for a per-model user pin (design §13.2).

    `CLUB3090_DEFAULT_<MODELID uppercased, non-alnum→_>`, e.g.
    qwen3.6-27b → CLUB3090_DEFAULT_QWEN3_6_27B.
    """
    return f"CLUB3090_DEFAULT_{_model_pin_suffix(model)}"


def model_thinking_pin_key(model):
    """The .env key for a per-model THINKING pin (#1014 follow-up).

    Same normalization as :func:`model_default_pin_key`:
    `CLUB3090_THINKING_<MODELID uppercased, non-alnum→_>`, e.g.
    qwen3.6-27b → CLUB3090_THINKING_QWEN3_6_27B. Values are the tri-state
    vocabulary on|off|inherit; switch.sh resolves it into the launch env
    (on/off → ENABLE_THINKING=true/false, inherit → nothing).
    """
    return f"CLUB3090_THINKING_{_model_pin_suffix(model)}"


def model_of_slug(slug):
    """The model-id a slug belongs to, or None if the slug is unknown."""
    entry = get_registry().get(slug)
    return entry.get("model") if entry else None

def slug_topology(slug):
    """The topology family a slug serves, derived from its compose_path.

    compose_path is `models/<model>/<engine>/compose/<topology>/<quant>/...`.
    Returns `single`/`dual`/`multi` (the ENGINE_PREFERENCE family) or None.
    """
    entry = get_registry().get(slug)
    if not entry:
        return None
    cp = entry.get("compose_path", "")
    if "/compose/" not in cp:
        return None
    after = cp.split("/compose/", 1)[1]
    topo = after.split("/", 1)[0]
    return _topology_family(topo)
