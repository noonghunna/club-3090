"""ModelSpec — ONE typed, versioned, provenance-labeled schema for model
geometry (the "one vocabulary" bridge between the deriver, the c3 BYO funnel,
the promote scaffold and the strict profile loader).

Design: local://modelspec-proposal.md (ModelSpecDesign wave-3 proposal,
§1 shape / §2 field tables / §5 versioning).  M1–M3 slice:

  M1  this module: ``Fact`` + ``ModelSpec`` (stdlib-only: dataclasses/typing —
      NO PyYAML, NO compose_registry import; safe to import from the deriver's
      ``python3 -m`` subprocess and from every shell-tool path).
  M2  ``deriver.derive()`` results expose ``res.model_spec`` (built by
      :meth:`ModelSpec.from_derive_result`); the deriver CLI emits it via
      ``--spec-json``; the GGUF header path (gguf_spec_facts) is integrated as
      a SOURCE via :meth:`ModelSpec.from_gguf_facts`.
  M3  ``ByoResult.facts`` becomes a typed ``ModelSpec`` and
      ``compute_promote_scaffold`` reads attributes instead of ``.get()``
      chains (placeholder logic unchanged).

Design rules (proposal §1):
- Every machine-filled value carries PROVENANCE.  A fallback default (e.g.
  ``max_ctx_supported`` falling back to 131072 when config.json has no
  ``max_position_embeddings``) is labeled ``"fallback"`` — never silently
  presented as card truth.
- ModelSpec holds NO policy fields: ``display_name``, human-named ``family``,
  ``attention_k_eq_v``, ``requires_genesis``, ``compatible_drafters``,
  ``decode_granularity`` stay human/curated inputs.
- Additive-only evolution within ``spec_version: 1``; bump the version when a
  field CHANGES MEANING (§5).  ``from_dict`` rejects unknown major versions
  loudly (same discipline as the strict loader's SUPPORTED_SCHEMA_VERSIONS).
- Family-specific extension sets (MoE / hybrid-GDN / SWA extractors) are the
  M5 increments — they attach additively; nothing here pre-registers them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Optional

MODEL_SPEC_VERSION = 1

#: Provenance kinds.  "fallback" MUST be surfaced, never silently presented as
#: card truth; "human" marks maintainer-entered values (write-boundary concern,
#: proposal §5 rule 2).
Provenance = Literal[
    "derived", "derived-estimate", "fallback", "curated", "human"
]


@dataclass(frozen=True)
class Fact:
    """One machine-derived (or curated) value + where it came from."""

    value: Any
    provenance: Provenance
    source: str  # e.g. "config.json:num_key_value_heads", "gguf-header:llama.block_count"


@dataclass(frozen=True)
class MtpFacts:
    """Multi-token-prediction head presence (deriver ``detect_mtp_head``:
    config DECLARES ∧ repo SHIPS the head weights)."""

    has_head: bool


@dataclass(frozen=True)
class GgufFacts:
    """GGUF container metadata (header probe) — provenance for the dims that
    rode in from a header instead of a config.json."""

    version: Optional[int] = None
    general_name: Optional[str] = None
    file_type: Optional[int] = None
    quant_label: Optional[str] = None
    quantization_version: Optional[int] = None
    kv_heads_assumed_equal: bool = False
    head_count_variable: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """The typed geometry spec.  See the module docstring + proposal §2 for
    the field-by-field mapping into the ModelProfile YAML."""

    spec_version: int = MODEL_SPEC_VERSION
    model_slug: str = ""  # HF org/Repo — scaffold input, NOT the profile id
    family: Optional[Fact] = None  # None until human/curated names it
    arch: Optional[str] = None  # architectures[0] — provenance only, never a YAML key
    confidence: str = ""  # exact | estimated-lower-bound | not-generic-dense-eligible

    # ── core dims (Table A) — absent ⇒ scaffold placeholder <int> ────────────
    hidden_size: Optional[Fact] = None
    num_hidden_layers: Optional[Fact] = None
    num_attn_heads: Optional[Fact] = None
    num_kv_heads: Optional[Fact] = None
    head_dim_attn: Optional[Fact] = None
    max_ctx_supported: Optional[Fact] = None
    valid_tp: tuple[int, ...] = (1, 2)  # policy default — never fabricated beyond
    weights_total_gb: Optional[Fact] = None
    vision_capable: Optional[Fact] = None

    # ── weights / provenance (Table B) — feed weights.<variant>.* / gates ────
    weight_format: Optional[Fact] = None
    torch_dtype: Optional[Fact] = None  # engine --dtype provenance, no YAML key
    effective_bpw: Optional[Fact] = None  # sanity-check vs size_gb, no YAML key
    footprint_gb: Optional[Fact] = None  # [C2a] disk gate input, no YAML key
    selected_weight_files: Optional[Fact] = None  # verify_glob/shards/files hints

    # ── family-specific extension sets (Table C — M5 fills these) ────────────
    mtp: Optional[MtpFacts] = None
    gguf: Optional[GgufFacts] = None  # header-probe provenance (route-G)

    # ── behavior ─────────────────────────────────────────────────────────────
    def __bool__(self) -> bool:
        """Truthy iff ANY consumable fact is present — preserves the old
        ``ByoResult.facts`` dict-truthiness semantics (empty dict == no facts
        ⇒ the scaffold keeps its placeholders)."""
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Fact) and v.value is not None:
                return True
        return False

    def arch_dims(self) -> dict[str, int]:
        """The present core dims as a plain {name: value} map — exactly the
        keys the promote scaffold renders into the profile YAML."""
        out: dict[str, int] = {}
        for name in (
            "hidden_size", "num_hidden_layers", "num_attn_heads",
            "num_kv_heads", "head_dim_attn", "max_ctx_supported",
        ):
            f = getattr(self, name)
            if f is not None and f.value is not None:
                out[name] = f.value
        return out

    def fallback_dims(self) -> dict[str, int]:
        """Core dims whose value is a FALLBACK default (never card truth) —
        the write boundary (M4) renders these with an explicit YAML comment."""
        return {
            name: f.value
            for name in self.arch_dims()
            for f in [getattr(self, name)]
            if f.provenance == "fallback"
        }

    def validate(self) -> list[str]:
        """Structural checks (proposal §5 rule 1).  Returns human-readable
        issues; ``warn:``-prefixed entries are advisory (e.g. asymmetric
        architectures) and do NOT invalidate the spec."""
        issues: list[str] = []
        for name, v in self.arch_dims().items():
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                issues.append(f"{name}: expected a positive int, got {v!r}")
        kv, attn = self.num_kv_heads, self.num_attn_heads
        if (
            kv is not None and attn is not None
            and isinstance(kv.value, int) and isinstance(attn.value, int)
            and kv.value > attn.value
        ):
            issues.append(
                f"warn: num_kv_heads ({kv.value}) > num_attn_heads "
                f"({attn.value}) — legal for some architectures, verify"
            )
        if not self.valid_tp or not all(t in (1, 2) for t in self.valid_tp):
            issues.append(f"valid_tp: policy default is a subset of [1, 2], got {self.valid_tp!r}")
        return issues

    # ── serialization (the deriver subprocess boundary is JSON) ──────────────
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Fact):
                out[f.name] = {
                    "value": v.value,
                    "provenance": v.provenance,
                    "source": v.source,
                }
            elif isinstance(v, (MtpFacts, GgufFacts)):
                out[f.name] = {g: getattr(v, g) for g in
                               (x.name for x in fields(v))}
            elif isinstance(v, tuple):
                out[f.name] = list(v)
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        """Inverse of :meth:`to_dict`.  Rejects unknown MAJOR versions loudly
        (§5); within v1, unrecognized keys are tolerated (additive-only)."""
        if not isinstance(d, dict):
            raise ValueError(f"ModelSpec: expected a JSON object, got {type(d).__name__}")
        ver = d.get("spec_version")
        if ver != MODEL_SPEC_VERSION:
            raise ValueError(
                f"ModelSpec: unsupported spec_version {ver!r} (this reader speaks "
                f"v{MODEL_SPEC_VERSION})"
            )
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name == "valid_tp":
                kw[f.name] = tuple(v or ())
            elif isinstance(v, dict) and {"value", "provenance", "source"} <= set(v):
                kw[f.name] = Fact(v["value"], v["provenance"], v["source"])
            elif f.name == "mtp" and isinstance(v, dict):
                kw[f.name] = MtpFacts(**v)
            elif f.name == "gguf" and isinstance(v, dict):
                kw[f.name] = GgufFacts(**v)
            else:
                kw[f.name] = v
        return cls(**kw)

    # ── builders ─────────────────────────────────────────────────────────────
    @classmethod
    def from_derive_result(cls, res: Any) -> "ModelSpec":
        """Build the spec from a ``deriver.DeriveResult`` (duck-typed — this
        module imports NOTHING from the package, so the deriver subprocess and
        compat can both call it without a cycle).

        Provenance mapping (Table A):
          - generic-dense spec dims  → ``config.json:<key>``
          - head_dim computed        → ``config.json:hidden_size//num_attention_heads``
          - max_ctx 131072 default   → ``fallback`` / ``default:131072`` — NEVER
            presented as card truth
          - tier-1 curated hit       → ``curated`` (the curated profile is
            authoritative; the spec exists to make that explicit)
          - GGUF header dims do NOT flow through here — they are a separate
            source (:meth:`from_gguf_facts`).
        """
        slug = getattr(res, "slug", "") or ""
        conf = getattr(res, "confidence", None)
        confidence = getattr(conf, "value", conf) if conf is not None else ""
        if getattr(res, "error", None) is not None:
            # Stratum-1 structured error: nothing is known, nothing is invented.
            return cls(spec_version=MODEL_SPEC_VERSION, model_slug=slug, confidence=confidence)

        profile = getattr(res, "profile", None) or {}
        spec = getattr(res, "spec", None) or {}

        tier1 = getattr(res, "tier1", None)
        curated = tier1 is not None
        prov: Provenance = "curated" if curated else "derived"
        src_prefix = f"profiles:{profile['model_id']}" if curated else "config.json"

        def fact(value: Any, provenance: Provenance, source: str) -> Optional[Fact]:
            return None if value is None else Fact(value, provenance, source)

        def dim(name: str, config_key: str) -> Optional[Fact]:
            v = spec.get(name)
            if v is None:
                v = profile.get(f"config_{config_key}")
            src = (
                f"{src_prefix}:{config_key}" if curated
                else f"config.json:{config_key}"
            )
            return fact(v, prov, src)

        # head_dim: config.json may carry it directly; else the deriver computed
        # hidden // heads (only when divisible).  The derivation path is the
        # Fact.source (Table A row 6).
        head_dim: Optional[Fact] = None
        hd = spec.get("head_dim_attn")
        if hd is not None:
            if profile.get("config_head_dim") == hd:
                head_dim = Fact(hd, prov, "config.json:head_dim")
            else:
                head_dim = Fact(
                    hd, prov, "config.json:hidden_size//num_attention_heads"
                )

        # max_ctx: the deriver's own fallback (131072) MUST stay labeled — a
        # missing config key is not a card fact (proposal §1 design rule).
        max_ctx: Optional[Fact] = None
        mc = spec.get("max_ctx_supported")
        if mc is not None:
            if profile.get("config_max_position_embeddings") == mc:
                max_ctx = Fact(mc, "derived", "config.json:max_position_embeddings")
            else:
                max_ctx = Fact(mc, "fallback", "default:max_position_embeddings‖131072")

        return cls(
            spec_version=MODEL_SPEC_VERSION,
            model_slug=slug,
            family=(
                fact(profile.get("family"), prov, f"{src_prefix}:family")
                if curated and profile.get("family") else None
            ),
            arch=profile.get("arch"),
            confidence=confidence,
            hidden_size=dim("hidden_size", "hidden_size"),
            num_hidden_layers=dim("num_hidden_layers", "num_hidden_layers"),
            num_attn_heads=dim("num_attn_heads", "num_attention_heads"),
            num_kv_heads=dim("num_kv_heads", "num_key_value_heads"),
            head_dim_attn=head_dim,
            max_ctx_supported=max_ctx,
            valid_tp=tuple(spec.get("valid_tp") or (1, 2)),
            weights_total_gb=fact(
                spec.get("weights_total_gb", profile.get("weights_variant_size_gb")),
                prov,
                f"{src_prefix}:size_gb" if curated else "safetensors:selected-blobs-sum",
            ),
            vision_capable=fact(
                profile.get("vision"), "derived", "config.json:vision-heuristic"
            ),
            weight_format=fact(
                profile.get("weight_format"), prov,
                f"{src_prefix}:weight_format" if curated
                else "quant-chain:quantization_config‖torch_dtype‖header-probe",
            ),
            torch_dtype=fact(profile.get("torch_dtype"), "derived", "config.json:torch_dtype"),
            effective_bpw=fact(profile.get("effective_bpw"), "derived",
                               "quant-chain:bits-per-weight"),
            footprint_gb=fact(profile.get("footprint_gb"), "derived",
                              "hf-api:download_set-sizes"),
            selected_weight_files=fact(profile.get("selected_weight_files"), "derived",
                                       "hf-api:siblings"),
            mtp=MtpFacts(has_head=bool(profile.get("has_mtp_head"))),
        )

    @classmethod
    def from_gguf_facts(cls, facts: dict[str, Any]) -> Optional["ModelSpec"]:
        """Integrate the GGUF header path (gguf_spec_facts shape — commit
        7d5ce310) as a SOURCE under the typed model: every mapped dim carries
        ``provenance="derived-estimate"`` with a ``gguf-header:<arch>.<kv>``
        source.  Returns None when the dict isn't a header-derived facts blob
        (no ``general.architecture`` ⇒ nothing maps honestly)."""
        if not isinstance(facts, dict) or not facts.get("arch"):
            return None
        arch = str(facts["arch"])
        g: dict[str, Any] = facts.get("gguf") or {}

        def f(key: str, source: str) -> Optional[Fact]:
            v = facts.get(key)
            return (
                None if v is None
                else Fact(v, "derived-estimate", f"gguf-header:{arch}.{source}")
            )

        kv_src = "attention.head_count_kv"
        if g.get("kv_heads_assumed_equal"):
            kv_src = "attention.head_count (kv omitted ⇒ MHA equality)"
        return cls(
            spec_version=MODEL_SPEC_VERSION,
            model_slug=str(facts.get("model_id") or ""),
            # No config.json exists for a GGUF repo → family/display stay human
            # placeholders downstream; never fabricated from the header.
            family=None,
            arch=arch,
            confidence=str(facts.get("confidence") or "estimated-lower-bound"),
            hidden_size=f("hidden_size", "embedding_length"),
            num_hidden_layers=f("num_hidden_layers", "block_count"),
            num_attn_heads=f("num_attn_heads", "attention.head_count"),
            num_kv_heads=f("num_kv_heads", kv_src),
            head_dim_attn=f("head_dim_attn", "attention.key_length"),
            max_ctx_supported=f("max_ctx_supported", "context_length"),
            valid_tp=tuple(facts.get("valid_tp") or (1, 2)),
            weights_total_gb=f("weights_total_gb", "file-size"),
            mtp=MtpFacts(has_head=bool(facts.get("has_mtp_head"))),
            gguf=GgufFacts(
                version=g.get("version"),
                general_name=g.get("general_name"),
                file_type=g.get("file_type"),
                quant_label=g.get("quant_label"),
                quantization_version=g.get("quantization_version"),
                kv_heads_assumed_equal=bool(g.get("kv_heads_assumed_equal")),
                head_count_variable=bool(g.get("head_count_variable")),
                truncated=bool(g.get("truncated")),
            ),
        )

    @classmethod
    def from_plain_spec(cls, spec: dict[str, Any], *, model_slug: str = "") -> "ModelSpec":
        """Legacy adapter: the pull-gate ``--json`` ``spec`` block (the kv-calc
        generic-dense dict — values only, no provenance).  Provenance is
        reconstructed conservatively: a max_ctx of exactly 131072 is labeled
        ``fallback`` (the deriver's documented default — an unlabeled dict
        cannot prove it came from config.json)."""
        if not isinstance(spec, dict):
            return cls(model_slug=model_slug)

        def f(key: str, config_key: str) -> Optional[Fact]:
            v = spec.get(key)
            return (
                None if v is None
                else Fact(v, "derived", f"pull-gate:config.json:{config_key}")
            )

        mc = spec.get("max_ctx_supported")
        max_ctx = None
        if mc is not None:
            max_ctx = (
                Fact(mc, "fallback", "pull-gate:default:131072")
                if mc == 131072
                else Fact(mc, "derived", "pull-gate:config.json:max_position_embeddings")
            )
        wtg = spec.get("weights_total_gb")
        return cls(
            spec_version=MODEL_SPEC_VERSION,
            model_slug=model_slug,
            hidden_size=f("hidden_size", "hidden_size"),
            num_hidden_layers=f("num_hidden_layers", "num_hidden_layers"),
            num_attn_heads=f("num_attn_heads", "num_attention_heads"),
            num_kv_heads=f("num_kv_heads", "num_key_value_heads"),
            head_dim_attn=f("head_dim_attn", "head_dim"),
            max_ctx_supported=max_ctx,
            valid_tp=tuple(spec.get("valid_tp") or (1, 2)),
            weights_total_gb=(
                Fact(wtg, "derived", "pull-gate:safetensors:selected-blobs-sum")
                if wtg is not None else None
            ),
        )
