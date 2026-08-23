"""ModelSpec — ONE typed, versioned, provenance-labeled schema for model
geometry (the "one vocabulary" bridge between the deriver, the c3 BYO funnel,
the promote scaffold and the strict profile loader).

Design: local://modelspec-proposal.md (ModelSpecDesign wave-3 proposal,
§1 shape / §2 field tables / §5 versioning).  Landed slices:

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

  M5  family extractors land additively — first slice: ``MoEFacts`` (MoE
      routing from config.json alias pairs + GGUF ``expert_count`` /
      ``expert_used_count``), consumed by the promote scaffold's auto-filled
      ``experts`` line.  Public surface for downstream tools (kv-calc):
      ``ModelSpec.moe: Optional[MoEFacts]`` whose four fields are each an
      ``Optional[Fact]`` (``num_experts`` / ``experts_per_tok`` /
      ``shared_experts`` / ``moe_intermediate_size``) plus
      :meth:`MoEFacts.summary` for the "N routed / K active (+S shared)"
      one-liner.
  M5 slice-2  the family extractors: ``hybrid_gdn`` (:class:`HybridGdnFacts`
      — qwen3-next linear-attn dims + full-vs-linear layer split),
      ``swa`` (:class:`SwaFacts` — gemma sliding-window split) and ``mla``
      (:class:`MlaFacts` — deepseek2 single-latent-KV, never fabricated into
      head counts).  dLLM families deliberately get NO extractor — decode
      granularity / canvas size are policy, not geometry (see the note above
      :class:`MlaFacts`).

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
class MoEFacts:
    """Mixture-of-experts routing (Table C extractor — M5 first slice).
    Every present value is a full ``Fact``: the alias pair ACTUALLY found in
    config.json (``num_local_experts`` vs ``num_experts``, …) is the recorded
    source, never silently normalized.  A dense model carries ``moe=None``
    on the spec — an all-None MoEFacts is never attached."""

    #: routed experts — config ``num_local_experts`` ‖ ``num_experts``;
    #: GGUF ``<arch>.expert_count``
    num_experts: Optional[Fact] = None
    #: active experts per token — config ``num_experts_per_tok`` ‖
    #: ``top_k_experts``; GGUF ``<arch>.expert_used_count``
    experts_per_tok: Optional[Fact] = None
    #: shared (always-on) expert count — config ``num_shared_experts`` ‖
    #: ``n_shared_experts``.  None also when a family declares only
    #: ``shared_expert_intermediate_size`` (a width is not a count — never
    #: fabricated).
    shared_experts: Optional[Fact] = None
    #: per-routed-expert FFN width — config ``moe_intermediate_size``
    moe_intermediate_size: Optional[Fact] = None

    def summary(self) -> str:
        """The scaffold's one-line form: "128 routed / 8 active (+1 shared)"
        (degrades gracefully when only some facts are present)."""
        parts: list[str] = []
        if self.num_experts is not None:
            parts.append(f"{self.num_experts.value} routed")
        if self.experts_per_tok is not None:
            parts.append(f"{self.experts_per_tok.value} active")
        out = " / ".join(parts)
        if self.shared_experts is not None:
            out += f" (+{self.shared_experts.value} shared)"
        return out



def _fact_block_from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Rebuild a frozen facts-block dataclass from its flattened ``to_dict``
    shape — nested Fact dicts become real Facts (the JSON boundary across
    the deriver subprocess stays lossless)."""
    kw: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        u = d[f.name]
        kw[f.name] = (
            Fact(u["value"], u["provenance"], u["source"])
            if isinstance(u, dict)
            and {"value", "provenance", "source"} <= set(u)
            else u
        )
    return cls(**kw)


@dataclass(frozen=True)
class HybridGdnFacts:
    """Qwen3-Next-style gated-delta-net hybrid (Table C extractor — M5
    slice-2): the linear-attention head geometry plus the full-vs-linear
    LAYER split.  Every present value is a full ``Fact`` naming the config /
    GGUF key actually found; a non-hybrid model carries ``hybrid_gdn=None``
    on the spec — an all-None block is never attached."""

    #: DeltaNet key heads — config ``linear_num_key_heads``;
    #: GGUF ``<arch>.ssm.group_count``
    linear_num_k_heads: Optional[Fact] = None
    #: DeltaNet value heads — config ``linear_num_value_heads``;
    #: GGUF ``<arch>.ssm.time_step_rank``
    linear_num_v_heads: Optional[Fact] = None
    #: DeltaNet key-head depth — config ``linear_key_head_dim``;
    #: GGUF ``<arch>.ssm.state_size``
    linear_k_head_dim: Optional[Fact] = None
    #: DeltaNet value-head depth — config ``linear_value_head_dim``; GGUF
    #: derived ``inner_size // time_step_rank`` (derived-estimate)
    linear_v_head_dim: Optional[Fact] = None
    #: short-conv width — config ``linear_conv_kernel_dim`` ‖
    #: ``conv_kernel``; GGUF ``<arch>.ssm.conv_kernel``
    linear_conv_kernel_dim: Optional[Fact] = None
    #: recurrent (GDN/DeltaNet) layer count — counted from
    #: ``layer_types``, else derived from ``full_attention_interval``
    num_gdn_layers: Optional[Fact] = None
    #: full-attention layer count — same sources as num_gdn_layers
    num_attn_layers: Optional[Fact] = None

    def summary(self) -> str:
        """The scaffold's one-liner: "30 GDN / 10 full-attn layers"
        (degrades gracefully when the split is unknown)."""
        parts: list[str] = []
        if self.num_gdn_layers is not None:
            parts.append(f"{self.num_gdn_layers.value} GDN")
        if self.num_attn_layers is not None:
            parts.append(f"{self.num_attn_layers.value} full-attn")
        return " / ".join(parts) + " layers" if parts else ""


@dataclass(frozen=True)
class SwaFacts:
    """Sliding-window attention split (gemma — Table C extractor, M5
    slice-2): window width, the global-vs-sliding layer counts and — where a
    release exposes them — per-class head dims / the asymmetric global KV
    count.  A uniform-attention model carries ``swa=None``."""

    #: window width — config ``sliding_window``;
    #: GGUF ``<arch>.attention.sliding_window``
    sliding_window: Optional[Fact] = None
    #: global-layer count — counted from ``layer_types`` / GGUF
    #: ``attention.sliding_window_pattern`` array / legacy int pattern
    num_full_attn_layers: Optional[Fact] = None
    #: sliding-layer count — same sources
    num_sliding_attn_layers: Optional[Fact] = None
    #: sliding-layer head dim — config ``head_dim_sliding``, else
    #: ``head_dim`` ONLY when a global/sliding split is independently
    #: evidenced (a plain uniform ``head_dim`` is never relabeled);
    #: GGUF ``<arch>.attention.key_length_swa``
    head_dim_sliding: Optional[Fact] = None
    #: global-layer head dim — config ``global_head_dim``; GGUF
    #: ``attention.key_length`` (gemma4's converter writes the GLOBAL dim
    #: there — gated on sliding evidence so it never leaks to other arches)
    global_head_dim: Optional[Fact] = None
    #: global-layer KV heads when asymmetric (gemma-4-26b-a4b: 8 sliding /
    #: 2 global) — config ``num_global_key_value_heads`` / GGUF aligned
    #: ``head_count_kv`` × ``sliding_window_pattern`` arrays.  Never
    #: fabricated when the header carries only a collapsed scalar.
    num_global_kv_heads: Optional[Fact] = None

    def summary(self) -> str:
        """The scaffold's one-liner: "1024-token window · 5 full / 25
        sliding layers" (each part drops out independently)."""
        parts: list[str] = []
        if self.sliding_window is not None:
            parts.append(f"{self.sliding_window.value}-token window")
        if (
            self.num_full_attn_layers is not None
            and self.num_sliding_attn_layers is not None
        ):
            parts.append(
                f"{self.num_full_attn_layers.value} full / "
                f"{self.num_sliding_attn_layers.value} sliding layers"
            )
        return " · ".join(parts)


@dataclass(frozen=True)
class MlaFacts:
    """Multi-head latent attention (deepseek2 — Table C extractor, M5
    slice-2): ONE compressed KV latent shared by every query head.  These
    are LATENT-geometry Facts — deliberately NEVER fabricated into
    ``num_kv_heads``/head counts (a deepseek2 GGUF converts to MQA with one
    KV group; that conversion artifact is not model geometry)."""

    #: compressed-KV latent rank — config/GGUF ``kv_lora_rank``
    kv_lora_rank: Optional[Fact] = None
    #: non-positional query-key head depth — config ``qk_nope_head_dim``;
    #: GGUF derived ``key_length_mla − rope.dimension_count``
    qk_nope_head_dim: Optional[Fact] = None
    #: RoPE-shared head depth — config ``qk_rope_head_dim``;
    #: GGUF ``<arch>.rope.dimension_count`` (the deepseek2 converter writes
    #: exactly that value there)
    qk_rope_head_dim: Optional[Fact] = None

    def summary(self) -> str:
        """The scaffold's one-liner: "latent rank 512 · qk 128+64(RoPE)"."""
        parts: list[str] = []
        nope = self.qk_nope_head_dim.value if self.qk_nope_head_dim else None
        rope = self.qk_rope_head_dim.value if self.qk_rope_head_dim else None
        if self.kv_lora_rank is not None:
            parts.append(f"latent rank {self.kv_lora_rank.value}")
        if nope is not None and rope is not None:
            parts.append(f"qk {nope}+{rope}(RoPE)")
        elif nope is not None:
            parts.append(f"qk_nope {nope}")
        elif rope is not None:
            parts.append(f"qk_rope {rope}")
        return " · ".join(parts)


# ── dLLM (diffusiongemma): documented NO-EXTRACTION (M5 slice-2).  Diffusion-LM
# configs expose NO structural geometry beyond the standard dense dims — decode
# granularity / canvas size are serving POLICY (proposal §2 marks them human).
# An empty DllmFacts would fabricate structure config.json does not carry, so
# there is deliberately none: those scaffold lines stay hand-fill placeholders.


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


#: The named facts-block slots (ModelSpec field name → block type).  Drives
#: the generic serialization/validation branches below — a new M5 extractor
#: registers here and inherits to_dict/from_dict/validate for free.
_FACT_BLOCKS: dict[str, type] = {
    "moe": MoEFacts,
    "mtp": MtpFacts,
    "gguf": GgufFacts,
    "hybrid_gdn": HybridGdnFacts,
    "swa": SwaFacts,
    "mla": MlaFacts,
}


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
    moe: Optional[MoEFacts] = None  # MoE routing (M5 slice-1 extractor)
    mtp: Optional[MtpFacts] = None
    hybrid_gdn: Optional[HybridGdnFacts] = None  # qwen3-next GDN/DeltaNet (M5 slice-2)
    swa: Optional[SwaFacts] = None  # gemma sliding-window split (M5 slice-2)
    mla: Optional[MlaFacts] = None  # deepseek2 single-latent-KV (M5 slice-2)
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
        # Family fact-blocks (Table C): every ATTACHED Fact must be a
        # positive int — a wrong-typed value is a bug, absence stays None.
        for blk_name in _FACT_BLOCKS:
            blk = getattr(self, blk_name)
            if blk is None:
                continue
            for bf in fields(blk):
                mf = getattr(blk, bf.name)
                # Only Fact-carrying fields are checked here — metadata
                # blocks (gguf) / presence flags (mtp.has_head) hold plain
                # values that are not positive-int geometry.
                if not isinstance(mf, Fact):
                    continue
                if (
                    not isinstance(mf.value, int)
                    or isinstance(mf.value, bool) or mf.value <= 0
                ):
                    issues.append(
                        f"{blk_name}.{bf.name}: expected a positive int, "
                        f"got {mf.value!r}"
                    )
        moe = self.moe
        if moe is not None:
            n_exp, k_act = moe.num_experts, moe.experts_per_tok
            if (
                n_exp is not None and k_act is not None
                and isinstance(n_exp.value, int) and isinstance(k_act.value, int)
                and k_act.value > n_exp.value
            ):
                issues.append(
                    f"warn: moe.experts_per_tok ({k_act.value}) > "
                    f"moe.num_experts ({n_exp.value}) — impossible routing, verify"
                )
        # Layer-split coherence: the family counts should sum to the total.
        nl = self.num_hidden_layers
        if nl is not None and isinstance(nl.value, int) and not isinstance(nl.value, bool):
            for blk_name, a_name, b_name in (
                ("hybrid_gdn", "num_gdn_layers", "num_attn_layers"),
                ("swa", "num_full_attn_layers", "num_sliding_attn_layers"),
            ):
                blk = getattr(self, blk_name)
                pa = getattr(blk, a_name, None) if blk is not None else None
                pb = getattr(blk, b_name, None) if blk is not None else None
                if (
                    pa is not None and pb is not None
                    and isinstance(pa.value, int) and isinstance(pb.value, int)
                    and pa.value + pb.value != nl.value
                ):
                    issues.append(
                        f"warn: {blk_name}.{a_name} + {b_name} "
                        f"({pa.value} + {pb.value}) != num_hidden_layers "
                        f"({nl.value}) — verify"
                    )
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
            elif isinstance(v, tuple(_FACT_BLOCKS.values())):
                # Facts-block fields are themselves Facts (or plain metadata)
                # — flatten each with the value/provenance/source shape so
                # the JSON boundary stays lossless (from_dict rebuilds via
                # _fact_block_from_dict).
                out[f.name] = {
                    g: (
                        {"value": fv.value, "provenance": fv.provenance,
                         "source": fv.source}
                        if isinstance(fv := getattr(v, g), Fact) else fv
                    )
                    for g in (x.name for x in fields(v))
                }
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
            elif f.name in _FACT_BLOCKS and isinstance(v, dict):
                kw[f.name] = _fact_block_from_dict(_FACT_BLOCKS[f.name], v)
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

        # MoE routing (Table C, M5 slice-1): the deriver threads config.json's
        # raw alias keys through as ``config_<key>`` profile entries — resolve
        # the alias pairs HERE so each Fact's source names the key that was
        # actually present ("config.json:num_local_experts" vs
        # "config.json:num_experts").  All-None ⇒ dense config ⇒ spec.moe
        # stays None (never attach an empty MoEFacts).
        def moe_fact(*alias_keys: str) -> Optional[Fact]:
            for k in alias_keys:
                v = profile.get(f"config_{k}")
                if v is not None:
                    return Fact(v, prov, f"{src_prefix}:{k}")
            return None

        moe = MoEFacts(
            num_experts=moe_fact("num_local_experts", "num_experts"),
            experts_per_tok=moe_fact("num_experts_per_tok", "top_k_experts"),
            shared_experts=moe_fact("num_shared_experts", "n_shared_experts"),
            moe_intermediate_size=moe_fact("moe_intermediate_size"),
        )
        if moe == MoEFacts():
            moe = None

        # ── Family extractors (Table C, M5 slice-2): GDN/DeltaNet hybrid,
        # SWA split, MLA latent.  The SAME alias-pair discipline as the MoE
        # block above: each Fact names the key ACTUALLY found, and an
        # all-None block is never attached.  dLLM families (diffusiongemma)
        # deliberately get NO extractor — decode granularity / canvas size
        # are policy, not geometry (see the module note above MlaFacts).
        cfg_fact = moe_fact  # identical rule: first PRESENT alias wins

        lt_raw = profile.get("config_layer_types")
        ltypes = [str(t) for t in lt_raw] if isinstance(lt_raw, list) else []
        n_full_lt = sum(t == "full_attention" for t in ltypes)
        n_layers_v = spec.get("num_hidden_layers")
        if n_layers_v is None:
            n_layers_v = profile.get("config_num_hidden_layers")

        # Layer splits: EXACT when config.json ships ``layer_types``
        # (counted); otherwise derived from the family's interval/pattern
        # key and labeled derived-estimate (it is arithmetic, not a fact).
        gdn_split: tuple[Optional[Fact], Optional[Fact]] = (None, None)
        swa_split: tuple[Optional[Fact], Optional[Fact]] = (None, None)
        if ltypes:
            split_src = f"{src_prefix}:layer_types"
            n_lin = sum(t in ("linear_attention", "gated_deltanet") for t in ltypes)
            n_slide = sum(
                t in ("sliding_attention", "sliding_window") for t in ltypes
            )
            if n_lin and n_full_lt:
                gdn_split = (
                    Fact(n_lin, prov, split_src),
                    Fact(n_full_lt, prov, split_src),
                )
            if n_slide and n_full_lt:
                swa_split = (
                    Fact(n_full_lt, prov, split_src),
                    Fact(n_slide, prov, split_src),
                )

        _k_h = cfg_fact("linear_num_key_heads")
        _v_h = cfg_fact("linear_num_value_heads")
        _k_d = cfg_fact("linear_key_head_dim")
        _v_d = cfg_fact("linear_value_head_dim")
        _conv = cfg_fact("linear_conv_kernel_dim", "conv_kernel")
        iv = profile.get("config_full_attention_interval")
        if gdn_split == (None, None) and any((_k_h, _v_h, _k_d, _v_d, _conv)) \
                and isinstance(iv, int) and not isinstance(iv, bool) and iv > 0 \
                and isinstance(n_layers_v, int) and not isinstance(n_layers_v, bool) \
                and n_layers_v > 0:
            est = f"{src_prefix}:num_hidden_layers÷full_attention_interval"
            n_attn = n_layers_v // iv
            gdn_split = (
                Fact(n_layers_v - n_attn, "derived-estimate", est),
                Fact(n_attn, "derived-estimate", est),
            )

        gdn = HybridGdnFacts(
            linear_num_k_heads=_k_h,
            linear_num_v_heads=_v_h,
            linear_k_head_dim=_k_d,
            linear_v_head_dim=_v_d,
            linear_conv_kernel_dim=_conv,
            num_gdn_layers=gdn_split[0],
            num_attn_layers=gdn_split[1],
        )
        if gdn == HybridGdnFacts():
            gdn = None

        window = cfg_fact("sliding_window")
        global_hd = cfg_fact("global_head_dim")
        pw = profile.get("config_sliding_window_pattern")
        if swa_split == (None, None) and isinstance(pw, int) \
                and not isinstance(pw, bool) and pw > 1 \
                and isinstance(n_layers_v, int) and not isinstance(n_layers_v, bool) \
                and n_layers_v > 0:
            est = f"{src_prefix}:num_hidden_layers÷sliding_window_pattern"
            n_glob = n_layers_v // pw
            swa_split = (
                Fact(n_glob, "derived-estimate", est),
                Fact(n_layers_v - n_glob, "derived-estimate", est),
            )
        # Gemma4 naming (upstream converter contract): when a per-class
        # global dim exists, the plain ``head_dim`` IS the sliding-layer
        # dim.  Without that split evidence a uniform head_dim is NEVER
        # relabeled as sliding-only.
        slide_hd = cfg_fact("head_dim_sliding") or (
            cfg_fact("head_dim")
            if (global_hd is not None or swa_split[0] is not None) else None
        )
        swa = SwaFacts(
            sliding_window=window,
            num_full_attn_layers=swa_split[0],
            num_sliding_attn_layers=swa_split[1],
            head_dim_sliding=slide_hd,
            global_head_dim=global_hd,
            num_global_kv_heads=cfg_fact("num_global_key_value_heads"),
        )
        if swa == SwaFacts():
            swa = None

        mla = MlaFacts(
            kv_lora_rank=cfg_fact("kv_lora_rank"),
            qk_nope_head_dim=cfg_fact("qk_nope_head_dim"),
            qk_rope_head_dim=cfg_fact("qk_rope_head_dim"),
        )
        if mla == MlaFacts():
            mla = None
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
            moe=moe,
            hybrid_gdn=gdn,
            swa=swa,
            mla=mla,
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
        moe_n = f("num_experts", "expert_count")
        moe_k = f("experts_per_tok", "expert_used_count")
        # A dense header carries neither expert KV ⇒ spec.moe stays None.
        moe = (
            MoEFacts(num_experts=moe_n, experts_per_tok=moe_k)
            if (moe_n is not None or moe_k is not None) else None
        )

        # M5 slice-2: family blocks ride the facts dict as {field: {value,
        # kv}} maps (gguf_spec_facts emits them ONLY when the header carries
        # the family's KVs).  Each entry becomes a derived-estimate Fact with
        # a gguf-header:<kv> source; absent ⇒ slot None.
        def fam_block(name: str, blk_cls: Any) -> Any:
            entries = facts.get(name)
            if not isinstance(entries, dict):
                return None
            kw = {
                k: Fact(e["value"], "derived-estimate", f"gguf-header:{e['kv']}")
                for k, e in entries.items()
                if isinstance(e, dict) and {"value", "kv"} <= set(e)
                and e.get("value") is not None
            }
            return blk_cls(**kw) if kw else None

        gdn = fam_block("hybrid_gdn", HybridGdnFacts)
        swa = fam_block("swa", SwaFacts)
        mla = fam_block("mla", MlaFacts)
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
            hybrid_gdn=gdn,
            swa=swa,
            mla=mla,
            valid_tp=tuple(facts.get("valid_tp") or (1, 2)),
            weights_total_gb=f("weights_total_gb", "file-size"),
            mtp=MtpFacts(has_head=bool(facts.get("has_mtp_head"))),
            moe=moe,
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
