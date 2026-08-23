#!/usr/bin/env python3
"""ModelSpec unit tests — the typed, provenance-labeled bridge schema
(ModelSpec proposal §1/§2/§5; M1–M3 slice).

    pytest scripts/tests/test_model_spec.py

NO live network: every DeriveResult is synthetic (the deriver's documented
test seam is the injectable fetcher; here we don't even need it — we build
DeriveResult objects directly and check ModelSpec's pure builders).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.lib.profiles import deriver as D  # noqa: E402
from scripts.lib.profiles.model_spec import (  # noqa: E402
    MODEL_SPEC_VERSION,
    Fact,
    HybridGdnFacts,
    MlaFacts,
    ModelSpec,
    MoEFacts,
    MtpFacts,
    SwaFacts,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic DeriveResults (eligible / not-eligible / tier-1 / error)
# ---------------------------------------------------------------------------
def _config(**over) -> dict:
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 64,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": None,          # forces the computed head-dim path by default
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
    }
    cfg.update(over)
    return cfg


def _derive_result(*, eligible=True, config=None, tier1=None, error=None) -> D.DeriveResult:
    config = config if config is not None else _config()
    res = D.DeriveResult(slug="org/Model")
    if error is not None:
        res.error = error
        res.confidence = None
        return res
    curated_id = getattr(tier1, "model_id", None)
    res.tier1 = tier1
    res.profile = {
        "model_id": curated_id or "org/Model",
        "weights_variant": None if tier1 is None else "main",
        "arch": (config.get("architectures") or [None])[0],
        "family": ("generic-dense" if eligible else None) if tier1 is None else "qwen3-dense",
        "auto_map": False,
        "weight_format": "fp8" if tier1 is None else "autoround",
        "torch_dtype": config.get("torch_dtype"),
        "effective_bpw": 8.0 if tier1 is None else 4.5,
        "weights_total_gb": 30.9 if tier1 is None else None,
        "footprint_gb": 31.2 if tier1 is None else None,
        "selected_weight_files": ["model-00001-of-00002.safetensors"] if tier1 is None else None,
        "_hf_api": {},
        "download_set": [],
        "config_hidden_size": config["hidden_size"],
        "config_num_hidden_layers": config["num_hidden_layers"],
        "config_num_attention_heads": config["num_attention_heads"],
        "config_num_key_value_heads": config["num_key_value_heads"],
        "config_head_dim": config.get("head_dim"),
        "config_max_position_embeddings": config.get("max_position_embeddings"),
        # ModelSpec M5 (additive): raw MoE alias keys — exactly what
        # deriver.derive() now emits into res.profile.
        **{
            f"config_{k}": config.get(k)
            for k in (
                "num_experts", "num_local_experts", "num_experts_per_tok",
                "top_k_experts", "moe_intermediate_size",
                "num_shared_experts", "n_shared_experts",
                # M5 slice-2: raw FAMILY keys — exactly what
                # deriver.derive() now emits into res.profile.
                "linear_num_key_heads", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim", "conv_kernel",
                "layer_types", "full_attention_interval",
                "sliding_window", "sliding_window_pattern",
                "global_head_dim", "head_dim_sliding",
                "num_global_key_value_heads",
                "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            )
        },
        "vision": False,
        "has_mtp_head": False,
        **({} if tier1 is None else {
            # tier-1 curated profile shape (Table A row 10)
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attn_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "weights_variant_size_gb": 18.0,
        }),
    }
    if tier1 is not None:
        res.confidence = D.Confidence.EXACT
        return res
    res.generic_dense_eligible = eligible
    res.confidence = (
        D.Confidence.ESTIMATED_LOWER_BOUND if eligible
        else D.Confidence.NOT_ELIGIBLE
    )
    if eligible:
        res.spec = D._build_generic_dense_spec("org/Model", config, 30.9)
    return res


# ---------------------------------------------------------------------------
# M1: Fact + shape basics
# ---------------------------------------------------------------------------
class TestFact:
    def test_fact_carries_value_provenance_source(self):
        f = Fact(64, "derived", "config.json:num_hidden_layers")
        assert f.value == 64
        assert f.provenance == "derived"
        assert f.source == "config.json:num_hidden_layers"

    def test_fallback_is_an_explicit_provenance(self):
        f = Fact(131072, "fallback", "default:max_position_embeddings‖131072")
        assert f.provenance == "fallback"

    def test_frozen(self):
        f = Fact(1, "derived", "x")
        with pytest.raises(Exception):
            f.value = 2


class TestShape:
    def test_version_constant(self):
        assert MODEL_SPEC_VERSION == 1

    def test_empty_spec_is_falsy(self):
        assert not ModelSpec(model_slug="org/X")
        assert bool(ModelSpec(hidden_size=Fact(4096, "derived", "config.json:hidden_size")))

    def test_no_policy_fields(self):
        """display_name / family-tag / drafters etc. are HUMAN inputs — they
        must NOT be machine fields on the spec (proposal §1 design rule)."""
        names = {f.name for f in ModelSpec.__dataclass_fields__.values()}
        for banned in ("display_name", "attention_k_eq_v", "requires_genesis",
                       "compatible_drafters", "decode_granularity"):
            assert banned not in names

    def test_arch_dims_and_fallback_dims(self):
        s = ModelSpec(
            hidden_size=Fact(4096, "derived", "config.json:hidden_size"),
            max_ctx_supported=Fact(131072, "fallback", "default:max_position_embeddings‖131072"),
        )
        assert s.arch_dims() == {"hidden_size": 4096, "max_ctx_supported": 131072}
        assert s.fallback_dims() == {"max_ctx_supported": 131072}


# ---------------------------------------------------------------------------
# M2: from_derive_result — provenance labeling (Tables A/B)
# ---------------------------------------------------------------------------
class TestFromDeriveResult:
    def test_eligible_dims_labeled_config_json(self):
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.model_slug == "org/Model"
        assert s.confidence == "estimated-lower-bound"
        assert s.arch == "Qwen3ForCausalLM"
        assert s.hidden_size == Fact(4096, "derived", "config.json:hidden_size")
        assert s.num_hidden_layers == Fact(64, "derived", "config.json:num_hidden_layers")
        assert s.num_attn_heads == Fact(32, "derived", "config.json:num_attention_heads")
        assert s.num_kv_heads == Fact(8, "derived", "config.json:num_key_value_heads")
        assert s.valid_tp == (1, 2)

    def test_head_dim_computed_path_is_recorded_in_source(self):
        """head_dim absent in config → hidden // heads; the derivation path is
        the Fact.source (Table A row 6)."""
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.head_dim_attn.value == 128   # 4096 // 32, the computed path
        assert s.head_dim_attn.source == "config.json:hidden_size//num_attention_heads"

    def test_head_dim_explicit_config_value(self):
        cfg = _config(head_dim=128)
        res = _derive_result(config=cfg)
        res.spec = D._build_generic_dense_spec("org/Model", cfg, 30.9)
        s = ModelSpec.from_derive_result(res)
        assert s.head_dim_attn == Fact(128, "derived", "config.json:head_dim")

    def test_max_ctx_from_config_is_derived(self):
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.max_ctx_supported == Fact(
            262144, "derived", "config.json:max_position_embeddings"
        )

    def test_max_ctx_fallback_IS_labeled_fallback(self):
        """THE acceptance rule: a missing config key ⇒ 131072 default labeled
        'fallback', never card truth."""
        cfg = _config(max_position_embeddings=None)
        res = _derive_result(config=cfg)
        res.profile["config_max_position_embeddings"] = None
        res.spec = D._build_generic_dense_spec("org/Model", cfg, 30.9)
        s = ModelSpec.from_derive_result(res)
        assert s.max_ctx_supported.value == 131072
        assert s.max_ctx_supported.provenance == "fallback"
        assert s.fallback_dims() == {"max_ctx_supported": 131072}

    def test_not_eligible_still_carries_config_dims(self):
        """The old embedded probe setdefault-filled dims from the profile's
        config_* duplicates even without a generic-dense spec — preserved."""
        res = _derive_result(eligible=False)
        s = ModelSpec.from_derive_result(res)
        assert s.spec_version == MODEL_SPEC_VERSION
        assert s.confidence == "not-generic-dense-eligible"
        assert s.hidden_size.value == 4096
        assert s.max_ctx_supported is None   # no spec ⇒ no ctx claim at all
        assert s.weight_format.provenance == "derived"

    def test_weights_and_provenance_block(self):
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.weights_total_gb == Fact(30.9, "derived", "safetensors:selected-blobs-sum")
        assert s.torch_dtype == Fact("bfloat16", "derived", "config.json:torch_dtype")
        assert s.effective_bpw.value == 8.0
        assert s.footprint_gb.value == 31.2
        assert s.selected_weight_files.value == ["model-00001-of-00002.safetensors"]
        assert s.vision_capable == Fact(False, "derived", "config.json:vision-heuristic")
        assert s.mtp == MtpFacts(has_head=False)

    def test_tier1_curated_hit(self):
        t1 = D.Tier1Match(model_id="qwen3-27b", weights_variant="main", slug="org/Model")
        s = ModelSpec.from_derive_result(_derive_result(tier1=t1))
        assert s.confidence == "exact"
        assert s.family == Fact("qwen3-dense", "curated", "profiles:qwen3-27b:family")
        assert s.hidden_size.provenance == "curated"
        assert s.weights_total_gb.value == 18.0
        assert s.weights_total_gb.source == "profiles:qwen3-27b:size_gb"

    def test_error_result_knows_nothing(self):
        s = ModelSpec.from_derive_result(
            _derive_result(error=D.DeriverError(D.DeriverErrorKind.REPO_NOT_FOUND, "404"))
        )
        assert not s
        assert s.arch_dims() == {}
        assert s.vision_capable is None

    def test_family_stays_none_for_derived(self):
        """generic-dense ELIGIBILITY is a pricing verdict, not a family —
        family stays None until a human/curator names it (proposal §1)."""
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.family is None


# ---------------------------------------------------------------------------
# GGUF header path integrated as a SOURCE
# ---------------------------------------------------------------------------
class TestFromGgufFacts:
    @staticmethod
    def _header(**over) -> dict:
        kv = {
            "general.architecture": "llama",
            "general.name": "Synth-7B",
            "general.file_type": 15,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.context_length": 131072,
        }
        kv.update(over)
        return D.gguf_spec_facts(
            {"version": 3, "kv": kv},
            model_id="org/Synth-7B-GGUF", weight_gb=4.2,
        )

    def test_header_dims_are_derived_estimate_with_gguf_sources(self):
        s = ModelSpec.from_gguf_facts(self._header())
        assert s is not None
        assert s.confidence == "estimated-lower-bound"
        assert s.num_hidden_layers == Fact(
            32, "derived-estimate", "gguf-header:llama.block_count"
        )
        assert s.hidden_size.source == "gguf-header:llama.embedding_length"
        assert s.num_kv_heads.source == "gguf-header:llama.attention.head_count_kv"
        assert s.max_ctx_supported.source == "gguf-header:llama.context_length"
        assert s.weights_total_gb.value == 4.2
        assert s.gguf.general_name == "Synth-7B"

    def test_mha_omitted_kv_heads_assumed_equal(self):
        facts = self._header(**{"llama.attention.head_count_kv": None})
        s = ModelSpec.from_gguf_facts(facts)
        assert s.num_kv_heads.value == 32
        assert "kv omitted" in s.num_kv_heads.source
        assert s.gguf.kv_heads_assumed_equal is True

    def test_no_architecture_maps_to_none(self):
        assert ModelSpec.from_gguf_facts({"kv": {}}) is None
        assert ModelSpec.from_gguf_facts(None) is None
        assert ModelSpec.from_gguf_facts({}) is None

    def test_family_never_fabricated_from_header(self):
        s = ModelSpec.from_gguf_facts(self._header())
        assert s.family is None


# ---------------------------------------------------------------------------
# Legacy pull-gate plain spec adapter
# ---------------------------------------------------------------------------
class TestFromPlainSpec:
    def test_plain_dims_labeled_pull_gate(self):
        spec = {
            "model_id": "org/M", "model_family": "generic-dense",
            "arch": "LlamaForCausalLM", "hidden_size": 4096,
            "num_hidden_layers": 32, "num_attn_heads": 32, "num_kv_heads": 8,
            "head_dim_attn": 128, "weights_total_gb": 4.2, "valid_tp": [1, 2],
            "max_ctx_supported": 262144,
        }
        s = ModelSpec.from_plain_spec(spec, model_slug="org/M")
        assert s.hidden_size == Fact(4096, "derived", "pull-gate:config.json:hidden_size")
        assert s.max_ctx_supported.provenance == "derived"

    def test_plain_131072_max_ctx_conservatively_fallback(self):
        spec = {"hidden_size": 4096, "max_ctx_supported": 131072}
        s = ModelSpec.from_plain_spec(spec)
        assert s.max_ctx_supported.provenance == "fallback"

    def test_empty_or_non_dict(self):
        assert not ModelSpec.from_plain_spec({})
        assert not ModelSpec.from_plain_spec(None)


# ---------------------------------------------------------------------------
# Serialization round-trip + validation (proposal §5)
# ---------------------------------------------------------------------------
class TestSerializationAndValidation:
    def test_round_trip_lossless(self):
        s = ModelSpec.from_derive_result(_derive_result())
        d = json.loads(json.dumps(s.to_dict()))          # through real JSON
        s2 = ModelSpec.from_dict(d)
        assert s2 == s

    def test_round_trip_gguf_and_mtp(self):
        s = ModelSpec.from_gguf_facts({
            "arch": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
            "confidence": "estimated-lower-bound", "weights_total_gb": 4.2,
            "gguf": {"version": 3, "quant_label": "Q4_K_M", "truncated": True},
        })
        s2 = ModelSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert s2 == s
        assert s2.gguf.quant_label == "Q4_K_M"
        assert s2.gguf.truncated is True

    def test_unknown_major_version_rejected_loudly(self):
        with pytest.raises(ValueError, match="spec_version"):
            ModelSpec.from_dict({"spec_version": 99})

    def test_validate_clean_spec(self):
        assert ModelSpec.from_derive_result(_derive_result()).validate() == []

    def test_validate_flags_nonpositive_int(self):
        s = ModelSpec(hidden_size=Fact(0, "derived", "x"))
        issues = s.validate()
        assert any("hidden_size" in i and "warn:" not in i for i in issues)

    def test_validate_warns_on_kv_gt_attn(self):
        s = ModelSpec(
            num_attn_heads=Fact(8, "derived", "c:a"),
            num_kv_heads=Fact(16, "derived", "c:k"),
        )
        issues = s.validate()
        assert any(i.startswith("warn:") and "num_kv_heads" in i for i in issues)

    def test_validate_flags_bad_valid_tp(self):
        s = ModelSpec(valid_tp=(1, 2, 4))
        assert any("valid_tp" in i for i in s.validate())


# ---------------------------------------------------------------------------
# THE losslessness guard (proposal §5 rule 5): every field the promote
# scaffold consumes must be present WITH provenance on a derived spec.
# ---------------------------------------------------------------------------
SCAFFOLD_CONSUMED_DIMS = (
    "hidden_size", "num_hidden_layers", "num_attn_heads",
    "num_kv_heads", "head_dim_attn", "max_ctx_supported",
)


class TestScaffoldRoundTrip:
    def test_every_consumed_field_present_with_provenance(self):
        s = ModelSpec.from_derive_result(_derive_result())
        dims = s.arch_dims()
        for key in SCAFFOLD_CONSUMED_DIMS:
            assert key in dims, f"scaffold field {key} dropped by ModelSpec"
            f = getattr(s, key)
            assert isinstance(f, Fact), key
            assert f.source, f"{key} has no source"
            assert f.provenance in ("derived", "derived-estimate", "fallback", "curated")
        assert list(s.valid_tp) == [1, 2]           # valid_tp survives
        assert s.weights_total_gb.value == 30.9     # weights.<variant>.size_gb
        assert s.vision_capable.value is False      # vision_capable key

    def test_json_cli_shape_round_trips_through_the_subprocess_boundary(self):
        """deriver --spec-json prints to_dict(); services parses from_dict().
        The whole pipeline must be value+provenance lossless."""
        s = ModelSpec.from_derive_result(_derive_result())
        blob = json.dumps(s.to_dict())               # what stdout carries
        s2 = ModelSpec.from_dict(json.loads(blob))
        for key in (*SCAFFOLD_CONSUMED_DIMS, "weights_total_gb",
                    "vision_capable", "valid_tp"):
            assert getattr(s2, key) == getattr(s, key), key

    def test_derive_result_property_builds_the_same_spec(self):
        res = _derive_result()
        assert res.model_spec == ModelSpec.from_derive_result(res)


# ---------------------------------------------------------------------------
# M5 slice-1: the MoE extractor (config.json alias pairs + GGUF expert KVs)
# ---------------------------------------------------------------------------
class TestMoeConfigPath:
    def test_dense_config_has_no_moe(self):
        """The default fixture is dense — spec.moe stays None and an all-None
        MoEFacts is never attached (proposal: absent ⇒ placeholders)."""
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.moe is None

    def _moe_result(self, **cfg_over) -> ModelSpec:
        return ModelSpec.from_derive_result(
            _derive_result(eligible=False, config=_config(**cfg_over))
        )

    def test_qwen_style_alias_pair_labeled_with_actual_key(self):
        s = self._moe_result(
            num_experts=128, num_experts_per_tok=8,
            moe_intermediate_size=1536,
        )
        assert s.moe.num_experts == Fact(128, "derived", "config.json:num_experts")
        assert s.moe.experts_per_tok == Fact(
            8, "derived", "config.json:num_experts_per_tok"
        )
        assert s.moe.moe_intermediate_size == Fact(
            1536, "derived", "config.json:moe_intermediate_size"
        )
        assert s.moe.shared_experts is None      # never fabricated from widths

    def test_deepseek_style_aliases_record_the_alias_actually_found(self):
        s = self._moe_result(
            num_local_experts=256, top_k_experts=8, n_shared_experts=2,
        )
        assert s.moe.num_experts.source == "config.json:num_local_experts"
        assert s.moe.experts_per_tok.source == "config.json:top_k_experts"
        assert s.moe.shared_experts == Fact(2, "derived", "config.json:n_shared_experts")
        assert s.moe.summary() == "256 routed / 8 active (+2 shared)"

    def test_summary_degrades_gracefully(self):
        s = self._moe_result(num_experts=64)
        assert s.moe.summary() == "64 routed"

    def test_partial_facts_still_attach(self):
        """Only some routing keys present ⇒ attach what IS known."""
        s = self._moe_result(moe_intermediate_size=1024)
        assert s.moe == MoEFacts(
            moe_intermediate_size=Fact(1024, "derived", "config.json:moe_intermediate_size")
        )


class TestMoeGgufPath:
    @staticmethod
    def _facts(**kv_over) -> dict:
        kv = {
            "general.architecture": "qwen3moe",
            "qwen3moe.embedding_length": 4096,
            "qwen3moe.block_count": 48,
        }
        kv.update(kv_over)
        return D.gguf_spec_facts({"version": 3, "kv": kv}, model_id="org/M")

    def test_expert_kvs_map_with_gguf_header_provenance(self):
        facts = self._facts(
            **{"qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8}
        )
        assert facts["num_experts"] == 128 and facts["experts_per_tok"] == 8
        s = ModelSpec.from_gguf_facts(facts)
        assert s.moe.num_experts == Fact(
            128, "derived-estimate", "gguf-header:qwen3moe.expert_count"
        )
        assert s.moe.experts_per_tok.provenance == "derived-estimate"
        assert s.moe.experts_per_tok.source == "gguf-header:qwen3moe.expert_used_count"
        assert s.moe.summary() == "128 routed / 8 active"

    def test_dense_header_has_no_moe(self):
        s = ModelSpec.from_gguf_facts(self._facts())
        assert s.moe is None
        facts = self._facts()
        assert facts["num_experts"] is None and facts["experts_per_tok"] is None


class TestMoeSerializationAndValidation:
    def _moe_spec(self) -> ModelSpec:
        return ModelSpec.from_derive_result(_derive_result(
            eligible=False,
            config=_config(num_local_experts=64, num_experts_per_tok=6,
                           n_shared_experts=1),
        ))

    def test_round_trip_lossless_through_real_json(self):
        s = self._moe_spec()
        s2 = ModelSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert s2 == s
        assert s2.moe.shared_experts.value == 1

    def test_validate_flags_nonpositive_expert_int(self):
        s = ModelSpec(moe=MoEFacts(
            num_experts=Fact(0, "derived", "x"),
        ))
        assert any("moe.num_experts" in i and "warn:" not in i
                   for i in s.validate())

    def test_validate_warns_on_more_active_than_routed(self):
        s = ModelSpec(moe=MoEFacts(
            num_experts=Fact(8, "derived", "x"),
            experts_per_tok=Fact(16, "derived", "y"),
        ))
        issues = s.validate()
        assert any(i.startswith("warn:") and "experts_per_tok" in i for i in issues)

    def test_dense_spec_validate_unchanged(self):
        assert ModelSpec.from_derive_result(_derive_result()).validate() == []


# ---------------------------------------------------------------------------
# M5 slice-2: the family extractors — GDN/DeltaNet (qwen3-next), SWA (gemma),
# MLA latent (deepseek2) + documented dLLM no-extraction.  Same contract as
# the MoE slice: per-Fact provenance, all-None ⇒ slot stays None.
# ---------------------------------------------------------------------------
class TestGdnConfigPath:
    def test_dense_config_has_no_gdn(self):
        s = ModelSpec.from_derive_result(_derive_result())
        assert s.hybrid_gdn is None

    def _gdn_result(self, **cfg_over) -> ModelSpec:
        return ModelSpec.from_derive_result(
            _derive_result(eligible=False, config=_config(**cfg_over))
        )

    def test_qwen3_next_layer_types_split_labeled_with_source(self):
        lt = ["linear_attention"] * 30 + ["full_attention"] * 10
        s = self._gdn_result(
            layer_types=lt,
            linear_num_key_heads=16, linear_num_value_heads=32,
            linear_key_head_dim=128, linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        g = s.hybrid_gdn
        assert g.linear_num_k_heads == Fact(16, "derived", "config.json:linear_num_key_heads")
        assert g.linear_num_v_heads == Fact(32, "derived", "config.json:linear_num_value_heads")
        assert g.linear_k_head_dim == Fact(128, "derived", "config.json:linear_key_head_dim")
        assert g.linear_v_head_dim == Fact(128, "derived", "config.json:linear_value_head_dim")
        assert g.linear_conv_kernel_dim == Fact(4, "derived", "config.json:linear_conv_kernel_dim")
        assert g.num_gdn_layers == Fact(30, "derived", "config.json:layer_types")
        assert g.num_attn_layers == Fact(10, "derived", "config.json:layer_types")
        assert g.summary() == "30 GDN / 10 full-attn layers"

    def test_conv_kernel_alias_records_the_key_actually_found(self):
        s = self._gdn_result(linear_num_key_heads=16, conv_kernel=4)
        assert s.hybrid_gdn.linear_conv_kernel_dim == Fact(
            4, "derived", "config.json:conv_kernel"
        )

    def test_interval_derived_split_is_labeled_estimate(self):
        """No layer_types ⇒ the full/linear split is arithmetic off
        full_attention_interval — derived-estimate, never card truth."""
        s = self._gdn_result(
            num_hidden_layers=48, full_attention_interval=4,
            linear_num_key_heads=16,
        )
        est_src = "config.json:num_hidden_layers÷full_attention_interval"
        assert s.hybrid_gdn.num_attn_layers == Fact(12, "derived-estimate", est_src)
        assert s.hybrid_gdn.num_gdn_layers == Fact(36, "derived-estimate", est_src)

    def test_interval_alone_never_fabricates_a_hybrid(self):
        """full_attention_interval WITHOUT any linear-attn key ⇒ no block —
        a split nobody's config claims is not invented."""
        s = self._gdn_result(num_hidden_layers=48, full_attention_interval=4)
        assert s.hybrid_gdn is None


class TestGdnGgufPath:
    @staticmethod
    def _facts(**kv_over) -> dict:
        kv = {
            "general.architecture": "qwen3next",
            "qwen3next.embedding_length": 2048,
            "qwen3next.block_count": 48,
        }
        kv.update(kv_over)
        return D.gguf_spec_facts({"version": 3, "kv": kv}, model_id="org/M")

    def test_ssm_kvs_map_with_gguf_header_provenance(self):
        facts = self._facts(
            **{
                "qwen3next.ssm.group_count": 16,
                "qwen3next.ssm.time_step_rank": 32,
                "qwen3next.ssm.state_size": 128,
                "qwen3next.ssm.conv_kernel": 4,
                "qwen3next.ssm.inner_size": 4096,
                "qwen3next.full_attention_interval": 4,
            }
        )
        s = ModelSpec.from_gguf_facts(facts)
        g = s.hybrid_gdn
        assert g.linear_num_k_heads == Fact(
            16, "derived-estimate", "gguf-header:qwen3next.ssm.group_count"
        )
        assert g.linear_num_v_heads.source.endswith("ssm.time_step_rank")
        assert g.linear_k_head_dim.value == 128
        # inner_size // time_step_rank — the derivation path IS the source.
        assert g.linear_v_head_dim == Fact(
            128, "derived-estimate",
            "gguf-header:qwen3next.ssm.inner_size//qwen3next.ssm.time_step_rank",
        )
        assert g.linear_conv_kernel_dim.value == 4
        assert g.summary() == "36 GDN / 12 full-attn layers"

    def test_dense_header_has_no_gdn_block(self):
        s = ModelSpec.from_gguf_facts(self._facts())
        assert s.hybrid_gdn is None
        assert "hybrid_gdn" not in self._facts()


class TestSwaConfigPath:
    def _swa_result(self, **cfg_over) -> ModelSpec:
        return ModelSpec.from_derive_result(
            _derive_result(eligible=False, config=_config(**cfg_over))
        )

    def test_gemma_layer_types_split_and_window(self):
        lt = ["sliding_attention"] * 25 + ["full_attention"] * 5
        s = self._swa_result(
            layer_types=lt, sliding_window=1024,
            head_dim=256, global_head_dim=512,
            num_global_key_value_heads=2,
        )
        w = s.swa
        assert w.sliding_window == Fact(1024, "derived", "config.json:sliding_window")
        assert w.num_full_attn_layers == Fact(5, "derived", "config.json:layer_types")
        assert w.num_sliding_attn_layers == Fact(25, "derived", "config.json:layer_types")
        # Gemma4 naming: with global_head_dim present, plain head_dim IS the
        # sliding-layer dim (upstream converter contract).
        assert w.head_dim_sliding == Fact(256, "derived", "config.json:head_dim")
        assert w.global_head_dim == Fact(512, "derived", "config.json:global_head_dim")
        assert w.num_global_kv_heads == Fact(
            2, "derived", "config.json:num_global_key_value_heads"
        )
        assert w.summary() == (
            "1024-token window · 5 full / 25 sliding layers"
        )

    def test_uniform_head_dim_is_never_relabeled_sliding(self):
        """sliding_window alone does NOT make head_dim a sliding-only dim."""
        s = self._swa_result(sliding_window=512, head_dim=128)
        assert s.swa.head_dim_sliding is None
        assert s.swa.global_head_dim is None
        assert s.swa.sliding_window.value == 512

    def test_legacy_int_pattern_is_a_derived_estimate(self):
        s = self._swa_result(
            num_hidden_layers=30, sliding_window_pattern=6, sliding_window=1024,
        )
        est_src = "config.json:num_hidden_layers÷sliding_window_pattern"
        assert s.swa.num_full_attn_layers == Fact(5, "derived-estimate", est_src)
        assert s.swa.num_sliding_attn_layers == Fact(25, "derived-estimate", est_src)


class TestSwaGgufPath:
    @staticmethod
    def _facts(**kv_over) -> dict:
        pattern = [False] + [True] * 5          # every 6th layer global
        kv = {
            "general.architecture": "gemma4",
            "gemma4.embedding_length": 3840,
            "gemma4.block_count": 30,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": pattern * 5,
            "gemma4.attention.key_length": 512,       # gemma4: GLOBAL dim
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.head_count_kv": [2, 8, 8, 8, 8, 8] * 5,
        }
        kv.update(kv_over)
        return D.gguf_spec_facts({"version": 3, "kv": kv}, model_id="org/G")

    def test_pattern_array_counts_and_asymmetric_global_kv(self):
        s = ModelSpec.from_gguf_facts(self._facts())
        w = s.swa
        assert w.sliding_window == Fact(
            1024, "derived-estimate", "gguf-header:gemma4.attention.sliding_window"
        )
        src = "gguf-header:gemma4.attention.sliding_window_pattern"
        assert w.num_full_attn_layers == Fact(5, "derived-estimate", src)
        assert w.num_sliding_attn_layers == Fact(25, "derived-estimate", src)
        assert w.global_head_dim.value == 512
        assert w.head_dim_sliding.value == 256
        assert w.num_global_kv_heads == Fact(
            2, "derived-estimate",
            "gguf-header:gemma4.attention.head_count_kv×sliding_window_pattern",
        )

    def test_plain_llama_header_has_no_swa_block(self):
        kv = {
            "general.architecture": "llama",
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
        }
        facts = D.gguf_spec_facts({"version": 3, "kv": kv}, model_id="x")
        assert "swa" not in facts
        assert ModelSpec.from_gguf_facts(facts).swa is None


class TestMlaConfigPath:
    def _mla_result(self, **cfg_over) -> ModelSpec:
        return ModelSpec.from_derive_result(
            _derive_result(eligible=False, config=_config(**cfg_over))
        )

    def test_deepseek_latent_facts_extracted_verbatim(self):
        s = self._mla_result(
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
        )
        m = s.mla
        assert m.kv_lora_rank == Fact(512, "derived", "config.json:kv_lora_rank")
        assert m.qk_nope_head_dim == Fact(128, "derived", "config.json:qk_nope_head_dim")
        assert m.qk_rope_head_dim == Fact(64, "derived", "config.json:qk_rope_head_dim")
        assert m.summary() == "latent rank 512 · qk 128+64(RoPE)"

    def test_mla_never_touches_core_head_counts(self):
        """The single-latent-KV geometry must NOT leak into num_kv_heads /
        head counts — the core dims stay exactly what config.json says
        (here: the dense fixture's 32/8), never derived from the latent."""
        s = self._mla_result(
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
        )
        assert s.mla is not None
        assert s.num_attn_heads == Fact(32, "derived", "config.json:num_attention_heads")
        assert s.num_kv_heads == Fact(8, "derived", "config.json:num_key_value_heads")


class TestMlaGgufPath:
    @staticmethod
    def _facts(**kv_over) -> dict:
        kv = {
            "general.architecture": "deepseek2",
            "deepseek2.block_count": 60,
        }
        kv.update(kv_over)
        return D.gguf_spec_facts({"version": 3, "kv": kv}, model_id="org/D")

    def test_deepseek2_kvs_map_with_derivation_provenance(self):
        facts = self._facts(
            **{
                "deepseek2.attention.kv_lora_rank": 512,
                "deepseek2.rope.dimension_count": 64,
                "deepseek2.attention.key_length_mla": 192,
            }
        )
        s = ModelSpec.from_gguf_facts(facts)
        m = s.mla
        assert m.kv_lora_rank == Fact(
            512, "derived-estimate", "gguf-header:deepseek2.attention.kv_lora_rank"
        )
        assert m.qk_rope_head_dim == Fact(
            64, "derived-estimate", "gguf-header:deepseek2.rope.dimension_count"
        )
        # qk_nope = key_length_mla − rope.dimension_count (labeled estimate).
        assert m.qk_nope_head_dim == Fact(
            128, "derived-estimate",
            "gguf-header:deepseek2.attention.key_length_mla"
            "-deepseek2.rope.dimension_count",
        )

    def test_rope_alone_is_not_mla_evidence(self):
        """rope.dimension_count exists on EVERY RoPE arch — without
        kv_lora_rank it must not fabricate an MLA block."""
        facts = self._facts(**{"deepseek2.rope.dimension_count": 64})
        assert "mla" not in facts
        assert ModelSpec.from_gguf_facts(facts).mla is None


class TestDllmNoExtraction:
    def test_no_dllm_extractor_exists_by_design(self):
        """dLLM families (diffusiongemma): decode granularity / canvas size
        are serving POLICY, not config.json geometry — there is deliberately
        no DllmFacts block to attach (documented no-extraction)."""
        import scripts.lib.profiles.model_spec as ms_mod

        assert not hasattr(ms_mod, "DllmFacts")
        res = _derive_result(eligible=False, config=_config(
            architectures=["DiffusionGemmaForCausalLM"],
        ))
        s = ModelSpec.from_derive_result(res)
        assert s.hybrid_gdn is None and s.swa is None and s.mla is None

    def _family_spec(self) -> ModelSpec:
        return ModelSpec.from_derive_result(_derive_result(
            eligible=False,
            config=_config(
                # Coherent hybrid: 36 GDN + 12 full-attn over 48 layers.
                layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
                linear_num_key_heads=16, linear_num_value_heads=32,
                sliding_window=1024, global_head_dim=512,
                kv_lora_rank=512, qk_rope_head_dim=64,
                num_hidden_layers=48,
            ),
        ))

    def test_round_trip_lossless_through_real_json(self):
        s = self._family_spec()
        s2 = ModelSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert s2 == s
        assert s2.hybrid_gdn.linear_num_k_heads.value == 16
        assert s2.swa.sliding_window.value == 1024
        assert s2.mla.kv_lora_rank.value == 512

    def test_validate_flags_nonpositive_family_fact(self):
        for blk in (
            HybridGdnFacts(linear_num_k_heads=Fact(-1, "derived", "x")),
            SwaFacts(sliding_window=Fact(0, "derived", "x")),
            MlaFacts(qk_rope_head_dim=Fact(-64, "derived", "x")),
        ):
            spec = ModelSpec(hybrid_gdn=blk if isinstance(blk, HybridGdnFacts) else None,
                             swa=blk if isinstance(blk, SwaFacts) else None,
                             mla=blk if isinstance(blk, MlaFacts) else None)
            issues = spec.validate()
            assert issues and issues[0].startswith(("hybrid_gdn.", "swa.", "mla."))

    def test_validate_warns_when_family_split_ignores_total_layers(self):
        s = ModelSpec(
            num_hidden_layers=Fact(48, "derived", "x"),
            hybrid_gdn=HybridGdnFacts(
                num_gdn_layers=Fact(30, "derived", "y"),
                num_attn_layers=Fact(10, "derived", "z"),
            ),
        )
        issues = s.validate()
        assert any(i.startswith("warn:") and "hybrid_gdn" in i for i in issues)

    def test_validate_clean_on_consistent_family_spec(self):
        assert self._family_spec().validate() == []
