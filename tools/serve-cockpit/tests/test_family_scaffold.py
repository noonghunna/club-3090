"""M5 slice-2 — family extractors render into the promote scaffold like the
MoE experts block:

  - GDN/DeltaNet + SWA: auto-filled summary comment + canonical
    compat.ModelProfile keys, riding the spec skeleton's arch dict too;
  - MLA: COMMENT-ONLY latent geometry (no ModelProfile YAML keys yet —
    inventing one would break strict-loader parity);
  - dLLM / dense specs: every slot None ⇒ byte-for-byte unchanged scaffold.
"""

from __future__ import annotations

import sys
from pathlib import Path

from club3090_cockpit.data import ByoResult, compute_promote_scaffold


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.lib.profiles.model_spec import (  # noqa: E402
    Fact,
    HybridGdnFacts,
    MlaFacts,
    ModelSpec,
    SwaFacts,
)


def _with_slots(mspec: ModelSpec, **slots) -> ModelSpec:
    """Rebuild a frozen spec with the given facts-block slots attached."""
    return ModelSpec(
        **{
            f.name: slots.get(f.name, getattr(mspec, f.name))
            for f in ModelSpec.__dataclass_fields__.values()
        }
    )


def _dense_spec(**family_slots) -> ModelSpec:
    """A minimal GGUF-route dense spec; family slots layered on top."""
    base = ModelSpec.from_gguf_facts({
        "model_id": "org/Synth-7B-GGUF",
        "arch": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attn_heads": 32,
        "num_kv_heads": 8,
        "head_dim_attn": 128,
        "max_ctx_supported": 131072,
        "weights_total_gb": 4.2,
        "valid_tp": [1, 2],
        "confidence": "estimated-lower-bound",
        "facts_provenance": "gguf-header",
    })
    assert base is not None
    return base if not family_slots else _with_slots(base, **family_slots)


def _byo(facts: ModelSpec) -> ByoResult:
    return ByoResult(
        repo="org/Synth-Hybrid-GGUF", profile_like="llama-cpp/q4km",
        arch="gguf", eligible=True, fit_verdict="fits-clean",
        route="G", sibling_slug="llama-cpp/q4km", quant_match="Q4_K_M",
        facts=facts,
    )


class TestFamilyScaffoldRender:
    def test_gdn_block_renders_like_experts_block(self):
        gdn = HybridGdnFacts(
            linear_num_k_heads=Fact(16, "derived-estimate",
                                    "gguf-header:qwen3next.ssm.group_count"),
            linear_num_v_heads=Fact(32, "derived-estimate",
                                    "gguf-header:qwen3next.ssm.time_step_rank"),
            num_gdn_layers=Fact(36, "derived-estimate",
                                "gguf-header:qwen3next.block_count÷"
                                "qwen3next.full_attention_interval"),
            num_attn_layers=Fact(12, "derived-estimate",
                                 "gguf-header:qwen3next.block_count÷"
                                 "qwen3next.full_attention_interval"),
        )
        sc = compute_promote_scaffold(byo=_byo(_dense_spec(hybrid_gdn=gdn)),
                                      measurement=None)
        assert not sc.error
        split_src = (
            "gguf-header:qwen3next.block_count÷"
            "qwen3next.full_attention_interval"
        )
        # The summary tag names the FIRST present fact's source
        # (num_gdn_layers leads the canonical key order).
        assert (
            f"# GDN/DeltaNet hybrid (auto-filled from {split_src}): "
            "36 GDN / 12 full-attn layers"
        ) in sc.profile_yaml
        for key, val in (("num_gdn_layers", 36), ("num_attn_layers", 12),
                         ("linear_num_k_heads", 16), ("linear_num_v_heads", 32)):
            assert f"{key}: {val}" in sc.profile_yaml, key
        # The keys ride the spec skeleton too.
        assert sc.spec["arch"]["num_gdn_layers"] == 36
        assert sc.spec["arch"]["linear_num_v_heads"] == 32

    def test_swa_block_renders_window_and_split(self):
        swa = SwaFacts(
            sliding_window=Fact(1024, "derived-estimate",
                                "gguf-header:gemma4.attention.sliding_window"),
            num_full_attn_layers=Fact(5, "derived-estimate",
                                      "gguf-header:gemma4.attention."
                                      "sliding_window_pattern"),
            num_sliding_attn_layers=Fact(25, "derived-estimate",
                                         "gguf-header:gemma4.attention."
                                         "sliding_window_pattern"),
            head_dim_sliding=Fact(256, "derived-estimate",
                                  "gguf-header:gemma4.attention.key_length_swa"),
            global_head_dim=Fact(512, "derived-estimate",
                                 "gguf-header:gemma4.attention.key_length"),
            num_global_kv_heads=Fact(2, "derived-estimate",
                                     "gguf-header:gemma4.attention.head_count_kv×"
                                     "sliding_window_pattern"),
        )
        sc = compute_promote_scaffold(byo=_byo(_dense_spec(swa=swa)),
                                      measurement=None)
        assert not sc.error
        assert (
            "# sliding-window attention (auto-filled from "
            "gguf-header:gemma4.attention.sliding_window): "
            "1024-token window · 5 full / 25 sliding layers"
        ) in sc.profile_yaml
        for key, val in (("sliding_window", 1024), ("num_full_attn_layers", 5),
                         ("num_sliding_attn_layers", 25),
                         ("head_dim_sliding", 256), ("global_head_dim", 512),
                         ("num_global_kv_heads", 2)):
            assert f"{key}: {val}" in sc.profile_yaml, key
        assert sc.spec["arch"]["num_global_kv_heads"] == 2

    def test_mla_renders_comment_only_never_yaml_keys(self):
        """MLA latent fields have NO compat.ModelProfile YAML key yet — the
        block documents them as comments so strict-loader parity holds."""
        mla = MlaFacts(
            kv_lora_rank=Fact(512, "derived-estimate",
                              "gguf-header:deepseek2.attention.kv_lora_rank"),
            qk_nope_head_dim=Fact(128, "derived-estimate",
                                  "gguf-header:deepseek2.attention.key_length_mla"
                                  "-deepseek2.rope.dimension_count"),
            qk_rope_head_dim=Fact(64, "derived-estimate",
                                  "gguf-header:deepseek2.rope.dimension_count"),
        )
        sc = compute_promote_scaffold(byo=_byo(_dense_spec(mla=mla)),
                                      measurement=None)
        assert not sc.error
        assert (
            "# multi-head latent attention (auto-filled): "
            "latent rank 512 · qk 128+64(RoPE)"
        ) in sc.profile_yaml
        assert "NOT head counts" in sc.profile_yaml
        # No YAML key may exist for the latent fields — comment-only.
        for key in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"):
            lines = [
                ln for ln in sc.profile_yaml.splitlines()
                if ln.startswith(f"{key}:")
            ]
            assert lines == [], key
            assert key not in sc.spec["arch"], key

    def test_non_family_scaffold_byte_identical(self):
        """Dense / dLLM specs keep every slot None — the rendered preview AND
        the spec skeleton are byte-for-byte what they were before slice-2."""
        plain = compute_promote_scaffold(byo=_byo(_dense_spec()),
                                         measurement=None)
        explicit_none = compute_promote_scaffold(
            byo=_byo(_dense_spec(hybrid_gdn=None, swa=None, mla=None)),
            measurement=None,
        )
        assert plain.profile_yaml == explicit_none.profile_yaml
        assert plain.spec["arch"] == explicit_none.spec["arch"]
        for token in ("GDN/DeltaNet hybrid", "sliding-window attention",
                      "multi-head latent attention"):
            assert token not in plain.profile_yaml
