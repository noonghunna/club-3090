#!/bin/sh
''':'
exec python3 "$0" "$@"
':'''
from __future__ import annotations

__doc__ = """kv-calc.py — predict per-card VRAM budget for vLLM composes.

Predicts (per card, after TP split):
  - Model weights
  - KV pool (attention layers only — recurrent / SSM states show up in activation)
  - Activation peak (model-specific: Qwen GDN forward, Gemma SWA + dense MLP)
  - Cudagraph + workspace overhead
  - Drafter overhead (MTP / DFlash)
  - Total vs available VRAM
  - Verdict: PASS / TIGHT / FAIL

Four models modelled:
  - Qwen 3.6 27B (DeltaNet hybrid: 16 full_attention + 48 GDN)
  - Qwen 3.6 35B-A3B (MoE + DeltaNet hybrid: 10 attention + 30 GDN)
  - Gemma 4 31B (SWA + dense MLP: 10 full_attention + 50 sliding_attention)
  - Gemma 4 26B-A4B (MoE + SWA: 5 full_attention + 25 sliding_attention)

vLLM rate-limits KV pool to fit available budget; this predictor models that
capping behavior. When the requested KV pool exceeds what fits, the verdict
is TIGHT (vLLM will cap pool — effective concurrency reduced) not FAIL.

Anchored to:
  - PerfMamba (arxiv 2511.22849) — Qwen GDN block-wise state materialization scaling
    https://arxiv.org/html/2511.22849
  - TurboQuant (arxiv 2504.19874, ICLR 2026) — TQ3 byte savings
    https://arxiv.org/abs/2504.19874
  - PagedAttention (arxiv 2309.06180) — KV pool layout
    https://arxiv.org/abs/2309.06180

Calibrated against measured BENCHMARKS.md rows per model. Coefficients reflect
club-3090's empirical findings. See docs/KV_MATH.md for the derivation +
calibration trace.

Usage:
  bash tools/kv-calc.py --compose dual-turbo --vram 24                                     # Qwen (default model)
  bash tools/kv-calc.py --model gemma-4-31b --compose gemma-dual-int8 --vram 24
  bash tools/kv-calc.py --model gemma-4-31b --solve-max-ctx --kv-format int8_per_token_head --tp 2 --vram 24
  bash tools/kv-calc.py --calibration  # all calibrated models, grouped per-model
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.profiles.compat import load_profiles
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY


# =============================================================================
# Model specs
# =============================================================================

def _load_profiles_silent():
    logger = logging.getLogger("compat")
    old_disabled, old_env = logger.disabled, os.environ.get("CLUB3090_LOG_LEVEL")
    logger.disabled = True
    os.environ["CLUB3090_LOG_LEVEL"] = "CRITICAL"
    try:
        return load_profiles()
    finally:
        logger.disabled = old_disabled
        if old_env is None:
            os.environ.pop("CLUB3090_LOG_LEVEL", None)
        else:
            os.environ["CLUB3090_LOG_LEVEL"] = old_env


PROFILES = _load_profiles_silent()


def _weight_size(model, variant):
    value = model.weights[variant]["size_gb"]
    if not isinstance(value, (int, float)):
        raise ValueError(f"weight size for {model.id}/{variant} is not numeric: {value}")
    return float(value)


def _load_model_specs_from_yaml(profiles):
    qwen, gemma = profiles.models["qwen3.6-27b"], profiles.models["gemma-4-31b"]
    qwen_moe, gemma_moe = profiles.models["qwen3.6-35b-a3b"], profiles.models["gemma-4-26b-a4b"]
    gemma12 = profiles.models["gemma-4-12b"]
    q_fields = ("hidden_size", "num_hidden_layers", "num_gdn_layers", "num_attn_layers", "num_attn_heads", "num_kv_heads", "head_dim_attn", "linear_num_v_heads", "linear_num_k_heads", "linear_v_head_dim", "linear_k_head_dim", "linear_conv_kernel_dim", "max_ctx_supported", "attention_k_eq_v")
    g_fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_full_attn_layers", "num_sliding_attn_layers", "num_attn_heads", "num_kv_heads", "head_dim_sliding", "global_head_dim", "sliding_window", "max_ctx_supported", "attention_k_eq_v")
    gm_fields = (*g_fields, "num_global_kv_heads", "num_experts", "num_experts_per_tok", "moe_intermediate_size", "active_params_b", "mtp_num_hidden_layers")
    qm_fields = (*q_fields, "num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size", "active_params_b", "mtp_num_hidden_layers")
    qspec = {"model_id": qwen.id, "model_family": qwen.family, **{k: getattr(qwen, k) for k in q_fields}, "valid_tp": list(qwen.valid_tp), "weights_total_gb": _weight_size(qwen, qwen.default_weight_variant), "mamba_state_bytes": 4, "chunk_size": 256, "mtp_n_default": profiles.drafters["qwen-mtp-builtin"].n_default}
    qmspec = {"model_id": qwen_moe.id, "model_family": qwen_moe.family, **{k: getattr(qwen_moe, k) for k in qm_fields}, "valid_tp": list(qwen_moe.valid_tp), "weights_total_gb": _weight_size(qwen_moe, qwen_moe.default_weight_variant), "weights_gptq_gb": _weight_size(qwen_moe, "gptq_int4"), "mamba_state_bytes": 4, "chunk_size": 256, "mtp_n_default": profiles.drafters["qwen-mtp-builtin"].n_default}
    gspec = {"model_id": gemma.id, "model_family": gemma.family, **{k: getattr(gemma, k) for k in g_fields}, "valid_tp": list(gemma.valid_tp), "weights_int4_gb": _weight_size(gemma, "autoround-int4"), "weights_awq_gb": _weight_size(gemma, "awq"), "weights_bf16_gb": _weight_size(gemma, "bf16"), "drafter_mtp_gb": float(profiles.drafters["gemma-it-assistant"].vram_footprint_gb), "drafter_dflash_gb": float(profiles.drafters["gemma-dflash"].vram_footprint_gb), "mtp_n_default": profiles.drafters["gemma-it-assistant"].n_default}
    gmspec = {"model_id": gemma_moe.id, "model_family": gemma_moe.family, **{k: getattr(gemma_moe, k) for k in gm_fields}, "valid_tp": list(gemma_moe.valid_tp), "weights_int4_gb": _weight_size(gemma_moe, "autoround-int4-mixed"), "weights_awq_gb": _weight_size(gemma_moe, "awq"), "drafter_mtp_gb": float(profiles.drafters["gemma-26b-it-assistant"].vram_footprint_gb), "mtp_n_default": profiles.drafters["gemma-26b-it-assistant"].n_default}
    # Gemma-4-12B (gemma4_unified arch). Its TEXT backbone is gemma4-swa-dense
    # shaped (same SWA KV math family as gemma-4-31b), so it rides the SAME
    # "gemma4-swa-dense" prediction path — we set model_family to that internal
    # KV-family tag (NOT its ModelProfile family "gemma4-unified", which is the
    # vLLM arch name). It ships bf16 weights + an Intel AutoRound INT8 variant
    # (~13 GB, W8A16) + a small assistant drafter; there is no int4/awq variant,
    # so weights_int4_gb/weights_awq_gb point at the bf16 blob (23.9 GB) to keep
    # _weights_per_card_gb()'s branch from KeyError'ing, while weights_int8_gb
    # carries the real INT8 footprint (vllm/gemma-12b-single-int8-mtp path).
    # Activation/overhead constants stay the SHARED Gemma dense constants (NOT
    # re-tuned). The ONE measured calibration is the growing KV per-token —
    # `measured_kv_growing_bpt_tp1` (see kv_pool_per_card_bytes): gemma4_unified's
    # global-layer KV measured 1.44x LOWER than the 31B-derived global-only
    # formula predicts, so we ride the measurement for the 12B only.
    g12_bf16 = _weight_size(gemma12, "bf16")
    g12_int8 = _weight_size(gemma12, "autoround-int8")
    g12spec = {"model_id": gemma12.id, "model_family": "gemma4-swa-dense", **{k: getattr(gemma12, k) for k in g_fields}, "valid_tp": list(gemma12.valid_tp), "weights_int4_gb": g12_bf16, "weights_awq_gb": g12_bf16, "weights_bf16_gb": g12_bf16, "weights_int8_gb": g12_int8, "drafter_mtp_gb": float(profiles.drafters["gemma-12b-it-assistant"].vram_footprint_gb), "measured_kv_growing_bpt_tp1": 45632, "mtp_n_default": profiles.drafters["gemma-12b-it-assistant"].n_default}
    return {
        "qwen3.6-27b": qspec,
        "qwen3.6-35b-a3b": qmspec,
        "gemma-4-31b": gspec,
        "gemma-4-26b-a4b": gmspec,
        "gemma-4-12b": g12spec,
    }


MODEL_SPECS = _load_model_specs_from_yaml(PROFILES)
QWEN36_27B = MODEL_SPECS["qwen3.6-27b"]
QWEN36_35B_A3B = MODEL_SPECS["qwen3.6-35b-a3b"]
GEMMA4_31B = MODEL_SPECS["gemma-4-31b"]
GEMMA4_26B_A4B = MODEL_SPECS["gemma-4-26b-a4b"]
GEMMA4_12B = MODEL_SPECS["gemma-4-12b"]


# =============================================================================
# KV format bytes-per-element
# =============================================================================
# (one element = one head dim of one head; K and V counted separately for
# models where attention_k_eq_v=False. For Gemma the K==V tying halves this
# at the formula level — see kv_pool_per_card_bytes.)
# Source: vLLM/HF docs + TurboQuant paper + PR #40391 (INT8 per-token-head).

KV_FORMAT_BYTES = {
    "fp16":                  2.0,
    "bf16":                  2.0,
    "fp8_e5m2":              1.0,
    "fp8_e4m3":              1.0,
    "int8_per_token_head":   1.01,        # 1.0 int8 + per-token-head fp16 scale (~1% amortized)
    "q4_0":                  0.5 + 0.0625, # 4-bit + per-group scale
    "k8v4":                  0.75,         # avg of K=int8 V=int4
    "turboquant_3bit_nc":    0.375 + 0.05, # 3 bits + small QJL overhead
}

INDEXER_FORMAT_BYTES = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8_e5m2": 1.0,
    "fp8_e4m3": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
}


# =============================================================================
# Per-model activation coefficients
# =============================================================================

# ---- Qwen GDN activation-peak per-layer per-token coefficient (bytes) ----
# Calibrated empirically against measured BENCHMARKS rows. PerfMamba's
# O(γ·D·N·L) scaling sets the form; this captures fla.ops.chunk
# implementation details + KV-format-dependent dequant overhead.
QWEN_GDN_ACTIVATION_COEF = {
    "fp16":               135,
    "bf16":               135,
    "fp8_e5m2":           130,
    "fp8_e4m3":           130,
    "int8_per_token_head": 130,
    "q4_0":               155,
    "k8v4":               155,
    "turboquant_3bit_nc": 165,
}

# ---- Qwen MoE activation + built-in MTP workspace ----
# Path-B low-anchor fit from the two v0.7.3 preview rows. The per-token GDN
# coefficient follows the dense-Qwen shape; the small constant captures MoE
# expert dispatch/router buffers. NOTE: vLLM shards both attention and MoE
# expert weights across TP ranks, so weights ARE divided by TP for
# qwen3-next-moe in _weights_per_card_gb() (fixed in #260). The ≈budget peak
# on the live rows is the elastic KV pool filling spare VRAM, not resident
# full-quant weights — the old no-/tp assumption over-predicted long-ctx
# configs (e.g. 262K) into a false FAIL.
QWEN_MOE_ACTIVATION_COEF = {
    "fp16":               110,
    "bf16":               110,
    "fp8_e5m2":           105,
    "fp8_e4m3":           105,
    "int8_per_token_head": 105,
    "q4_0":               130,
    "k8v4":               130,
    "turboquant_3bit_nc": 140,
}
QWEN_MOE_EXPERT_DISPATCH_GB = 0.20
QWEN_MOE_BUILTIN_MTP_WORKSPACE_GB = 0.10

# ---- Gemma activation peak (mostly constant in ctx) ----
# Unlike Qwen GDN, Gemma's activation peak comes from dense MLP forward +
# SWA windowed-attention prefill, both bounded by chunked-prefill chunk_size.
# Result: roughly CONSTANT in max_ctx. Calibrated as a per-TP base (GB) plus
# a small per-token term to capture residual ctx scaling.
GEMMA_ACTIVATION_CONST_GB = 1.5     # per card at TP=1 — calibrated, ~scales as 1/TP
GEMMA_ACTIVATION_PER_TOKEN_BYTES = 8  # tiny ctx scaling term to keep solver well-behaved

# ---- Gemma MoE activation peak ----
# Low-anchor fit from awq.yml and awq-mtp.yml. MoE dispatch is folded into the
# constant term; external assistant weights are modelled separately via
# drafter_gb.
GEMMA_MOE_ACTIVATION_CONST_GB = 1.8
GEMMA_MOE_ACTIVATION_PER_TOKEN_BYTES = 12

# ---- generic-dense activation peak (conservative lower-bound on fit) ----
# P1 / v0.8.0 Pull-Gate stratum-5/6: for an uncurated standard-dense
# (MHA/GQA) model we have no measured anchor, so we deliberately OVER-price
# the activation peak to guarantee generic-dense never predicts more fit
# than a curated exact branch would for an equivalent shape.
#
# The form is a per-token coefficient × hidden_size × max_ctx (per card,
# divided by TP). The coefficient is chosen to be at least as conservative
# as the most conservative curated dense activation term in this file. The
# most conservative curated dense per-token term is Qwen GDN (turboquant)
# at 165 bytes/layer/token; expressed per hidden-unit that is
# 165 * num_gdn_layers / hidden_size ≈ 165 * 64 / 2048 ≈ 5.16 bytes per
# hidden-unit-token for the dense Qwen 27B shape. We round UP to 6.0 to stay
# strictly on the conservative (lower-bound-on-fit) side for arbitrary
# dense shapes, and add a fixed per-card floor at least as large as Gemma's
# constant dense activation term (1.5 GB) so small-ctx configs are never
# under-priced relative to the curated Gemma dense branch.
GENERIC_DENSE_ACTIVATION_COEF_PER_HIDDEN = 6.0  # bytes per (hidden_size × token); ≥ most conservative curated dense
GENERIC_DENSE_ACTIVATION_FLOOR_GB = 1.5         # ≥ Gemma dense constant activation term


# =============================================================================
# Compose presets (per-model)
# =============================================================================
COMPOSE_ALIAS_TEXT = {
    "qwen3.6-27b": "minimal=vllm/minimal dual=vllm/dual",
    "qwen3.6-35b-a3b": "qwen-a3b-preview-single=vllm/qwen-a3b-preview-single qwen-35b-a3b-dual=vllm/qwen-35b-a3b-dual",
    "gemma-4-31b": "gemma-dual=vllm/gemma-bf16-mtp gemma-dual-int8=vllm/gemma-int8-mtp gemma-single=vllm/gemma-mtp-tp1",
    # gemma-4-12b legacy alias namespace is keyed by model id, so reusing the
    # bare `gemma-dual` string here is harmless — compat + the CLI always pass
    # an explicit --model, and the reverse map is keyed by (unique) registry
    # slug. `gemma-dual` → the MTP dual; `gemma-no-mtp` → the no-drafter dual.
    "gemma-4-12b": "gemma-dual=vllm/gemma-12b-dual-bf16-mtp gemma-single-int8-mtp=vllm/gemma-12b-single-int8-mtp",
    "gemma-4-26b-a4b": "gemma-a4b-single=vllm/gemma-a4b-single gemma-a4b=vllm/gemma-a4b gemma-a4b-awq=vllm/gemma-a4b-awq gemma-a4b-awq-mtp=vllm/gemma-a4b-awq-mtp",
}
COMPOSE_ALIASES = {model: tuple(part.split("=", 1) for part in text.split()) for model, text in COMPOSE_ALIAS_TEXT.items()}

REGISTRY_TO_LEGACY_COMPOSE = {
    registry: legacy
    for aliases in COMPOSE_ALIASES.values()
    for legacy, registry in aliases
}

COMPOSE_COMPAT_OVERRIDES = {
    ("qwen3.6-27b", "minimal"): {"max_num_seqs": 4, "mem_util": 0.90},
    ("qwen3.6-27b", "dual"): {"mem_util": 0.95},
    ("gemma-4-31b", "gemma-single"): {"kv_format": "fp8_e5m2"},
}


def _compose_cfg_from_registry(profiles, model_id, legacy_name, registry_name):
    entry = COMPOSE_REGISTRY[registry_name]
    drafter = profiles.drafters[entry["drafter"]] if entry.get("drafter") else None
    cfg = {k: entry[k] for k in ("max_ctx", "max_num_seqs", "tp", "kv_format", "mem_util")}
    cfg["mtp"] = drafter is not None and drafter.spec_method in ("mtp", "mtp_assistant")
    if drafter is not None and drafter.spec_method == "dflash":
        cfg.update({"mtp": False, "dflash_draft_gb": float(drafter.vram_footprint_gb)})
    if drafter is not None:
        cfg["mtp_n"] = int(drafter.n_default)
    # gemma-4-12b rides the shared Gemma dense path (model_family
    # gemma4-swa-dense in MODEL_SPECS); treat its drafter + weights the same way.
    if model_id in ("gemma-4-31b", "gemma-4-12b") and drafter is not None:
        cfg["drafter_gb"] = float(drafter.vram_footprint_gb)
    if model_id in ("gemma-4-31b", "gemma-4-12b"):
        cfg["weights_variant"] = {"awq": "awq", "bf16": "bf16", "autoround-int8": "int8"}.get(entry["weights_variant"], "int4")
    if model_id == "gemma-4-26b-a4b":
        cfg["weights_variant"] = "awq" if entry["weights_variant"] == "awq" else "int4"
        if drafter is not None:
            cfg["drafter_gb"] = float(drafter.vram_footprint_gb)
    if model_id == "qwen3.6-35b-a3b":
        cfg["weights_variant"] = "gptq" if entry["weights_variant"] == "gptq_int4" else "default"
    cfg.update(COMPOSE_COMPAT_OVERRIDES.get((model_id, legacy_name), {}))
    return cfg


COMPOSES = {
    model_id: {legacy: _compose_cfg_from_registry(PROFILES, model_id, legacy, registry) for legacy, registry in aliases}
    for model_id, aliases in COMPOSE_ALIASES.items()
}


# =============================================================================
# Calibration: measured BENCHMARKS rows (peak per-card VRAM during bench)
# =============================================================================

CALIBRATION = {
    model_id: [
        (REGISTRY_TO_LEGACY_COMPOSE[row["compose"]], row["vram_gb"], row["measured_peak_gb"], row.get("ctx_override"), row.get("source", ""))
        for row in cal.rows
    ]
    for model_id, cal in PROFILES.calibration.items()
}


# =============================================================================
# Prediction
# =============================================================================

@dataclass
class Prediction:
    model: str
    weights_gb: float
    kv_pool_requested_gb: float
    kv_pool_actual_gb: float          # capped at available budget (vLLM behavior)
    kv_pool_sliding_fixed_gb: float   # Gemma sliding-window fixed term (0 for Qwen)
    activation_gb: float
    cudagraph_overhead_gb: float
    drafter_gb: float
    total_gb: float
    vram_gb: float
    budget_gb: float
    pct_of_vram: float
    verdict: str
    notes: list[str]


@dataclass
class CacheBreakdown:
    layout: str
    sequences: int
    gpus: int
    tp: int
    attention_kv_growing_gb: float
    attention_kv_sliding_fixed_gb: float
    recurrent_state_gb: float
    compressed_kv_gb: float
    indexer_cache_gb: float
    draft_kv_gb: float
    total_cache_gb: float
    notes: list[str]


def _weights_per_card_gb(spec, tp, weights_variant="default"):
    """Return per-card weights footprint in GB after TP split."""
    if spec["model_family"] == "qwen3-next-hybrid":
        return spec["weights_total_gb"] / tp
    elif spec["model_family"] == "qwen3-next-moe":
        # vLLM shards attention + MoE expert weights across TP ranks, so
        # per-card weights are /tp (#260). The old no-/tp assumption was
        # mis-inferred from the live ≈budget peak — but that peak is the
        # elastic KV pool filling spare VRAM (vLLM allocates all available
        # KV memory as blocks), not resident full-quant weights. Not dividing
        # over-predicted the fixed footprint and turned long-ctx configs
        # (e.g. 262K dual, which fits) into a false FAIL.
        if weights_variant == "gptq":
            return spec["weights_gptq_gb"] / tp
        return spec["weights_total_gb"] / tp
    elif spec["model_family"] == "gemma4-swa-dense":
        if weights_variant == "awq":
            return spec["weights_awq_gb"] / tp
        elif weights_variant == "bf16":
            return spec["weights_bf16_gb"] / tp
        elif weights_variant == "int8":
            return spec["weights_int8_gb"] / tp
        else:  # int4 default
            return spec["weights_int4_gb"] / tp
    elif spec["model_family"] == "gemma4-swa-moe":
        # NOT /tp'd (unlike qwen3-next-moe, fixed in #260) — deliberately left
        # pending its own re-calibration. The v0.7.3 AWQ rows peak at 23.45–
        # 23.50 GB/card (TP=2, ~0.95 GB above the 22.08 budget), so the current
        # no-/tp footprint produces a *protective* TIGHT verdict there. Dividing
        # by TP without a measured fixed-footprint anchor would relax that to
        # PASS for a config that genuinely runs at 23.5/24. Re-do with a proper
        # boot-log anchor (KV-pool tokens → fixed = peak − pool) before /tp'ing.
        if weights_variant == "int4":
            return spec["weights_int4_gb"]
        return spec["weights_awq_gb"]
    elif spec["model_family"] == "generic-dense":
        # Weight authority is the caller-supplied summed selected-blob size
        # (HF siblings, populated by P2). generic-dense NEVER recomputes
        # weights from params — it trusts the provided blob-size GB. Dense
        # weights are sharded evenly across TP ranks.
        return spec["weights_total_gb"] / tp
    raise ValueError(f"Unknown model_family: {spec['model_family']}")


def kv_pool_per_card_bytes(spec, kv_format, max_ctx, max_num_seqs, tp, mtp_n=0):
    """Per-card KV pool bytes (growing portion only).

    Returns a tuple (growing_per_card_bytes, sliding_fixed_per_card_bytes).
    Sliding term is zero for models without sliding-window layers.

    For Qwen 3.6 (DeltaNet hybrid):
      Only the 16 full_attention layers grow KV. GDN layers have a fixed-size
      recurrent state (not seq-len-dependent), so they show up in activation.
      K and V stored independently → ×2 factor.

    For Gemma 4 (SWA + dense MLP):
      Only the full_attention layers grow KV (at global_head_dim); the count is
      read from the spec, not hardcoded — 10/50 full/sliding on the 31B, 8/40 on
      the 12B. The sliding_attention layers hold a FIXED window (constant in ctx,
      separate small term). K==V tying IS exploited by vLLM's allocator → ×1
      factor (calibrated against BENCHMARKS data; see docs/KV_MATH.md).
      gemma4_unified (12B) overrides the growing per-token with a MEASURED
      constant (`measured_kv_growing_bpt_tp1`) — see the branch below.
    """
    bpe = KV_FORMAT_BYTES[kv_format]

    if spec["model_family"] == "qwen3-next-hybrid":
        # K and V stored independently
        per_token = (
            spec["num_attn_layers"]
            * spec["num_kv_heads"]
            * spec["head_dim_attn"]
            * 2  # K + V
            * bpe
        )
        effective_ctx = max_ctx + mtp_n * 32
        growing = (per_token / tp) * effective_ctx * max_num_seqs
        return growing, 0.0

    elif spec["model_family"] == "qwen3-next-moe":
        # K and V stored independently. GDN recurrent state is fixed-size and
        # per-stream, not context-linear.
        per_token = (
            spec["num_attn_layers"]
            * spec["num_kv_heads"]
            * spec["head_dim_attn"]
            * 2
            * bpe
        )
        effective_ctx = max_ctx + mtp_n * 32
        growing = (per_token / tp) * effective_ctx * max_num_seqs
        recurrent_per_stream = (
            spec["num_gdn_layers"]
            * (
                spec["linear_num_v_heads"] * spec["linear_v_head_dim"]
                + spec["linear_num_k_heads"] * spec["linear_k_head_dim"]
                + spec["linear_conv_kernel_dim"] * spec["hidden_size"]
            )
            * 2  # recurrent state kept in bf16 on this stack
        )
        recurrent_fixed = recurrent_per_stream * max_num_seqs
        return growing, recurrent_fixed

    elif spec["model_family"] == "gemma4-swa-dense":
        # K==V tied → ×1 storage
        measured_bpt_tp1 = spec.get("measured_kv_growing_bpt_tp1")
        if measured_bpt_tp1 is not None:
            # gemma4_unified (gemma-4-12b) MEASURED calibration. The 31B-derived
            # global-only formula (below) predicts 65,536 B/tok TP1 for the 12B
            # (8 full x 8 kv x 512 global_head_dim x bpe2), but the LIVE
            # gemma4_unified pool measured 22,816 B/tok/card = 45,632 TP1
            # (8.16 GiB available KV / 384,019 tokens @ 131K / TP2 / mem-util
            # 0.90, MTP, 2026-06-04) — 1.44x LOWER. Cause: gemma4_unified's
            # unified-KV global layers + vLLM's hybrid-SWA pool accounting differ
            # from the 31B's gemma4-swa-dense. bf16 baseline; scale by bpe for
            # quantized KV. Single-anchor measured calibration; the precise arch
            # decomposition is a TODO (needs vLLM kv_cache_utils source or >1
            # anchor). Set ONLY on the 12B spec, so the 31B keeps the formula.
            per_token_growing = measured_bpt_tp1 * (bpe / 2.0)
        else:
            per_token_growing = (
                spec["num_full_attn_layers"]
                * spec["num_kv_heads"]
                * spec["global_head_dim"]
                * 1  # K==V tied; vLLM stores once
                * bpe
            )
        # No MTP draft-token bump on Gemma — drafter is a separate model
        growing = (per_token_growing / tp) * max_ctx * max_num_seqs

        # Sliding-window fixed term — 50 layers × window × head_dim × 1 × bpe
        sliding_fixed_total = (
            spec["num_sliding_attn_layers"]
            * spec["num_kv_heads"]
            * spec["head_dim_sliding"]
            * 1  # K==V tied here too
            * bpe
            * spec["sliding_window"]
        )
        sliding_per_card = sliding_fixed_total / tp
        return growing, sliding_per_card

    elif spec["model_family"] == "gemma4-swa-moe":
        # K==V tied. Global layers use their own KV-head count; sliding
        # layers keep the windowed KV head count.
        per_token_growing = (
            spec["num_full_attn_layers"]
            * spec["num_global_kv_heads"]
            * spec["global_head_dim"]
            * 1
            * bpe
        )
        growing = (per_token_growing / tp) * max_ctx * max_num_seqs
        sliding_fixed_total = (
            spec["num_sliding_attn_layers"]
            * spec["num_kv_heads"]
            * spec["head_dim_sliding"]
            * 1
            * bpe
            * spec["sliding_window"]
        )
        sliding_per_card = sliding_fixed_total / tp
        return growing, sliding_per_card

    elif spec["model_family"] == "generic-dense":
        # Standard MHA/GQA: every hidden layer grows KV. K and V stored
        # independently (no K==V tying assumption for an arbitrary uncurated
        # model — the conservative choice → ×2). Same shape as the curated
        # Qwen dense branch (num_layers × num_kv_heads × head_dim × 2 × bpe),
        # generalized to all layers since a pure-dense model has no GDN/SWA
        # split. No sliding-window fixed term for generic-dense.
        per_token = (
            spec["num_hidden_layers"]
            * spec["num_kv_heads"]
            * spec["head_dim_attn"]
            * 2  # K + V stored independently (conservative; no K==V tying)
            * bpe
        )
        growing = (per_token / tp) * max_ctx * max_num_seqs
        return growing, 0.0

    raise ValueError(f"Unknown model_family: {spec['model_family']}")



def _auto_kv_layout(spec) -> str:
    family = spec["model_family"]
    if family in {"qwen3-next-hybrid", "qwen3-next-moe"}:
        return "hybrid_mamba"
    if family in {"gemma4-swa-dense", "gemma4-swa-moe"}:
        return "sliding_window"
    return "dense_gqa"


def _recurrent_state_per_card_bytes(spec, max_num_seqs, tp):
    """Non-attention recurrent/SSM state, reported separately from KV.

    Qwen3-Next GDN/DeltaNet state is not attention KV. It is replicated per
    running sequence and should be visible in planning output because it
    competes for the same per-card VRAM budget as the attention KV pool.
    """
    if spec["model_family"] not in {"qwen3-next-hybrid", "qwen3-next-moe"}:
        return 0.0
    required = (
        "num_gdn_layers", "linear_num_v_heads", "linear_v_head_dim",
        "linear_num_k_heads", "linear_k_head_dim", "linear_conv_kernel_dim",
        "hidden_size",
    )
    if any(spec.get(k) is None for k in required):
        return 0.0
    bytes_per_elem = spec.get("mamba_state_bytes", 2)
    recurrent_per_stream = (
        spec["num_gdn_layers"]
        * (
            spec["linear_num_v_heads"] * spec["linear_v_head_dim"]
            + spec["linear_num_k_heads"] * spec["linear_k_head_dim"]
            + spec["linear_conv_kernel_dim"] * spec["hidden_size"]
        )
        * bytes_per_elem
    )
    # Runtime state is per rank, not sharded like dense weights.
    return recurrent_per_stream * max_num_seqs


def _indexer_cache_per_card_bytes(
    *,
    max_ctx,
    max_num_seqs,
    tp,
    indexer_ratio_layers=0,
    indexer_compress_ratio=4,
    indexer_head_dim=None,
    indexer_format="fp4",
):
    if not indexer_ratio_layers:
        return 0.0
    if indexer_head_dim is None or indexer_head_dim <= 0:
        raise ValueError("--indexer-ratio-layers requires --indexer-head-dim")
    if indexer_compress_ratio <= 0:
        raise ValueError("--indexer-compress-ratio must be positive")
    bpe = INDEXER_FORMAT_BYTES[indexer_format]
    indexed_tokens = max_ctx // indexer_compress_ratio
    return (
        indexer_ratio_layers
        * indexed_tokens
        * indexer_head_dim
        * bpe
        * max_num_seqs
        / tp
    )


def _compressed_kv_per_card_bytes(
    *,
    max_ctx,
    max_num_seqs,
    tp,
    compressed_layers=0,
    compression_ratio=128,
    compressed_head_dim=None,
    kv_format="fp8_e5m2",
):
    if not compressed_layers:
        return 0.0
    if compressed_head_dim is None or compressed_head_dim <= 0:
        raise ValueError("--compressed-layers requires --compressed-head-dim")
    if compression_ratio <= 0:
        raise ValueError("--compression-ratio must be positive")
    bpe = KV_FORMAT_BYTES[kv_format]
    compressed_tokens = max_ctx // compression_ratio
    return (
        compressed_layers
        * compressed_tokens
        * compressed_head_dim
        * bpe
        * max_num_seqs
        / tp
    )


def architecture_cache_breakdown(
    spec,
    kv_format,
    max_ctx,
    max_num_seqs,
    tp,
    *,
    gpus=1,
    layout="auto",
    mtp=False,
    include_draft_kv=False,
    draft_kv_gb=0.0,
    compressed_layers=0,
    compression_ratio=128,
    compressed_head_dim=None,
    indexer_ratio_layers=0,
    indexer_compress_ratio=4,
    indexer_head_dim=None,
    indexer_format="fp4",
) -> CacheBreakdown:
    """Raw per-card cache/state breakdown for home-rig planning.

    This is intentionally reporting-only. predict() remains the calibrated
    fit authority used by existing scripts; this function exposes the extra
    architecture buckets that generic KV calculators show explicitly.
    """
    _validate_tp_for_spec(spec, tp)
    if gpus < 1 or gpus > 8:
        raise ValueError("--gpus must be between 1 and 8 for single-node home/workstation planning")
    if max_num_seqs < 1:
        raise ValueError("sequences/max_num_seqs must be positive")

    mtp_n = int(spec.get("mtp_n_default", 3)) if mtp else 0
    growing_b, fixed_b = kv_pool_per_card_bytes(
        spec, kv_format, max_ctx, max_num_seqs, tp, mtp_n=mtp_n,
    )

    family = spec["model_family"]
    sliding_fixed_b = fixed_b if family in {"gemma4-swa-dense", "gemma4-swa-moe"} else 0.0
    recurrent_b = fixed_b if family == "qwen3-next-moe" else _recurrent_state_per_card_bytes(spec, max_num_seqs, tp)

    compressed_b = _compressed_kv_per_card_bytes(
        max_ctx=max_ctx,
        max_num_seqs=max_num_seqs,
        tp=tp,
        compressed_layers=compressed_layers,
        compression_ratio=compression_ratio,
        compressed_head_dim=compressed_head_dim,
        kv_format=kv_format,
    )
    indexer_b = _indexer_cache_per_card_bytes(
        max_ctx=max_ctx,
        max_num_seqs=max_num_seqs,
        tp=tp,
        indexer_ratio_layers=indexer_ratio_layers,
        indexer_compress_ratio=indexer_compress_ratio,
        indexer_head_dim=indexer_head_dim,
        indexer_format=indexer_format,
    )

    draft_kv_b = draft_kv_gb * 1e9 if include_draft_kv else 0.0
    layout = _auto_kv_layout(spec) if layout == "auto" else layout
    total_b = growing_b + sliding_fixed_b + recurrent_b + compressed_b + indexer_b + draft_kv_b

    notes = []
    if include_draft_kv and draft_kv_gb <= 0:
        notes.append("draft KV requested, but no --draft-kv-gb override was provided; built-in MTP token bump is already included when --mtp is set")
    if compressed_layers or indexer_ratio_layers:
        notes.append("compressed/indexer buckets are architecture-level estimates; calibrated fit verdict still comes from the main prediction path")
    if gpus != tp:
        notes.append(f"planning {gpus} GPU(s) with TP={tp}; per-card memory is sharded by TP, not by idle estate cards")

    return CacheBreakdown(
        layout=layout,
        sequences=max_num_seqs,
        gpus=gpus,
        tp=tp,
        attention_kv_growing_gb=growing_b / 1e9,
        attention_kv_sliding_fixed_gb=sliding_fixed_b / 1e9,
        recurrent_state_gb=recurrent_b / 1e9,
        compressed_kv_gb=compressed_b / 1e9,
        indexer_cache_gb=indexer_b / 1e9,
        draft_kv_gb=draft_kv_b / 1e9,
        total_cache_gb=total_b / 1e9,
        notes=notes,
    )


def activation_peak_per_card_bytes(spec, kv_format, max_ctx, tp):
    """Per-card peak activation during prefill forward.

    For Qwen 3.6 (DeltaNet GDN): linear in seq_len, KV-format-dependent
      coefficient (PerfMamba O(γ·D·N·L) form, fla.ops.chunk implementation
      details calibrated empirically).

    For Gemma 4 (dense MLP + SWA): mostly CONSTANT in seq_len because chunked
      prefill bounds the MLP intermediate. Small per-token residual to keep
      the solver smooth.
    """
    if spec["model_family"] == "qwen3-next-hybrid":
        coef = QWEN_GDN_ACTIVATION_COEF[kv_format]
        return (coef * spec["num_gdn_layers"] * max_ctx) / tp

    elif spec["model_family"] == "qwen3-next-moe":
        coef = QWEN_MOE_ACTIVATION_COEF[kv_format]
        return (coef * spec["num_gdn_layers"] * max_ctx) / tp + QWEN_MOE_EXPERT_DISPATCH_GB * 1e9

    elif spec["model_family"] == "gemma4-swa-dense":
        const_bytes = GEMMA_ACTIVATION_CONST_GB * 1e9
        per_token = GEMMA_ACTIVATION_PER_TOKEN_BYTES * max_ctx
        return (const_bytes + per_token) / tp

    elif spec["model_family"] == "gemma4-swa-moe":
        const_bytes = GEMMA_MOE_ACTIVATION_CONST_GB * 1e9
        per_token = GEMMA_MOE_ACTIVATION_PER_TOKEN_BYTES * max_ctx
        return (const_bytes + per_token) / tp

    elif spec["model_family"] == "generic-dense":
        # Conservative hidden_size-based coefficient × max_ctx, plus a fixed
        # per-card floor ≥ the most conservative curated dense constant
        # (Gemma 1.5 GB). The coefficient (6.0 B / hidden-unit / token) is
        # ≥ the most conservative curated dense per-token term re-expressed
        # per hidden-unit (Qwen GDN turboquant ≈ 5.16). This guarantees a
        # lower-bound on fit vs. an equivalent curated dense shape.
        per_token = (
            GENERIC_DENSE_ACTIVATION_COEF_PER_HIDDEN
            * spec["hidden_size"]
            * max_ctx
        )
        floor_bytes = GENERIC_DENSE_ACTIVATION_FLOOR_GB * 1e9
        return (floor_bytes + per_token) / tp

    raise ValueError(f"Unknown model_family: {spec['model_family']}")


def cudagraph_overhead_gb(mem_util, tp):
    """vLLM cudagraph capture + flashinfer workspace overhead per card.
    Roughly linear with mem_util (higher mem-util → more graphs captured).
    TP increases per-card overhead slightly due to NCCL workspaces.
    """
    base = 0.5 + 1.0 * mem_util
    tp_bump = 0.0 if tp == 1 else 0.3 * (tp - 1)
    return base + tp_bump


def _validate_tp_for_spec(spec, tp):
    valid_tp = spec.get("valid_tp")
    if valid_tp and tp not in valid_tp:
        raise ValueError(
            f"TP={tp} invalid for {spec['model_id']} "
            f"(num_kv_heads={spec['num_kv_heads']} cannot be divided across TP cleanly). "
            f"Valid TP values: {valid_tp}"
        )


def predict(
    spec=QWEN36_27B,
    kv_format="fp8_e5m2",
    max_ctx=180000,
    max_num_seqs=1,
    tp=1,
    mem_util=0.95,
    vram_gb=24,
    dflash_draft_gb=0.0,
    drafter_gb=0.0,
    mtp=False,
    weights_variant="default",
) -> Prediction:
    """Predict per-card VRAM usage.

    vLLM caps KV pool to (budget - fixed_components), so the prediction
    reflects what actually gets allocated. When requested > available,
    verdict is TIGHT with a note about effective concurrency reduction.

    Args:
      drafter_gb: total drafter weight (MTP / DFlash) — split by TP.
      dflash_draft_gb: legacy alias — folded into drafter_gb if set.
    """
    _validate_tp_for_spec(spec, tp)

    weights_gb = _weights_per_card_gb(spec, tp, weights_variant)

    mtp_n = int(spec.get("mtp_n_default", 3)) if mtp else 0
    growing_b, sliding_b = kv_pool_per_card_bytes(
        spec, kv_format, max_ctx, max_num_seqs, tp,
        mtp_n=mtp_n,
    )
    kv_pool_requested_gb = growing_b / 1e9
    kv_pool_sliding_fixed_gb = sliding_b / 1e9

    activation_gb = activation_peak_per_card_bytes(spec, kv_format, max_ctx, tp) / 1e9
    overhead_gb = cudagraph_overhead_gb(mem_util, tp)

    # Drafter: prefer drafter_gb; fall back to legacy dflash_draft_gb.
    drafter_total = drafter_gb if drafter_gb > 0 else dflash_draft_gb
    if mtp and spec["model_family"] == "qwen3-next-moe":
        drafter_total += QWEN_MOE_BUILTIN_MTP_WORKSPACE_GB
    drafter_per_card = drafter_total / tp if tp > 1 else drafter_total

    fixed_gb = weights_gb + activation_gb + overhead_gb + drafter_per_card + kv_pool_sliding_fixed_gb
    budget_gb = mem_util * vram_gb
    available_for_kv = max(0.0, budget_gb - fixed_gb)

    # vLLM caps the KV pool to fit available budget (PagedAttention allocator).
    kv_pool_actual_gb = min(kv_pool_requested_gb, available_for_kv)

    total_gb = fixed_gb + kv_pool_actual_gb
    pct = 100 * total_gb / budget_gb if budget_gb > 0 else 999.0

    notes = []

    # Verdict logic:
    #   - FAIL: fixed components alone exceed budget (no room even for minimum KV).
    #   - TIGHT: requested KV pool exceeds available — vLLM will cap, effective
    #            concurrency reduced (BOOT OK, but `--max-num-seqs` may not be
    #            honored at full max_ctx).
    #   - PASS: requested KV fits with room to spare.
    # The Qwen A3B preview is KV-light enough that the live 16K rows boot with
    # <0.1 GB requested growing KV. Keep the older 1 GB guard for dense/long-KV
    # models, but avoid false FAILs on this MoE family.
    MIN_KV_GB = 0.05 if spec["model_family"] == "qwen3-next-moe" else 1.0
    if available_for_kv < MIN_KV_GB:
        verdict = "FAIL"
        notes.append(
            f"fixed components ({fixed_gb:.1f} GB) leave only {available_for_kv:.1f} GB for KV pool "
            f"(need ≥{MIN_KV_GB:.1f} GB minimum); vLLM pre-check will refuse — "
            f"lower max_ctx, drop a drafter, or raise mem_util"
        )
    elif kv_pool_requested_gb > available_for_kv * 1.05:
        verdict = "TIGHT"
        notes.append(
            f"requested KV pool ({kv_pool_requested_gb:.1f} GB) > available ({available_for_kv:.1f} GB) — "
            f"vLLM will cap to {available_for_kv:.1f} GB; effective concurrency may be lower than "
            f"--max-num-seqs={max_num_seqs} at full max_ctx={max_ctx:,}"
        )
    else:
        verdict = "PASS"

    # Model-specific advisory notes (preserved from v1)
    if kv_format == "turboquant_3bit_nc" and vram_gb < 24 and spec["model_family"] == "qwen3-next-hybrid":
        notes.append("⚠ TQ3 KV on <24 GB cards: consider --kv-format fp8_e5m2 (see docs/HARDWARE.md, #47)")
    if max_ctx > 50000 and tp == 1 and spec["model_family"] == "qwen3-next-hybrid" and kv_format != "fp16":
        notes.append("⚠ single-card vLLM at >50K single-prompt: Cliff 2 territory (DeltaNet GDN forward); see docs/CLIFFS.md")
    if spec["model_family"] == "gemma4-swa-dense" and kv_format == "fp8_e4m3":
        notes.append("⚠ fp8_e4m3 on Ampere (sm_86): Triton `fp8e4nv` kernel unsupported; use int8_per_token_head instead (PR #40391 via #42102)")
    if spec["model_family"] == "gemma4-swa-dense" and tp == 1 and vram_gb < 32:
        notes.append("⚠ Gemma 4 31B TP=1 needs ≥32 GB VRAM; 24 GB Ampere boot-OOMs (model weights + drafter + min KV)")
    if spec["model_family"] in {"qwen3-next-moe", "gemma4-swa-moe"}:
        notes.append("MoE projection uses low-anchor calibration; add max_ctx/max_num_seqs A/B rows before treating this as production-grade.")
    if (
        spec["model_family"] == "qwen3-next-moe"
        and kv_pool_requested_gb > 0
        and available_for_kv > kv_pool_requested_gb * 1.5
    ):
        notes.append(
            f"DeltaNet-hybrid KV is cheap — one max_ctx={max_ctx:,} seq needs only "
            f"{kv_pool_requested_gb:.2f} GB, but vLLM fills the ~{available_for_kv:.1f} GB available pool "
            f"to budget (≈{available_for_kv / kv_pool_requested_gb:.1f}× concurrency headroom). The sub-budget "
            f"'predicted total' is the single-seq floor, not spare VRAM you can repurpose."
        )
    if tp > 4:
        notes.append("TP > 4 predictions are extrapolated; report deltas via scripts/report.sh --bench")

    return Prediction(
        model=spec["model_id"],
        weights_gb=weights_gb,
        kv_pool_requested_gb=kv_pool_requested_gb,
        kv_pool_actual_gb=kv_pool_actual_gb,
        kv_pool_sliding_fixed_gb=kv_pool_sliding_fixed_gb,
        activation_gb=activation_gb,
        cudagraph_overhead_gb=overhead_gb,
        drafter_gb=drafter_per_card,
        total_gb=total_gb,
        vram_gb=vram_gb,
        budget_gb=budget_gb,
        pct_of_vram=pct,
        verdict=verdict,
        notes=notes,
    )


# =============================================================================
# v0.8.0 Pull-Gate — generic-dense eligibility predicate + verdict adapter
# =============================================================================
#
# Import path for P2/P3/P4: `tools/kv-calc.py` is not a dotted-importable
# module name (hyphen). Load it via importlib, e.g.:
#
#     import importlib.util, pathlib, sys
#     _p = pathlib.Path(REPO_ROOT) / "tools" / "kv-calc.py"
#     spec = importlib.util.spec_from_file_location("kv_calc", _p)
#     kv_calc = importlib.util.module_from_spec(spec)
#     sys.modules["kv_calc"] = kv_calc   # required: @dataclass resolves
#                                        # cls.__module__ via sys.modules
#     spec.loader.exec_module(kv_calc)
#     kv_calc.is_generic_dense_eligible(config)   # stratum-5 predicate
#     kv_calc.raw_verdict(...)                     # [B]→[C1] adapter
#
# (This mirrors how the existing test harness loads the module.)

# Markers that make a config NOT a standard dense (MHA/GQA) transformer.
_SSM_RWKV_CONFIG_KEYS = (
    "ssm_cfg",            # mamba / mamba2
    "mamba_d_state",
    "mamba_expand",
    "mamba_n_heads",
    "rwkv",               # rwkv-family marker
    "time_mix_extra_dim",
    "linear_attn_config", # qwen3-next / deltanet hybrid
    "linear_num_value_heads",
    "linear_num_key_heads",
    "full_attention_interval",  # hybrid attention layout
)
_SSM_RWKV_MODEL_TYPE_SUBSTRINGS = (
    "mamba", "rwkv", "ssm", "qwen3_next", "qwen3next",
    "jamba", "zamba", "recurrentgemma", "griffin", "hgrn",
)


def is_generic_dense_eligible(config: dict) -> bool:
    """Stratum-5 pre-`[B]` eligibility: True iff `config` is a standard
    dense MHA/GQA transformer the generic-dense pricing branch can price.

    Eligible iff ALL of:
      - has hidden_size, num_hidden_layers, num_attention_heads,
        num_key_value_heads
      - head_dim present OR derivable as hidden_size / num_attention_heads
        with no remainder
    AND NONE of (ineligible — return False, never emit a number):
      - SSM / Mamba / RWKV / linear-attention (DeltaNet hybrid) markers
      - MoE (`num_local_experts` or `num_experts`)
      - sliding-window-only attention (`sliding_window` set without any
        full-attention layers)
    """
    if not isinstance(config, dict):
        return False

    required = ("hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads")
    for key in required:
        v = config.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            return False

    hidden_size = config["hidden_size"]
    num_attn_heads = config["num_attention_heads"]

    # head_dim explicit OR cleanly derivable.
    head_dim = config.get("head_dim")
    if not (isinstance(head_dim, int) and not isinstance(head_dim, bool) and head_dim > 0):
        if num_attn_heads <= 0 or hidden_size % num_attn_heads != 0:
            return False

    # --- exclusions ---
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(
        str(a) for a in config.get("architectures", []) if a is not None
    ).lower()
    for sub in _SSM_RWKV_MODEL_TYPE_SUBSTRINGS:
        if sub in model_type or sub in architectures:
            return False
    for key in _SSM_RWKV_CONFIG_KEYS:
        if config.get(key) is not None:
            return False

    # MoE.
    if config.get("num_local_experts") is not None:
        return False
    if config.get("num_experts") is not None:
        return False

    # Sliding-window-only attention: a sliding_window is set and there is no
    # evidence of any full-attention layers. (A model that mixes full +
    # sliding — e.g. has a full-attention interval — is already excluded
    # above via the hybrid-layout markers; here we reject pure-SWA.)
    sliding_window = config.get("sliding_window")
    if sliding_window is not None and sliding_window:
        # Heuristics that indicate at least some full-attention layers.
        has_full_attn = bool(
            config.get("sliding_window_pattern")          # e.g. Gemma "every Nth is global"
            or config.get("max_window_layers")            # Qwen2-style: first N layers full
            or config.get("global_attn_layers")
            or config.get("layer_types")                  # explicit per-layer attn type list
        )
        if not has_full_attn:
            return False

    return True


# Map predict()'s coarse verdict → the design's [B]→[C1] vocabulary.
_RAW_VERDICT_MAP = {
    "FAIL": "wont-fit",
    "TIGHT": "fits-constrained",
    "PASS": "fits-clean",
}


def raw_verdict(
    spec,
    kv_format="fp8_e5m2",
    max_ctx=180000,
    max_num_seqs=1,
    tp=1,
    mem_util=0.95,
    vram_gb=24,
    dflash_draft_gb=0.0,
    drafter_gb=0.0,
    mtp=False,
    weights_variant="default",
) -> dict:
    """Thin `[B]`→`[C1]` adapter over predict().

    Returns a JSON-serializable dict exposing predict()'s result in the
    design's vocabulary:
      FAIL  → "wont-fit"
      TIGHT → "fits-constrained"
      PASS  → "fits-clean"

    The structured dict (verdict + key GB breakdown + notes) is what the
    P3/P4 `[C1]` §4.1 total function consumes alongside the confidence tier.
    Confidence assignment is NOT done here (that is the deriver / stratum-5
    layer in later STEPs); this adapter only translates the raw fit verdict.
    """
    p = predict(
        spec=spec,
        kv_format=kv_format,
        max_ctx=max_ctx,
        max_num_seqs=max_num_seqs,
        tp=tp,
        mem_util=mem_util,
        vram_gb=vram_gb,
        dflash_draft_gb=dflash_draft_gb,
        drafter_gb=drafter_gb,
        mtp=mtp,
        weights_variant=weights_variant,
    )
    return {
        "raw_verdict": _RAW_VERDICT_MAP[p.verdict],
        "predict_verdict": p.verdict,
        "model": p.model,
        "breakdown_gb": {
            "weights": round(p.weights_gb, 4),
            "kv_pool_requested": round(p.kv_pool_requested_gb, 4),
            "kv_pool_actual": round(p.kv_pool_actual_gb, 4),
            "kv_pool_sliding_fixed": round(p.kv_pool_sliding_fixed_gb, 4),
            "activation": round(p.activation_gb, 4),
            "cudagraph_overhead": round(p.cudagraph_overhead_gb, 4),
            "drafter": round(p.drafter_gb, 4),
            "total": round(p.total_gb, 4),
        },
        "budget_gb": round(p.budget_gb, 4),
        "vram_gb": p.vram_gb,
        "pct_of_vram": round(p.pct_of_vram, 2),
        "notes": list(p.notes),
    }



def fmt_cache_breakdown(b: CacheBreakdown) -> str:
    lines = []
    lines.append("Cache architecture breakdown")
    lines.append("----------------------------")
    lines.append(f"  Layout:                   {b.layout}")
    lines.append(f"  Scope:                    {b.gpus} GPU(s), TP={b.tp}, sequences={b.sequences}")
    lines.append(f"  Attention KV — growing:   {b.attention_kv_growing_gb:>6.2f} GB / card")
    if b.attention_kv_sliding_fixed_gb > 0.001:
        lines.append(f"  Attention KV — sliding:   {b.attention_kv_sliding_fixed_gb:>6.2f} GB / card")
    if b.recurrent_state_gb > 0.001:
        lines.append(f"  Recurrent / SSM state:    {b.recurrent_state_gb:>6.2f} GB / card")
    if b.compressed_kv_gb > 0.001:
        lines.append(f"  Compressed KV estimate:   {b.compressed_kv_gb:>6.2f} GB / card")
    if b.indexer_cache_gb > 0.001:
        lines.append(f"  Indexer cache estimate:   {b.indexer_cache_gb:>6.2f} GB / card")
    if b.draft_kv_gb > 0.001:
        lines.append(f"  Draft KV override:        {b.draft_kv_gb:>6.2f} GB / card")
    lines.append(f"  Cache/state subtotal:     {b.total_cache_gb:>6.2f} GB / card")
    for note in b.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def fmt_prediction(p: Prediction, header: str = "") -> str:
    lines = []
    if header:
        lines.append(header)
        lines.append("-" * len(header))
    lines.append(f"  Model:                    {p.model}")
    lines.append(f"  Weights:                  {p.weights_gb:>6.2f} GB / card")
    if p.kv_pool_sliding_fixed_gb > 0.01:
        lines.append(f"  KV pool — sliding fixed:  {p.kv_pool_sliding_fixed_gb:>6.2f} GB / card  (constant, doesn't grow with ctx)")
    if abs(p.kv_pool_requested_gb - p.kv_pool_actual_gb) > 0.05:
        lines.append(f"  KV pool — growing (req):  {p.kv_pool_requested_gb:>6.2f} GB / card  (requested)")
        lines.append(f"  KV pool — growing (cap):  {p.kv_pool_actual_gb:>6.2f} GB / card  (vLLM-capped to fit)")
    else:
        lines.append(f"  KV pool — growing:        {p.kv_pool_actual_gb:>6.2f} GB / card")
    lines.append(f"  Activation peak:          {p.activation_gb:>6.2f} GB / card")
    lines.append(f"  Cudagraph + workspace:    {p.cudagraph_overhead_gb:>6.2f} GB / card")
    if p.drafter_gb > 0:
        lines.append(f"  Drafter (MTP / DFlash):   {p.drafter_gb:>6.2f} GB / card")
    lines.append(f"  ─────────────────────────────────────")
    lines.append(f"  Predicted total:          {p.total_gb:>6.2f} GB / card  ({p.pct_of_vram:.0f}% of {p.budget_gb:.1f} GB engine budget)")
    lines.append(f"  Verdict:                  {p.verdict}")
    for note in p.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


# =============================================================================
# Calibration runner
# =============================================================================

def _resolve_compose_for_predict(model_key, compose_id, vram, ctx_override=None):
    """Resolve a compose preset to predict() kwargs, applying optional ctx override."""
    spec = MODEL_SPECS[model_key]
    cfg = COMPOSES[model_key][compose_id]
    max_ctx = ctx_override if ctx_override is not None else cfg["max_ctx"]

    kwargs = dict(
        spec=spec,
        kv_format=cfg["kv_format"],
        max_ctx=max_ctx,
        max_num_seqs=cfg["max_num_seqs"],
        tp=cfg["tp"],
        mem_util=cfg["mem_util"],
        vram_gb=vram,
        mtp=cfg.get("mtp", False),
        weights_variant=cfg.get("weights_variant", "default"),
        drafter_gb=cfg.get("drafter_gb", 0.0),
        dflash_draft_gb=cfg.get("dflash_draft_gb", 0.0),
    )
    return kwargs


def _calibration_block(model_key: str) -> tuple[int, int]:
    """Print calibration table for one model. Returns (correct, total)."""
    rows = CALIBRATION.get(model_key, [])
    if not rows:
        return 0, 0

    spec = MODEL_SPECS[model_key]
    print(f"== {spec['model_id']} ==")
    print(f"  {'compose':<26s} {'predicted':>10s} {'budget':>9s} {'measured':>10s} {'verdict':>8s}")
    print(f"  {'─'*25:<26s} {'─'*9:>10s} {'─'*8:>9s} {'─'*9:>10s} {'─'*7:>8s}")

    correct = 0
    for row in rows:
        compose, vram, measured, ctx_override, _src = row
        kwargs = _resolve_compose_for_predict(model_key, compose, vram, ctx_override)
        p = predict(**kwargs)
        # Verdict is "correct" if (PASS/TIGHT and measured fits) or (FAIL and would OOM).
        # We don't have negative (FAIL) data points in BENCHMARKS — every row booted —
        # so verdict_correct simplifies to: PASS/TIGHT and measured < vram.
        if p.verdict in ("PASS", "TIGHT") and measured < vram:
            mark = "✓"
            correct += 1
        elif p.verdict == "FAIL" and measured >= vram:
            mark = "✓"
            correct += 1
        else:
            mark = "⨯"
        compose_disp = compose if ctx_override is None else f"{compose}@{ctx_override//1024}K"
        print(f"  {compose_disp:<26s} {p.total_gb:>8.2f} GB {p.budget_gb:>7.2f} GB {measured:>8.2f} GB {p.verdict:>7s} {mark}")

    print()
    print(f"  Verdict accuracy: {correct}/{len(rows)} ({100*correct/len(rows):.0f}%)")
    print()
    return correct, len(rows)


def run_calibration():
    print("=" * 88)
    print("Calibration — predicted per-card VRAM vs measured BENCHMARKS rows")
    print("=" * 88)
    print()
    print("  Predicted = weights + activation + overhead + drafter + (KV capped at available).")
    print("  Budget = mem_util × VRAM. Measured = nvidia-smi peak during bench (target ≈ budget).")
    print("  Verdict ✓ iff PASS/TIGHT and measured < VRAM (boot OK).")
    print()

    total_c, total_n = 0, 0
    for model_key in MODEL_SPECS:
        c, n = _calibration_block(model_key)
        total_c += c
        total_n += n

    if total_n > 0:
        print(f"Overall: {total_c}/{total_n} ({100*total_c/total_n:.0f}%)")
        print()
    print("Notes:")
    print("  - This is a directional estimator (±1.5 GB error band on the breakdown).")
    print("  - vLLM's `gpu_worker.py` boot log is the authoritative source.")
    print("  - If predicted PASS but measured > budget, file an issue with `scripts/report.sh --bench`.")


# =============================================================================
# Max-ctx solver
# =============================================================================

def solve_max_ctx(spec, kv_format, max_num_seqs, tp, mem_util, vram_gb,
                  drafter_gb=0.0, dflash_draft_gb=0.0, mtp=False, weights_variant="default"):
    """Binary search for the largest max_ctx that keeps the verdict at PASS or TIGHT."""
    lo, hi = 1024, spec.get("max_ctx_supported", 262144)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = (mid // 1024) * 1024  # round to nearest 1024 for cleaner numbers
        if mid == 0:
            break
        p = predict(
            spec=spec, kv_format=kv_format, max_ctx=mid, max_num_seqs=max_num_seqs,
            tp=tp, mem_util=mem_util, vram_gb=vram_gb,
            drafter_gb=drafter_gb, dflash_draft_gb=dflash_draft_gb,
            mtp=mtp, weights_variant=weights_variant,
        )
        if p.verdict in ("PASS", "TIGHT"):
            best = mid
            lo = mid + 1024
        else:
            hi = mid - 1024
    return best


# =============================================================================
# CLI
# =============================================================================

def _all_compose_choices() -> list[str]:
    """Flat list of compose names across all models for argparse choices."""
    out = []
    for model_key in COMPOSES:
        out.extend(COMPOSES[model_key].keys())
    return sorted(set(out))


def _resolve_compose_model(compose_name: str, explicit_model: Optional[str]) -> str:
    """Infer model from compose name if --model not given.

    Composes are namespaced by prefix; Qwen uses bare names, Gemma uses gemma-*.
    """
    if explicit_model:
        return explicit_model
    for model_key, composes in COMPOSES.items():
        if compose_name in composes:
            return model_key
    return "qwen3.6-27b"  # back-compat default


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=sorted(MODEL_SPECS.keys()),
                   help="Which model to predict for. Default: qwen3.6-27b (back-compat) or inferred from --compose.")
    p.add_argument("--compose", choices=_all_compose_choices(),
                   help="Use a shipped compose's defaults. Override individual flags below.")
    p.add_argument("--kv-format", choices=sorted(KV_FORMAT_BYTES.keys()),
                   help="KV cache format. Default: from --compose, or fp8_e5m2.")
    p.add_argument("--max-ctx", type=int, help="max_model_len. Default: from --compose, or 180000.")
    p.add_argument("--max-num-seqs", type=int, help="max_num_seqs. Default: from --compose, or 1.")
    p.add_argument("--sequences", type=int, help="Alias for --max-num-seqs, for KV planning calculators.")
    p.add_argument("--tp", type=int, choices=[1, 2, 4, 8, 16], help="tensor_parallel_size. Default: from --compose, or 1.")
    p.add_argument("--gpus", type=int, default=None, help="Physical GPUs in the local single-node rig (1-8). Informational; TP controls sharding. Default: TP.")
    p.add_argument("--mem-util", type=float, help="gpu_memory_utilization. Default: from --compose, or 0.95.")
    p.add_argument("--vram", type=float, default=24, help="VRAM per card in GB. Default 24.")
    p.add_argument("--mtp", action="store_true", default=None, help="MTP enabled (Qwen: n=3 built-in; Gemma: external drafter).")
    p.add_argument("--no-mtp", dest="mtp", action="store_false")
    p.add_argument("--drafter-gb", type=float, default=None,
                   help="Drafter model size in GB (MTP / DFlash). 0 if not using a drafter.")
    p.add_argument("--dflash-draft-gb", type=float, default=None,
                   help="(deprecated alias for --drafter-gb)")
    p.add_argument("--weights-variant", choices=["default", "int4", "awq", "bf16", "int8"], default=None,
                   help="Gemma 4 only: which weight quant variant. Default: from --compose, or int4.")
    p.add_argument("--calibration", action="store_true", help="Print predicted vs measured for all calibrated models.")
    p.add_argument("--solve-max-ctx", action="store_true", help="Binary-search for the largest max_ctx that fits.")
    p.add_argument("--json", action="store_true", help="Output prediction as JSON.")
    p.add_argument("--kv-breakdown", action="store_true",
                   help="Also print/report raw architecture cache buckets (attention KV, SSM state, optional indexer/draft estimates).")
    p.add_argument("--kv-layout", choices=["auto", "dense_gqa", "hybrid_mamba", "sliding_window", "compressed_sparse"],
                   default="auto", help="Reporting label for --kv-breakdown. Default: infer from model family.")
    p.add_argument("--include-draft-kv", action="store_true",
                   help="Include --draft-kv-gb in --kv-breakdown. Built-in MTP token bump still comes from --mtp.")
    p.add_argument("--draft-kv-gb", type=float, default=0.0,
                   help="Per-card draft KV estimate to include with --include-draft-kv. Default 0.")
    p.add_argument("--compressed-layers", type=int, default=0,
                   help="Optional compressed/sparse KV layer count for --kv-breakdown reporting. Default 0.")
    p.add_argument("--compression-ratio", type=int, default=128,
                   help="Token compression ratio for --compressed-layers. Default 128.")
    p.add_argument("--compressed-head-dim", type=int,
                   help="Effective compressed KV head dimension for --compressed-layers.")
    p.add_argument("--indexer-ratio-layers", type=int, default=0,
                   help="Optional indexer-cache layer count for --kv-breakdown reporting. Default 0.")
    p.add_argument("--indexer-compress-ratio", type=int, default=4,
                   help="Token compression ratio for indexer cache. Default 4.")
    p.add_argument("--indexer-head-dim", type=int,
                   help="Effective indexer head dimension for --indexer-ratio-layers.")
    p.add_argument("--indexer-format", choices=sorted(INDEXER_FORMAT_BYTES.keys()), default="fp4",
                   help="Indexer precision for --kv-breakdown. Default fp4.")
    args = p.parse_args()

    if args.sequences is not None and args.max_num_seqs is not None and args.sequences != args.max_num_seqs:
        print("ERROR: --sequences and --max-num-seqs disagree; pass only one value.", file=sys.stderr)
        return 2
    if args.sequences is not None:
        args.max_num_seqs = args.sequences

    if args.calibration:
        run_calibration()
        return 0

    # Resolve model: explicit --model > inferred from --compose > qwen3.6-27b
    model_key = _resolve_compose_model(args.compose, args.model) if args.compose else (args.model or "qwen3.6-27b")
    spec = MODEL_SPECS[model_key]

    # Resolve compose-derived defaults
    if args.compose:
        # Compose must belong to the resolved model
        if args.compose not in COMPOSES[model_key]:
            print(f"ERROR: --compose {args.compose} is not in --model {model_key}'s compose list.", file=sys.stderr)
            print(f"       Available for {model_key}: {', '.join(sorted(COMPOSES[model_key].keys()))}", file=sys.stderr)
            return 2
        cfg = COMPOSES[model_key][args.compose]
        kv_format = args.kv_format or cfg["kv_format"]
        max_ctx = args.max_ctx or cfg["max_ctx"]
        max_num_seqs = args.max_num_seqs or cfg["max_num_seqs"]
        tp = args.tp or cfg["tp"]
        mem_util = args.mem_util if args.mem_util is not None else cfg["mem_util"]
        mtp = args.mtp if args.mtp is not None else cfg.get("mtp", False)
        drafter_gb = args.drafter_gb if args.drafter_gb is not None else cfg.get("drafter_gb", 0.0)
        dflash_gb = args.dflash_draft_gb if args.dflash_draft_gb is not None else cfg.get("dflash_draft_gb", 0.0)
        weights_variant = args.weights_variant or cfg.get("weights_variant", "default")
        header = f"Predicted budget — {model_key} / {args.compose} on {args.vram} GB VRAM (kv={kv_format}, ctx={max_ctx:,}, seqs={max_num_seqs}, TP={tp}, mem={mem_util})"
    else:
        kv_format = args.kv_format or "fp8_e5m2"
        max_ctx = args.max_ctx or 180000
        max_num_seqs = args.max_num_seqs or 1
        tp = args.tp or 1
        mem_util = args.mem_util if args.mem_util is not None else 0.95
        mtp = bool(args.mtp) if args.mtp is not None else False
        drafter_gb = args.drafter_gb or 0.0
        dflash_gb = args.dflash_draft_gb or 0.0
        weights_variant = args.weights_variant or "default"
        header = f"Predicted budget — {model_key} custom config on {args.vram} GB VRAM (kv={kv_format}, ctx={max_ctx:,}, seqs={max_num_seqs}, TP={tp}, mem={mem_util})"

    try:
        _validate_tp_for_spec(spec, tp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    gpus = args.gpus if args.gpus is not None else tp
    if gpus < 1 or gpus > 8:
        print("ERROR: --gpus must be between 1 and 8 for single-node home/workstation planning.", file=sys.stderr)
        return 2

    if args.solve_max_ctx:
        best = solve_max_ctx(
            spec, kv_format=kv_format, max_num_seqs=max_num_seqs,
            tp=tp, mem_util=mem_util, vram_gb=args.vram,
            drafter_gb=drafter_gb, dflash_draft_gb=dflash_gb, mtp=mtp,
            weights_variant=weights_variant,
        )
        if best > 0:
            pred_at_best = predict(
                spec=spec, kv_format=kv_format, max_ctx=best, max_num_seqs=max_num_seqs,
                tp=tp, mem_util=mem_util, vram_gb=args.vram,
                drafter_gb=drafter_gb, dflash_draft_gb=dflash_gb, mtp=mtp,
                weights_variant=weights_variant,
            )
            if args.json:
                out = pred_at_best.__dict__.copy()
                out["solved_max_ctx"] = best
                print(json.dumps(out, indent=2))
            else:
                print(f"Max-ctx solver — {model_key} / {kv_format}, seqs={max_num_seqs}, TP={tp}, mem_util={mem_util}, VRAM={args.vram} GB")
                print(f"  Largest max_ctx that fits: {best:,} tokens")
                print(f"  At that ctx: predicted = {pred_at_best.total_gb:.2f} GB / card ({pred_at_best.pct_of_vram:.0f}% of budget)")
                print(f"  Verdict at that ctx: {pred_at_best.verdict}")
                for note in pred_at_best.notes:
                    print(f"  Note: {note}")
                print()
                print("Note: this is a directional estimate (±1.5 GB error band). The vLLM engine")
                print("pre-check (gpu_worker.py boot log) is authoritative.")
        else:
            print(f"No max_ctx fits at this config on {args.vram} GB. Reduce TP, swap KV format, or get bigger cards.")
        return 0

    pred = predict(
        spec=spec, kv_format=kv_format, max_ctx=max_ctx, max_num_seqs=max_num_seqs,
        tp=tp, mem_util=mem_util, vram_gb=args.vram,
        drafter_gb=drafter_gb, dflash_draft_gb=dflash_gb, mtp=mtp,
        weights_variant=weights_variant,
    )

    breakdown = None
    if args.kv_breakdown:
        try:
            breakdown = architecture_cache_breakdown(
                spec=spec,
                kv_format=kv_format,
                max_ctx=max_ctx,
                max_num_seqs=max_num_seqs,
                tp=tp,
                gpus=gpus,
                layout=args.kv_layout,
                mtp=mtp,
                include_draft_kv=args.include_draft_kv,
                draft_kv_gb=args.draft_kv_gb,
                compressed_layers=args.compressed_layers,
                compression_ratio=args.compression_ratio,
                compressed_head_dim=args.compressed_head_dim,
                indexer_ratio_layers=args.indexer_ratio_layers,
                indexer_compress_ratio=args.indexer_compress_ratio,
                indexer_head_dim=args.indexer_head_dim,
                indexer_format=args.indexer_format,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if args.json:
        out = pred.__dict__.copy()
        if breakdown is not None:
            out["cache_breakdown"] = breakdown.__dict__
        print(json.dumps(out, indent=2))
    else:
        print(fmt_prediction(pred, header=header))
        if breakdown is not None:
            print()
            print(fmt_cache_breakdown(breakdown))
        print()
        print("Run `tools/kv-calc.py --calibration` to see predicted-vs-measured for all anchors.")
        print("Run `tools/kv-calc.py --solve-max-ctx ...` to find the largest max_ctx that fits.")
        print("See docs/KV_MATH.md for math + per-model architecture details.")
    return 0 if pred.verdict in ("PASS", "TIGHT") else 1


if __name__ == "__main__":
    sys.exit(main())
