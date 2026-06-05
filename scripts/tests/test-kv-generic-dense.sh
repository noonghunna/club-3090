#!/usr/bin/env bash
# v0.8.0 Pull-Gate P1 — generic-dense family + eligibility predicate + raw_verdict adapter.
#
# Asserts:
#   1. tools/kv-calc.py --calibration still reports the exact line "Overall: 9/9 (100%)"
#      (curated branches byte-unchanged — the regression contract).
#   2. generic-dense on a standard-dense reference fixture (Llama-class 7B) is a
#      conservative LOWER BOUND on fit:
#        - obviously-fits config (24 GB card)  -> fits-clean
#        - obviously-cannot config (tiny VRAM) -> wont-fit
#        - predicted free VRAM <= what an equivalent-shape curated Qwen dense
#          branch would predict (generic-dense never over-promises fit).
#   3. is_generic_dense_eligible: True for the dense fixture; False for MoE / SSM
#      / SWA-only fixtures (no number emitted on ineligible).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --- 1. calibration regression contract: exact 17/17 line --------------------
CALIB_OUT="$(tools/kv-calc.py --calibration 2>&1)"
if ! grep -qxF "Overall: 9/9 (100%)" <<<"$CALIB_OUT"; then
  echo "FAIL: --calibration did not report the exact line 'Overall: 9/9 (100%)'" >&2
  echo "----- last 6 lines of --calibration output -----" >&2
  tail -6 <<<"$CALIB_OUT" >&2
  exit 1
fi
echo "PASS: --calibration reports 'Overall: 9/9 (100%)' (curated branches unchanged)"

# --- 2 + 3. generic-dense pricing + eligibility predicate --------------------
python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
kv_path = root / "tools" / "kv-calc.py"
spec = importlib.util.spec_from_file_location("kv_calc", kv_path)
kv = importlib.util.module_from_spec(spec)
# Register before exec_module: kv-calc.py uses @dataclass, which resolves
# cls.__module__ via sys.modules during class processing.
sys.modules["kv_calc"] = kv
spec.loader.exec_module(kv)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---- standard-dense reference fixture: Llama-class ~7B ----------------------
# 32 layers, 32 attn heads, 8 KV heads (GQA), head_dim 128, hidden 4096.
# Blob-size ~13.5 GB (bf16 7B). This obviously fits a 24 GB card.
DENSE_SPEC = {
    "model_id": "generic-dense-fixture-7b",
    "model_family": "generic-dense",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attn_heads": 32,
    "num_kv_heads": 8,
    "head_dim_attn": 128,
    "weights_total_gb": 13.5,
    "valid_tp": [1, 2],
    "max_ctx_supported": 131072,
}

# (a) obviously fits a 24 GB card at a modest ctx.
rv_fit = kv.raw_verdict(
    spec=DENSE_SPEC, kv_format="fp8_e5m2", max_ctx=8192,
    max_num_seqs=1, tp=1, mem_util=0.90, vram_gb=24,
)
check(
    rv_fit["raw_verdict"] == "fits-clean",
    f"dense 7B fixture @ 8K ctx / 24 GB -> fits-clean (got {rv_fit['raw_verdict']}, "
    f"total {rv_fit['breakdown_gb']['total']} GB)",
)

# (b) obviously cannot fit a tiny card (8 GB) with the same 13.5 GB weights.
rv_nofit = kv.raw_verdict(
    spec=DENSE_SPEC, kv_format="fp16", max_ctx=131072,
    max_num_seqs=4, tp=1, mem_util=0.90, vram_gb=8,
)
check(
    rv_nofit["raw_verdict"] == "wont-fit",
    f"dense 7B fixture @ 128K ctx / 8 GB -> wont-fit (got {rv_nofit['raw_verdict']})",
)

# (c) conservative lower-bound: generic-dense predicted free VRAM must be
#     <= what a curated dense branch (Qwen 3.6 27B, exact) predicts for an
#     equivalent shape & config. We compare the SAME runtime knobs; the
#     generic-dense pricing is intentionally >= the curated exact term, so
#     its remaining free VRAM must be <= the curated branch's.
KNOBS = dict(kv_format="fp8_e5m2", max_ctx=32768, max_num_seqs=1,
             tp=2, mem_util=0.95, vram_gb=24)

# Build a generic-dense spec that mirrors the curated Qwen 27B shape so the
# comparison is apples-to-apples (same layers / kv-heads / head_dim / weights).
qwen = kv.QWEN36_27B
EQUIV_DENSE = {
    "model_id": "generic-dense-equiv-qwen27b",
    "model_family": "generic-dense",
    "hidden_size": qwen["hidden_size"],
    "num_hidden_layers": qwen["num_hidden_layers"],
    "num_attn_heads": qwen["num_attn_heads"],
    "num_kv_heads": qwen["num_kv_heads"],
    "head_dim_attn": qwen["head_dim_attn"],
    "weights_total_gb": qwen["weights_total_gb"],
    "valid_tp": list(qwen["valid_tp"]),
    "max_ctx_supported": qwen.get("max_ctx_supported", 262144),
}

p_curated = kv.predict(spec=qwen, **KNOBS)
p_generic = kv.predict(spec=EQUIV_DENSE, **KNOBS)

# Free VRAM = budget - total. generic-dense must leave <= curated's free
# VRAM (i.e. it must predict a total >= the curated exact total for the
# equivalent shape — never claim MORE fit).
free_curated = p_curated.budget_gb - p_curated.total_gb
free_generic = p_generic.budget_gb - p_generic.total_gb
check(
    free_generic <= free_curated + 1e-6,
    f"generic-dense is a lower bound on fit: free_generic={free_generic:.3f} GB "
    f"<= free_curated={free_curated:.3f} GB "
    f"(generic total {p_generic.total_gb:.2f} >= curated total {p_curated.total_gb:.2f})",
)

# ---- eligibility predicate -------------------------------------------------
DENSE_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    # head_dim omitted on purpose -> must derive 4096/32 = 128 cleanly.
}
check(
    kv.is_generic_dense_eligible(DENSE_CONFIG) is True,
    "is_generic_dense_eligible(dense Llama, head_dim derived) -> True",
)

DENSE_EXPLICIT_HEADDIM = dict(DENSE_CONFIG, head_dim=128, model_type="qwen2")
check(
    kv.is_generic_dense_eligible(DENSE_EXPLICIT_HEADDIM) is True,
    "is_generic_dense_eligible(dense Qwen2, explicit head_dim) -> True",
)

MOE_CONFIG = {
    "model_type": "mixtral",
    "architectures": ["MixtralForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_local_experts": 8,
}
check(
    kv.is_generic_dense_eligible(MOE_CONFIG) is False,
    "is_generic_dense_eligible(MoE Mixtral) -> False (no number emitted)",
)

SSM_CONFIG = {
    "model_type": "mamba2",
    "architectures": ["Mamba2ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "mamba_d_state": 128,
}
check(
    kv.is_generic_dense_eligible(SSM_CONFIG) is False,
    "is_generic_dense_eligible(SSM Mamba2) -> False (no number emitted)",
)

SWA_ONLY_CONFIG = {
    "model_type": "mistral",
    "architectures": ["MistralForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "sliding_window": 4096,  # set, with no full-attention-layer marker
}
check(
    kv.is_generic_dense_eligible(SWA_ONLY_CONFIG) is False,
    "is_generic_dense_eligible(sliding-window-only Mistral) -> False",
)

DELTANET_HYBRID_CONFIG = {
    "model_type": "qwen3_next",
    "architectures": ["Qwen3NextForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 64,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "linear_attn_config": {"linear_num_value_heads": 32},
}
check(
    kv.is_generic_dense_eligible(DELTANET_HYBRID_CONFIG) is False,
    "is_generic_dense_eligible(DeltaNet hybrid Qwen3-Next) -> False",
)

# ---- optional home/workstation KV breakdown mode ---------------------------
breakdown = kv.architecture_cache_breakdown(
    spec=kv.QWEN36_27B,
    kv_format="fp8_e5m2",
    max_ctx=32768,
    max_num_seqs=2,
    tp=2,
    gpus=4,
    layout="auto",
)
check(
    breakdown.layout == "hybrid_mamba" and breakdown.gpus == 4 and breakdown.sequences == 2,
    "architecture_cache_breakdown reports inferred layout + explicit rig/sequences metadata",
)
check(
    breakdown.attention_kv_growing_gb > 0 and breakdown.total_cache_gb >= breakdown.attention_kv_growing_gb,
    "architecture_cache_breakdown reports positive attention KV subtotal without changing fit prediction",
)

sparse = kv.architecture_cache_breakdown(
    spec=kv.QWEN36_27B,
    kv_format="fp8_e5m2",
    max_ctx=32768,
    max_num_seqs=1,
    tp=2,
    gpus=2,
    layout="compressed_sparse",
    compressed_layers=31,
    compression_ratio=128,
    compressed_head_dim=131072,
    indexer_ratio_layers=30,
    indexer_compress_ratio=4,
    indexer_head_dim=65536,
    indexer_format="fp4",
)
check(
    sparse.compressed_kv_gb > 0 and sparse.indexer_cache_gb > 0,
    "architecture_cache_breakdown can report optional compressed-KV and indexer-cache buckets",
)

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll generic-dense assertions passed.")
PY

echo "test-kv-generic-dense.sh OK"
