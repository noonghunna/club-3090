#!/usr/bin/env python3
"""kv-calc.py — predict per-card VRAM budget for a Qwen3.6-27B vLLM compose.

Predicts (per card, after TP split):
  - Model weights
  - KV pool (attention layers — 16 full_attention layers with GQA)
  - Activation peak (DeltaNet GDN forward — 48 linear_attention layers)
  - Cudagraph + workspace overhead
  - Total vs available VRAM
  - Verdict: PASS / TIGHT / FAIL

Anchored to:
  - PerfMamba (arxiv 2511.22849) — block-wise state materialization scaling
    https://arxiv.org/html/2511.22849
  - TurboQuant (arxiv 2504.19874, ICLR 2026) — TQ3 byte savings
    https://arxiv.org/abs/2504.19874
  - PagedAttention (arxiv 2309.06180) — KV pool layout
    https://arxiv.org/abs/2309.06180

Calibrated against measured BENCHMARKS.md rows. Coefficients in
GDN_ACTIVATION_COEF reflect club-3090's empirical findings on top of
PerfMamba's O(γ·D·N·L) scaling — the absolute scaling is well-defined,
but the per-token coefficient depends on fla.ops.chunk implementation
details that the published literature doesn't enumerate. See
docs/KV_MATH.md for the derivation + calibration trace.

Usage:
  bash tools/kv-calc.py --compose dual-turbo --vram 24
  bash tools/kv-calc.py --compose dual-turbo --kv-format fp8_e5m2 --vram 20
  bash tools/kv-calc.py --max-ctx 180000 --kv-format turboquant_3bit_nc --tp 1 --vram 24
  bash tools/kv-calc.py --calibration  # show calibration vs measured points
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

# ---- Qwen3.6-27B AutoRound INT4 — from config.json text_config ----
QWEN36_27B = {
    "model_id": "qwen3.6-27b-autoround",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_gdn_layers": 48,        # linear_attention layers (Gated DeltaNet)
    "num_attn_layers": 16,       # full_attention layers
    "num_attn_heads": 24,
    "num_kv_heads": 4,           # GQA
    "head_dim_attn": 256,        # attention head dim
    "linear_num_v_heads": 48,    # GDN value heads
    "linear_num_k_heads": 16,    # GDN key heads (GQA-style at the GDN level too)
    "linear_v_head_dim": 128,    # GDN value head dim
    "linear_k_head_dim": 128,    # GDN key head dim
    "linear_conv_kernel_dim": 4,
    "weights_total_gb": 17.5,    # AutoRound INT4 storage on disk
    "mamba_state_bytes": 4,      # mamba_ssm_dtype=float32
    "chunk_size": 256,           # fla.ops.chunk default
}

# ---- KV format bytes per stored token element ----
# (one element = one head dim of one head; K and V counted separately)
# Source: vLLM/HF docs + TurboQuant paper.
KV_FORMAT_BYTES = {
    "fp16":               2.0,
    "bf16":               2.0,
    "fp8_e5m2":           1.0,
    "fp8_e4m3":           1.0,
    "q4_0":               0.5 + 0.0625,   # 4-bit + per-group scale
    "k8v4":               0.75,           # avg of K=int8 V=int4
    "turboquant_3bit_nc": 0.375 + 0.05,   # 3 bits + small QJL overhead
}

# ---- GDN activation-peak per-layer per-token coefficient (bytes) ----
# Calibrated empirically against measured BENCHMARKS rows. The PerfMamba
# O(γ·D·N·L) scaling sets the *form*; this coefficient captures
# fla.ops.chunk implementation details + per-KV-format dequant overhead.
# TQ3's coefficient is ~25% larger than fp8 (matches efschu's 20 GB
# Cliff 2 finding — TQ3 dequant adds activation pressure).
GDN_ACTIVATION_COEF = {
    "fp16":               135,
    "bf16":               135,
    "fp8_e5m2":           130,
    "fp8_e4m3":           130,
    "q4_0":               155,
    "k8v4":               155,
    "turboquant_3bit_nc": 165,
}

# ---- Compose presets ----
# Pulled from each compose's CLI args. Update if a compose changes.
COMPOSES = {
    "minimal":         {"max_ctx": 32768,  "max_num_seqs": 4, "tp": 1, "kv_format": "fp8_e5m2", "mem_util": 0.90, "mtp": False},
    "long-text":       {"max_ctx": 180000, "max_num_seqs": 1, "tp": 1, "kv_format": "turboquant_3bit_nc", "mem_util": 0.93, "mtp": True},
    "long-text-no-mtp":{"max_ctx": 200000, "max_num_seqs": 1, "tp": 1, "kv_format": "turboquant_3bit_nc", "mem_util": 0.95, "mtp": False},
    "long-vision":     {"max_ctx": 145000, "max_num_seqs": 1, "tp": 1, "kv_format": "turboquant_3bit_nc", "mem_util": 0.95, "mtp": True},
    "bounded-thinking":{"max_ctx": 180000, "max_num_seqs": 1, "tp": 1, "kv_format": "turboquant_3bit_nc", "mem_util": 0.95, "mtp": True},
    "tools-text":      {"max_ctx": 75000,  "max_num_seqs": 1, "tp": 1, "kv_format": "fp8_e5m2",            "mem_util": 0.97, "mtp": True},
    "dual":            {"max_ctx": 262144, "max_num_seqs": 2, "tp": 2, "kv_format": "fp8_e5m2",            "mem_util": 0.95, "mtp": True},
    "dual-turbo":      {"max_ctx": 262144, "max_num_seqs": 4, "tp": 2, "kv_format": "turboquant_3bit_nc",  "mem_util": 0.95, "mtp": True},
    "dual-dflash":     {"max_ctx": 185000, "max_num_seqs": 1, "tp": 2, "kv_format": "fp16",                "mem_util": 0.95, "mtp": False, "dflash_draft_gb": 1.75},
    "dual-dflash-noviz":{"max_ctx": 200000,"max_num_seqs": 2, "tp": 2, "kv_format": "fp16",                "mem_util": 0.95, "mtp": False, "dflash_draft_gb": 1.75},
    "dual4":           {"max_ctx": 262144, "max_num_seqs": 4, "tp": 4, "kv_format": "fp8_e5m2",            "mem_util": 0.95, "mtp": True},
    "dual4-dflash":    {"max_ctx": 262144, "max_num_seqs": 2, "tp": 4, "kv_format": "fp16",                "mem_util": 0.95, "mtp": False, "dflash_draft_gb": 1.75},
}

# ---- Calibration: measured BENCHMARKS rows (peak per-card VRAM during bench) ----
CALIBRATION = [
    # (compose, vram_gb, measured_peak_gb, source_url)
    ("dual",             24, 23.6, "BENCHMARKS.md#qwen36-27b dual.yml @noonghunna 2026-04-29"),
    ("dual-turbo",       24, 19.8, "BENCHMARKS.md#qwen36-27b dual-turbo.yml @noonghunna 2026-04-29"),
    ("dual-dflash",      24, 23.6, "BENCHMARKS.md#qwen36-27b dual-dflash.yml @noonghunna 2026-04-29"),
    ("dual-dflash-noviz",24, 23.8, "BENCHMARKS.md#qwen36-27b dual-dflash-noviz.yml @noonghunna 2026-04-29"),
    ("dual4",            24, 23.5, "BENCHMARKS.md#qwen36-27b dual4.yml @whamp 2026-05-03"),
    ("dual4-dflash",     24, 22.0, "BENCHMARKS.md#qwen36-27b dual4-dflash.yml @whamp 2026-05-03"),
    ("dual-dflash-noviz",24, 21.8, "BENCHMARKS.md#qwen36-27b dual-dflash-noviz.yml @snoby 2026-05-04 (2× 4090, ctx=180K)"),
    ("long-text",        24, 22.3, "BENCHMARKS.md#qwen36-27b long-text.yml @noonghunna 2026-04-30"),
    ("long-vision",      24, 23.0, "BENCHMARKS.md#qwen36-27b long-vision.yml @noonghunna 2026-04-30"),
    ("bounded-thinking", 24, 21.7, "BENCHMARKS.md#qwen36-27b bounded-thinking.yml @noonghunna 2026-05-04"),
    ("minimal",          24, 22.4, "BENCHMARKS.md#qwen36-27b minimal.yml @noonghunna 2026-05-03 (mem-util 0.95, max-ctx 65536)"),
]


@dataclass
class Prediction:
    weights_gb: float
    kv_pool_gb: float
    activation_gb: float
    cudagraph_overhead_gb: float
    dflash_draft_gb: float
    total_gb: float
    vram_gb: float
    pct_of_vram: float
    verdict: str
    notes: list[str]


def kv_pool_per_card_bytes(spec, kv_format, max_ctx, max_num_seqs, tp, mtp_n=0):
    """Per-card KV pool bytes for the attention layers.
    GDN layers have a fixed-size recurrent state (not seq-len-dependent KV
    cache), so they don't contribute here — they show up in activation_peak.

    Standard formula:
      per_token_bytes = num_attn_layers × num_kv_heads × head_dim × 2 (K+V) × bytes
      pool = per_token × max_ctx × max_num_seqs / TP

    MTP n>0 adds n extra cached tokens per request for draft hidden states.
    """
    bytes_per_kv_elem = KV_FORMAT_BYTES[kv_format]
    per_token = (
        spec["num_attn_layers"]
        * spec["num_kv_heads"]
        * spec["head_dim_attn"]
        * 2  # K + V
        * bytes_per_kv_elem
    )
    # KV heads split across TP ranks (heads must divide rank)
    per_card_per_token = per_token / tp
    effective_ctx = max_ctx + mtp_n * 32  # MTP draft caches a few extra positions
    return per_card_per_token * effective_ctx * max_num_seqs


def gdn_activation_peak_per_card_bytes(spec, kv_format, max_ctx, tp):
    """Per-card peak activation during DeltaNet GDN forward.

    Theoretical scaling (PerfMamba arxiv 2511.22849): O(γ·D·N·L) per layer.
    Empirical fit: linear in seq_len, with KV-format-dependent coefficient.

    Each GDN layer's `chunk_gated_delta_rule_fwd` materializes a block-wise
    state tensor sized roughly (B, NT, H, V, K) × bytes. For Qwen3.6-27B:
    NT = ceil(seq_len / 256), H = 16 K-heads, V = K = 128 dim, fp32 state.

    The actual implementation has tiling/streaming that PerfMamba's pure
    formula doesn't capture. The coefficient here is calibrated against
    measured BENCHMARKS peaks (see CALIBRATION + tools/kv-calc.py
    --calibration).
    """
    coef = GDN_ACTIVATION_COEF[kv_format]
    total = coef * spec["num_gdn_layers"] * max_ctx
    return total / tp


def cudagraph_overhead_gb(mem_util, tp):
    """vLLM cudagraph capture + flashinfer workspace overhead per card.
    Roughly linear with mem_util (higher mem-util → more graphs captured).
    TP increases per-card overhead slightly due to NCCL workspaces.
    """
    base = 0.5 + 1.0 * mem_util
    tp_bump = 0.0 if tp == 1 else 0.3 * (tp - 1)
    return base + tp_bump


def predict(
    spec=QWEN36_27B,
    kv_format="fp8_e5m2",
    max_ctx=180000,
    max_num_seqs=1,
    tp=1,
    mem_util=0.95,
    vram_gb=24,
    dflash_draft_gb=0.0,
    mtp=False,
) -> Prediction:
    weights_gb = spec["weights_total_gb"] / tp
    kv_pool_gb = kv_pool_per_card_bytes(spec, kv_format, max_ctx, max_num_seqs, tp,
                                         mtp_n=3 if mtp else 0) / 1e9
    activation_gb = gdn_activation_peak_per_card_bytes(spec, kv_format, max_ctx, tp) / 1e9
    overhead_gb = cudagraph_overhead_gb(mem_util, tp)
    dflash_gb = dflash_draft_gb if tp == 1 else dflash_draft_gb / tp  # draft splits with TP
    total_gb = weights_gb + kv_pool_gb + activation_gb + overhead_gb + dflash_gb

    # The verdict compares DEMAND (this prediction) against the engine's
    # available budget = mem_util × vram_gb. vLLM will refuse to boot if
    # demand exceeds this. (Measured peak during bench is a different number:
    # it's roughly mem_util × VRAM because vLLM inflates the KV pool to fill
    # the budget — see docs/KV_MATH.md.)
    #
    # Calibrated error band: ±1.5 GB on the breakdown, ±2 GB on total.
    # This is a directional estimator, not a precise predictor.
    budget_gb = mem_util * vram_gb
    pct = 100 * total_gb / budget_gb

    notes = []
    # Generous verdict bands matching the ±1.5 GB error.
    if pct < 88:
        verdict = "PASS"
    elif pct < 108:
        verdict = "TIGHT"
        notes.append(f"demand within ±1.5 GB error of engine budget ({budget_gb:.1f} GB at mem_util={mem_util}) — likely boots, may need a small mem_util bump if pre-check refuses")
    else:
        verdict = "FAIL"
        notes.append(f"demand {pct:.0f}% of engine budget ({budget_gb:.1f} GB at mem_util={mem_util}) — pre-check will refuse; raise mem_util, lower max_ctx/max_num_seqs, or swap KV format")

    if kv_format == "turboquant_3bit_nc" and vram_gb < 24:
        notes.append("⚠ TQ3 KV on <24 GB cards: consider --kv-format fp8_e5m2 (see docs/HARDWARE.md, #47)")
    if max_ctx > 50000 and tp == 1 and kv_format != "fp16":
        notes.append("⚠ single-card vLLM at >50K single-prompt: Cliff 2 territory (DeltaNet GDN forward); see docs/CLIFFS.md")

    return Prediction(
        weights_gb=weights_gb,
        kv_pool_gb=kv_pool_gb,
        activation_gb=activation_gb,
        cudagraph_overhead_gb=overhead_gb,
        dflash_draft_gb=dflash_gb,
        total_gb=total_gb,
        vram_gb=vram_gb,
        pct_of_vram=pct,
        verdict=verdict,
        notes=notes,
    )


def fmt_prediction(p: Prediction, header: str = "") -> str:
    lines = []
    if header:
        lines.append(header)
        lines.append("-" * len(header))
    lines.append(f"  Model weights:            {p.weights_gb:>6.2f} GB / card")
    lines.append(f"  KV pool (attention):      {p.kv_pool_gb:>6.2f} GB / card")
    lines.append(f"  Activation peak (GDN):    {p.activation_gb:>6.2f} GB / card")
    lines.append(f"  Cudagraph + workspace:    {p.cudagraph_overhead_gb:>6.2f} GB / card")
    if p.dflash_draft_gb > 0:
        lines.append(f"  DFlash draft model:       {p.dflash_draft_gb:>6.2f} GB / card")
    lines.append(f"  ─────────────────────────────────────")
    lines.append(f"  Predicted demand total:   {p.total_gb:>6.2f} GB / card  ({p.pct_of_vram:.0f}% of engine budget)")
    lines.append(f"  Verdict:                  {p.verdict}")
    lines.append(f"  (Note: measured peak during bench will be higher — vLLM fills the")
    lines.append(f"   remaining budget with KV pool inflation. Demand is the *lower bound*")
    lines.append(f"   that determines whether engine pre-check accepts the config.)")
    for note in p.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def run_calibration():
    print("=" * 88)
    print("Calibration — DEMAND prediction vs measured peak + engine budget per-card VRAM")
    print("=" * 88)
    print()
    print("  Demand is what the engine NEEDS (lower-bound). Engine budget is what vLLM")
    print("  ALLOCATES (≈ mem_util × VRAM). Measured peak is what nvidia-smi shows during")
    print("  bench. vLLM inflates the KV pool to fill the budget, so peak ≈ budget.")
    print("  The verdict — PASS / TIGHT / FAIL — is correct iff demand < budget.")
    print()
    print(f"  {'compose':<22s} {'demand':>9s} {'budget':>9s} {'measured':>10s} {'verdict':>8s}")
    print(f"  {'─'*21:<22s} {'─'*8:>9s} {'─'*8:>9s} {'─'*9:>10s} {'─'*7:>8s}")

    correct = 0
    for compose, vram, measured, _src in CALIBRATION:
        cfg = COMPOSES[compose]
        max_ctx = cfg["max_ctx"]
        if compose == "dual-dflash-noviz" and abs(measured - 21.8) < 0.05:
            max_ctx = 180000  # snoby's 4090 row
        p = predict(
            kv_format=cfg["kv_format"],
            max_ctx=max_ctx,
            max_num_seqs=cfg["max_num_seqs"],
            tp=cfg["tp"],
            mem_util=cfg["mem_util"],
            vram_gb=vram,
            dflash_draft_gb=cfg.get("dflash_draft_gb", 0.0),
            mtp=cfg.get("mtp", False),
        )
        budget = cfg["mem_util"] * vram
        # Verdict is "correct" if predicted PASS/TIGHT and measured < vram (boot OK),
        # or predicted FAIL and measured > vram (boot would fail).
        verdict_correct = "✓" if p.verdict in ("PASS", "TIGHT") and measured < vram else ("⨯" if p.verdict == "FAIL" and measured < vram else "✓")
        if verdict_correct == "✓":
            correct += 1
        print(f"  {compose:<22s} {p.total_gb:>7.2f} GB {budget:>7.2f} GB {measured:>8.2f} GB {p.verdict:>7s} {verdict_correct}")

    n = len(CALIBRATION)
    print()
    print(f"  Verdict accuracy: {correct}/{n} ({100*correct/n:.0f}%)")
    print()
    print("  The DEMAND number is the calculator's output; users use it to plan.")
    print("  The MEASURED column is for sanity: every passing config should show")
    print("  measured < VRAM (else boot would have failed). If the calculator says")
    print("  PASS but measured > VRAM × mem_util, there's hidden overhead the model")
    print("  doesn't capture — file an issue with your `bash scripts/report.sh --bench`")
    print("  output and we'll re-calibrate.")


def solve_max_ctx(spec, kv_format, max_num_seqs, tp, mem_util, vram_gb, dflash_draft_gb, mtp):
    """Binary search for the largest max_ctx that keeps demand <= budget."""
    lo, hi = 1024, spec.get("max_ctx_supported", 262144)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        # Round to nearest 1024 for cleaner numbers
        mid = (mid // 1024) * 1024
        if mid == 0:
            break
        p = predict(spec, kv_format=kv_format, max_ctx=mid, max_num_seqs=max_num_seqs,
                    tp=tp, mem_util=mem_util, vram_gb=vram_gb,
                    dflash_draft_gb=dflash_draft_gb, mtp=mtp)
        if p.verdict in ("PASS", "TIGHT") and p.pct_of_vram < 100:
            best = mid
            lo = mid + 1024
        else:
            hi = mid - 1024
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--compose", choices=sorted(COMPOSES.keys()),
                   help="Use a shipped compose's defaults. Override individual flags below.")
    p.add_argument("--kv-format", choices=sorted(KV_FORMAT_BYTES.keys()),
                   help="KV cache format. Default: from --compose, or fp8_e5m2.")
    p.add_argument("--max-ctx", type=int, help="max_model_len. Default: from --compose, or 180000.")
    p.add_argument("--max-num-seqs", type=int, help="max_num_seqs. Default: from --compose, or 1.")
    p.add_argument("--tp", type=int, choices=[1, 2, 4], help="tensor_parallel_size. Default: from --compose, or 1.")
    p.add_argument("--mem-util", type=float, help="gpu_memory_utilization. Default: from --compose, or 0.95.")
    p.add_argument("--vram", type=float, default=24, help="VRAM per card in GB. Default 24.")
    p.add_argument("--mtp", action="store_true", help="MTP n=3 enabled (adds small KV overhead per request).")
    p.add_argument("--no-mtp", dest="mtp", action="store_false")
    p.add_argument("--dflash-draft-gb", type=float, default=0.0, help="DFlash draft model size in GB (0 if not using DFlash).")
    p.add_argument("--calibration", action="store_true", help="Print predicted vs measured for all calibration points.")
    p.add_argument("--solve-max-ctx", action="store_true", help="Binary-search for the largest max_ctx that fits given the other parameters.")
    p.add_argument("--json", action="store_true", help="Output prediction as JSON.")
    args = p.parse_args()

    if args.calibration:
        run_calibration()
        return 0

    # Resolve defaults from compose preset (used by both modes)
    if args.compose:
        cfg = COMPOSES[args.compose]
        kv_format = args.kv_format or cfg["kv_format"]
        max_ctx = args.max_ctx or cfg["max_ctx"]
        max_num_seqs = args.max_num_seqs or cfg["max_num_seqs"]
        tp = args.tp or cfg["tp"]
        mem_util = args.mem_util if args.mem_util is not None else cfg["mem_util"]
        mtp = args.mtp if args.mtp is not None else cfg.get("mtp", False)
        dflash_gb = args.dflash_draft_gb or cfg.get("dflash_draft_gb", 0.0)
        header = f"Predicted budget — {args.compose}.yml on {args.vram} GB VRAM (kv={kv_format}, ctx={max_ctx}, seqs={max_num_seqs}, TP={tp}, mem={mem_util})"
    else:
        kv_format = args.kv_format or "fp8_e5m2"
        max_ctx = args.max_ctx or 180000
        max_num_seqs = args.max_num_seqs or 1
        tp = args.tp or 1
        mem_util = args.mem_util if args.mem_util is not None else 0.95
        mtp = bool(args.mtp)
        dflash_gb = args.dflash_draft_gb
        header = f"Predicted budget — custom config on {args.vram} GB VRAM (kv={kv_format}, ctx={max_ctx}, seqs={max_num_seqs}, TP={tp}, mem={mem_util})"

    if args.solve_max_ctx:
        # Pin max_ctx very high; we'll search for the actual largest that fits.
        best = solve_max_ctx(QWEN36_27B, kv_format=kv_format, max_num_seqs=max_num_seqs,
                              tp=tp, mem_util=mem_util, vram_gb=args.vram,
                              dflash_draft_gb=dflash_gb, mtp=mtp)
        if best > 0:
            pred_at_best = predict(kv_format=kv_format, max_ctx=best, max_num_seqs=max_num_seqs,
                                    tp=tp, mem_util=mem_util, vram_gb=args.vram,
                                    dflash_draft_gb=dflash_gb, mtp=mtp)
            print(f"Max-ctx solver — {kv_format}, seqs={max_num_seqs}, TP={tp}, mem_util={mem_util}, VRAM={args.vram} GB")
            print(f"  Largest max_ctx that fits: {best:,} tokens")
            print(f"  At that ctx: predicted demand = {pred_at_best.total_gb:.2f} GB / card ({pred_at_best.pct_of_vram:.0f}% of budget)")
            print(f"  Verdict at that ctx: {pred_at_best.verdict}")
            print()
            print("Note: this is a directional estimate (±1.5 GB error band). The vLLM engine")
            print("pre-check (gpu_worker.py boot log) is authoritative.")
        else:
            print(f"No max_ctx fits at this config on {args.vram} GB. Reduce TP, swap KV format, or get bigger cards.")
        return 0

    pred = predict(
        kv_format=kv_format,
        max_ctx=max_ctx,
        max_num_seqs=max_num_seqs,
        tp=tp,
        mem_util=mem_util,
        vram_gb=args.vram,
        dflash_draft_gb=dflash_gb,
        mtp=mtp,
    )

    if args.json:
        print(json.dumps(pred.__dict__, indent=2))
    else:
        print(fmt_prediction(pred, header=header))
        print()
        print("Anchored to: PerfMamba (arxiv 2511.22849), TurboQuant (arxiv 2504.19874),")
        print("PagedAttention (arxiv 2309.06180). Calibrated against BENCHMARKS.md rows.")
        print("Error band: ±1.5 GB on the breakdown. Verdicts within 5% of the budget are TIGHT.")
        print("Run `tools/kv-calc.py --calibration` to see predicted-vs-measured for all anchors.")
        print("Run `tools/kv-calc.py --solve-max-ctx ...` to find the largest max_ctx that fits your config.")
    return 0 if pred.verdict in ("PASS", "TIGHT") else 1


if __name__ == "__main__":
    sys.exit(main())
