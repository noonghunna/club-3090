"""Qwen3.8-27B: SGLang vs vLLM on 2x RTX 3090 (TP=2) — decode + prefill.

Companion to qwen38-throughput.png (the vLLM-only DFlash2 tier chart from
discussion #1076). This one is the two-engine comparison for discussion #1237's
sgl/ tier.

⚠️⚠️ THE PREFILL PANEL IS NOT LIKE-FOR-LIKE AND THE CHART SAYS SO. The vLLM
speed tiers ship W4A8 (int8 activations, vendored Marlin patch); stock SGLang
has no W4A8 path for a dense auto-round 4-bit checkpoint. Against vLLM at
matched W4A16 (1219.5 prefill@10K) SGLang's 1183 is ~-3%, not -33%. The hatched
bar shows SGLang WITH @A1RM4X's W4A8 patch (club-3090#1226), which reaches
prefill parity.

Data: BENCHMARKS.md, canonical bench (3 warm + 5 measured), ENABLE_THINKING=1.
SGLang rows are single-boot; superfast decode figures are MEDIANS (its narrative
leg had CV 20%). vLLM rows are the published dual-tier table.

Re-run:  python3 docs/img/qwen38-sglang-vs-vllm.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
VLLM, SGL, SGLW = "#4C78A8", "#E45756", "#F2A25C"

# tier, vllm_narr, vllm_code, sgl_narr, sgl_code, note
decode = [
    ("dual-fast\n(int4 · MTP)",        73, 100,  89.2, 112.5, ""),
    ("dual-superfast\n(int4 · DFlash2)",78, 141, 115.1, 171.2, "†"),
    ("dual-max\n(fp8 · MTP)",          69,  87,  67.3,  75.6, ""),
    ("dual-supermax\n(fp8 · DFlash2)", 68, 130,  None,  None, "‡"),
]
# tier, vllm_prefill(W4A8), sgl_prefill(W4A16), sgl_prefill(W4A8 patched)
prefill = [
    ("dual-fast",      1778, 1183, 1710),
    ("dual-superfast", 1851, 1234, 1779),
    ("dual-max",       1178, 1121, None),
    ("dual-supermax",  1216, None, None),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10.5), height_ratios=[1.25, 1])
fig.suptitle("Qwen3.8-27B — SGLang vs vLLM on 2× RTX 3090 (TP=2)",
             fontsize=16, fontweight="bold", y=0.985)

# ---- decode panel ----
x = np.arange(len(decode)); w = 0.2
for i, (off, key, col, lab) in enumerate([
        (-1.5*w, 1, VLLM, "vLLM narrative"), (-0.5*w, 2, VLLM, "vLLM code"),
        ( 0.5*w, 3, SGL,  "SGLang narrative"), ( 1.5*w, 4, SGL, "SGLang code")]):
    vals = [(r[key] if r[key] is not None else 0) for r in decode]
    alpha = 0.6 if "narrative" in lab else 1.0
    b = ax1.bar(x+off, vals, w, color=col, alpha=alpha, label=lab,
                edgecolor="white", linewidth=0.6)
    for rect, v in zip(b, vals):
        if v: ax1.text(rect.get_x()+rect.get_width()/2, v+2, f"{v:g}",
                       ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([r[0]+r[5] for r in decode], fontsize=9.5)
ax1.set_ylabel("decode tok/s", fontsize=11)
ax1.set_title("Decode — SGLang wins the int4 speed tiers, loses the fp8 fidelity tier",
              fontsize=11.5, pad=8)
ax1.legend(fontsize=9, ncol=4, loc="upper left"); ax1.grid(axis="y", alpha=0.25)
ax1.set_ylim(0, 200)
ax1.text(3.2, 26, "SGLang:\nnot shipped\non dual ‡", ha="center", fontsize=8.5,
         style="italic", color="#666")
ax1.text(2.0, 112, "SGLang caps at 131K here\n(VRAM-bound, measured)",
         ha="center", fontsize=8.5, color="#B03030",
         bbox=dict(fc="#FFF0F0", ec="#E0B0B0", lw=0.6, pad=3))

# ---- prefill panel ----
x2 = np.arange(len(prefill)); w2 = 0.26
bv = ax2.bar(x2-w2, [r[1] or 0 for r in prefill], w2, color=VLLM,
             label="vLLM (W4A8)", edgecolor="white", linewidth=0.6)
bs = ax2.bar(x2,     [r[2] or 0 for r in prefill], w2, color=SGL,
             label="SGLang (W4A16 — stock)", edgecolor="white", linewidth=0.6)
bw = ax2.bar(x2+w2,  [r[3] or 0 for r in prefill], w2, color=SGLW, hatch="//",
             label="SGLang + W4A8 patch (#1226)", edgecolor="white", linewidth=0.6)
for bars in (bv, bs, bw):
    for rect in bars:
        v = rect.get_height()
        if v: ax2.text(rect.get_x()+rect.get_width()/2, v+22, f"{v:g}",
                       ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x2); ax2.set_xticklabels([r[0] for r in prefill], fontsize=9.5)
ax2.set_ylabel("prefill tok/s @10K", fontsize=11)
ax2.set_title("Prefill — ⚠ NOT like-for-like: vLLM ships W4A8, stock SGLang is W4A16",
              fontsize=11.5, pad=8, color="#B03030")
ax2.legend(fontsize=9, loc="upper right"); ax2.grid(axis="y", alpha=0.25)
ax2.set_ylim(0, 2250)
ax2.text(0.5, 1980, "matched-quant (both W4A16): 1219.5 vs 1183 ≈ −3%, not −33%",
         fontsize=9, color="#B03030",
         bbox=dict(fc="#FFF0F0", ec="#E0B0B0", lw=0.6, pad=3))

fig.text(0.5, 0.012,
         "† medians (superfast narrative CV 20%).  ‡ dual-supermax not shipped on SGLang — "
         "the external drafter leaves too little KV pool at TP=2.  "
         "Single boot per config; sub-5% gaps are noise on this rig.",
         ha="center", fontsize=8.2, color="#555")
fig.tight_layout(rect=[0, 0.028, 1, 0.975])
for ext in ("png", "svg"):
    fig.savefig(OUT/f"qwen38-sglang-vs-vllm.{ext}", dpi=170 if ext=="png" else None,
                bbox_inches="tight")
print("wrote", OUT/"qwen38-sglang-vs-vllm.png")
