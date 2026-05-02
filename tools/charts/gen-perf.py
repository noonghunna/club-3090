"""Generate TPS comparison bar-charts for Qwen3.6-27B on club-3090.

Outputs (in docs/img/):
  performance.svg + .png        — single + dual configs, used by top README
  performance-single.svg + .png — 6 single-card configs, used by docs/SINGLE_CARD.md
  performance-dual.svg + .png   — 4 dual-card configs, used by docs/DUAL_CARD.md
  performance-quad-pairs.svg + .png
                                  — 4-card routed-pairs results, used by docs/QUAD_CARD.md

Source data: results/v0.20-migration/*.summary (post-migration n=5 benches).
Quad-pairs source: bench-results/bench-20260501-195114.md.

Re-run:  python3 tools/charts/gen-perf.py
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/club3090-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "img"
EXPORT_DPI = 180

# (label, narr_tps, code_tps, group)
# Single-card + dual-turbo numbers re-benched 2026-05-01 PM on the v0.20
# (0.20.1rc1.dev16+g7a1eb8ac2) + Genesis v7.66 dev tip (commit fc89395)
# substrate, n=5 measured runs after 3 warmups. Dual.yml / dual-dflash*
# pending re-bench; their dev205-era numbers carry forward as estimates
# (steady-state TPS is the same regime — fp8 paths weren't TPS-changed
# by the v0.20 migration).
# Luce DFlash measured 2026-04-30 PM on Qwen3.6-27B Q4_K_M + matched 3.6
# draft (TQ3 KV, max_ctx=65K, greedy only). Group "single-luce-watch" =
# experimental / not recommended for shipping yet; see docs/UPSTREAM.md.
configs_all = [
    ("v714 48K\n(default)",       55.00, 70.50, "single-vllm"),
    ("long-vision 145K\n+ vision",       50.32, 66.12, "single-vllm"),
    ("long-text 180K\ntext-only",      49.74, 67.39, "single-vllm"),
    ("tools-text 75K\nfp8 IDE-agent",  53.32, 69.66, "single-vllm"),
    ("bounded-thinking 180K\nstructured-CoT",  49.77, 65.80, "single-vllm"),
    ("minimal\n(no spec-dec)",    32.41, 32.56, "single-vllm"),
    ("llama.cpp Q3_K_XL\n262K + vision", 21.22, 20.79, "single-llama"),
    ("llama.cpp Q4_K_M\n+ ngram-mod 32K",22.04, 26.11, "single-llama"),
    ("Luce DFlash 3.6+3.6*\nTQ3, 65K, greedy", 40.00, 71.65, "single-luce-watch"),
    ("dual.yml\n262K + vision",   69.05, 88.58, "dual-vllm"),
    ("dual-turbo\n4 streams 262K",58.33, 76.01, "dual-vllm"),
    ("dual-dflash\n185K + vision",81.94, 124.93, "dual-vllm"),
    ("dual-dflash-noviz\n200K",   78.19, 126.99, "dual-vllm"),
]

quad_pair_configs = [
    ("Pair A\nGPUs 0,1",            189.08, 229.71, "quad-vllm"),
    ("Pair B\nGPUs 2,3",            184.95, 225.87, "quad-vllm"),
    ("Combined\n2× TP=2 pairs",     369.84, 444.57, "quad-vllm-aggregate"),
]

GROUP_COLORS = {
    "single-vllm":         ("#9ec5e8", "#2c7fb8"),
    "single-llama":        ("#fdd0a2", "#e6550d"),
    "single-luce-watch":   ("#dadaeb", "#807dba"),
    "dual-vllm":           ("#a1d99b", "#2c8a2c"),
    "quad-vllm":           ("#9dd9d2", "#2b8c83"),
    "quad-vllm-aggregate": ("#f2c078", "#c05a28"),
}
GROUP_LABELS = {
    "single-vllm":         "1× 3090\nvLLM patched",
    "single-llama":        "1× 3090\nllama.cpp",
    "single-luce-watch":   "1× 3090\nLuce DFlash *experimental*",
    "dual-vllm":           "2× 3090\nvLLM (TP=2)",
    "quad-vllm":           "4× 3090\npair endpoints",
    "quad-vllm-aggregate": "4× 3090\ncombined",
}


def make_chart(configs, out_stem, title_subject, figsize, measured_date="2026-04-30"):
    labels = [c[0] for c in configs]
    narr = [c[1] for c in configs]
    code = [c[2] for c in configs]
    groups = [c[3] for c in configs]

    x = np.arange(len(configs))
    w = 0.38

    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    narr_colors = [GROUP_COLORS[g][0] for g in groups]
    code_colors = [GROUP_COLORS[g][1] for g in groups]

    bars1 = ax.bar(x - w/2, narr, w, color=narr_colors, edgecolor="#333", linewidth=0.5)
    bars2 = ax.bar(x + w/2, code, w, color=code_colors, edgecolor="#333", linewidth=0.5)

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8.5, color="#333")
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8.5, color="#000", fontweight="bold")

    boundaries = [i - 0.5 for i in range(1, len(groups)) if groups[i] != groups[i-1]]
    for b in boundaries:
        ax.axvline(b, color="#999", linestyle="--", linewidth=0.7, alpha=0.6)

    groupseen = {}
    for i, g in enumerate(groups):
        groupseen.setdefault(g, []).append(i)
    y_band = max(max(narr), max(code)) * 1.18
    for g, idxs in groupseen.items():
        mid = (idxs[0] + idxs[-1]) / 2
        ax.text(mid, y_band, GROUP_LABELS[g], ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("TPS  (3 warm + 5 measured, canonical bench)", fontsize=10)
    ax.set_title(f"Qwen3.6-27B  —  measured TPS {title_subject}  on  noonghunna/club-3090  ({measured_date})",
                 fontsize=12, pad=36)
    ax.set_ylim(0, max(max(narr), max(code)) * 1.30)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#cccccc", edgecolor="#333", label="narrative (essay prompt, 1000 tok)"),
        Patch(facecolor="#666666", edgecolor="#333", label="code (quicksort prompt, 800 tok)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9, frameon=False)

    if any(g.startswith("quad-") for g in groups):
        substrate_parts = [
            "vLLM nightly dev205+g07351e088",
            "quad-pairs: two TP=2 endpoints + MTP n=3",
            "8 concurrent requests per pair",
            "RTX 3090 sm_86, two NVLink pairs",
        ]
    else:
        substrate_parts = ["vLLM 0.20.1rc1.dev16+g7a1eb8ac2 + Genesis v7.66 dev (fc89395)"]
    if any(g == "single-llama" for g in groups):
        substrate_parts.append("llama.cpp mainline 0d0764dfd")
    if any(g == "single-luce-watch" for g in groups):
        substrate_parts.append("Luce DFlash dflash@f12a87c (greedy only)")
    if not any(g.startswith("quad-") for g in groups):
        substrate_parts.append("RTX 3090 sm_86, PCIe-only, 230W")
    ax.text(0.5, -0.22,
            "Substrate: " + "  •  ".join(substrate_parts),
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555", style="italic")
    if any(g == "single-luce-watch" for g in groups):
        ax.text(0.5, -0.30,
                "* Luce DFlash 3.6+3.6 = experimental: matched draft still under training (z-lab 2026-04-26 snapshot), greedy-only sampling, no vision, daemon-mode bugs. Not yet recommended for shipping. See docs/UPSTREAM.md.",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#777", style="italic", wrap=True)
    if any(g.startswith("quad-") for g in groups):
        ax.text(0.5, -0.30,
                "Source: bench-results/bench-20260501-195114.md; 3 warmup + 5 measured batches, direct pair endpoints now default to :8021 and :8022.",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#777", style="italic", wrap=True)

    plt.tight_layout()
    svg_path = OUT / f"{out_stem}.svg"
    png_path = OUT / f"{out_stem}.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path.name} + {png_path.name}")


single_configs = [c for c in configs_all if c[3].startswith("single-")]
dual_configs   = [c for c in configs_all if c[3].startswith("dual-")]

make_chart(configs_all,    "performance",        "per config",          figsize=(18, 7.5))
make_chart(single_configs, "performance-single", "(single 3090)",       figsize=(13, 6.5))
make_chart(dual_configs,   "performance-dual",   "(2× 3090, TP=2)",     figsize=(8.5, 6.5))
make_chart(quad_pair_configs, "performance-quad-pairs",
           "(4× 3090, two TP=2 pairs)", figsize=(10.5, 6.5),
           measured_date="2026-05-01")

# Tweet-asset: just the two recommended single-card vLLM routes
# (long-vision + long-text). Clean visual match for the launch tweet.
tweet_configs = [c for c in configs_all if c[0].startswith(("long-vision", "long-text"))]
make_chart(tweet_configs,  "performance-single-vllm",
           "(single 3090, vLLM patched)", figsize=(7.5, 6.5))
