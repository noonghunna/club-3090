"""Generate TPS comparison bar-charts for Qwen3.6-27B on club-3090.

Outputs (in docs/img/):
  performance.svg + .png        — all 10 configs (single + dual), used by top README
  performance-single.svg + .png — 6 single-card configs, used by docs/SINGLE_CARD.md
  performance-dual.svg + .png   — 4 dual-card configs, used by docs/DUAL_CARD.md

Source data: /opt/ai/BENCHMARKS.md rows R2/R3'/R3'''/R6, L1, ngram-mod, C1-C4.

Re-run:  python3 tools/charts/gen-perf.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "img"

# (label, narr_tps, code_tps, group)
configs_all = [
    ("v714 48K\n(default)",       55.00, 70.50, "single-vllm"),
    ("v714 192K\n+ vision",       50.93, 67.69, "single-vllm"),
    ("v714 205K\ntext-only",      50.11, 65.84, "single-vllm"),
    ("minimal\n(no spec-dec)",    32.41, 32.56, "single-vllm"),
    ("llama.cpp Q3_K_XL\n262K + vision", 21.22, 20.79, "single-llama"),
    ("llama.cpp Q4_K_M\n+ ngram-mod 32K",22.04, 26.11, "single-llama"),
    ("dual.yml\n262K + vision",   69.05, 88.58, "dual-vllm"),
    ("dual-turbo\n4 streams 262K",53.65, 72.93, "dual-vllm"),
    ("dual-dflash\n185K + vision",81.94, 124.93, "dual-vllm"),
    ("dual-dflash-noviz\n200K",   78.19, 126.99, "dual-vllm"),
]

GROUP_COLORS = {
    "single-vllm":  ("#9ec5e8", "#2c7fb8"),
    "single-llama": ("#fdd0a2", "#e6550d"),
    "dual-vllm":    ("#a1d99b", "#2c8a2c"),
}
GROUP_LABELS = {
    "single-vllm":  "single 3090 — vLLM patched",
    "single-llama": "single 3090 — llama.cpp",
    "dual-vllm":    "2× 3090 — vLLM (TP=2)",
}


def make_chart(configs, out_stem, title_subject, figsize):
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
    y_band = max(max(narr), max(code)) * 1.10
    for g, idxs in groupseen.items():
        mid = (idxs[0] + idxs[-1]) / 2
        ax.text(mid, y_band, GROUP_LABELS[g], ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("TPS  (3 warm + 5 measured, canonical bench)", fontsize=10)
    ax.set_title(f"Qwen3.6-27B  —  measured TPS {title_subject}  on  noonghunna/club-3090  (2026-04-28)",
                 fontsize=12, pad=22)
    ax.set_ylim(0, max(max(narr), max(code)) * 1.20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#cccccc", edgecolor="#333", label="narrative (essay prompt, 1000 tok)"),
        Patch(facecolor="#666666", edgecolor="#333", label="code (quicksort prompt, 800 tok)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9, frameon=False)

    ax.text(0.5, -0.22,
            "Substrate: vLLM nightly dev205+g07351e088 + Genesis 917519b (v7.62.x)  •  llama.cpp mainline 0d0764dfd  •  RTX 3090 sm_86, PCIe-only, 230W",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555", style="italic")

    plt.tight_layout()
    svg_path = OUT / f"{out_stem}.svg"
    png_path = OUT / f"{out_stem}.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path.name} + {png_path.name}")


single_configs = [c for c in configs_all if c[3].startswith("single-")]
dual_configs   = [c for c in configs_all if c[3].startswith("dual-")]

make_chart(configs_all,    "performance",        "per config",          figsize=(14, 6.5))
make_chart(single_configs, "performance-single", "(single 3090)",       figsize=(11, 6.0))
make_chart(dual_configs,   "performance-dual",   "(2× 3090, TP=2)",     figsize=(8.5, 6.0))
