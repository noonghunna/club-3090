"""Generate TPS comparison bar-charts for Qwen3.6-27B on club-3090.

Outputs (in docs/img/):
  performance.svg + .png        — all 10 configs (single + dual), used by top README
  performance-single.svg + .png — 6 single-card configs, used by docs/SINGLE_CARD.md

  † = slug RETIRED 2026-08-12 (needs --force). The measurement is real and kept as
  the historical record; the marker stops the chart from reading as a current
  recommendation. See docs/SINGLE_CARD.md for what actually runs today.
  performance-dual.svg + .png   — 4 dual-card configs, used by docs/DUAL_CARD.md

Source data: results/v0.20-migration/*.summary (post-migration n=5 benches).

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
# Single-card + dual-turbo numbers re-benched 2026-05-01 PM on the v0.20
# (0.20.1rc1.dev16+g7a1eb8ac2) + Genesis v7.69 dev tip (commit 2db18df, the
# 2026-05-02 PM cutover) + local vllm#35975 inputs_embeds backport substrate,
# n=5 measured runs after 3 warmups. Decode TPS is unchanged by the v7.69
# bump (PN32/P103/PN30-part3 fix Cliff 2 prefill envelope, not steady-state
# decode). Dual.yml / dual-dflash* pending re-bench; their dev205-era numbers
# carry forward as estimates (fp8 paths weren't TPS-changed by the v0.20
# migration).
# Luce DFlash measured 2026-04-30 PM on Qwen3.6-27B Q4_K_M + matched 3.6
# draft (TQ3 KV, max_ctx=65K, greedy only). Group "single-luce-watch" =
# experimental / not recommended for shipping yet; see docs/UPSTREAM.md.
# (compose-name, narr_tps, code_tps, group, description)
# Labels on the chart show the compose name only; description rendered as a
# legend block below the chart so x-axis stays readable at any density.
configs_all = [
    ("v714",                55.00,  70.50, "single-vllm",       "48K default"),
    ("long-vision",         50.32,  66.12, "single-vllm",       "145K +vision"),
    ("long-text",           49.74,  67.39, "single-vllm",       "180K MTP"),
    # long-text-no-mtp 200K (Max-context) bench pending — not on chart yet.
    # Estimated ~33 narr / ~40 code TPS based on no-spec-decode regime.
    ("tools-text",          53.32,  69.66, "single-vllm",       "75K fp8 IDE-agent"),
    ("bounded-thinking",    49.77,  65.80, "single-vllm",       "180K structured-CoT"),
    ("minimal",             32.41,  32.56, "single-vllm",       "no spec-dec"),
    ("llamacpp/mtp †",      51.28,  59.72, "single-llama",      "131K Q4_K_M + MTP"),
    ("llamacpp/mtp-vision †", 56.52, 66.17, "single-llama",     "49K Q4_K_M + MTP + vision"),
    ("llamacpp/default †",  21.22,  20.79, "single-llama",      "262K Q3_K_XL + vision"),
    ("llama.cpp+ngram",     22.04,  26.11, "single-llama",      "32K Q4_K_M + ngram-mod"),
    ("Luce DFlash*",        40.00,  71.65, "single-luce-watch", "65K TQ3, greedy"),
    ("dual.yml",            69.05,  88.58, "dual-vllm",         "262K + vision"),
    ("dual-turbo",          58.33,  76.01, "dual-vllm",         "262K, 4 streams"),
    ("dual-dflash",         81.94, 124.93, "dual-vllm",         "185K + vision"),
    ("dual-dflash-noviz",   78.19, 126.99, "dual-vllm",         "200K"),
]

GROUP_COLORS = {
    "single-vllm":         ("#9ec5e8", "#2c7fb8"),
    "single-llama":        ("#fdd0a2", "#e6550d"),
    "single-luce-watch":   ("#dadaeb", "#807dba"),
    "dual-vllm":           ("#a1d99b", "#2c8a2c"),
}
GROUP_LABELS = {
    "single-vllm":         "1× 3090\nvLLM patched",
    "single-llama":        "1× 3090\nllama.cpp",
    "single-luce-watch":   "1× 3090\nLuce DFlash *experimental*",
    "dual-vllm":           "2× 3090\nvLLM (TP=2)",
}


def make_chart(configs, out_stem, title_subject, figsize):
    labels = [c[0] for c in configs]
    narr = [c[1] for c in configs]
    code = [c[2] for c in configs]
    groups = [c[3] for c in configs]
    descriptions = [c[4] if len(c) > 4 else "" for c in configs]

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
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("TPS  (3 warm + 5 measured, canonical bench)", fontsize=10)
    ax.set_title(f"Qwen3.6-27B  —  measured TPS {title_subject}  on  noonghunna/club-3090  (updated 2026-05-20)",
                 fontsize=12, pad=36)
    ax.set_ylim(0, max(max(narr), max(code)) * 1.30)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#cccccc", edgecolor="#333", label="narrative (essay prompt, 1000 tok)"),
        Patch(facecolor="#666666", edgecolor="#333", label="code (quicksort prompt, 800 tok)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#ccc")

    # Compose-name legend: groups → "name = description · name = description"
    # rendered below the x-axis labels. Keeps x-axis short while preserving
    # the per-compose detail.
    legend_group_order = ["single-vllm", "single-llama", "single-luce-watch", "dual-vllm"]
    legend_group_titles = {
        "single-vllm":       "1× vLLM",
        "single-llama":      "1× llama.cpp",
        "single-luce-watch": "1× Luce*",
        "dual-vllm":         "2× vLLM",
    }
    legend_y = -0.22  # below x-axis labels (rotated 30° extend to ~-0.15)

    for g in legend_group_order:
        entries = [(labels[i], descriptions[i]) for i in range(len(labels)) if groups[i] == g and descriptions[i]]
        if not entries:
            continue
        body = "  ·  ".join(f"$\\bf{{{name.replace('_', chr(92)+'_')}}}$ = {desc}" for name, desc in entries)
        line = f"{legend_group_titles[g]}:  {body}"
        ax.text(0.5, legend_y, line,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#333")
        legend_y -= 0.05

    substrate_parts = ["vLLM 0.20.1rc1.dev16+g7a1eb8ac2 + Genesis v7.69 dev (2db18df) + vllm#35975 backport"]
    if any(g == "single-llama" for g in groups):
        substrate_parts.append("llama.cpp mainline d14ce3dab (build 9235, MTP)")
    if any(g == "single-luce-watch" for g in groups):
        substrate_parts.append("Luce DFlash dflash@f12a87c (greedy only)")
    substrate_parts.append("RTX 3090 sm_86, PCIe-only, 230W")
    # † marks slugs RETIRED 2026-08-12 (launchable with --force). Folded into the
    # existing substrate line rather than added as another text row: extra rows
    # re-flow tight_layout and squeeze the axes until the group headers collide.
    if any("†" in str(l) for l in labels):
        substrate_parts.append("† = retired 2026-08-12, --force to launch")
    ax.text(0.5, legend_y - 0.02,
            "Substrate: " + "  •  ".join(substrate_parts),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555", style="italic")
    if any(g == "single-luce-watch" for g in groups):
        ax.text(0.5, legend_y - 0.09,
                "* Luce DFlash 3.6+3.6 = experimental: matched draft still under training (z-lab 2026-04-26 snapshot), greedy-only sampling, no vision, daemon-mode bugs. Not yet recommended for shipping. See docs/UPSTREAM.md.",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#777", style="italic", wrap=True)

    plt.tight_layout()
    svg_path = OUT / f"{out_stem}.svg"
    png_path = OUT / f"{out_stem}.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path.name} + {png_path.name}")


single_configs = [c for c in configs_all if c[3].startswith("single-")]
dual_configs   = [c for c in configs_all if c[3].startswith("dual-")]

make_chart(configs_all,    "performance",        "per config",          figsize=(18, 7.5))
make_chart(single_configs, "performance-single", "(single 3090)",       figsize=(13, 6.5))
make_chart(dual_configs,   "performance-dual",   "(2× 3090, TP=2)",     figsize=(8.5, 6.5))

# Tweet-asset: just the two recommended single-card vLLM routes
# (long-vision + long-text). Clean visual match for the launch tweet.
tweet_configs = [c for c in configs_all if c[0].startswith(("long-vision", "long-text"))]
make_chart(tweet_configs,  "performance-single-vllm",
           "(single 3090, vLLM patched)", figsize=(7.5, 6.5))
