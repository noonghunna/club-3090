"""Generate per-card VRAM breakdown diagrams for Qwen3.6-27B configs.

Outputs (in docs/img/):
  vram-budget-combined.svg + .png — both single + dual sections (used in model README)
  vram-budget-single.svg + .png   — single-card configs only (used in docs/SINGLE_CARD.md)
  vram-budget-dual.svg + .png     — dual-card configs only (used in docs/DUAL_CARD.md)

Numbers from /opt/ai/BENCHMARKS.md (single-card R3', dual rebench C1-C4 + Q3_K_XL L1).
Component splits estimated from architectural math + boot-log inspection;
totals are measured.

Re-run:  python3 tools/charts/gen-vram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "img"

# Color palette
C_W   = "#3a6ea5"   # Model weights (deep blue)
C_KV  = "#e6843a"   # KV cache (orange)
C_VIS = "#7bb872"   # Vision tower / mmproj (green)
C_DFL = "#b070b8"   # DFlash draft (purple)
C_ACT = "#cccccc"   # Activations / workspace (light gray)
C_FREE= "#f4f4f4"   # Free / headroom (very light)

rows_single = [
    ("vLLM single 48K\n(default, fp8 KV)",
        [(7.0, "weights", C_W), (3.5, "KV (fp8)", C_KV), (0.5, "vision", C_VIS),
         (10.0, "activations + workspace", C_ACT), (3.0, "free", C_FREE)]),
    ("vLLM single 145K + vision\nTQ3 KV, mem-util 0.95 (v0.20 + v7.69)",
        [(7.0, "weights", C_W), (5.5, "KV (TQ3, 264K tok pool)", C_KV), (0.5, "vision", C_VIS),
         (9.3, "activations + workspace", C_ACT), (1.7, "free", C_FREE)]),
    ("vLLM single 180K Balanced MTP\nTQ3 KV, mem-util 0.93 (v0.20 + v7.69 + #35975)",
        [(7.0, "weights", C_W), (5.7, "KV (TQ3, 285K tok pool)", C_KV),
         (10.0, "activations + workspace", C_ACT), (1.3, "free", C_FREE)]),
    ("vLLM single 200K Max-context\nTQ3 KV, mem-util 0.95, no MTP (v0.20 + v7.69 + #35975)",
        [(7.0, "weights", C_W), (6.6, "KV (TQ3, 316K tok pool)", C_KV),
         (10.4, "activations + workspace", C_ACT), (0.0, "free", C_FREE)]),
    ("llama.cpp single 262K\nQ3_K_XL + mmproj + q4_0 KV",
        [(14.0, "weights", C_W), (4.5, "KV (q4_0)", C_KV), (0.8, "vision", C_VIS),
         (0.9, "activations", C_ACT), (3.8, "free", C_FREE)]),
]

rows_dual = [
    ("dual.yml  (default)\n262K + vision · 2 streams · fp8 KV",
        [(7.0, "weights", C_W), (10.0, "KV (fp8)", C_KV), (0.5, "vision", C_VIS),
         (6.1, "activations", C_ACT), (0.4, "free", C_FREE)]),
    ("dual-turbo\n262K · 4 streams · TQ3 KV (v0.20 + v7.69)",
        [(7.0, "weights", C_W), (6.0, "KV (TQ3, 1.5M tok pool, 4.67×)", C_KV), (0.5, "vision", C_VIS),
         (6.3, "activations", C_ACT), (4.2, "free", C_FREE)]),
    ("dual-dflash\n185K · 1 stream · FP16 KV",
        [(7.0, "weights", C_W), (10.5, "KV (FP16)", C_KV), (0.5, "vision", C_VIS),
         (1.75, "DFlash draft", C_DFL), (3.85, "activations", C_ACT), (0.4, "free", C_FREE)]),
    ("dual-dflash-noviz\n200K · 1 stream · FP16 KV",
        [(7.0, "weights", C_W), (11.5, "KV (FP16)", C_KV),
         (1.75, "DFlash draft", C_DFL), (3.55, "activations", C_ACT), (0.2, "free", C_FREE)]),
]

LEGEND = [
    Patch(facecolor=C_W,   edgecolor="#333", label="Model weights (Lorbus AutoRound INT4 / Q3_K_XL)"),
    Patch(facecolor=C_KV,  edgecolor="#333", label="KV cache"),
    Patch(facecolor=C_VIS, edgecolor="#333", label="Vision tower (mmproj)"),
    Patch(facecolor=C_DFL, edgecolor="#333", label="DFlash draft"),
    Patch(facecolor=C_ACT, edgecolor="#333", label="Activations / workspace / cudagraph pools"),
    Patch(facecolor=C_FREE, edgecolor="#333", label="Free headroom"),
]


def draw_panel(ax, rows, title):
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    for i, (label, segs) in enumerate(rows):
        x = 0
        for size, name, color in segs:
            if size <= 0:
                continue
            ax.barh(i, size, left=x, color=color, edgecolor="#333", linewidth=0.5, height=0.55)
            if size >= 1.0 and color != C_FREE:
                ax.text(x + size/2, i, name, ha="center", va="center",
                        fontsize=7.5, color="#fff" if color in (C_W, C_KV, C_DFL) else "#222")
            x += size
        ax.text(-0.3, i, label, ha="right", va="center", fontsize=8.5)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("VRAM per card  (GB / 24 GB)", fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, loc="left")
    ax.axvline(24, color="#aa3333", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(24.1, -0.5, "24 GB ceiling", color="#aa3333", fontsize=8, va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.3)


def save(fig, stem):
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path.name} + {png_path.name}")


# ----- combined (single + dual) -----
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7.5), dpi=110,
                                      gridspec_kw={"height_ratios": [3, 4]})
draw_panel(ax_top, rows_single, "Single 3090 — what fits on one card (TP=1)")
draw_panel(ax_bot, rows_dual,   "Dual 3090 — vLLM (TP=2), per-card breakdown — all 4 configs run vLLM; dual-dflash uses vLLM's DFlash spec-decode")
fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Qwen3.6-27B on RTX 3090 — VRAM allocation across configs (single + dual TP=2)",
             fontsize=12, y=0.99)
fig.text(0.5, -0.07,
         "Component sizes are approximate (architectural math + boot-log inspection); per-card totals are measured.\n"
         "TP=2 splits weights and KV symmetrically across both cards — both bars in the dual section are identical.\n"
         "DFlash draft adds ~1.75 GB / card; vision adds ~0.5 GB / card; turbo's 4 streams inflate cudagraph pools.",
         ha="center", va="top", fontsize=7.5, color="#555", style="italic")
plt.tight_layout(rect=[0.18, 0.06, 1, 0.96])
save(fig, "vram-budget-combined")

# ----- single-only -----
fig, ax = plt.subplots(figsize=(11, 4.0), dpi=110)
draw_panel(ax, rows_single, "Single 3090 — what fits on one card (TP=1)")
fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Qwen3.6-27B on 1× RTX 3090 — VRAM allocation per config",
             fontsize=12, y=0.99)
fig.text(0.5, -0.18,
         "Component sizes are approximate (architectural math + boot-log inspection); per-card totals are measured.\n"
         "Cliffs fire when 'activations' peak exceeds free headroom — see docs/FAQ.md for details.",
         ha="center", va="top", fontsize=7.5, color="#555", style="italic")
plt.tight_layout(rect=[0.18, 0.10, 1, 0.96])
save(fig, "vram-budget-single")

# ----- dual-only -----
fig, ax = plt.subplots(figsize=(13, 5.0), dpi=110)
draw_panel(ax, rows_dual, "Dual 3090 — vLLM (TP=2), per-card breakdown — all 4 configs run vLLM; dual-dflash uses vLLM's DFlash spec-decode")
fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Qwen3.6-27B on 2× RTX 3090 — VRAM allocation per config (TP=2)",
             fontsize=12, y=0.99)
fig.text(0.5, -0.16,
         "Both cards are identical under TP=2 — we show one card's VRAM split.\n"
         "DFlash draft adds ~1.75 GB / card; vision adds ~0.5 GB / card; turbo's 4 streams inflate cudagraph pools.",
         ha="center", va="top", fontsize=7.5, color="#555", style="italic")
plt.tight_layout(rect=[0.18, 0.08, 1, 0.96])
save(fig, "vram-budget-dual")
