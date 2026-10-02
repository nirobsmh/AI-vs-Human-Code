#!/usr/bin/env python3
"""IEEE one-column pipeline figure for the canonical70k hybrid detector.

Writes:
  experiments/canonical70k/figures/pipeline_ieee.pdf
  experiments/canonical70k/figures/pipeline_ieee.png

Run from the repository root:
  python experiments/canonical70k/figures/plot_pipeline_ieee.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent


def add_box(
    ax,
    x,
    y,
    text,
    facecolor="white",
    fontsize=6.2,
    lw=0.9,
    ls="-",
    pad=1.20,
):
    """Label with an auto-sized surrounding box (pad is a fraction of fontsize)."""
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.35,
        zorder=3,
        clip_on=False,
        bbox={
            "boxstyle": f"round,pad={pad}",
            "facecolor": facecolor,
            "edgecolor": "black",
            "linewidth": lw,
            "linestyle": ls,
        },
    )


def add_arrow(ax, x1, y1, x2, y2, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.85,
            color="black",
            linestyle=ls,
            shrinkA=6,
            shrinkB=6,
            clip_on=False,
            zorder=1,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 3.80))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    # Column centers — spaced for larger auto-padded boxes
    src_x, feat_x, clf_x, p_x, out_x = 0.85, 3.00, 5.55, 7.10, 8.65
    top_y, mid_y, bot_y = 7.65, 5.20, 2.75
    test_y, rob_y = 9.25, 0.95

    add_box(ax, src_x, mid_y, "Source\ncode", fontsize=6.5)

    add_box(
        ax,
        feat_x,
        top_y,
        "25 handcrafted\nfeatures",
        facecolor="0.94",
        fontsize=6.1,
    )
    add_box(
        ax,
        feat_x,
        bot_y,
        "CodeBERT\n768-d\nembedding",
        facecolor="0.94",
        fontsize=6.1,
    )

    add_arrow(ax, 1.55, mid_y + 0.2, 2.35, top_y)
    add_arrow(ax, 1.55, mid_y - 0.2, 2.35, bot_y)

    add_box(ax, clf_x, top_y, "Random\nForest", facecolor="0.87", fontsize=6.1)
    add_box(
        ax,
        clf_x,
        bot_y,
        "Logistic\nRegression",
        facecolor="0.87",
        fontsize=6.1,
    )

    add_arrow(ax, 3.95, top_y, 4.85, top_y)
    add_arrow(ax, 3.95, bot_y, 4.85, bot_y)

    ax.text(p_x, top_y, r"$p_f(x)$", fontsize=6.8, ha="center", va="center", zorder=4)
    ax.text(p_x, bot_y, r"$p_e(x)$", fontsize=6.8, ha="center", va="center", zorder=4)

    add_arrow(ax, 7.45, top_y, 7.95, mid_y + 0.55)
    add_arrow(ax, 7.45, bot_y, 7.95, mid_y - 0.55)

    # Keep the fusion equation on two short lines so the box stays compact.
    add_box(
        ax,
        out_x,
        mid_y,
        "Score-level fusion\n"
        r"$0.875\,p_f$" "\n"
        r"$+\,0.125\,p_e$",
        facecolor="0.80",
        fontsize=5.9,
        pad=1.30,
    )

    add_box(
        ax,
        out_x,
        test_y,
        "Held-out test\nperformance",
        fontsize=5.9,
    )
    add_box(
        ax,
        out_x,
        rob_y,
        "Robustness evaluation\nBlack / AST /\ncomments removed",
        fontsize=5.6,
        ls="--",
        pad=1.50,
    )

    add_arrow(ax, out_x, mid_y + 0.95, out_x, test_y - 0.55)
    add_arrow(ax, out_x, mid_y - 0.95, out_x, rob_y + 0.70, ls="--")

    ax.text(out_x + 1.15, 7.35, "fixed models", fontsize=5.2, ha="left", va="center")
    ax.text(out_x + 1.15, 3.05, "no retraining", fontsize=5.2, ha="left", va="center")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "pipeline_ieee.pdf"
    png_path = OUT_DIR / "pipeline_ieee.png"
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
