#!/usr/bin/env python3
"""IEEE Figure 2 — robustness ROC-AUC bar chart (canonical70k fixed cohort).

Writes:
  experiments/canonical70k/figures/robustness_roc_ieee_color.pdf
  experiments/canonical70k/figures/robustness_roc_ieee_color.png

Run from the repository root:
  python experiments/canonical70k/figures/plot_robustness_roc_ieee.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent


def add_labels(axis, bars, dy=0.006, fontsize=6.8):
    for bar in bars:
        h = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def grouped_bars(ax, x, width, series, colors, hatches, linewidth=0.75, with_labels=True):
    bars = []
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, (label, values) in zip(offsets, series.items()):
        b = ax.bar(
            x + offset * width,
            values,
            width,
            label=label if with_labels else None,
            color=colors[label],
            edgecolor="black",
            hatch=hatches[label],
            linewidth=linewidth,
        )
        bars.append(b)
    return bars


def main() -> None:
    models = ["Features", "CodeBERT", "Hybrid"]

    original = [0.9968, 0.9860, 0.9967]
    black = [0.6055, 0.6936, 0.6807]
    ast_norm = [0.5884, 0.6066, 0.6320]
    comments_removed = [0.9962, 0.9871, 0.9962]

    series = {
        "Original": original,
        "Black": black,
        "AST norm.": ast_norm,
        "Comments removed": comments_removed,
    }

    x = np.arange(len(models))
    width = 0.18

    colors = {
        "Original": "#4C78A8",
        "Black": "#F2A65A",
        "AST norm.": "#72B7B2",
        "Comments removed": "#F3A6A6",
    }
    hatches = {
        "Original": "///",
        "Black": "\\\\\\",
        "AST norm.": "...",
        "Comments removed": "xx",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Side-by-side panels avoid inset/label overlap on the main bars.
    fig, (ax, axz) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.55),
        gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.28},
    )

    bars_main = grouped_bars(ax, x, width, series, colors, hatches, linewidth=0.75)
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.50, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) Full range", fontsize=8, pad=6)

    for b in bars_main:
        add_labels(ax, b, dy=0.010, fontsize=6.3)

    # Zoom panel: only conditions near 1.0 (Original + Comments removed)
    zoom_width = 0.28
    gap = 0.08
    bars_z1 = axz.bar(
        x - 0.5 * zoom_width - gap / 2,
        original,
        zoom_width,
        color=colors["Original"],
        edgecolor="black",
        hatch=hatches["Original"],
        linewidth=0.7,
    )
    bars_z2 = axz.bar(
        x + 0.5 * zoom_width + gap / 2,
        comments_removed,
        zoom_width,
        color=colors["Comments removed"],
        edgecolor="black",
        hatch=hatches["Comments removed"],
        linewidth=0.7,
    )

    axz.set_ylim(0.980, 1.005)
    axz.set_xticks(x)
    axz.set_xticklabels(["Feat.", "CodeBERT", "Hybrid"], fontsize=7)
    axz.tick_params(axis="y", labelsize=7)
    axz.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5)
    axz.set_axisbelow(True)
    axz.spines["top"].set_visible(False)
    axz.spines["right"].set_visible(False)
    axz.set_title("(b) Zoom near 1.0", fontsize=8, pad=4)
    axz.set_ylabel("ROC-AUC", fontsize=8)

    # Stagger vertical offsets so near-equal heights do not collide.
    add_labels(axz, bars_z1, dy=0.0009, fontsize=5.8)
    add_labels(axz, bars_z2, dy=0.00015, fontsize=5.8)

    fig.legend(
        handles=[b[0] for b in bars_main],
        labels=list(series.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.98)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "robustness_roc_ieee_color.pdf"
    png_path = OUT_DIR / "robustness_roc_ieee_color.png"
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
