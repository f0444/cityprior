"""Figures for Part 5 (pNEUMA, cross-day transfer)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .plots import AQUA, BLUE, GRAY, GRID, INK, ORANGE  # noqa: E402

VIOLET = "#4a3aa7"
COLORS = {"same day": ORANGE, "other days": BLUE, "other days+": VIOLET, "all": AQUA}
LABELS = {"same day": "same morning,\nother half-hours\n(1.5 h)",
          "other days": "other mornings,\nsame half-hour\n(1.5 h)",
          "other days+": "other mornings,\nall half-hours\n(6 h)",
          "all": "everything but\nthe test\n(7.5 h)"}


def fig_cross_day(path: Path, res: dict):
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, tname, title in ((axs[0], "speed in 3 s", "a  Speed class 3 s ahead"),
                             (axs[1], "heading in 3 s", "b  Heading 3 s ahead (moving vehicles)")):
        r = res["cross_day"][tname]
        names = list(COLORS)
        vals = [r[n]["info_gain_bits"][0] for n in names]
        lo = [r[n]["info_gain_bits"][0] - r[n]["info_gain_bits"][1] for n in names]
        hi = [r[n]["info_gain_bits"][2] - r[n]["info_gain_bits"][0] for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, 0.62, color=[COLORS[n] for n in names], yerr=[lo, hi], capsize=3, ecolor=INK)
        for xi, n, v in zip(x, names, vals):
            ax.text(xi, v * 1.02 + 0.005, f"{v:.3f}\n{100 * r[n]['top1']:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        ax.text(0.99, 0.97, f"area-wide statistics: {100 * r['area-wide statistics']['top1']:.1f}% correct",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        ax.set_xticks(x, [LABELS[n] for n in names], fontsize=8.5)
        ax.set_ylabel("information gain vs. area-wide statistics (bits)")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y", color=GRID, lw=0.6)

    ax = axs[2]
    for tname, color in (("speed in 3 s", BLUE), ("heading in 3 s", ORANGE)):
        c = res["learning_curve"][tname]
        same = res["cross_day"][tname]["same day"]["info_gain_bits"][0]
        n = [int(k) for k in c]
        rel = [100 * c[k][0] / same for k in c]
        ax.plot(n, rel, color=color, lw=2, marker="o", ms=6, label=tname)
        dy = -15 if tname.startswith("speed") else 7
        for xi, yi in zip(n, rel):
            ax.annotate(f"{yi:.0f}%", (xi, yi), xytext=(0, dy), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
    ax.axhline(100, color=GRAY, lw=1.5, ls="--")
    ax.text(2.97, 98.5, "memory of the same morning", fontsize=8.5, color=INK, va="top", ha="right")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("other mornings in memory (all half-hours)")
    ax.set_ylabel("information gain, % of same-morning memory")
    ax.set_ylim(80, 150)
    ax.set_title("c  More mornings, better memory", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
