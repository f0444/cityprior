"""Figures for Part 7 (inD, pedestrian memory across days)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import ind as D  # noqa: E402
from .plots import BLUE, GRID, INK, ORANGE, SEQ_ORANGE  # noqa: E402

VIOLET = "#4a3aa7"
COLORS = {"same session": ORANGE, "other sessions": BLUE, "other sessions+": VIOLET}
LABELS = {"same session": "same session", "other sessions": "other days,\nequal volume",
          "other sessions+": "other days,\nall"}


def fig_cross_day(path: Path, res: dict, maps: dict):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax = axs[0]
    groups = [("all", res["pooled"])] + [(f"site {k}", v) for k, v in sorted(res["locations"].items())]
    w = 0.26
    for i, v in enumerate(COLORS):
        xs, ys, lo, hi = [], [], [], []
        for j, (_, r) in enumerate(groups):
            if v not in r:
                continue
            m, a, b = r[v]["info_gain_bits"]
            xs.append(j + (i - 1) * w), ys.append(m), lo.append(m - a), hi.append(b - m)
        ax.bar(xs, ys, w * 0.92, color=COLORS[v], yerr=[lo, hi], capsize=2, ecolor=INK, label=LABELS[v].replace("\n", " "))
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylabel("information gain vs. uniform over the road (bits / entry)")
    ax.set_title("a  Where pedestrians step onto the road, predicted from memory", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6)

    ax = axs[1]
    loc = max(maps, key=lambda k: len(maps[k][2]))
    grid, road, entries, rec_session = maps[loc]
    sessions = sorted(set(rec_session.values()))
    train = entries[entries["recording"].map(rec_session) == sessions[0]]
    test = entries[entries["recording"].map(rec_session) != sessions[0]]
    p = D.density(train, grid, road, uniform_share=0.0)
    ext = (grid.x0, grid.x0 + grid.nx * D.CELL, grid.y0, grid.y0 + grid.ny * D.CELL)
    ax.imshow(np.where(road, 1.0, np.nan), extent=ext, origin="lower", cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
    top = np.quantile(p[road], 0.90)                     # the 10% of road area memory ranks highest
    pm = np.ma.masked_where(p < top, p)
    ax.imshow(pm, extent=ext, origin="lower", cmap=SEQ_ORANGE, vmin=top, vmax=p.max(), interpolation="nearest", alpha=0.9)
    ax.scatter(test["x"], test["y"], s=5, c=BLUE, alpha=0.5, lw=0)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"b  Site {loc}: one day's top 10% (orange), the other day's entries (blue)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
