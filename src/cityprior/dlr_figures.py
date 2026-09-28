"""Figures for Part 4 (DLR UT)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from . import dlr  # noqa: E402
from .dlr_experiments import CELL_M  # noqa: E402
from .plots import BLUE, GRAY, GRID, INK, INK2, ORANGE, ROAD  # noqa: E402

VIOLET = "#4a3aa7"
MODEL_COLORS = {"area-wide statistics": GRAY, "live signal only": ORANGE, "location memory only": BLUE,
                "memory + live signal": VIOLET}


def _background(ax):
    t = pq.read_table(dlr.CACHE, columns=["x", "y", "frame"]).to_pandas()
    t = t[t["frame"] % 10 == 0]
    ax.hexbin(t["x"], t["y"], gridsize=160, bins="log", cmap=ROAD, mincnt=1, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_signal_memory(path: Path, res: dict):
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    ax = axs[0]
    _background(ax)
    heads = dlr.light_heads_local()
    gov = res["governing"]
    for key, g in gov.items():
        key = int(key)
        cx, cy = (key // 100_000 + 0.5) * CELL_M, ((key // 100) % 1000 + 0.5) * CELL_M
        h = heads[heads["id"] == g["light_id"]]
        if h.empty:
            continue
        hx, hy = h[["x", "y"]].mean()
        ax.plot([cx, hx], [cy, hy], color=BLUE, lw=0.6, alpha=0.35)
        ax.scatter([cx], [cy], s=6, color=BLUE, zorder=3)
    ax.scatter(heads["x"], heads["y"], s=22, marker="s", color=INK, zorder=4)
    x0, x1 = heads["x"].min() - 70, heads["x"].max() + 70
    y0, y1 = heads["y"].min() - 60, heads["y"].max() + 60
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_title(f"a  Memory learned which light governs each place\n"
                 f"({res['E10_E11']['n_governed_places']} places; black: signal heads)", loc="left")

    sub = res["E10_E11"]["subsets"]
    models = list(MODEL_COLORS)
    ax = axs[1]
    groups = ["all vehicles", "near a light, moving", "near a light, light green"]
    x = np.arange(len(groups))
    w = 0.2
    for i, mname in enumerate(models):
        vals = [100 * sub[g][mname]["top1"] for g in groups]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=MODEL_COLORS[mname], label=mname)
        for xi, vv in zip(x, vals):
            ax.text(xi + (i - 1.5) * w, vv + 1, f"{vv:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["all vehicles", "near a light,\nmoving", "near a light,\nlight green"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("speed class in 3 s predicted correctly (%)")
    ax.set_title("b  Memory and live signal phase add up", loc="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)

    ax = axs[2]
    g = "near a light, moving"
    names = ["constant velocity"] + models
    vals = [sub[g]["cv_lon_error_3s_m"]] + [sub[g][m]["lon_error_3s_m"] for m in models]
    cols = ["#c9c8c3"] + [MODEL_COLORS[m] for m in models]
    ax.barh(np.arange(len(names))[::-1], vals, color=cols)
    for yi, vv in zip(np.arange(len(names))[::-1], vals):
        ax.text(vv + 0.05, yi, f"{vv:.2f} m", va="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(names))[::-1], names)
    ax.set_xlabel("mean error of distance travelled in 3 s, m")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title(f"c  Moving vehicles near a light (n = {sub[g]['n']:,})", loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_conflicts(path: Path, res: dict):
    c = res["E12"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    ax = axs[0]
    _background(ax)
    xy = c["xy"]
    xs, ys, cl = np.array(xy["x"]), np.array(xy["y"]), np.array(xy["cls"])
    for k, color, label in (("micro", BLUE, "cyclist"), ("ped", ORANGE, "pedestrian")):
        m = cl == k
        ax.scatter(xs[m], ys[m], s=10, color=color, alpha=0.7, lw=0, label=f"{label} ({m.sum()})")
    heads = dlr.light_heads_local()
    ax.set_xlim(heads["x"].min() - 70, heads["x"].max() + 70), ax.set_ylim(heads["y"].min() - 60, heads["y"].max() + 60)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"a  Vehicle-VRU conflicts, TTC ≤ 1.5 s, one day (n = {c['n_conflicts']})", loc="left")

    ax = axs[1]
    ax.plot([0, 100], [0, 100], color=GRAY, lw=1.5, ls="--", label="no memory (uniform)")
    ax.plot(100 * np.array(c["curve_area"]), 100 * np.array(c["curve_captured"]), color=BLUE, lw=2,
            label=f"memory of even hours: {100 * c['capture_10pct']:.0f}% in top 10% of area")
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    ax.set_xlabel("share of road area, ranked by memory (%)")
    ax.set_ylabel(f"conflicts of odd hours captured (%, n = {c['n_test']})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("b  Conflict hot spots repeat across hours", loc="left")
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[2]
    h = np.arange(24)
    ax.bar(h, c["per_hour"], color=INK2, width=0.8)
    ax.set_xlabel("hour of the day (UTC, Sunday 24 Sep 2023)")
    ax.set_ylabel("conflicts per hour")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("c  When conflicts happen", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(out: Path, res: dict):
    fig_signal_memory(out / "dlr1_signal_memory.png", res)
    fig_conflicts(out / "dlr2_conflicts.png", res)
