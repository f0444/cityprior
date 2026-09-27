"""Figures for the fleet-vs-infrastructure comparison. Blue = fixed camera, orange = fleet."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from .plots import AQUA, BLUE, GRAY, GRID, INK, INK2, ORANGE  # noqa: E402

ORANGE_STEPS = {"fleet 100%": "#8e3510", "fleet 10%": "#eb6834", "fleet 3%": "#f7b394"}


def _pen_axis(ax):
    ax.set_xscale("log")
    ticks = [3, 10, 30, 100]
    ax.set_xticks(ticks, [f"{t}%" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("share of cars that collect data (fleet penetration)")


def _stat(rows, f):
    v = np.array([f(r) for r in rows])
    return v.mean(), v.min(), v.max()


def fig_real(path: Path, res: dict):
    real = res["real"]
    fleet = real["fleet"]
    pens = np.array([100 * f["penetration"] for f in fleet])
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axs[0]
    for key, ls, marker, label in (("share_seen", "-", "o", "ever seen by the fleet"),
                                   ("share_seen_off_crosswalk", "--", "s", "…outside crosswalks"),
                                   ("share_seen_within_2s", ":", "^", "seen within 2 s of stepping out")):
        st = np.array([_stat(f["draws"], lambda r: 100 * r[key]) for f in fleet])
        ax.errorbar(pens, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], color=ORANGE, ls=ls,
                    marker=marker, ms=6, lw=2, capsize=3, label=label)
    ax.axhline(100, color=BLUE, lw=1.5)
    ax.text(3.1, 101.5, "fixed cameras: every pedestrian", color=INK2, fontsize=9)
    _pen_axis(ax)
    ax.set_ylim(0, 108)
    ax.set_ylabel("pedestrians stepping onto the road (%)")
    ax.set_title("a  What a fleet gets to see")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    for subset, ls, label in (("all", "-", "all entries"), ("off_crosswalk", "--", "outside crosswalks")):
        st = np.array([_stat(f["draws"], lambda r: r[subset]["gain"]) for f in fleet])
        ax.errorbar(pens, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], color=ORANGE, ls=ls,
                    marker="o", ms=6, lw=2, capsize=3, label=f"fleet memory, {label}")
        ax.axhline(real["infrastructure"][subset]["gain"], color=BLUE, ls=ls, lw=1.8,
                   label=f"camera memory, {label}")
    _pen_axis(ax)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.set_ylabel("held-out information gain (bits / entry)")
    ax.set_title("b  How well the memory predicts the last 40 min")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[2]
    lc = real["learning_curve"]
    m = [r["minutes"] for r in lc]
    ax.plot(m, [r["infrastructure"]["gain"] for r in lc], color=BLUE, lw=2, marker="o", ms=6, label="fixed cameras")
    ax.plot(m, [r["fleet_1.0"] for r in lc], color=ORANGE_STEPS["fleet 100%"], lw=2, marker="o", ms=6,
            label="fleet, every car")
    ax.plot(m, [r["fleet_0.1"] for r in lc], color=ORANGE_STEPS["fleet 10%"], lw=2, marker="s", ms=6, ls="--",
            label="fleet, 10% of cars")
    ax.set_xscale("log")
    ax.set_xticks(m, [str(v) for v in m])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("minutes of history (log scale)")
    ax.set_ylabel("held-out information gain (bits / entry)")
    ax.set_title("c  Learning speed")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sim(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    obs = res["observability"]
    xs = np.array(next(iter(obs.values()))["x"])
    ax.plot(xs, np.full(len(xs), 95.0), color=BLUE, lw=2, label="fixed camera (assumed)")
    for name, v in obs.items():
        ax.plot(v["x"], 100 * np.array(v["q"]), color=ORANGE_STEPS[name], lw=2, label=name)
    ax.set_ylim(0, 105)
    ax.set_xlabel("position along the block, m")
    ax.set_ylabel("share of time the kerb is watched (%)")
    ax.set_title("a  Observability of the block (from real traffic)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    sim = res["sim"]
    ax.axhline(sim["reference_risk"], color=GRAY, lw=1.5, ls="--")
    ax.axhline(sim["oracle_risk"], color=AQUA, lw=1.5, ls="--")
    for name, src in sim["sources"].items():
        h = [r["hours"] for r in src["rows"]]
        y = [np.mean(r["risk"]) for r in src["rows"]]
        if name.startswith("infrastructure"):
            color, ls = BLUE, "-"
        else:
            color = ORANGE_STEPS[name.split(" (")[0]]
            ls = "--" if "naive" in name else "-"
        ax.plot(h, y, color=color, lw=2, marker="o", ms=5, ls=ls, label=name)
    ax.text(1.05, sim["reference_risk"], "no memory", va="bottom", fontsize=9, color=INK2)
    ax.text(1.05, sim["oracle_risk"], "perfect memory", va="top", fontsize=9, color=INK2)
    ax.set_xscale("log")
    h = [r["hours"] for r in next(iter(sim["sources"].values()))["rows"]]
    ax.set_xticks(h, [f"{v:g}" for v in h])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("calendar hours of history (log scale)")
    ax.set_ylabel("collisions per 10,000 at equal trip time")
    ax.set_title("b  Robotaxi safety with memory from cameras vs fleet")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(out: Path, res: dict):
    out.mkdir(parents=True, exist_ok=True)
    fig_real(out / "fleet1_real_data.png", res)
    fig_sim(out / "fleet2_simulator.png", res)
