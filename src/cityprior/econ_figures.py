"""Figures for Part 6 (cost-benefit)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .plots import AQUA, BLUE, GRAY, GRID, INK, ORANGE, RED  # noqa: E402

VIOLET = "#4a3aa7"
COLORS = {"cctv memory": AQUA, "new memory": GRAY, "live trusted": VIOLET, "live guaranteed": ORANGE}
LABELS = {"none": "nothing (fleet memory only)", "cctv memory": "memory from existing CCTV",
          "new memory": "new cameras, memory only", "live trusted": "live camera, trusted",
          "live guaranteed": "live camera, never worse than none"}
PNAMES = {"discount_rate": "discount rate", "horizon_years": "service life", "vtts": "value of riders' time", "free_flow_share": "free-flow share",
          "riders_per_traversal": "riders per traversal", "vehicle_hour": "value of a vehicle-hour",
          "cost_injured": "cost of an injury", "fatal_share": "fatal share", "ped_crash_rate": "crash rate",
          "site_exposure": "site exposure", "addressable_share": "addressable share",
          "cameras_per_site": "cameras per site", "camera_unit": "camera unit cost", "site_works": "site works",
          "rsu": "roadside unit", "edge_compute": "edge compute", "integration": "integration",
          "analytics_per_camera": "analytics per camera", "maintenance_share": "maintenance share",
          "changes_per_year": "changes per year", "operators": "operators", "active_hours": "active hours"}


def fig_value(path: Path, res: dict):
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [0.9, 1.1, 1.0]})

    ax = axs[0]
    pt = res["per_traversal"]
    u = 1 - np.array(pt["c"])
    ax.plot(u, pt["live_seconds"], color=VIOLET, lw=2, label="live camera + memory")
    ax.plot(u, pt["memory_seconds"], color=BLUE, lw=2, label="memory only")
    ax.axvline(1 - res["c_tgsim"], color=GRAY, lw=1, ls="--")
    ax.text(1 - res["c_tgsim"] + 0.01, 0.2, "this street", fontsize=8.5, color=INK)
    ax.set_xlabel("unevenness of pedestrian activity (1 - ceiling)")
    ax.set_ylabel("seconds saved per traversal at equal risk")
    sec = ax.secondary_yaxis("right", functions=(lambda s: 100 * s * res["value_per_second"],
                                                 lambda v: v / (100 * res["value_per_second"])))
    sec.set_ylabel("cents per traversal")
    ax.set_title("a  What one traversal of the block gains", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    n = np.array(res["npv_curves"]["n_day"])
    for o in ("live trusted", "live guaranteed", "new memory", "cctv memory"):
        v = np.array(res["npv_curves"][o]) / 1000
        ax.plot(n, v, color=COLORS[o], lw=2, label=LABELS[o])
        be = res["break_even_base"][o]
        if np.isfinite(be) and be < n.max():
            ax.plot([be], [0], "o", color=COLORS[o], ms=6)
            ax.annotate(f"{be:,.0f} / day", (be, 0), xytext=(-8, 10), textcoords="offset points", ha="right",
                        fontsize=8.5, color=COLORS[o])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-120, 400)
    ax.set_xlabel("robotaxi traversals of the block per day")
    ax.set_ylabel("net present value over 10 years (thousand $)")
    ax.set_title("b  Only live perception pays, at high volume", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[2]
    mi = res["memory_increment"]
    n = np.array(mi["n_day"])
    ax.plot(n, mi["memory_full_$"], color=BLUE, lw=2, label="value of memory (if fleets had none)")
    ax.plot(n, mi["operators_3"], color=AQUA, lw=2, ls="--", label="camera memory beyond fleet memory, 3 fleets")
    ax.plot(n, mi["operators_1"], color=AQUA, lw=2, label="camera memory beyond fleet memory, 1 fleet")
    ax.axhline(mi["cctv_opex"], color=RED, lw=1.5, ls=":")
    ax.text(n[1], mi["cctv_opex"] * 1.15, "running cost of CCTV analytics", fontsize=8.5, color=RED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    ax.set_xlabel("robotaxi traversals of the block per day")
    ax.set_ylabel("$ per year")
    ax.set_title("c  Memory is worth having, not buying cameras for", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_decision(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axs[0]
    dm = res["decision_map"]
    order = ["none", "cctv memory", "new memory", "live guaranteed", "live trusted"]
    cols = ["#efeeea", AQUA, GRAY, ORANGE, VIOLET]
    z = np.array([[order.index(b) for b in row] for row in dm["best"]])
    u = 1 - np.array(dm["c"])
    n = np.array(dm["n_day"])
    ax.pcolormesh(n, u, z, cmap=ListedColormap(cols), vmin=-0.5, vmax=len(order) - 0.5, shading="nearest")
    be = res["break_even_by_c"]
    ax.plot(be["live trusted"], 1 - np.array(be["c"]), color=INK, lw=1.2)
    ax.axhline(1 - res["c_tgsim"], color=INK, lw=0.8, ls="--")
    ax.text(n[1], 1 - res["c_tgsim"] + 0.01, "this street", fontsize=8.5, color=INK)
    present = sorted(set(z.ravel()))
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[i]) for i in present]
    ax.legend(handles, [LABELS[order[i]] for i in present], loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("robotaxi traversals of the block per day")
    ax.set_ylabel("unevenness of pedestrian activity (1 - ceiling)")
    ax.set_title("a  Best option per block face (base values)", loc="left")

    ax = axs[1]
    t = res["tornado_live_trusted"]
    base = t[0]["base"]
    rows = [r for r in t[1:] if np.isfinite(r["swing"]) and r["swing"] > 0.02][:10][::-1]
    for i, r in enumerate(rows):
        lo, hi = r["break_even_low"], r["break_even_high"]
        ax.barh(i, np.log10(lo) - np.log10(base), left=np.log10(base), color=BLUE, height=0.6)
        ax.barh(i, np.log10(hi) - np.log10(base), left=np.log10(base), color=ORANGE, height=0.6)
    ax.axvline(np.log10(base), color=INK, lw=1)
    ax.set_yticks(range(len(rows)), [PNAMES.get(r["param"], r["param"]) for r in rows], fontsize=9)
    ticks = [300, 1000, 3000, 10000, 30000]
    ax.set_xticks(np.log10(ticks), [f"{x:,}" for x in ticks])
    ax.set_xlabel("break-even robotaxi traversals per day (live camera, trusted)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              ["parameter at its low value", "parameter at its high value"], loc="lower right", fontsize=9)
    ax.set_title(f"b  What moves the break-even (base {base:,.0f} / day)", loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(out: Path, res: dict):
    out.mkdir(parents=True, exist_ok=True)
    fig_value(out / "econ1_value.png", res)
    fig_decision(out / "econ2_decision.png", res)
