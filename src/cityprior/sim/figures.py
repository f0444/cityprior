"""Figures for the closed-loop experiments (same visual language as ../plots.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from ..plots import AQUA, BLUE, GRAY, GRID, INK, INK2, ORANGE, RED  # noqa: E402  (also sets rcParams)

KMH = 3.6


def _curve(ax, curve, key, color, label, ls="-", marker="o"):
    t = [r["trip_time"] for r in curve]
    y = [r[key] for r in curve]
    ax.plot(t, y, color=color, lw=2, ls=ls, marker=marker, ms=5, label=label)
    if key == "collisions_per_10k":
        lo = [r["collisions_per_10k_ci"][0] for r in curve]
        hi = [r["collisions_per_10k_ci"][1] for r in curve]
        ax.fill_between(t, lo, hi, color=color, alpha=0.12, linewidth=0)


def fig_pareto(path: Path, res: dict):
    e1 = res["E1"]
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.1], height_ratios=[1, 1.4], hspace=0.08, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    _curve(ax, e1["no_memory"], "collisions_per_10k", GRAY, "no memory (knows the street's average only)")
    _curve(ax, e1["memory"], "collisions_per_10k", BLUE, f"location memory ({res['world']['memory_hours']:.0f} h of history)")
    _curve(ax, e1["memory_adds_caution_only"], "collisions_per_10k", BLUE, "memory may only add caution",
           ls=":", marker="s")
    _curve(ax, e1["oracle"], "collisions_per_10k", AQUA, "perfect memory (upper bound)", ls="--", marker=None)
    wc = e1["worst_case"][0]
    ax.scatter([wc["trip_time"]], [wc["collisions_per_10k"]], color=INK, s=50, zorder=5)
    ax.annotate(f"worst-case occlusion planner\n({res['world']['worst_case_speed'] * KMH:.0f} km/h past parked cars)",
                (wc["trip_time"], wc["collisions_per_10k"]), xytext=(-10, 30), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    s = e1["summary"]
    ref = s["reference"]
    ax.scatter([ref["trip_time"]], [ref["collisions_per_10k"]], s=140, facecolors="none", edgecolors=INK, lw=1.5, zorder=6)
    if np.isfinite(s["memory_time_at_reference_risk"]):
        ax.annotate("", xy=(s["memory_time_at_reference_risk"], ref["collisions_per_10k"]),
                    xytext=(ref["trip_time"], ref["collisions_per_10k"]),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        ax.text(ref["trip_time"] + 0.4, ref["collisions_per_10k"] * 1.04,
                f"same risk: {ref['trip_time'] - s['memory_time_at_reference_risk']:.1f} s faster", fontsize=9, color=INK)
    ax.annotate("", xy=(ref["trip_time"], s["memory_risk_at_reference_time"]),
                xytext=(ref["trip_time"], ref["collisions_per_10k"]),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    red, lo, hi = s["memory_reduction_at_reference_time_ci"]
    ax.text(ref["trip_time"] + 0.4, s["memory_risk_at_reference_time"] * 0.9,
            f"same time: {100 * red:.0f}% fewer collisions\n(95% CI {100 * lo:.0f}…{100 * hi:.0f}%)",
            fontsize=9, color=INK, va="top")
    ax.set_xlabel("expected time to drive the block, s")
    ax.set_ylabel("collisions with pedestrians per 10,000 traversals")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("a  Safety vs trip time (each dot: one caution level)")
    ax.grid(True, color=GRID, lw=0.6)

    ax = fig.add_subplot(gs[:, 1])
    _curve(ax, e1["no_memory"], "near_misses_per_10k", GRAY, "no memory")
    _curve(ax, e1["memory"], "near_misses_per_10k", BLUE, "location memory")
    _curve(ax, e1["oracle"], "near_misses_per_10k", AQUA, "perfect memory", ls="--", marker=None)
    ax.set_xlabel("expected time to drive the block, s")
    ax.set_ylabel("near misses (braking ≥ 4 m/s² needed) per 10,000")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("b  Near misses")
    ax.grid(True, color=GRID, lw=0.6)

    pr = res["profiles"]
    x = np.array(pr["x"])
    street = (x >= -10) & (x <= 100)
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.fill_between(x[street], 0, 3600 * np.array(pr["lam_true"])[street], color=ORANGE, alpha=0.35, linewidth=0,
                     label="true (from TGSIM, 99 real events)")
    ax1.plot(x[street], 3600 * np.array(pr["lam_memory"])[street], color=ORANGE, lw=1.8, label="memory estimate")
    ax1.set_ylabel("pedestrians / m / h")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("c  Where people step out → how fast to drive")
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, color=GRID, lw=0.6)
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    ax2.plot(x[street], KMH * np.array(pr["v_no_memory"])[street], color=GRAY, lw=2, label="no memory")
    ax2.plot(x[street], KMH * np.array(pr["v_memory"])[street], color=BLUE, lw=2, label="location memory")
    ax2.plot(x[street], KMH * np.array(pr["v_worst"])[street], color=INK, lw=1.2, ls=":", label="worst case")
    ax2.axvspan(0, 90, color=GRID, alpha=0.35, linewidth=0)
    ax2.set_ylabel("target speed, km/h")
    ax2.set_xlabel("position along the block, m (parked cars in 0–90 m)")
    ax2.set_ylim(0, 45)
    ax2.legend(loc="lower left", fontsize=8, ncol=3)
    ax2.grid(True, color=GRID, lw=0.6)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_coverage_history(path: Path, res: dict):
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axs[0]
    e2 = res["E2"]
    cov = [100 * r["coverage"] for r in e2]
    ax.plot(cov, [r["reference_risk"] for r in e2], color=GRAY, lw=2, marker="o", ms=6, label="live cameras only")
    ax.plot(cov, [r["memory_risk_at_reference_time"] for r in e2], color=BLUE, lw=2, marker="o", ms=6,
            label="live cameras + location memory")
    for c, a, b in zip(cov, [r["reference_risk"] for r in e2], [r["memory_risk_at_reference_time"] for r in e2]):
        if a > 0.1:
            ax.text(c, b, f"{100 * (b / a - 1):+.0f}%", ha="center", va="top", fontsize=8.5, color=INK,
                    transform=ax.transData, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xlabel("share of the block seen by infrastructure cameras (%)")
    ax.set_ylabel("collisions per 10,000 at equal trip time")
    ax.set_ylim(bottom=0)
    ax.set_title("a  Memory matters most where cameras don't see")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    e3 = res["E3"]
    h = [r["hours"] for r in e3]
    m = [np.mean(r["risk_at_reference_time"]) for r in e3]
    lo = [np.min(r["risk_at_reference_time"]) for r in e3]
    hi = [np.max(r["risk_at_reference_time"]) for r in e3]
    ref = res["E1"]["summary"]["reference"]["collisions_per_10k"]
    orc = res["E1"]["summary"]["oracle_risk_at_reference_time"]
    ax.axhline(ref, color=GRAY, lw=1.5, ls="--")
    ax.text(h[0], ref, " no memory", va="bottom", fontsize=9, color=INK2)
    ax.axhline(orc, color=AQUA, lw=1.5, ls="--")
    ax.text(h[0], orc, " perfect memory", va="bottom", fontsize=9, color=INK2)
    ax.fill_between(h, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(h, m, color=BLUE, lw=2, marker="o", ms=6, label="location memory (3 random histories)")
    ax.set_xscale("log")
    ax.set_xticks(h, [f"{v:g}" for v in h])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("hours of history in memory (log scale)")
    ax.set_ylabel("collisions per 10,000 at equal trip time")
    ax.set_ylim(bottom=0)
    ax.set_title("b  How much history is enough?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[2]
    e6 = res["E6"]
    het = np.array([100 * (1 - r["ceiling"]) for r in e6])
    top = het.max() * 1.1
    for key, color, label in (("oracle", AQUA, "perfect memory"), ("memory", BLUE, "10 h of memory")):
        y = np.array([100 * r[f"{key}_reduction_ci"][0] for r in e6])
        err = np.array([[100 * (r[f"{key}_reduction_ci"][0] - r[f"{key}_reduction_ci"][1]),
                         100 * (r[f"{key}_reduction_ci"][2] - r[f"{key}_reduction_ci"][0])] for r in e6]).T
        ax.errorbar(het, y, yerr=err, color=color, lw=2, marker="o", ms=6, capsize=3, label=label)
    real = next(r for r in e6 if r["kappa"] == 1.0)
    ax.annotate("this street\n(TGSIM)", (100 * (1 - real["ceiling"]), 100 * real["memory_reduction_ci"][0]),
                xytext=(25, -30), textcoords="offset points", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set_xlabel("heterogeneity of pedestrian activity along the block: 1 − (E√λ)² / Eλ  (%)")
    ax.set_ylabel("fewer collisions at equal trip time (%)")
    ax.set_xlim(-1, top)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_title("c  Memory pays off where activity is concentrated")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bars(ax, items, title):
    names = [n for n, _, _ in items]
    vals = [r["collisions_per_10k"] for _, r, _ in items]
    cols = [c for _, _, c in items]
    x = np.arange(len(items))
    ax.bar(x, vals, 0.62, color=cols)
    for xi, (_, r, _) in zip(x, items):
        ax.text(xi, r["collisions_per_10k"], f"{r['collisions_per_10k']:.2f}\n{r['trip_time']:.1f} s",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("collisions per 10,000 traversals")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title(title)
    ax.grid(True, axis="y", color=GRID, lw=0.6)


def fig_robustness(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    e4 = res["E4"]
    _bars(axs[0], [
        ("no memory", e4["no_memory"], GRAY),
        ("stale memory", e4["stale_memory_floor_0.1"], BLUE),
        ("stale, caution\nfloor 60%", e4["stale_memory_floor_0.6"], BLUE),
        ("stale, may only\nadd caution", e4["stale_memory_adds_caution_only"], BLUE),
        ("stale + camera\non 50% of block", e4["stale_memory_plus_camera_50pct"], ORANGE),
        ("fresh memory", e4["fresh_memory"], AQUA),
        ("fresh, may only\nadd caution", e4["fresh_memory_adds_caution_only"], AQUA),
    ], f"a  Hotspot moved {abs(res.get('stale_shift_m', 0)):.0f} m to the quietest spot; memory did not")
    e5 = res["E5"]
    _bars(axs[1], [
        ("camera up", e5["camera_up"], ORANGE),
        ("camera down,\ncar not told", e5["silent_outage"], RED),
        ("camera down,\nheartbeat → memory", e5["outage_with_heartbeat"], BLUE),
    ], "b  Camera outage: trust must expire")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(out: Path, res: dict):
    out.mkdir(parents=True, exist_ok=True)
    fig_pareto(out / "sim1_pareto.png", res)
    fig_coverage_history(out / "sim2_coverage_history.png", res)
    fig_robustness(out / "sim3_robustness.png", res)


STRATEGY_STYLE = {
    "static · camera": ("#4a3aa7", "--", "never updated"),
    "cumulative · camera": ("#eda100", "-", "adds all observations"),
    "forgetting · camera": ("#e87ba4", "-", "forgets (half-life 4 h)"),
    "detect · camera": (BLUE, "-", "detects change + resets (camera)"),
    "detect · fleet 10%": (ORANGE, "-", "detects change + resets (fleet 10%)"),
}


def _time_average(hours, values, until: float = 24.0) -> float:
    h, v = np.asarray(hours, float), np.asarray(values, float)
    keep = h <= until
    return float(np.trapezoid(v[keep], h[keep]) / until)


def fig_change(path: Path, res: dict):
    hours = np.array(res["eval_hours"])
    xplot = np.where(hours == 0, 0.125, hours)
    ref = res["references"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.25, 1.1, 1.0]})

    ax = axs[0]
    ax.axhline(ref["no_memory"]["collisions_per_10k"], color=GRAY, lw=1.5, ls="--")
    ax.axhline(ref["perfect_memory"]["collisions_per_10k"], color=AQUA, lw=1.5, ls="--")
    ax.text(0.13, ref["no_memory"]["collisions_per_10k"], "no memory", va="bottom", fontsize=9, color=INK2)
    ax.text(0.13, ref["perfect_memory"]["collisions_per_10k"], "fresh memory", va="top", fontsize=9, color=INK2)
    for key, runs in res["curves"].items():
        color, ls, label = STRATEGY_STYLE[key]
        y = np.mean([[r["collisions_per_10k"] for r in run] for run in runs], axis=0)
        ax.plot(xplot, y, color=color, ls=ls, lw=2, marker="o", ms=4, label=label)
    ax.set_xscale("log")
    ticks = [0.125, 0.5, 1, 2, 4, 8, 24, 48]
    ax.set_xticks(ticks, ["0", "0.5", "1", "2", "4", "8", "24", "48"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("hours since the hotspot moved (log scale)")
    ax.set_ylabel("collisions per 10,000 traversals")
    ax.set_title("a  Recovery after the hotspot moved")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    keys = list(res["curves"])
    after = [np.mean([_time_average(hours, [r["collisions_per_10k"] for r in run]) for run in res["curves"][k]])
             for k in keys]
    steady = [np.mean([v["collisions_per_10k"] for v in res["steady_state_no_change"][k]]) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.19, after, 0.36, color=INK2, label="first 24 h after a change")
    ax.bar(x + 0.19, steady, 0.36, color="#c9c8c3", label="no change (price of vigilance)")
    for xi, a, b in zip(x, after, steady):
        ax.text(xi - 0.19, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.text(xi + 0.19, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(ref["no_memory"]["collisions_per_10k"], color=GRAY, lw=1.2, ls="--")
    short = {"static · camera": "never\nupdated", "cumulative · camera": "adds all\nobservations",
             "forgetting · camera": "forgets\n(half-life 4 h)", "detect · camera": "detect + reset\n(camera)",
             "detect · fleet 10%": "detect + reset\n(fleet 10%)"}
    ax.set_xticks(x, [short[k] for k in keys], fontsize=8.5)
    ax.set_ylabel("collisions per 10,000 traversals")
    ax.set_ylim(0, max(after + steady) * 1.25)
    ax.set_title("b  Cost after a change vs cost when nothing changes")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, axis="y", color=GRID, lw=0.6)

    ax = axs[2]
    colors = {"camera": BLUE, "fleet 10%": ORANGE, "fleet 3%": "#f7b394"}
    for name, st in res["detector"].items():
        d = np.sort(np.array([v if v is not None else np.inf for v in st["delays"]]))
        finite = d[np.isfinite(d)]
        yv = np.arange(1, len(finite) + 1) / len(d)
        ax.step(np.r_[0, finite], np.r_[0, yv], where="post", color=colors[name], lw=2,
                label=(f"{name}: median {st['median_delay_h']:.2f} h" if st["median_delay_h"] is not None
                       else f"{name}: {100 * st['share_detected_24h']:.0f}% found in 24 h")
                + f", {st['false_alarms_per_30_days']:.2f} false alarms / 30 days")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("hours until the change is detected")
    ax.set_ylabel("share of changes detected")
    ax.set_title("c  Detection delay (200 simulated changes each)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
