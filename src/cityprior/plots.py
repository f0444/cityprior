"""Figures for the README / paper. Static PNGs, light theme, one message per panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .memory import LocationMemory  # noqa: E402

# reference palette (dataviz skill): categorical slots + one-hue sequential ramps
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, INK, INK2, GRID = "#9b9a96", "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fde3d7", "#f7b394", "#f08a5d", "#eb6834", "#c24e1e", "#8e3510"])
ROAD = LinearSegmentedColormap.from_list("road", [SURFACE, "#e9e8e4", "#d9d8d3"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": SURFACE, "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": False, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def _road(ax, mem: LocationMemory):
    occ = sum(mem.occupancy.values())
    ax.imshow(np.log1p(occ), extent=mem.grid.extent, origin="upper", cmap=ROAD, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _masked(a, mask):
    return np.ma.masked_where(~mask, a)


def fig_memory(path: Path, df: pd.DataFrame, events: dict, mem: LocationMemory):
    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    g = mem.grid

    # (a) vehicle flow field
    ax = axs[0, 0]
    _road(ax, mem)
    cnt = mem.flow_count["veh"]
    best = cnt.argmax(axis=2)
    tot = cnt.max(axis=2)
    vec = np.take_along_axis(mem.flow_vec["veh"], best[..., None, None], axis=2)[..., 0, :]
    iy, ix = np.mgrid[1:g.ny:3, 1:g.nx:3]
    keep = tot[iy, ix] >= 30
    u, v = vec[iy, ix, 0], vec[iy, ix, 1]
    n = np.hypot(u, v) + 1e-9
    sp = np.nan_to_num(mem.veh_speed_p85[iy, ix])
    xs, ys = g.x0 + (ix + 0.5) * g.cell, g.y0 + (iy + 0.5) * g.cell
    q = ax.quiver(xs[keep], ys[keep], (u / n)[keep], (v / n)[keep], sp[keep], cmap=SEQ_BLUE,
                  angles="xy", scale_units="xy", scale=0.45, width=0.0028, clim=(0, 14))
    cb = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("85th-percentile speed, m/s")
    ax.set_title("a  Vehicle flow memory: dominant direction and typical speed")

    # (b) pedestrian emergence
    ax = axs[0, 1]
    _road(ax, mem)
    rate = mem.emergence_rate(sigma_m=1.5)
    im = ax.imshow(_masked(rate, rate > 0.05), extent=g.extent, origin="upper", cmap=SEQ_ORANGE,
                   norm=LogNorm(0.05, rate.max()), interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("pedestrians entering the road, per hour per 10 m²")
    ax.set_title("b  Where pedestrians step onto the road")

    # (c) hard braking: region posterior painted over cells
    ax = axs[1, 0]
    _road(ax, mem)
    rr = mem.region_rate("hard_brake").set_index("region")
    h = df[df["cls"] == "veh"]
    cell = g.flat(h["x"].to_numpy(), h["y"].to_numpy())
    mode = pd.Series(h["region"].to_numpy()).groupby(cell).agg(lambda s: s.value_counts().index[0])
    region_map = np.full(g.size, -1)
    region_map[mode.index[mode.index >= 0]] = mode[mode.index >= 0].to_numpy()
    pm = pd.Series(rr["p_mean"]).reindex(region_map).to_numpy().reshape(g.shape)
    vmask = (mem.occupancy["veh"] > 2) & ~np.isnan(pm)
    im = ax.imshow(_masked(100 * np.nan_to_num(pm), vmask), extent=g.extent, origin="upper",
                   cmap=SEQ_BLUE, interpolation="nearest", vmin=0)
    hb = events["hard_brake"]
    ax.scatter(hb["x"], hb["y"], s=3, c=INK, alpha=0.35, linewidths=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("P(hard brake) per vehicle passage, % (posterior mean)")
    ax.set_title("c  Hard-braking risk per lane segment (dots: events)")

    # (d) conflicts
    ax = axs[1, 1]
    _road(ax, mem)
    c = events["conflict"]
    sc = ax.scatter(c["x"], c["y"], c=c["ttc"], cmap=SEQ_ORANGE.reversed(), s=36, vmin=0, vmax=1.5,
                    edgecolors="white", linewidths=0.8, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("minimum TTC, s")
    ax.set_title(f"d  Vehicle–VRU conflicts, TTC ≤ 1.5 s (n = {len(c)} in 2 h)")

    fig.suptitle("Location memory built from 12 fixed infrastructure cameras (TGSIM Foggy Bottom, Washington DC)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_heldout(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    e_all, e_off = res["emergence"]["all"], res["emergence"]["off_crosswalk"]
    ax.plot([0, 100], [0, 100], color=GRAY, lw=1.5, ls="--", label="no memory (uniform over road)")
    ax.plot(100 * e_all["curve_area"], 100 * e_all["curve_captured"], color=BLUE, lw=2,
            label=f"all entries — {100 * e_all['capture_at_10pct_area']:.0f}% in top 10% of area")
    ax.plot(100 * e_off["curve_area"], 100 * e_off["curve_captured"], color=ORANGE, lw=2,
            label=f"outside crosswalks — {100 * e_off['capture_at_10pct_area']:.0f}% in top 10%")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(10, color=GRID, lw=1)
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    ax.set_xlabel("share of road area, ranked by memory (%)")
    ax.set_ylabel("share of held-out pedestrian entries captured (%)")
    ax.set_title("a  Memory from the first 79 min predicts the last 40 min")
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    lc = pd.DataFrame(res["learning_curve"])
    m = [g[0] for g in lc["gain_bits"]]
    lo = [g[1] for g in lc["gain_bits"]]
    hi = [g[2] for g in lc["gain_bits"]]
    ax.fill_between(lc["minutes"], lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(lc["minutes"], m, color=BLUE, lw=2, marker="o", ms=6)
    for x, y in zip(lc["minutes"], m):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(lc["minutes"], [str(v) for v in lc["minutes"]])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, max(hi) * 1.2)
    ax.set_xlabel("minutes of history in memory (log scale)")
    ax.set_ylabel("information gain vs. no memory (bits / entry)")
    ax.set_title("b  More history → better prior (not saturated at 79 min)")
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_rates(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title in ((axs[0], "hard_brake", "a  Hard braking per vehicle passage"),
                           (axs[1], "ped_conflict", "b  Conflict (TTC ≤ 1.5 s) per pedestrian passage: too rare in 2 h")):
        r = res[key]
        cal = r["calibration"]
        top = max(cal["predicted"].max(), cal["observed"].max()) * 100 * 1.15
        ax.plot([0, top], [0, top], color=GRAY, lw=1.5, ls="--")
        ax.axhline(100 * r["p_global"], color=ORANGE, lw=1.5)
        ax.text(top * 0.02, 100 * r["p_global"] + top * 0.02, "pooled rate (no memory)", color=INK2, fontsize=9)
        ax.plot(100 * cal["predicted"], 100 * cal["observed"], color=BLUE, lw=2, marker="o", ms=7)
        ax.set_xlim(0, top), ax.set_ylim(0, top)
        ax.set_xlabel("memory's predicted probability (%)")
        ax.set_ylabel("observed frequency in held-out 40 min (%)")
        g = r["info_gain_bits_per_passage"]
        ax.text(0.98, 0.04,
                f"passages {r['n_passages']:,}, events {r['n_events']}\n"
                f"AUC {r['auc']:.2f}   Brier skill {r['brier_skill']:+.3f}\n"
                f"info gain {1000 * g[0]:.1f} mbit/passage [{1000 * g[1]:.1f}, {1000 * g[2]:.1f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
        ax.set_title(title)
        ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_prediction(path: Path, res: dict):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axs[0]
    labels, no_mem, with_mem = [], [], []
    for c, name in (("veh", "vehicles"), ("ped", "pedestrians")):
        m = res["manoeuvre"][c]
        labels.append(f"{name}\n(n = {m['n_turning']:,})")
        no_mem.append(100 * m["top1_acc_no_memory_turning"])
        with_mem.append(100 * m["top1_acc_memory_turning"])
    x = np.arange(len(labels))
    ax.bar(x - 0.19, no_mem, 0.36, color=GRAY, label="no memory (area-wide statistics)")
    ax.bar(x + 0.19, with_mem, 0.36, color=BLUE, label="location memory")
    for xi, a, b in zip(x, no_mem, with_mem):
        ax.text(xi - 0.19, a + 1, f"{a:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(xi + 0.19, b + 1, f"{b:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("direction in 3 s guessed correctly (%)")
    ax.set_title("a  Road users that change direction: which way next?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6)

    ax = axs[1]
    groups, cv, mod = [], [], []
    for c, name in (("veh", "vehicles"), ("ped", "pedestrians")):
        v = res["prediction"][c]["tuned_turning"]
        for sub in ("straight", "turning"):
            groups.append(f"{name}\n{sub}")
            cv.append(v[f"fde_3s_{sub}"]["cv"])
            mod.append(v[f"fde_3s_{sub}"]["mod"])
    x = np.arange(len(groups))
    ax.bar(x - 0.19, cv, 0.36, color=GRAY, label="constant velocity")
    ax.bar(x + 0.19, mod, 0.36, color=BLUE, label="CV steered by flow memory")
    for xi, a, b in zip(x, cv, mod):
        ax.text(xi + 0.19, b, f"{100 * (b / a - 1):+.0f}%", ha="center", va="bottom", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_xticks(x, groups)
    ax.set_ylabel("final displacement error at 3 s, m")
    ax.set_title("b  Single 3 s rollout, tuned on validation for curved motion")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_occlusion(path: Path, obs: pd.DataFrame, res: dict):
    from .occlusion import encounter_lead_times
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    bins = np.arange(0, 45, 5)
    obs = obs.assign(dist=pd.cut(obs["lon"], bins))
    for flag, color, label in (("occluded_all", BLUE, "blocked by any vehicle (upper bound)"),
                               ("occluded_tall", ORANGE, "blocked by bus / truck (lower bound)")):
        s = obs.groupby("dist", observed=True)[flag].mean() * 100
        ax.plot(bins[:-1] + 2.5, s.to_numpy(), color=color, lw=2, marker="o", ms=6, label=label)
    ax.set_xlabel("distance ahead of the vehicle, m")
    ax.set_ylabel("pedestrians on / entering the path that are hidden (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("a  Hidden from the car, visible to the infrastructure")
    ax.grid(True, color=GRID, lw=0.6)

    ax = axs[1]
    for flag, color, label in (("occluded_all", BLUE, "any vehicle"), ("occluded_tall", ORANGE, "bus / truck")):
        enc = encounter_lead_times(obs, flag)
        lead = np.sort(enc.loc[enc["lead_s"] > 0, "lead_s"].to_numpy())
        if len(lead):
            ax.plot(lead, 1 - np.arange(len(lead)) / len(lead), color=color, lw=2,
                    label=f"{label} (n = {len(lead):,} encounters)")
    ax.set_xlabel("infrastructure lead time over the car's own line of sight, s")
    ax.set_ylabel("share of hidden-at-first encounters ≥ x")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("b  How much earlier the City Model knows")
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _box(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                        [-length / 2, -width / 2], [-length / 2, width / 2]])
    return np.c_[x + corners[:, 0] * c - corners[:, 1] * s, y + corners[:, 0] * s + corners[:, 1] * c]


def fig_message(path: Path, df: pd.DataFrame, mem: LocationMemory, msg: dict, ego: pd.Series,
                obs: pd.DataFrame, res: dict):
    frame = int(ego["frame"])
    now = df[df["frame"] == frame]
    fig, ax = plt.subplots(figsize=(11, 9))
    _road(ax, mem)
    rate = mem.emergence_rate()
    layer = np.where(msg["mask"], rate, 0)
    ax.imshow(_masked(layer, layer > 0.05), extent=mem.grid.extent, origin="upper", cmap=SEQ_ORANGE,
              norm=LogNorm(0.05, max(rate.max(), 0.1)), interpolation="bilinear", alpha=0.9)
    # corridor: 80 m ahead, 10 m behind, +-12 m (see message.corridor_mask)
    c, s_ = np.cos(ego["heading"]), np.sin(ego["heading"])
    corners = [(-10, -12), (80, -12), (80, 12), (-10, 12)]
    ax.add_patch(Polygon([(ego["x"] + a * c - b * s_, ego["y"] + a * s_ + b * c) for a, b in corners],
                         closed=True, fill=False, edgecolor=INK2, lw=1.3, ls="--", zorder=3))

    for _, v in now[now["cls"] == "veh"].iterrows():
        is_ego = v["track"] == ego["track"]
        ax.add_patch(Polygon(_box(v["x"], v["y"], v["heading"] if v["speed"] > 0.5 else _lane_heading(df, v),
                                  v["length"], v["width"]),
                             closed=True, facecolor=BLUE if is_ego else "#bdbcb6",
                             edgecolor=INK if is_ego else "#8f8e89", lw=1.2, zorder=4))
    hid = obs[(obs["frame"] == frame) & (obs["track_e"] == ego["track"]) & obs["occluded_all"]]["track_p"]
    peds = now[now["cls"].isin(["ped", "micro"])]
    sx = ego["x"] + np.cos(ego["heading"]) * ego["length"] / 4
    sy = ego["y"] + np.sin(ego["heading"]) * ego["length"] / 4
    for _, p in peds.iterrows():
        hidden = p["track"] in set(hid)
        if hidden:
            ax.plot([sx, p["x"]], [sy, p["y"]], color=RED, lw=1, ls=":", zorder=3)
        ax.scatter(p["x"], p["y"], s=110 if hidden else 30, color=RED if hidden else INK,
                   edgecolors="white", linewidths=1.2, zorder=5)
        if np.hypot(p["vx"], p["vy"]) > 0.3:
            ax.arrow(p["x"], p["y"], p["vx"] * 2, p["vy"] * 2, color=RED if hidden else INK,
                     width=0.12, head_width=0.8, length_includes_head=True, zorder=5)

    pad = 45
    ax.set_xlim(ego["x"] - pad, ego["x"] + pad)
    ax.set_ylim(ego["y"] + pad, ego["y"] - pad)
    ref = res["message"]["reference"]
    txt = (f"message for this car, t = {ego['t']:.1f} s\n"
           f"• static memory tile ({msg['tile_shape'][0]}×{msg['tile_shape'][1]} m, "
           f"{len(msg['tile_layers'])} layers): {msg['tile_bytes_zlib'] / 1024:.1f} KB once, cached\n"
           f"• live objects in corridor: {msg['n_objects']} ({msg['n_vru']} VRU) → "
           f"{msg['live_kbit_s_at_10hz']:.0f} kbit/s at 10 Hz\n"
           f"• vs raw video of 12 cameras ≈ {ref['raw_video_all_cameras_mbit_s']:.0f} Mbit/s\n"
           f"• hidden from this car right now: {len(set(hid))} pedestrian(s) (red)")
    ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=9.5, va="bottom", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID))
    who = "the automated test vehicle" if res["message"].get("ego_is_automated_test_vehicle") else "a vehicle"
    ax.set_title(f"What the City Model sends {who} (blue)\n"
                 "corridor ahead (dashed) · pedestrian-entry prior from memory (orange) · live objects",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _lane_heading(df: pd.DataFrame, v: pd.Series) -> float:
    """Parked / queued vehicles have no velocity heading: use the track's mean heading."""
    tr = df[(df["track"] == v["track"]) & (df["speed"] > 0.5)]
    if len(tr):
        return float(np.arctan2(np.sin(tr["heading"]).mean(), np.cos(tr["heading"]).mean()))
    return float(np.arctan2(v["width"], v["length"]) * 0)


def make_all(out: Path, df, events, mem, res, pred_store, obs, msg, ego):
    fig_memory(out / "fig1_location_memory.png", df, events, mem)
    fig_heldout(out / "fig2_emergence_heldout.png", res)
    fig_rates(out / "fig3_region_rates_calibration.png", res)
    fig_prediction(out / "fig4_prediction.png", res)
    fig_occlusion(out / "fig5_occlusion_virtual_ego.png", obs, res)
    fig_message(out / "fig6_robotaxi_message.png", df, mem, msg, ego, obs, res)
