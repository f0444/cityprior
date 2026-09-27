"""Held-out evaluation: does memory of the past predict what happens later?

Every score compares the location memory with the *location-agnostic* prior that a
vehicle arriving for the first time would have to use (uniform over the road
area, or the pooled city-wide rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .memory import MOVING_SPEED, N_BINS, LocationMemory, heading_bin, transition_pairs

def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 1000, q=(2.5, 97.5), seed: int = 0):
    """Point estimate and percentile bootstrap interval (deterministic per call)."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    lo, hi = np.percentile(boots, q)
    return float(stat(values)), float(lo), float(hi)


# ------------------------------------------------------------- emergence
def emergence_eval(mem: LocationMemory, test_events: pd.DataFrame, sigma_m: float = 1.5,
                   alpha: float = 0.05) -> dict:
    """Where will pedestrians step onto the road?

    * info gain: mean log2 p_memory(x) - log2 p_uniform(x) over held-out
      emergence locations (bits per event; 0 = memory is useless);
    * capture curve: share of held-out emergences inside the top-k% of road area
      ranked by memory density (like crime hot-spot evaluation).
    """
    dens = mem.emergence_density(sigma_m, alpha)
    sup = mem.support
    n_sup = sup.sum()
    idx = mem.grid.flat(test_events["x"].to_numpy(), test_events["y"].to_numpy())
    flat_d, flat_s = dens.ravel(), sup.ravel()
    inside = (idx >= 0) & flat_s[np.clip(idx, 0, None)]
    # both densities live on the road area seen in memory; events outside it
    # (new places) count as misses in the capture curve
    gain = np.log2(flat_d[idx[inside]]) - np.log2(1.0 / n_sup)

    order = np.argsort(-flat_d[flat_s])
    sup_cells = np.flatnonzero(flat_s)[order]
    rank = np.full(mem.grid.size, len(sup_cells))
    rank[sup_cells] = np.arange(len(sup_cells))
    ev_rank = np.where(inside, rank[np.clip(idx, 0, None)], len(sup_cells))
    area = np.linspace(0, 1, 201)
    captured = np.array([(ev_rank < a * len(sup_cells)).mean() for a in area])
    return {
        "n_test_events": int(len(idx)),
        "share_outside_memory_area": float(1 - inside.mean()),
        "info_gain_bits": bootstrap_ci(gain),
        "capture_at_5pct_area": float(captured[10]),
        "capture_at_10pct_area": float(captured[20]),
        "capture_at_20pct_area": float(captured[40]),
        "curve_area": area,
        "curve_captured": captured,
    }


# ------------------------------------------------------- per-passage rates
def passage_outcomes(df: pd.DataFrame, events: pd.DataFrame, cls: str, event_track_col: str,
                     t0: float, t1: float) -> pd.DataFrame:
    """One row per (track, region) passage in [t0, t1) with y=1 if the event
    happened to that track in that region."""
    h = df[(df["t"] >= t0) & (df["t"] < t1) & (df["cls"] == cls)]
    pas = h[["track", "region"]].drop_duplicates()
    e = events[(events["t"] >= t0) & (events["t"] < t1)]
    hit = set(zip(e[event_track_col], e["region"]))
    pas["y"] = [int((t, r) in hit) for t, r in zip(pas["track"], pas["region"])]
    return pas.reset_index(drop=True)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties get half credit)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(scores)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rate_eval(mem: LocationMemory, what: str, outcomes: pd.DataFrame) -> dict:
    """Probability that a passage through a region produces the event:
    region posterior from memory vs pooled rate. Brier skill > 0 and info gain
    > 0 mean the memory knows something a newcomer does not."""
    rr = mem.region_rate(what).set_index("region")
    p_glob = rr.attrs["p_global"]
    p_loc = outcomes["region"].map(rr["p_mean"]).fillna(p_glob).to_numpy()
    y = outcomes["y"].to_numpy()
    eps = 1e-6
    ll = lambda p: -(y * np.log2(np.clip(p, eps, 1)) + (1 - y) * np.log2(np.clip(1 - p, eps, 1)))
    brier_loc, brier_glob = np.mean((p_loc - y) ** 2), np.mean((p_glob - y) ** 2)
    gain = ll(np.full_like(p_loc, p_glob)) - ll(p_loc)
    bins = np.quantile(p_loc, np.linspace(0, 1, 6))
    b = np.clip(np.searchsorted(bins, p_loc, side="right") - 1, 0, 4)
    calib = pd.DataFrame({"p": p_loc, "y": y, "bin": b}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return {
        "n_passages": int(len(y)), "n_events": int(y.sum()),
        "p_global": float(p_glob), "prior_strength": float(rr.attrs["prior_strength"]),
        "brier_skill": float(1 - brier_loc / brier_glob),
        "info_gain_bits_per_passage": bootstrap_ci(gain),
        "auc": auc(p_loc, y),
        "calibration": calib,
    }


# ------------------------------------------------------------- manoeuvres
def manoeuvre_eval(mem: LocationMemory, df_test: pd.DataFrame, cls: str, alpha: float = 2.0) -> dict:
    """Which way will a road user be heading 3 s from now?

    Memory: heading-transition counts of this cell, shrunk towards the pooled
    transitions with `alpha` pseudo-observations. No memory: pooled transitions
    over the whole area (what a newcomer knows about traffic in general)."""
    tp = transition_pairs(df_test[df_test["cls"] == cls], MOVING_SPEED[cls])
    iy, ix, ok = mem.grid.ij(tp["x"].to_numpy(), tp["y"].to_numpy())
    tp, iy, ix = tp[ok], iy[ok], ix[ok]
    b0 = heading_bin(tp["heading"].to_numpy(), N_BINS)
    b1 = heading_bin(tp["heading_f"].to_numpy(), N_BINS)
    T = mem.transition[cls]
    glob = T.sum(axis=(0, 1)) + 0.5
    p_glob = glob / glob.sum(axis=1, keepdims=True)                  # (B, B)
    cnt = T[iy, ix, b0].astype(float)                                # (n, B)
    p_loc = (cnt + alpha * p_glob[b0]) / (cnt.sum(axis=1, keepdims=True) + alpha)
    rows = np.arange(len(b0))
    gain = np.log2(p_loc[rows, b1]) - np.log2(p_glob[b0, b1])
    turning = b1 != b0
    return {
        "n": int(len(b0)), "n_turning": int(turning.sum()),
        "info_gain_bits": bootstrap_ci(gain),
        "info_gain_bits_turning": bootstrap_ci(gain[turning]),
        "top1_acc_memory": float((p_loc.argmax(1) == b1).mean()),
        "top1_acc_no_memory": float((p_glob[b0].argmax(1) == b1).mean()),
        "top1_acc_memory_turning": float((p_loc.argmax(1) == b1)[turning].mean()),
        "top1_acc_no_memory_turning": float((p_glob[b0].argmax(1) == b1)[turning].mean()),
        "p_true_turning_memory": float(p_loc[rows, b1][turning].mean()),
        "p_true_turning_no_memory": float(p_glob[b0, b1][turning].mean()),
    }
