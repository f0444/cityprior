"""Location memory: what a fixed infrastructure sensor learns about a place.

The memory holds sufficient statistics, not raw trajectories:
  * per grid cell and road-user class: occupancy, exposure (distinct passages),
    a discretised flow field (heading bins with mean direction and speed, in the
    spirit of CLiFF maps of dynamics) and vehicle speed quantiles;
  * per cell: counts of pedestrian emergences, hard-braking onsets, conflicts;
  * per region (lane / crosswalk polygon from the dataset): exposure-normalised
    event rates with empirical-Bayes Beta posteriors.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import betabinom

from .geometry import Grid, wrap_angle

CLASSES = ("ped", "micro", "veh")
MOVING_SPEED = {"ped": 0.3, "micro": 1.0, "veh": 1.0}  # m/s, for the flow field
N_BINS = 8


def heading_bin(heading: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    return (np.floor((np.asarray(heading) + np.pi) / (2 * np.pi / n_bins)).astype(int)) % n_bins


def transition_pairs(d: pd.DataFrame, min_speed: float, horizon_frames: int = 30,
                     stride: int = 5) -> pd.DataFrame:
    """(state now, heading `horizon_frames` later) pairs of the same track, for
    manoeuvre statistics: which way does traffic here go next?"""
    d = d.loc[d["frame"] % stride == 0, ["track", "frame", "x", "y", "heading", "speed"]]
    fut = d[["track", "frame", "heading", "speed"]].assign(frame=d["frame"] - horizon_frames)
    m = d.merge(fut, on=["track", "frame"], suffixes=("", "_f"))
    return m[(m["speed"] >= min_speed) & (m["speed_f"] >= min_speed)].reset_index(drop=True)


def fit_prior_strength(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Empirical Bayes for rates k/n across regions: Beta(m*p, m*(1-p)) prior with
    p = pooled rate and m chosen by beta-binomial marginal likelihood."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = max(k.sum() / max(n.sum(), 1.0), 1e-6)
    ms = np.logspace(0, 4, 80)
    ll = [betabinom.logpmf(k, n, m * p, m * (1 - p)).sum() for m in ms]
    return p, float(ms[int(np.argmax(ll))])


def window(df: pd.DataFrame, events: dict[str, pd.DataFrame], spans: list[tuple[float, float]]):
    """Restrict trajectories and events to a union of time spans."""
    def m(t):
        return np.logical_or.reduce([(t >= a) & (t < b) for a, b in spans])
    return df[m(df["t"])], {k: e[m(e["t"])] for k, e in events.items()}


@dataclass
class LocationMemory:
    grid: Grid
    hours: float
    occupancy: dict[str, np.ndarray] = field(default_factory=dict)   # agent-seconds
    passages: dict[str, np.ndarray] = field(default_factory=dict)    # distinct tracks
    flow_count: dict[str, np.ndarray] = field(default_factory=dict)  # (ny, nx, B)
    flow_vec: dict[str, np.ndarray] = field(default_factory=dict)    # (ny, nx, B, 2)
    flow_speed: dict[str, np.ndarray] = field(default_factory=dict)  # (ny, nx, B)
    transition: dict[str, np.ndarray] = field(default_factory=dict)  # (ny, nx, B, B) heading now -> +3 s
    veh_speed_p50: np.ndarray | None = None
    veh_speed_p85: np.ndarray | None = None
    emergence: np.ndarray | None = None
    hard_brake: np.ndarray | None = None
    conflicts: np.ndarray | None = None
    regions: pd.DataFrame | None = None

    @property
    def support(self) -> np.ndarray:
        """Cells where any road user was ever observed: the monitored road area."""
        return sum(self.occupancy[c] for c in CLASSES) > 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, df: pd.DataFrame, events: dict[str, pd.DataFrame],
              spans: list[tuple[float, float]], grid: Grid, dt: float = 0.1) -> "LocationMemory":
        """Build memory from the observations inside `spans` (list of [t0, t1))."""
        h, ev = window(df, events, spans)
        mem = cls(grid=grid, hours=sum(b - a for a, b in spans) / 3600.0)
        B = N_BINS
        for c in CLASSES:
            d = h[h["cls"] == c]
            idx = grid.flat(d["x"].to_numpy(), d["y"].to_numpy())
            ok = idx >= 0
            mem.occupancy[c] = np.bincount(idx[ok], minlength=grid.size).reshape(grid.shape) * dt
            cells = pd.DataFrame({"track": d["track"].to_numpy()[ok], "cell": idx[ok]}).drop_duplicates()
            mem.passages[c] = np.bincount(cells["cell"], minlength=grid.size).reshape(grid.shape)

            mv = ok & (d["speed"].to_numpy() >= MOVING_SPEED[c])
            b = heading_bin(d["heading"].to_numpy()[mv], B)
            key = idx[mv] * B + b
            size = grid.size * B
            cnt = np.bincount(key, minlength=size)
            hd = d["heading"].to_numpy()[mv]
            vx = np.bincount(key, weights=np.cos(hd), minlength=size)
            vy = np.bincount(key, weights=np.sin(hd), minlength=size)
            sp = np.bincount(key, weights=d["speed"].to_numpy()[mv], minlength=size)
            with np.errstate(invalid="ignore", divide="ignore"):
                mem.flow_count[c] = cnt.reshape(grid.ny, grid.nx, B)
                mem.flow_vec[c] = np.stack([vx, vy], -1).reshape(grid.ny, grid.nx, B, 2)
                mem.flow_speed[c] = np.where(cnt > 0, sp / cnt, 0.0).reshape(grid.ny, grid.nx, B)

            tp = transition_pairs(d, MOVING_SPEED[c])
            ti = grid.flat(tp["x"].to_numpy(), tp["y"].to_numpy())
            tk = ti >= 0
            key = (ti[tk] * B + heading_bin(tp["heading"].to_numpy()[tk], B)) * B \
                + heading_bin(tp["heading_f"].to_numpy()[tk], B)
            mem.transition[c] = np.bincount(key, minlength=grid.size * B * B).reshape(grid.ny, grid.nx, B, B)

        v = h[(h["cls"] == "veh") & (h["speed"] >= 1.0)]
        vi = grid.flat(v["x"].to_numpy(), v["y"].to_numpy())
        spd = pd.Series(v["speed"].to_numpy()[vi >= 0]).groupby(vi[vi >= 0])
        for name, level in (("veh_speed_p50", 0.5), ("veh_speed_p85", 0.85)):
            arr = np.full(grid.size, np.nan)
            if len(v):
                q = spd.quantile(level)
                arr[q.index.to_numpy()] = q.to_numpy()
            setattr(mem, name, arr.reshape(grid.shape))

        mem.emergence = grid.accumulate(ev["emergence"]["x"], ev["emergence"]["y"])
        mem.hard_brake = grid.accumulate(ev["hard_brake"]["x"], ev["hard_brake"]["y"])
        mem.conflicts = grid.accumulate(ev["conflict"]["x"], ev["conflict"]["y"])
        mem.regions = region_table(h, ev)
        return mem

    # ---------------------------------------------------------------- queries
    def emergence_density(self, sigma_m: float = 1.5, alpha: float = 0.05) -> np.ndarray:
        """Spatial distribution of pedestrian emergence over the road area (sums to 1).
        `alpha` pseudo-counts per cell keep unseen places possible."""
        sup = self.support
        s = self.grid.smooth(self.emergence, sigma_m) * sup + alpha * sup
        return s / s.sum()

    def emergence_rate(self, sigma_m: float = 1.5) -> np.ndarray:
        """Pedestrian emergences per hour per 10 m^2 (smoothed)."""
        area = self.grid.cell ** 2
        return self.grid.smooth(self.emergence, sigma_m) / self.hours / area * 10.0

    def region_rate(self, what: str) -> pd.DataFrame:
        """Per-region posterior of the per-passage probability of an event.

        what: 'hard_brake' (per vehicle passage) or 'ped_conflict' (per pedestrian passage).
        """
        r = self.regions
        k, n = r[f"{what}_tracks"].to_numpy(), r[f"{what}_exposure"].to_numpy()
        p, m = fit_prior_strength(k, n)
        a, b = k + m * p, (n - k) + m * (1 - p)
        out = pd.DataFrame({
            "region": r["region"], "type": r["type"], "k": k, "n": n,
            "p_mean": a / (a + b),
            "p_lo": beta_dist.ppf(0.05, a, b), "p_hi": beta_dist.ppf(0.95, a, b),
        })
        out.attrs.update(p_global=p, prior_strength=m)
        return out

    def flow_mode(self, c: str, x, y, heading, kappa: float = 2.0, min_count: int = 5):
        """Dominant local motion direction compatible with the current heading.

        Returns (direction [rad], mean speed [m/s], valid mask) for each query.
        """
        iy, ix, ok = self.grid.ij(x, y)
        iy, ix = np.clip(iy, 0, self.grid.ny - 1), np.clip(ix, 0, self.grid.nx - 1)
        cnt = self.flow_count[c][iy, ix]                                 # (n, B)
        vec = self.flow_vec[c][iy, ix]                                   # (n, B, 2)
        direction = np.arctan2(vec[..., 1], vec[..., 0])
        score = cnt * np.exp(kappa * np.cos(wrap_angle(direction - np.asarray(heading)[:, None])))
        score = np.where(cnt >= min_count, score, -1.0)
        best = score.argmax(axis=1)
        rows = np.arange(len(best))
        valid = ok & (score[rows, best] > 0)
        return direction[rows, best], self.flow_speed[c][iy, ix][rows, best], valid

    # ------------------------------------------------------------- footprint
    def layers_uint8(self) -> dict[str, np.ndarray]:
        """Static layers quantised to one byte per cell (what would be shipped)."""
        def q(a, hi):
            a = np.nan_to_num(np.asarray(a, float))
            return np.clip(np.round(a / hi * 255), 0, 255).astype(np.uint8)

        fc = self.flow_count["veh"] + self.flow_count["ped"]
        dom = fc.argmax(axis=2).astype(np.uint8)
        return {
            "support": self.support.astype(np.uint8),
            "ped_emergence_rate": q(self.emergence_rate(), max(self.emergence_rate().max(), 1e-9)),
            "hard_brake_density": q(self.grid.smooth(self.hard_brake, 2.0), max(self.grid.smooth(self.hard_brake, 2.0).max(), 1e-9)),
            "conflict_density": q(self.grid.smooth(self.conflicts, 2.0), max(self.grid.smooth(self.conflicts, 2.0).max(), 1e-9)),
            "veh_speed_p85": q(self.veh_speed_p85, 25.0),
            "dominant_flow_bin": dom,
        }

    def footprint_bytes(self) -> dict[str, int]:
        layers = self.layers_uint8()
        raw = sum(a.nbytes for a in layers.values())
        packed = len(zlib.compress(b"".join(a.tobytes() for a in layers.values()), 9))
        return {"cells": self.grid.size, "layers": len(layers), "raw_bytes": raw, "zlib_bytes": packed}


def region_table(h: pd.DataFrame, ev: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Exposure and event counts per dataset region (lane / crosswalk polygon)."""
    share = h.groupby("region")["cls"].value_counts(normalize=True).unstack(fill_value=0.0)
    r = pd.DataFrame(index=share.index)
    r["ped_share"] = share.get("ped", 0.0)
    r["type"] = np.where(r["ped_share"] >= 0.5, "crosswalk", "roadway")
    veh = h[h["cls"] == "veh"].groupby("region")["track"].nunique()
    ped = h[h["cls"] == "ped"].groupby("region")["track"].nunique()
    hb = ev["hard_brake"].groupby("region")["track"].nunique()
    pc = ev["conflict"][ev["conflict"]["cls"] == "ped"].groupby("region")["track_p"].nunique()
    em = ev["emergence"].groupby("region").size()
    r["hard_brake_exposure"] = veh.reindex(r.index).fillna(0).astype(int)
    r["hard_brake_tracks"] = hb.reindex(r.index).fillna(0).astype(int)
    r["ped_conflict_exposure"] = ped.reindex(r.index).fillna(0).astype(int)
    r["ped_conflict_tracks"] = pc.reindex(r.index).fillna(0).astype(int)
    r["emergences"] = em.reindex(r.index).fillna(0).astype(int)
    # a track can be counted as an event without being counted as exposure only
    # through edge effects at the window border; keep k <= n
    for w in ("hard_brake", "ped_conflict"):
        r[f"{w}_tracks"] = np.minimum(r[f"{w}_tracks"], r[f"{w}_exposure"])
    return r.reset_index()
