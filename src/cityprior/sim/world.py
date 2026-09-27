"""Mid-block street with a parked row: geometry, parameters, pedestrian profile.

Coordinates: x along the street in the robotaxi's driving direction, y lateral.
    y in [-3.5, 0)   sidewalk (pedestrians approach the curb here)
    y in [0.1, 1.9]  parked cars (occluders), lane-side edge at y = 2.0
    y in [3.0, 4.9]  robotaxi swath (1.0 m clearance from the parked row)
The street section with parking and mid-block pedestrian activity is x in [0, L].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.05
    street_len: float = 90.0
    x_start: float = -40.0            # robotaxi front at t = 0
    exit_margin: float = 20.0         # finished when the rear passes street_len + exit_margin
    dx: float = 0.5                   # resolution of speed profiles / intensity grids

    # geometry
    parked_y: tuple[float, float] = (0.1, 1.9)
    occlusion_edge: float = 2.0
    clearance: float = 1.0            # parked row edge -> robotaxi side
    av_length: float = 4.8
    av_width: float = 1.9
    sensor_back: float = 1.2          # roof sensor behind the front bumper

    # robotaxi dynamics / perception
    v_limit: float = 11.2             # 25 mph
    a_acc: float = 2.0
    a_track_dec: float = 3.0
    a_plan: float = 1.5               # anticipatory deceleration used in profiles
    a_emergency: float = 7.0
    t_perception: float = 0.25        # continuous visibility needed before reacting
    sensor_range: float = 80.0
    stop_margin: float = 1.0
    hard_brake: float = 3.0           # |a| >= this counts as a hard-braking event

    # pedestrians
    ped_radius: float = 0.3
    y_spawn: float = -3.0
    y_end: float = 8.5
    v_design: float = 1.6             # walking speed assumed by the planner's risk model
    gap_acceptance: float = 3.0       # attentive pedestrians wait if the car is closer (s)
    p_inattentive: float = 0.1

    # infrastructure camera
    infra_latency: float = 0.3
    infra_p_detect: float = 0.9       # per 0.1 s frame
    infra_q: float = 0.95             # P(an approaching pedestrian is reported before the curb)

    # parked row
    car_len: tuple[float, float] = (4.3, 5.2)
    gap_small: tuple[float, float] = (0.6, 2.0)
    gap_large: tuple[float, float] = (4.0, 8.0)
    p_gap_large: float = 0.15

    @property
    def y_av(self) -> float:
        return self.occlusion_edge + self.clearance + self.av_width / 2

    @property
    def swath(self) -> tuple[float, float]:
        return self.y_av - self.av_width / 2, self.y_av + self.av_width / 2

    @property
    def x_end(self) -> float:
        return self.street_len + self.exit_margin

    @property
    def grid(self) -> np.ndarray:
        return np.arange(self.x_start - 5.0, self.x_end + self.av_length + 5.0 + 1e-9, self.dx)

    @property
    def t_ped_to_swath(self) -> float:
        """Planner model: time for a pedestrian at the occlusion edge to touch the swath."""
        return (self.clearance - self.ped_radius) / self.v_design


@dataclass
class Profile:
    """Pedestrian emergence intensity along the street, per metre per second."""

    x: np.ndarray
    lam: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def total_rate(self) -> float:
        """Emergences per second on the whole street section."""
        return float(np.trapezoid(self.lam, self.x))

    @property
    def mean(self) -> float:
        return self.total_rate / (self.x[-1] - self.x[0])

    def on(self, grid: np.ndarray) -> np.ndarray:
        return np.interp(grid, self.x, self.lam, left=0.0, right=0.0)

    def mirrored(self) -> "Profile":
        return Profile(self.x, self.lam[::-1].copy(), {**self.meta, "mirrored": True})

    def shifted(self, dx_m: float) -> "Profile":
        """Same activity pattern moved along the block (wrapping at the ends)."""
        step = self.x[1] - self.x[0]
        return Profile(self.x, np.roll(self.lam, int(round(dx_m / step))), {**self.meta, "shift_m": dx_m})

    def concentrated(self, kappa: float) -> "Profile":
        """lam^kappa rescaled to the same total rate: kappa=0 uniform, 1 as is, >1 sharper."""
        lam = np.maximum(self.lam, 1e-12) ** kappa
        lam = lam / np.trapezoid(lam, self.x) * self.total_rate
        return Profile(self.x, lam, {**self.meta, "kappa": kappa})

    def ceiling(self) -> float:
        """(E sqrt(lam))^2 / E lam: best achievable risk x time relative to ignoring
        where pedestrians appear (1 = no benefit possible)."""
        return float(np.mean(np.sqrt(self.lam)) ** 2 / np.mean(self.lam))

    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (self.lam[1:] + self.lam[:-1]) * np.diff(self.x))])
        return np.interp(u * cdf[-1], cdf, self.x)


def profile_from_tgsim(events: pd.DataFrame, x_range=(70.0, 160.0), y_range=(28.0, 50.0),
                       hours: float = 7139.8 / 3600, sigma_m: float = 3.0, street_len: float = 90.0,
                       crosswalk_regions=(37, 41)) -> Profile:
    """Mid-block pedestrian emergence profile of the TGSIM top street (between two
    signalised intersections), mapped to x in [0, street_len]."""
    e = events[(events["x"] >= x_range[0]) & (events["x"] < x_range[1])
               & (events["y"] >= y_range[0]) & (events["y"] < y_range[1])
               & ~events["region"].isin(crosswalk_regions)]
    xs = (e["x"].to_numpy() - x_range[0]) / (x_range[1] - x_range[0]) * street_len
    x = np.arange(0.0, street_len + 1e-9, 1.0)
    counts = np.histogram(xs, bins=np.append(x - 0.5, x[-1] + 0.5))[0].astype(float)
    lam = gaussian_filter1d(counts, sigma_m, mode="nearest") / (hours * 3600.0)
    return Profile(x, lam, {"source": "TGSIM Foggy Bottom mid-block", "n_events": int(len(e)), "hours": hours})


def ped_speed_quantiles(df: pd.DataFrame, tracks) -> np.ndarray:
    """101 quantiles of walking speed of the given pedestrian tracks (moving samples)."""
    p = df[(df["cls"] == "ped") & df["track"].isin(tracks) & (df["speed"] > 0.5) & (df["speed"] < 3.5)]
    return np.quantile(p["speed"].to_numpy(), np.linspace(0, 1, 101))


def sample_parked_row(rng: np.random.Generator, n: int, p: SimParams, max_cars: int = 40):
    """Random parked rows covering [-5, L + 5]; returns (x0, x1) arrays (n, max_cars)."""
    x0 = np.full((n, max_cars), -1e3)
    x1 = np.full((n, max_cars), -1e3 + 1e-3)
    pos = rng.uniform(-8.0, -5.0, n)
    for k in range(max_cars):
        length = rng.uniform(*p.car_len, n)
        active = pos < p.street_len + 5.0
        x0[active, k] = pos[active]
        x1[active, k] = pos[active] + length[active]
        big = rng.random(n) < p.p_gap_large
        gap = np.where(big, rng.uniform(*p.gap_large, n), rng.uniform(*p.gap_small, n))
        pos = pos + length + gap
    return x0, x1


def snap_to_gap(x_raw: np.ndarray, x0: np.ndarray, x1: np.ndarray, body: float = 0.4) -> np.ndarray:
    """Pedestrians emerge through a gap between parked cars: move each emergence
    point into the nearest gap (clipped so the body fits)."""
    order = np.argsort(x0, axis=1)
    a0, a1 = np.take_along_axis(x0, order, 1), np.take_along_axis(x1, order, 1)
    g0, g1 = a1[:, :-1], a0[:, 1:]                       # gap [end of car k, start of car k+1]
    valid = ((g1 - g0) > 2 * body) & (g0 > -100.0)   # ignore padding slots
    lo, hi = g0 + body, g1 - body
    clipped = np.clip(x_raw[:, None], lo, hi)
    dist = np.where(valid, np.abs(clipped - x_raw[:, None]), np.inf)
    k = dist.argmin(axis=1)
    return clipped[np.arange(len(x_raw)), k]
