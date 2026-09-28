"""Vectorised closed-loop simulation: one robotaxi, one pedestrian per episode.

Rare-event design: every episode contains exactly one pedestrian who emerges
from the parked row somewhere in a window W around the moment the robotaxi
passes that spot. Per-traversal rates are then
    rate = (Lambda * W) * [p_inattentive * P(event | inattentive) + (1 - p) * P(event | attentive)]
with Lambda the street's emergence rate. All planners are run on the *same*
sampled episodes (common random numbers), so differences are paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import Profile, SimParams, sample_parked_row, snap_to_gap

TAU_RANGE = (-9.0, 2.0)            # curb-crossing time relative to the robotaxi's nominal arrival
WINDOW = TAU_RANGE[1] - TAU_RANGE[0]
NEAR_MISS_DRAC = 4.0               # m/s^2
T_REFUTE = 0.4                     # s of own clear view without a detection that refutes a camera report


@dataclass
class Encounters:
    x0: np.ndarray          # (N, M) parked cars
    x1: np.ndarray
    u: np.ndarray           # (N,) quantile of the emergence location
    tau: np.ndarray         # (N,)
    v_p: np.ndarray         # (N,)
    attentive: np.ndarray   # (N,) bool
    seed: int               # for infrastructure detection randomness

    def __len__(self) -> int:
        return len(self.u)


def sample_encounters(rng: np.random.Generator, n: int, p: SimParams, speed_quantiles: np.ndarray,
                      attentive: bool) -> Encounters:
    x0, x1 = sample_parked_row(rng, n, p)
    v_p = np.interp(rng.random(n), np.linspace(0, 1, len(speed_quantiles)), speed_quantiles)
    return Encounters(x0, x1, rng.random(n), rng.uniform(*TAU_RANGE, n), v_p,
                      np.full(n, attentive), int(rng.integers(1 << 31)))


@dataclass
class Infra:
    coverage: np.ndarray | None = None     # on p.grid, 0/1
    up: bool = False                       # camera actually working
    # --- faults and attacks (defaults: an honest camera)
    delete_share: float = 0.0              # share of real pedestrians the camera never reports
    ghost: bool = False                    # the episode's pedestrian exists only in camera reports
    report_from_y: float | None = None     # reports start only once the object is this far out (pop-up)
    # --- defence
    onboard_priority: bool = False         # where the car sees for itself, camera reports are ignored


def nominal_run(profile: np.ndarray, p: SimParams, t_max: float = 400.0):
    """Robotaxi alone: time to finish and arrival time of the front at each x."""
    grid = p.grid
    x, v, t = p.x_start, float(np.interp(p.x_start, grid, profile)), 0.0
    xs, ts = [x], [t]
    while x - p.av_length <= p.x_end and t < t_max:
        a = np.clip(1.5 * (np.interp(x, grid, profile) - v), -p.a_track_dec, p.a_acc)
        v_new = max(0.0, v + a * p.dt)
        x += 0.5 * (v + v_new) * p.dt
        v, t = v_new, t + p.dt
        xs.append(x)
        ts.append(t)
    return t, np.asarray(xs), np.asarray(ts)


def simulate(enc: Encounters, truth: Profile, profile: np.ndarray, p: SimParams,
             infra: Infra | None = None, t_extra: float = 30.0) -> dict[str, np.ndarray]:
    """Run all episodes in parallel. `profile` is the target-speed map the planner
    uses; `truth` is where pedestrians really emerge."""
    infra = infra or Infra()
    n = len(enc)
    grid = p.grid
    rng = np.random.default_rng(enc.seed)
    t_nom, xs_nom, ts_nom = nominal_run(profile, p)

    x_p = snap_to_gap(truth.inverse_cdf(enc.u), enc.x0, enc.x1)
    t_arrive = np.interp(x_p, xs_nom, ts_nom)
    t_spawn = t_arrive + enc.tau - (0.0 - p.y_spawn) / enc.v_p
    lo, hi = p.swath
    r = p.ped_radius
    y_wait = p.occlusion_edge

    # pedestrians already walking at t = 0
    y = p.y_spawn + np.clip(-t_spawn, 0, None) * enc.v_p
    committed = ~enc.attentive | (y > y_wait)
    y = np.where(enc.attentive & ~committed, np.minimum(y, y_wait), y)
    vy = np.zeros(n)

    xf = np.full(n, p.x_start)
    v = np.full(n, float(np.interp(p.x_start, grid, profile)))
    vis_timer = np.zeros(n)
    known_ob = np.zeros(n, bool)
    covered = np.zeros(n, bool) if infra.coverage is None else np.interp(x_p, grid, infra.coverage) > 0.5
    first_det = np.full(n, np.inf)
    k_lat = max(1, int(round(p.infra_latency / p.dt)))
    hist_y = np.repeat(y[None], k_lat + 1, axis=0)
    hist_vy = np.zeros((k_lat + 1, n))
    p_frame = 1 - (1 - p.infra_p_detect) ** (p.dt / 0.1)
    # separate generator so that honest-camera runs keep their exact random stream
    deleted = np.random.default_rng(enc.seed + 1).random(n) < infra.delete_share
    ghost = bool(infra.ghost)
    refute_timer = np.zeros(n)
    refuted = np.zeros(n, bool)
    unreported_seen = np.zeros(n, bool)

    collided = np.zeros(n, bool)
    near = np.zeros(n, bool)
    impact_v = np.zeros(n)
    hard = np.zeros(n, int)
    was_hard = np.zeros(n, bool)
    max_decel = np.zeros(n)
    t_finish = np.full(n, np.nan)
    first_known_by_infra = np.zeros(n, bool)
    first_known_time = np.full(n, np.inf)

    n_steps = int((t_nom + t_extra) / p.dt)
    for k in range(n_steps):
        t = (k + 1) * p.dt
        running = ~collided & np.isnan(t_finish)
        if not running.any():
            break

        # ---- pedestrian
        active = (t >= t_spawn) & (y < p.y_end)
        gap = x_p - r - xf
        passed = xf - p.av_length > x_p + r
        tta = np.where(v > 0.1, gap / np.maximum(v, 0.1), np.inf)
        car_close = ~passed & (gap > -p.av_length) & (tta < p.gap_acceptance)
        at_line = active & enc.attentive & ~committed & (y >= y_wait - 1e-9)
        waiting = at_line & car_close
        committed |= at_line & ~car_close
        moving = active & ~waiting
        vy = np.where(moving, enc.v_p, 0.0)
        y_next = y + vy * p.dt
        y_next = np.where(enc.attentive & ~committed, np.minimum(y_next, y_wait), y_next)
        # nobody walks into the side of a car that is passing right in front of them
        alongside = (gap < 0) & ~passed
        y_next = np.where(alongside & (y + r <= lo), np.minimum(y_next, lo - r - 0.02), y_next)
        vy = np.where(y_next > y, vy, 0.0)
        y = y_next
        present = active & (y < p.y_end)

        # ---- onboard perception: ray from the roof sensor through the parked band
        sx = xf - p.sensor_back
        below = y < p.parked_y[1]
        denom = np.where(below, p.y_av - y, 1.0)
        s_in = (p.y_av - p.parked_y[1]) / denom
        s_out = np.where(y < p.parked_y[0], (p.y_av - p.parked_y[0]) / denom, 1.0)
        xa, xb = sx + s_in * (x_p - sx), sx + s_out * (x_p - sx)
        seg_lo, seg_hi = np.minimum(xa, xb), np.maximum(xa, xb)
        blocked = below & ((enc.x0 < seg_hi[:, None]) & (enc.x1 > seg_lo[:, None])).any(axis=1)
        geo_vis = present & ~blocked & (np.abs(x_p - sx) < p.sensor_range)
        vis = geo_vis & (not ghost)
        vis_timer = np.where(vis, vis_timer + p.dt, 0.0)
        new_ob = ~known_ob & (vis_timer >= p.t_perception - 1e-9)
        known_ob |= vis_timer >= p.t_perception - 1e-9

        # ---- infrastructure camera (latency + CV extrapolation)
        hist_y[k % (k_lat + 1)] = y
        hist_vy[k % (k_lat + 1)] = vy
        if infra.up:
            reportable = present & covered & ~deleted
            if infra.report_from_y is not None:
                reportable &= y >= infra.report_from_y
            detect = reportable & (rng.random(n) < p_frame)
            first_det = np.where(detect & np.isinf(first_det), t, first_det)
        infra_known = t >= first_det + p.infra_latency
        # evidence for a fleet audit: the car saw someone the covering camera had not reported
        # (only pedestrians who appeared during the episode: the camera saw them from the start)
        unreported_seen |= new_ob & covered & infra.up & ~infra_known & (t_spawn > 0)
        if infra.onboard_priority:
            refute_timer = np.where(geo_vis & infra_known & ~known_ob, refute_timer + p.dt, 0.0)
            refuted |= refute_timer >= T_REFUTE - 1e-9
            infra_known = infra_known & ~refuted & ~(geo_vis & ~known_ob)
        slot = (k - k_lat) % (k_lat + 1)
        y_inf = hist_y[slot] + hist_vy[slot] * p.infra_latency
        vy_inf = hist_vy[slot]
        newly = np.isinf(first_known_time) & (known_ob | infra_known)
        first_known_by_infra |= newly & ~known_ob
        first_known_time = np.where(newly, t, first_known_time)

        # ---- planner: react to known pedestrians, otherwise track the profile
        knows = present & (known_ob | infra_known)
        y_e = np.where(known_ob, y, y_inf)
        vy_e = np.where(known_ob, vy, vy_inf)
        in_band = (y_e + r > lo) & (y_e - r < hi)
        crossing = vy_e > 0.2
        with np.errstate(divide="ignore", invalid="ignore"):
            t_in = np.where(in_band, 0.0, np.where(crossing, (lo - r - y_e) / vy_e, np.inf))
            t_out = np.where(crossing, (hi + r - y_e) / vy_e, np.where(in_band, np.inf, -np.inf))
        v_eff = np.maximum(v, 0.5)
        t_arr = np.maximum(gap, 0.0) / v_eff
        t_clear = t_arr + (p.av_length + 2 * r) / v_eff
        conflict = knows & (gap > 0) & (t_arr < t_out + 0.5) & (t_clear > t_in - 0.5)
        stop_dist = gap - p.stop_margin
        with np.errstate(divide="ignore"):
            a_req = np.where(stop_dist > 0.05, v ** 2 / (2 * stop_dist), np.inf)
        a_react = np.where(conflict, -np.minimum(np.maximum(a_req, 1.0), p.a_emergency), np.inf)
        a_track = np.clip(1.5 * (np.interp(xf, grid, profile) - v), -p.a_track_dec, p.a_acc)
        a = np.maximum(np.minimum(a_track, a_react), -p.a_emergency)
        a = np.where(running, a, 0.0)

        v_new = np.clip(v + a * p.dt, 0.0, None)
        a = (v_new - v) / p.dt
        xf = np.where(running, xf + 0.5 * (v + v_new) * p.dt, xf)
        v = np.where(running, v_new, 0.0)

        is_hard = running & (a <= -p.hard_brake)
        hard += is_hard & ~was_hard
        was_hard = is_hard
        max_decel = np.maximum(max_decel, np.where(running, -a, 0.0))

        # ---- outcomes
        dxc = np.clip(x_p, xf - p.av_length, xf) - x_p
        dyc = np.clip(y, lo, hi) - y
        touch = running & present & (not ghost) & (np.hypot(dxc, dyc) < r) & (v > 0.3)
        impact_v = np.where(touch & ~collided, v, impact_v)
        collided |= touch
        # near miss: deceleration required to avoid the pedestrian (DRAC) >= 4 m/s^2
        in_path = (y + r > lo) & (y - r < hi)
        with np.errstate(divide="ignore"):
            drac = np.where(gap > 0.05, v ** 2 / (2 * gap), np.inf)
        near |= touch | (running & present & (not ghost) & in_path & (gap > 0.05) & (drac >= NEAR_MISS_DRAC)
                         & (v > 0.5))
        t_finish = np.where(running & ~collided & (xf - p.av_length > p.x_end), t, t_finish)

    return {
        "collided": collided, "impact_v": impact_v, "near": near, "hard_brakes": hard,
        "max_decel": max_decel, "t_finish": t_finish, "t_nominal": np.full(n, t_nom),
        "known_by_infra_first": first_known_by_infra, "x_p": x_p,
        "unreported_seen": unreported_seen, "refuted": refuted, "deleted": deleted,
    }


def summarize(res_i: dict, res_a: dict, truth: Profile, p: SimParams) -> dict:
    """Per-traversal expectations from inattentive / attentive episode strata."""
    exposure = truth.total_rate * WINDOW            # pedestrians interacting per traversal
    w_i, w_a = p.p_inattentive, 1 - p.p_inattentive
    t_nom = float(res_i["t_nominal"][0])

    def m(key, res):
        return float(np.mean(res[key]))

    def delay(res):
        ok = ~res["collided"] & ~np.isnan(res["t_finish"])
        return float(np.mean(res["t_finish"][ok] - t_nom)) if ok.any() else np.nan

    k, n = int(res_i["collided"].sum()), len(res_i["collided"])
    lo, hi = _wilson(k, n)
    per = 1e4 * exposure
    return {
        "t_nominal": t_nom,
        "trip_time": t_nom + exposure * (w_i * delay(res_i) + w_a * delay(res_a)),
        "collisions_per_10k": per * (w_i * m("collided", res_i) + w_a * m("collided", res_a)),
        "collisions_per_10k_ci": (per * w_i * lo, per * w_i * hi),
        "near_misses_per_10k": per * (w_i * m("near", res_i) + w_a * m("near", res_a)),
        "hard_brakes_per_1k": 1e3 * exposure * (w_i * m("hard_brakes", res_i) + w_a * m("hard_brakes", res_a)),
        "mean_impact_speed": float(res_i["impact_v"][res_i["collided"]].mean()) if k else 0.0,
        "p_collision_given_inattentive": k / n,
        "n_inattentive": n,
        "n_attentive": len(res_a["collided"]),
        "exposure_per_traversal": exposure,
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), c + h
