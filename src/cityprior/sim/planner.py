"""Occlusion-aware speed planning with and without a location prior.

Risk model (per metre of parked row). A pedestrian stepping out of the occlusion
edge at distance d ahead of the front bumper is unavoidable if the car can
neither stop before it nor pass before the pedestrian reaches the swath:

    v * T_p  <  d  <  v * t_r + v^2 / (2 a_e)

With emergences at rate lam(x) [1/(m s)], the car is exposed at x for
    band(v) = max(0, t_r + v / (2 a_e) - T_p)   seconds,
so expected unavoidable conflicts = integral lam(x) * band(v(x)) dx.

Minimising  integral 1/v dx + w * integral lam * band(v) dx  pointwise gives
    v*(x) = sqrt(2 a_e / (w lam(x))),   bounded below by the worst-case speed
    v_wc = 2 a_e (T_p - t_r), at which band = 0, and above by the speed limit.

All planners share w (the exchange rate between time and risk); they differ
only in the intensity map lam they believe:
    worst case       - no probabilities, v = v_wc everywhere along the parked row
    no memory        - the street's mean intensity, everywhere (knows *how many*, not *where*)
    memory           - the location memory's lam(x)
    + live camera    - lam suppressed where a healthy camera would report approaching pedestrians
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .world import Profile, SimParams


def worst_case_speed(p: SimParams) -> float:
    return max(0.0, 2 * p.a_emergency * (p.t_ped_to_swath - p.t_perception))


def band(v: np.ndarray, p: SimParams) -> np.ndarray:
    return np.maximum(0.0, p.t_perception + v / (2 * p.a_emergency) - p.t_ped_to_swath)


def weight_for_reference_speed(v_ref: float, lam_mean: float, p: SimParams) -> float:
    """w such that a planner believing lam = lam_mean everywhere drives at v_ref."""
    return 2 * p.a_emergency / (v_ref ** 2 * lam_mean)


def speed_profile(lam_believed: np.ndarray, w: float | None, p: SimParams) -> np.ndarray:
    """Target speed on p.grid. w=None -> worst case along the parked row."""
    x = p.grid
    on_street = (x >= 0.0) & (x <= p.street_len)
    v = np.full(x.shape, p.v_limit)
    if w is None:
        v[on_street] = worst_case_speed(p)
    else:
        with np.errstate(divide="ignore"):
            v_star = np.sqrt(2 * p.a_emergency / (w * np.maximum(lam_believed, 0.0)))
        v[on_street] = np.clip(v_star[on_street], worst_case_speed(p), p.v_limit)
    # anticipate slower sections with a comfortable deceleration
    for i in range(len(x) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * p.a_plan * p.dx))
    return v


def believed_intensity(prior: Profile, p: SimParams, coverage: np.ndarray | None = None,
                       camera_trusted: bool = False, floor: float = 0.0,
                       floor_ref: float | None = None) -> np.ndarray:
    """Intensity map the planner acts on (on p.grid)."""
    lam = prior.on(p.grid)
    if floor > 0.0:
        ref = prior.mean if floor_ref is None else floor_ref
        on_street = (p.grid >= 0) & (p.grid <= p.street_len)
        lam = np.where(on_street, np.maximum(lam, floor * ref), lam)
    if camera_trusted and coverage is not None:
        lam = lam * (1.0 - p.infra_q * coverage)
    return lam


def expected_risk(v: np.ndarray, lam_true: np.ndarray, p: SimParams) -> float:
    """Model-based expected unavoidable conflicts per traversal (all pedestrians)."""
    return float(np.trapezoid(lam_true * band(v, p), p.grid))


def memory_from_counts(x: np.ndarray, counts: np.ndarray, seconds: float, q: float | np.ndarray = 0.95,
                       exposure_corrected: bool = True, sigma_m: float = 3.0, prior_seconds: float = 1800.0,
                       fallback_mean: float | None = None, meta: dict | None = None) -> Profile:
    """Location memory from observed emergence counts per metre gathered over
    `seconds` of (possibly discounted) watching: smoothed, divided by the
    observability q(x), and shrunk towards the street mean with `prior_seconds`
    of pseudo-observation. Nearly no data -> nearly the street mean."""
    q = np.broadcast_to(np.asarray(q, float), np.shape(counts))
    counts = np.asarray(counts, float)
    if exposure_corrected:
        smooth = gaussian_filter1d(counts / np.maximum(q, 0.02), sigma_m, mode="nearest")
    else:
        smooth = gaussian_filter1d(counts, sigma_m, mode="nearest") / max(q.mean(), 1e-6)
    if seconds > 0 and smooth.sum() > 0:
        lam_bar = smooth.mean() / seconds
    else:
        lam_bar = fallback_mean if fallback_mean is not None else 0.0
    lam = (smooth + prior_seconds * lam_bar) / (seconds + prior_seconds)
    return Profile(np.asarray(x), lam, meta or {})


def estimate_memory(truth: Profile, hours: float, rng: np.random.Generator,
                    p_obs: float | np.ndarray = 0.95, exposure_corrected: bool = True,
                    sigma_m: float = 3.0, prior_seconds: float = 1800.0) -> Profile:
    """Location memory learned by watching the street for `hours`.

    p_obs: probability that an emergence at x is observed at all (scalar for a
    fixed camera; a profile q(x) for a fleet, which only sees what passing cars
    see). exposure_corrected: divide by q(x) per location (the estimator knows
    how long each place was watched) instead of by its average."""
    T = hours * 3600.0
    q = np.broadcast_to(np.asarray(p_obs, float), truth.lam.shape)
    counts = rng.poisson(truth.lam * T * q).astype(float)
    return memory_from_counts(truth.x, counts, T, q, exposure_corrected, sigma_m, prior_seconds,
                              fallback_mean=truth.mean,
                              meta={"hours": hours, "events_observed": int(counts.sum())})


def camera_coverage(fraction: float, p: SimParams) -> np.ndarray:
    """Cameras at both intersections see `fraction` of the block, half from each
    end; mid-block is the last to be covered (as in real deployments)."""
    x = p.grid
    reach = fraction * p.street_len / 2
    return ((x < reach) | (x > p.street_len - reach)).astype(float) * (fraction > 0)
