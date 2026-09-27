import numpy as np

from cityprior.sim.engine import Encounters, Infra, nominal_run, sample_encounters, simulate
from cityprior.sim.planner import (band, believed_intensity, camera_coverage, estimate_memory,
                                   speed_profile, weight_for_reference_speed, worst_case_speed)
from cityprior.sim.world import Profile, SimParams, sample_parked_row, snap_to_gap

P = SimParams()


def _flat_profile(rate_per_hour: float = 50.0) -> Profile:
    x = np.arange(0.0, P.street_len + 1e-9, 1.0)
    return Profile(x, np.full(len(x), rate_per_hour / 3600 / P.street_len))


def _hotspot_profile() -> Profile:
    x = np.arange(0.0, P.street_len + 1e-9, 1.0)
    lam = 0.2 + 5.0 * np.exp(-0.5 * ((x - 40) / 4) ** 2)
    return Profile(x, lam / np.trapezoid(lam, x) * 50 / 3600)


def test_worst_case_speed_zeroes_the_risk_band():
    v = worst_case_speed(P)
    assert 1.0 < v < 5.0
    assert np.isclose(band(np.array([v]), P)[0], 0.0, atol=1e-9)
    assert band(np.array([P.v_limit]), P)[0] > 0.3


def test_profile_limits_and_comfortable_anticipation():
    lam = believed_intensity(_hotspot_profile(), P)
    w = weight_for_reference_speed(8.0, _hotspot_profile().mean, P)
    v = speed_profile(lam, w, P)
    assert v.min() >= worst_case_speed(P) - 1e-9 and v.max() <= P.v_limit + 1e-9
    decel = (v[:-1] ** 2 - v[1:] ** 2) / (2 * P.dx)
    assert decel.max() <= P.a_plan + 1e-6
    i_hot, i_cold = np.argmin(np.abs(P.grid - 40)), np.argmin(np.abs(P.grid - 80))
    assert v[i_hot] < v[i_cold]                    # slower where people step out


def test_nominal_run_time_matches_kinematics():
    v = np.full(P.grid.shape, 10.0)
    t, xs, ts = nominal_run(v, P)
    expected = (P.x_end + P.av_length - P.x_start) / 10.0
    assert abs(t - expected) < 0.2
    assert np.all(np.diff(xs) >= 0)


def test_pedestrians_emerge_through_gaps():
    rng = np.random.default_rng(1)
    x0, x1 = sample_parked_row(rng, 200, P)
    xp = snap_to_gap(rng.uniform(0, P.street_len, 200), x0, x1)
    inside_car = ((x0 < xp[:, None] + 0.3) & (x1 > xp[:, None] - 0.3)).any(axis=1)
    assert not inside_car.any()


def _encounters(n: int, v_p: float, attentive: bool, seed: int = 0) -> Encounters:
    rng = np.random.default_rng(seed)
    e = sample_encounters(rng, n, P, np.full(101, v_p), attentive)
    return e


def test_worst_case_planner_never_hits_design_speed_pedestrians():
    """Consistency of planner model and simulator: at the worst-case speed an
    inattentive pedestrian walking no faster than assumed is always avoided."""
    enc = _encounters(1500, v_p=P.v_design - 0.1, attentive=False)
    res = simulate(enc, _flat_profile(), speed_profile(None, None, P), P)
    assert res["collided"].sum() == 0


def test_fast_car_hits_inattentive_pedestrians_sometimes():
    enc = _encounters(1500, v_p=1.6, attentive=False)
    res = simulate(enc, _flat_profile(), np.full(P.grid.shape, P.v_limit), P)
    assert 0.01 < res["collided"].mean() < 0.5
    assert res["impact_v"][res["collided"]].min() > 0.3


def test_attentive_pedestrians_are_not_hit_at_full_speed():
    enc = _encounters(1000, v_p=1.6, attentive=True)
    res = simulate(enc, _flat_profile(), np.full(P.grid.shape, P.v_limit), P)
    assert res["collided"].sum() == 0


def test_camera_reports_prevent_collisions_at_full_speed():
    enc = _encounters(1500, v_p=1.6, attentive=False)
    fast = np.full(P.grid.shape, P.v_limit)
    cov = camera_coverage(1.0, P)
    blind = simulate(enc, _flat_profile(), fast, P, Infra(cov, up=False))
    seen = simulate(enc, _flat_profile(), fast, P, Infra(cov, up=True))
    assert seen["collided"].sum() < 0.2 * max(1, blind["collided"].sum())
    assert seen["known_by_infra_first"].mean() > 0.5


def test_memory_estimate_converges():
    rng = np.random.default_rng(0)
    truth = _hotspot_profile()
    err = [np.abs(estimate_memory(truth, h, rng).lam - truth.lam).mean() / truth.mean for h in (0.5, 50)]
    assert err[1] < err[0] and err[1] < 0.3


def test_add_caution_only_profile_is_never_faster_and_feasible():
    w = weight_for_reference_speed(8.0, _hotspot_profile().mean, P)
    v_mem = speed_profile(believed_intensity(_hotspot_profile(), P), w, P)
    v_flat = speed_profile(believed_intensity(_flat_profile(), P), w, P)
    v = np.minimum(v_mem, v_flat)
    assert np.all(v <= v_flat + 1e-12)
    decel = (v[:-1] ** 2 - v[1:] ** 2) / (2 * P.dx)
    assert decel.max() <= P.a_plan + 1e-6


def test_profile_shift_and_concentration_preserve_total_rate():
    prof = _hotspot_profile()
    assert np.isclose(prof.shifted(20).total_rate, prof.total_rate, rtol=0.02)
    for k in (0.0, 2.0):
        assert np.isclose(prof.concentrated(k).total_rate, prof.total_rate)
    assert np.isclose(prof.concentrated(0.0).ceiling(), 1.0)
    assert prof.concentrated(2.0).ceiling() < prof.ceiling() < 1.0


def test_near_misses_only_count_pedestrians_ahead():
    """A pedestrian crossing behind the car is not a near miss."""
    enc = _encounters(800, v_p=1.6, attentive=True)
    enc.tau[:] = 1.5            # pedestrians reach the curb after the car's front has passed
    res = simulate(enc, _flat_profile(), np.full(P.grid.shape, P.v_limit), P)
    assert res["near"].mean() < 0.02
