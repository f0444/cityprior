import numpy as np
import pytest

from cityprior import econ as M


def fake_effects():
    rows = [{"ceiling": c, "reference": {"collisions_per_10k": 2.0},
             "sources": {"memory": {"seconds_saved_at_equal_risk": s, "reduction_at_equal_time": r},
                         "live camera, trusted": {"seconds_saved_at_equal_risk": 3.5, "reduction_at_equal_time": 1.0},
                         "live camera, adds caution only": {"seconds_saved_at_equal_risk": 2.0,
                                                            "reduction_at_equal_time": 1.0}}}
            for c, s, r in ((0.5, 2.0, 0.5), (1.0, 0.0, 0.0))]
    return M.Effects({"rows": rows, "street_len_m": 90.0})


def test_annuity_matches_closed_form():
    P = {**M.base_params(), "discount_rate": 0.07, "horizon_years": 10}
    assert M.annuity(P) == pytest.approx((1 - 1.07 ** -10) / 0.07)


def test_effects_interpolate_in_ceiling():
    E = fake_effects()
    assert E.seconds("memory", 0.75) == pytest.approx(1.0)
    assert E.reduction("memory", 1.0) == 0.0


def test_stale_hours_hits_anchors_and_falls_with_passes():
    for n, t in M.STALE_ANCHORS:
        assert M.stale_hours(n) == pytest.approx(t, rel=1e-6)
    hours = [M.stale_hours(n) for n in (1, 3, 10, 30, 100, 300)]
    assert all(a >= b for a, b in zip(hours, hours[1:]))
    assert M.stale_hours(0.1, max_hours=500) == 500


def test_observability_fits_part3():
    assert M.observability(18.4) == pytest.approx(0.26, abs=0.01)
    assert M.observability(5.5) == pytest.approx(0.084, abs=0.005)


def test_live_npv_linear_in_volume_and_break_even_consistent():
    E = fake_effects()
    P = M.base_params()
    a, b, c = (M.npv("live trusted", n, 0.75, P, E) for n in (1000, 2000, 3000))
    assert b - a == pytest.approx(c - b)
    be = M.break_even("live trusted", 0.75, P, E)
    assert M.npv("live trusted", be * 1.01, 0.75, P, E) > 0 > M.npv("live trusted", be * 0.99, 0.75, P, E)


def test_camera_memory_adds_nothing_when_fleet_is_as_fast():
    E = fake_effects()
    P = M.base_params()
    assert M.camera_memory_increment(184 * P["active_hours"], 0.75, P, E) == 0.0
    more_fleets = M.camera_memory_increment(300, 0.75, {**P, "operators": 3}, E)
    assert more_fleets > M.camera_memory_increment(300, 0.75, P, E) > 0


def test_sampled_params_stay_in_range():
    for P in M.sample_params(np.random.default_rng(1), 50):
        for k, v in M.PARAMS.items():
            assert min(v.low, v.high) - 1e-9 <= P[k] <= max(v.low, v.high) + 1e-9
