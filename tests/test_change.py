import numpy as np

from cityprior.sim.change import STEP_H, detect_changes, g_statistic, memory_at, observe
from cityprior.sim.world import Profile

X = np.arange(0.0, 91.0)


def _hotspot(center: float) -> Profile:
    lam = 0.2 + 5.0 * np.exp(-0.5 * ((X - center) / 4) ** 2)
    return Profile(X, lam / np.trapezoid(lam, X) * 50 / 3600)


def test_g_statistic_zero_when_observed_equals_expected():
    k = np.array([3.0, 0.0, 5.0])
    assert np.isclose(g_statistic(k, k + 1e-12), 0.0, atol=1e-6)
    assert g_statistic(np.array([10.0, 0.0]), np.array([1.0, 9.0])) > 20


def test_detector_finds_a_moved_hotspot_quickly_with_a_camera():
    q = np.full(len(X), 0.95)
    delays = []
    for seed in range(20):
        st = observe(np.random.default_rng(seed), _hotspot(35), _hotspot(75), q, post_h=12)
        d = detect_changes(st, _hotspot(35).mean)
        delays.append(d[0] if d else np.inf)
    assert np.median(delays) <= 1.0
    assert np.isfinite(delays).all()


def test_detector_rarely_fires_without_a_change():
    q = np.full(len(X), 0.95)
    alarms = sum(len(detect_changes(observe(np.random.default_rng(100 + s), _hotspot(35), None, q, post_h=96),
                                    _hotspot(35).mean)) for s in range(10))
    assert alarms <= 1          # 40 days of checks


def test_update_strategies_after_a_change():
    before, after = _hotspot(35), _hotspot(75)
    st = observe(np.random.default_rng(1), before, after, np.full(len(X), 0.95), post_h=24)
    peak = lambda m: m.x[np.argmax(m.lam)]  # noqa: E731
    assert abs(peak(memory_at(st, 24.0, "static", before.mean)) - 35) < 5
    assert abs(peak(memory_at(st, 24.0, "forgetting", before.mean)) - 75) < 5
    det = detect_changes(st, before.mean)
    assert abs(peak(memory_at(st, 24.0, "detect", before.mean, det)) - 75) < 5
    cum = memory_at(st, 24.0, "cumulative", before.mean)
    assert cum.lam[np.argmin(np.abs(X - 35))] > 0.3 * cum.lam.max()   # still remembers the old hotspot


def test_static_memory_ignores_post_change_data():
    st = observe(np.random.default_rng(2), _hotspot(35), _hotspot(75), np.full(len(X), 0.95), post_h=4)
    a = memory_at(st, 0.0, "static", 1e-4).lam
    b = memory_at(st, 4.0, "static", 1e-4).lam
    assert np.allclose(a, b) and len(st.t0) == int((10 + 4) / STEP_H)
