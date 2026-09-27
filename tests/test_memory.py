import numpy as np
import pandas as pd

from cityprior.geometry import Grid
from cityprior.memory import LocationMemory, fit_prior_strength, heading_bin


def test_heading_bins_cover_circle():
    b = heading_bin(np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi - 1e-9]))
    assert list(b) == [0, 2, 4, 6, 7]


def test_empirical_bayes_pooled_rate():
    k = np.array([1, 2, 30, 0])
    n = np.array([100, 100, 100, 5])
    p, m = fit_prior_strength(k, n)
    assert np.isclose(p, 33 / 305)
    assert m > 0


def _toy_memory():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        "t": np.arange(n) * 0.1, "frame": np.arange(n), "track": np.repeat(np.arange(n // 20), 20),
        "x": rng.uniform(0, 20, n), "y": rng.uniform(0, 10, n),
        "speed": 1.2, "heading": 0.0, "cls": "ped", "region": 1,
    })
    ev = {
        "emergence": pd.DataFrame({"t": [1.0, 2.0, 3.0], "x": [2.0, 2.2, 15.0], "y": [5.0, 5.1, 5.0],
                                   "track": [0, 1, 2], "region": 1}),
        "hard_brake": pd.DataFrame({"t": [1.0], "x": [3.0], "y": [3.0], "track": [0], "region": 1}),
        "conflict": pd.DataFrame({"t": [1.0], "x": [3.0], "y": [3.0], "track_p": [0], "region": 1, "cls": "ped"}),
    }
    return LocationMemory.build(df, ev, [(0.0, n * 0.1)], Grid(0.0, 0.0, 20, 10, 1.0))


def test_emergence_density_is_a_distribution_with_hotspot():
    mem = _toy_memory()
    d = mem.emergence_density()
    assert np.isclose(d.sum(), 1.0)
    assert d[5, 2] > d[5, 10]          # the place where two people appeared
    assert (d[~mem.support] == 0).all()


def test_flow_mode_follows_observed_direction():
    mem = _toy_memory()
    direction, speed, ok = mem.flow_mode("ped", np.array([10.0]), np.array([5.0]), np.array([0.3]))
    assert ok[0] and abs(direction[0]) < 1e-6 and np.isclose(speed[0], 1.2)
