import numpy as np
import pandas as pd

from cityprior.events import _ttc_disc_model, hard_braking


def _pair(x_p, y_p, vx_p=0.0, vy_p=0.0):
    return pd.DataFrame({
        "x_v": [0.0], "y_v": [0.0], "vx_v": [10.0], "vy_v": [0.0], "heading_v": [0.0],
        "length": [4.5], "width": [1.8], "x_p": [x_p], "y_p": [y_p], "vx_p": [vx_p], "vy_p": [vy_p],
    })


def test_ttc_head_on_static_pedestrian():
    taus = np.arange(0.1, 3.0 + 1e-9, 0.1)
    # front disc centre at +1.5 m, radius 0.9 + 0.3: contact when 10 t + 2.7 >= 20 -> t = 1.73
    ttc = _ttc_disc_model(_pair(20.0, 0.0), taus, 0.3)
    assert np.isclose(ttc[0], 1.8)


def test_ttc_no_conflict_when_offset():
    taus = np.arange(0.1, 3.0 + 1e-9, 0.1)
    assert np.isnan(_ttc_disc_model(_pair(20.0, 6.0), taus, 0.3)[0])


def test_ttc_crossing_pedestrian():
    taus = np.arange(0.1, 3.0 + 1e-9, 0.1)
    # pedestrian walks into the lane from the side and meets the car around t=2 s
    ttc = _ttc_disc_model(_pair(20.0, -3.0, vy_p=1.5), taus, 0.3)
    assert 1.0 < ttc[0] < 2.5


def _track(speeds, tid=1):
    n = len(speeds)
    return pd.DataFrame({"track": tid, "t": np.arange(n) * 0.1, "x": 0.0, "y": 0.0,
                         "region": 1, "speed": speeds, "cls": "veh"})


def test_hard_braking_detects_sustained_deceleration():
    speeds = np.r_[np.full(20, 10.0), 10.0 - 0.5 * np.arange(1, 13), np.full(20, 4.0)]  # -5 m/s^2 for 1.2 s
    ev = hard_braking(_track(speeds))
    assert len(ev) == 1 and ev["a_min"].iloc[0] < -3.5


def test_hard_braking_ignores_gentle_and_constant():
    gentle = np.r_[np.full(20, 10.0), 10.0 - 0.15 * np.arange(1, 30), np.full(20, 5.65)]  # -1.5 m/s^2
    assert len(hard_braking(_track(gentle))) == 0
    assert len(hard_braking(_track(np.full(60, 8.0)))) == 0
