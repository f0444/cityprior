import numpy as np
import pandas as pd

from cityprior import ind_predict as P


def walkers(turn: bool, session: int, n: int = 6):
    """Pedestrians walking east at 1.25 m/s from x = 0; with turn=True they turn north at x = 10."""
    rows = []
    for i in range(n):
        x, y = 0.0, 0.2 * i
        for k in range(60):
            rows.append((session, i, "pedestrian", k * 5, k / 5, x, y))
            if turn and x >= 10:
                y += 0.25
            else:
                x += 0.25
    df = pd.DataFrame(rows, columns=["recording", "track", "cls", "frame", "t", "x", "y"])
    return df


def meta(sessions):
    return pd.DataFrame({"recordingId": sessions, "locationId": [1] * len(sessions), "session": sessions})


def test_anchor_targets_of_a_straight_walker():
    a = P.anchors(walkers(False, 0), meta([0]))
    assert len(a) > 0
    assert np.allclose(a["f15_x"], 3.75) and np.allclose(a["f15_y"], 0, atol=1e-9)
    assert np.allclose(a["speed"], 1.25) and not a["manoeuvre"].any()


def test_memory_predicts_the_turn_that_kinematics_cannot_see():
    old, new = P.anchors(walkers(True, 0), meta([0])), P.anchors(walkers(True, 1), meta([1]))
    before_turn = new[(new["x"] >= 8) & (new["x"] < 10)]
    m = P.memory_features(before_turn, old)
    # straight-ahead constant velocity misses the turn; memory says "they go left" (positive agent-frame y)
    assert (m["m1_15_y"] > 0.5).all() and (m["m1_15_x"] < 0).all()
