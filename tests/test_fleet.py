import numpy as np
import pandas as pd

from cityprior.data import stable_heading
from cityprior.fleet import fleet_emergences, line_of_sight
from cityprior.sim.planner import estimate_memory
from cityprior.sim.world import Profile


def _frame(**cols):
    return pd.DataFrame({"frame": [0], **{k: [v] for k, v in cols.items()}})


def test_line_of_sight_blocked_by_vehicle_between():
    observer = _frame(track=1, x=0.0, y=0.0, heading=0.0, length=4.0)
    target = _frame(tid=7, x=20.0, y=0.0)
    bus = _frame(track=2, x=10.0, y=0.0, heading=0.0, length=12.0, width=2.5)
    aside = _frame(track=2, x=10.0, y=6.0, heading=0.0, length=12.0, width=2.5)
    assert len(line_of_sight(observer, target, bus)) == 0
    assert len(line_of_sight(observer, target, aside)) == 1
    far = _frame(tid=7, x=80.0, y=0.0)
    assert len(line_of_sight(observer, far, aside, sensor_range=50.0)) == 0


def test_observer_does_not_block_itself():
    observer = _frame(track=1, x=0.0, y=0.0, heading=0.0, length=4.0)
    target = _frame(tid=7, x=0.0, y=10.0)
    assert len(line_of_sight(observer, target, observer.assign(width=1.8))) == 1


def test_fleet_records_first_sighting_not_emergence():
    vis = pd.DataFrame({"frame": [50, 60], "t": [5.0, 6.0], "observer": [1, 2], "track": [9, 9],
                        "x": [3.0, 3.5], "y": [4.0, 6.0], "region": [1, 1]})
    em = pd.DataFrame({"track": [9, 10], "t": [3.0, 4.0], "x": [3.0, 8.0], "y": [1.0, 1.0], "region": [1, 1]})
    fe = fleet_emergences(vis, em, observers=np.array([1, 2]))
    assert len(fe) == 1 and fe["delay"].iloc[0] == 2.0 and np.isclose(fe["loc_error"].iloc[0], 3.0)
    assert len(fleet_emergences(vis, em, observers=np.array([2]))) == 1   # only car 2 saw it, later


def test_stable_heading_keeps_orientation_when_stopped():
    df = pd.DataFrame({"track": [1] * 4, "speed": [5.0, 0.0, 0.0, 5.0],
                       "heading": [1.0, -2.5, 3.0, 1.1]})
    h = stable_heading(df)
    assert np.allclose(h.to_numpy(), [1.0, 1.0, 1.0, 1.1])


def test_exposure_correction_removes_observability_bias():
    x = np.arange(0.0, 91.0)
    truth = Profile(x, np.full(len(x), 50 / 3600 / 90))
    q = np.where(x < 45, 0.9, 0.1)                      # fleet sees the first half much more
    rng = np.random.default_rng(0)
    naive = estimate_memory(truth, 500.0, rng, q, exposure_corrected=False).lam
    fixed = estimate_memory(truth, 500.0, rng, q, exposure_corrected=True).lam
    assert naive[:40].mean() > 3 * naive[50:].mean()     # naive thinks the watched half is busier
    assert abs(fixed[:40].mean() / fixed[50:].mean() - 1) < 0.3
