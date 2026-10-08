import numpy as np
import pandas as pd

from cityprior import ind as D


def site():
    """A 20 m east-west road (y in [0, 6)) with cars on it and pedestrians crossing at x = 10."""
    rows = []
    for car in range(5):
        for k, x in enumerate(np.linspace(-20, 40, 61)):
            rows.append((0, car, "car", k, k / 5, x, 3.0))
    for p in range(8):
        for k, y in enumerate(np.linspace(-6, 12, 37)):
            rows.append((0, 100 + p, "pedestrian", k, k / 5, 10.0 + 0.2 * p, y))
    return pd.DataFrame(rows, columns=["recording", "track", "cls", "frame", "t", "x", "y"])


def test_road_and_entries():
    s = site()
    g = D.SiteGrid(s["x"].to_numpy(), s["y"].to_numpy())
    road = D.roadway(s, g)
    iy, ix = g.cells([5.0, 5.0], [3.0, 10.0])
    assert road[iy[0], ix[0]] and not road[iy[1], ix[1]]
    e = D.road_entries(s, g, road)
    assert len(e) == 8                                   # each pedestrian steps onto the road once
    assert np.allclose(e["x"], 10 + 0.2 * np.arange(8)) and (e["y"] < 3).all()


def test_memory_beats_uniform_where_people_cross():
    s = site()
    g = D.SiteGrid(s["x"].to_numpy(), s["y"].to_numpy())
    road = D.roadway(s, g)
    e = D.road_entries(s, g, road)
    p = D.density(e.iloc[:4], g, road)
    r = D.score(p, road, e.iloc[4:], g)
    assert (r["gain"] > 2).all() and r["in_top"].all()
