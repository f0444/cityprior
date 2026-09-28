"""DLR Urban Traffic dataset (DLR UT): one day at an instrumented intersection.

AIM Research Intersection, Braunschweig: 14 infrastructure multi-sensor systems,
trajectories at 20 Hz, all 30 traffic-light states at 1 Hz, weather, for
24 Sep 2023 (a Sunday), 00:00-24:00 UTC. CC BY-NC-SA 4.0,
doi:10.5281/zenodo.15754836.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline as PL

ROOT = PL.ROOT / "data/raw/dlr_ut/DLR-Urban-Traffic-dataset_v1-3-0"
CACHE = PL.PROC / "dlr_ut_10hz.parquet"
ORIGIN = (604600.0, 5792600.0)          # local metric frame: x east, y north
CLASSES = ("pedestrian", "bicycle", "motorbike", "car", "van", "truck")
GREEN_STATES = (5, 6)
STOP_STATES = (2, 3, 4, 7, 8)


def road_user_class(name: pd.Series) -> np.ndarray:
    return np.select([name == "pedestrian", name == "bicycle"], ["ped", "micro"], "veh")


def load(cache: Path = CACHE) -> pd.DataFrame:
    """All trajectories of the day at 10 Hz in the project's common schema."""
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    t0 = pd.Timestamp("2023-09-24 00:00:00", tz="UTC")
    for f in sorted(glob.glob(str(ROOT / "raw_data/trajectories/*.csv"))):
        d = pd.read_csv(f)
        t = (pd.to_datetime(d["timestamp"], format="ISO8601") - t0).dt.total_seconds().to_numpy()
        frame = np.round(t / 0.1).astype(np.int64)
        keep = np.abs(t - frame * 0.1) < 0.026            # 20 Hz -> 10 Hz
        d, t, frame = d[keep], t[keep], frame[keep]
        probs = d[[f"classifications_{c}" for c in CLASSES]].to_numpy()
        name = pd.Series(np.array(CLASSES)[probs.argmax(axis=1)], index=d.index)
        parts.append(pd.DataFrame({
            "track": d["id"].to_numpy(), "t": t.astype(np.float64), "frame": frame,
            "x": (d["center_easting"] - ORIGIN[0]).to_numpy(np.float32),
            "y": (d["center_northing"] - ORIGIN[1]).to_numpy(np.float32),
            "vx": d["velocity_easting"].to_numpy(np.float32), "vy": d["velocity_northing"].to_numpy(np.float32),
            "speed": d["velocity_magnitude"].to_numpy(np.float32),
            "yaw": np.deg2rad(d["yaw"].to_numpy(np.float32)),
            "length": d["dimension_length"].to_numpy(np.float32), "width": d["dimension_width"].to_numpy(np.float32),
            "type_name": name.to_numpy(), "class_conf": probs.max(axis=1).astype(np.float32),
            "interpolated": d["interpolated"].to_numpy(bool),
        }))
    df = pd.concat(parts, ignore_index=True)
    # one class per track: the most frequent per-frame argmax
    df["type_name"] = df.groupby("track")["type_name"].transform(lambda s: s.mode().iloc[0])
    df["cls"] = road_user_class(df["type_name"])
    df["heading"] = np.arctan2(df["vy"], df["vx"]).astype(np.float32)
    df = df.sort_values(["track", "t"]).drop_duplicates(["track", "frame"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_signals() -> pd.DataFrame:
    """Traffic-light states, one row per second and light id (1 Hz)."""
    t0 = pd.Timestamp("2023-09-24 00:00:00", tz="UTC")
    s = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "raw_data/traffic_lights/*.csv"))))
    s["t"] = (pd.to_datetime(s["timestamp"], format="ISO8601") - t0).dt.total_seconds()
    s["sec"] = np.floor(s["t"]).astype(int)
    return s.pivot_table(index="sec", columns="id", values="state", aggfunc="last").sort_index()


# Traffic-light heads (id, easting, northing) from the dataset documentation, Table 3.
LIGHT_HEADS = [
    (1, 604755.480, 5792814.102), (1, 604749.966, 5792812.211), (2, 604761.424, 5792815.793),
    (3, 604790.745, 5792814.817), (3, 604792.294, 5792809.095), (3, 604793.765, 5792806.195),
    (4, 604795.332, 5792803.053), (4, 604796.300, 5792798.214), (5, 604793.626, 5792779.886),
    (5, 604787.472, 5792776.873), (6, 604784.549, 5792776.031), (7, 604780.221, 5792775.418),
    (7, 604775.881, 5792774.066), (8, 604748.909, 5792774.532), (8, 604748.134, 5792780.177),
    (9, 604746.834, 5792783.266), (9, 604745.843, 5792786.123), (9, 604744.122, 5792790.989),
    (14, 604761.321, 5792815.296), (15, 604750.229, 5792812.302), (16, 604772.230, 5792818.380),
    (17, 604761.760, 5792815.453), (18, 604796.560, 5792798.306), (19, 604790.644, 5792814.821),
    (20, 604798.934, 5792789.475), (21, 604796.300, 5792798.214), (22, 604775.398, 5792773.930),
    (23, 604793.655, 5792780.032), (24, 604765.148, 5792770.611), (25, 604775.342, 5792774.082),
    (26, 604744.237, 5792790.707), (27, 604749.730, 5792774.493), (28, 604741.777, 5792800.018),
    (29, 604743.983, 5792791.443), (30, 604745.725, 5792791.731),
]


def light_heads_local() -> pd.DataFrame:
    h = pd.DataFrame(LIGHT_HEADS, columns=["id", "e", "n"])
    return h.assign(x=h["e"] - ORIGIN[0], y=h["n"] - ORIGIN[1])[["id", "x", "y"]]
