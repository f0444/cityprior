"""Loading and cleaning of the TGSIM Foggy Bottom infrastructure-camera dataset.

TGSIM Foggy Bottom (FHWA / USDOT, CC0): 12 stationary 4K infrastructure cameras
covering four intersections in Washington, DC; 2 hours (15:00-17:00), 10 Hz,
all road-user classes, positions in metres in the reference-image frame
(x to the right, y downwards).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = (
    "https://data.transportation.gov/api/views/brzy-6zfh/rows.csv?accessType=DOWNLOAD"
)
DT = 0.1  # seconds between frames

RAW_COLUMNS = {
    "id": "id",
    "time": "t",
    "xloc_kf": "x",
    "yloc_kf": "y",
    "lane_kf": "region",
    "speed_kf_x": "vx",
    "speed_kf_y": "vy",
    "acceleration_kf_x": "ax",
    "acceleration_kf_y": "ay",
    "length_smoothed": "length",
    "width_smoothed": "width",
    "type_most_common": "type",
}

TYPE_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "scooter",
    3: "car",
    4: "automated_vehicle",
    5: "motorcycle",
    6: "bus",
    7: "truck",
}
TALL_TYPES = (6, 7)  # bus, truck: occlude even a roof-mounted sensor


def road_user_class(type_code: np.ndarray) -> np.ndarray:
    """Collapse the 8 dataset types into ped / micro (bike, scooter) / veh."""
    return np.select([type_code == 0, type_code <= 2], ["ped", "micro"], "veh")


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading TGSIM Foggy Bottom (~340 MB) -> {dest}")
        urllib.request.urlretrieve(DATASET_URL, dest)
    return dest


def load(raw_csv: Path, cache: Path | None = None) -> pd.DataFrame:
    """Load raw CSV (or cached parquet), rename columns, add derived fields."""
    if cache is not None and cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = pd.read_csv(raw_csv).rename(columns=RAW_COLUMNS)
        df = _derive(df)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "t"]).reset_index(drop=True)
    df["frame"] = np.round(df["t"] / DT).astype(np.int64)
    df["cls"] = road_user_class(df["type"].to_numpy())
    df["speed"] = np.hypot(df["vx"], df["vy"])
    df["heading"] = np.arctan2(df["vy"], df["vx"])
    # Vehicles with missing box size get a typical passenger-car footprint.
    veh = df["cls"] == "veh"
    df.loc[veh & (df["length"] <= 0), "length"] = 4.5
    df.loc[veh & (df["width"] <= 0), "width"] = 1.8
    return df


def stitch_fragments(
    df: pd.DataFrame, cls: str = "ped", max_gap: float = 1.0, max_dist: float = 2.0
) -> pd.DataFrame:
    """Link track fragments caused by tracker re-identification failures.

    A track that starts within `max_gap` seconds after another track of the same
    class ended, within `max_dist` metres of that track's constant-velocity
    extrapolation, is treated as its continuation. Adds column `track`.
    """
    df = df.copy()
    df["track"] = df["id"]
    sub = df[df["cls"] == cls]
    g = sub.groupby("id")
    first, last = g.head(1).set_index("id"), g.tail(1).set_index("id")

    last = last.sort_values("t")
    lt = last["t"].to_numpy()
    lx, ly = last["x"].to_numpy(), last["y"].to_numpy()
    lvx, lvy = last["vx"].to_numpy(), last["vy"].to_numpy()
    lid = last.index.to_numpy()

    parent: dict[int, int] = {}
    used: set[int] = set()
    for tid, row in first.sort_values("t").iterrows():
        lo = np.searchsorted(lt, row["t"] - max_gap)
        hi = np.searchsorted(lt, row["t"], side="left")  # strictly earlier ends: no cycles
        if hi <= lo:
            continue
        gap = row["t"] - lt[lo:hi]
        d = np.hypot(lx[lo:hi] + lvx[lo:hi] * gap - row["x"],
                     ly[lo:hi] + lvy[lo:hi] * gap - row["y"])
        cand = np.flatnonzero((d < max_dist) & (lid[lo:hi] != tid))
        cand = [c for c in cand[np.argsort(d[cand])] if lid[lo + c] not in used]
        if cand:
            prev = int(lid[lo + cand[0]])
            used.add(prev)
            parent[int(tid)] = prev

    def root(i: int) -> int:
        seen = {i}
        while i in parent and parent[i] not in seen:
            i = parent[i]
            seen.add(i)
        return i

    mapping = {tid: root(tid) for tid in parent}
    df.loc[df["cls"] == cls, "track"] = sub["id"].map(lambda i: mapping.get(i, i))
    return df.sort_values(["track", "t"]).reset_index(drop=True)


def drop_flicker(df: pd.DataFrame, min_duration: float = 1.0) -> pd.DataFrame:
    """Remove tracks shorter than `min_duration` s (mostly false detections)."""
    key = "track" if "track" in df else "id"
    dur = df.groupby(key)["t"].transform(lambda s: s.max() - s.min())
    return df[dur >= min_duration].reset_index(drop=True)


def stable_heading(df: pd.DataFrame, min_speed: float = 1.0) -> pd.Series:
    """Vehicle orientation that survives stops: the heading of the last sample
    where the vehicle moved (forward/backward filled within the track). The
    velocity heading of a stopped vehicle is noise, which would spin its footprint."""
    moving = df["speed"] >= min_speed
    h = df["heading"].where(moving)
    key = "track" if "track" in df else "id"
    h = h.groupby(df[key]).transform(lambda s: s.ffill().bfill())
    return h.fillna(df["heading"])
