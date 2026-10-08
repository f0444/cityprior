"""inD: drone recordings of four German intersections (Aachen), several sessions per site.

Bock, Krajewski, Moers, Runde, Vater & Eckstein, "The inD Dataset", IEEE IV 2020,
doi:10.1109/IV47402.2020.9304839. Free for non-commercial use; the data may not be
redistributed (only abstract derivatives such as the statistics in results/).

All recordings of one location share one local metric frame, so a memory built on one
session can be laid over another. A session is a run of consecutive recordings with the
same weekday label at one location.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, gaussian_filter

from . import pipeline as PL

ROOT = PL.ROOT / "data/raw/inD/data"
CACHE = PL.PROC / "ind_5hz.parquet"
EVERY = 5                       # 25 Hz -> 5 Hz
CELL = 1.0                      # m
VEH_CLASSES = ("car", "truck_bus")


def recordings() -> pd.DataFrame:
    m = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "*_recordingMeta.csv"))))
    m = m.sort_values("recordingId").reset_index(drop=True)
    # sessions: consecutive recordings with the same weekday label at the same location
    new = (m["locationId"] != m["locationId"].shift()) | (m["weekday"] != m["weekday"].shift())
    m["session"] = new.cumsum() - 1
    return m


def load(cache=CACHE) -> pd.DataFrame:
    """All tracks at 5 Hz: recording, track, class, frame, t (s), x, y (m)."""
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for f in sorted(glob.glob(str(ROOT / "*_tracks.csv"))):
        rid = int(f.split("/")[-1][:2])
        tr = pd.read_csv(f, usecols=["recordingId", "trackId", "frame", "xCenter", "yCenter"])
        tr = tr[tr["frame"] % EVERY == 0]
        meta = pd.read_csv(f.replace("_tracks.csv", "_tracksMeta.csv"), usecols=["trackId", "class"])
        tr = tr.merge(meta, on="trackId")
        parts.append(pd.DataFrame({"recording": rid, "track": tr["trackId"].to_numpy(), "cls": tr["class"].to_numpy(),
                                   "frame": tr["frame"].to_numpy(), "t": tr["frame"].to_numpy() / 25.0,
                                   "x": tr["xCenter"].astype(np.float32).to_numpy(),
                                   "y": tr["yCenter"].astype(np.float32).to_numpy()}))
    df = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


class SiteGrid:
    """1 m grid over one location."""

    def __init__(self, x: np.ndarray, y: np.ndarray, pad: float = 5.0):
        self.x0, self.y0 = np.floor(x.min() - pad), np.floor(y.min() - pad)
        self.nx = int(np.ceil((x.max() + pad - self.x0) / CELL))
        self.ny = int(np.ceil((y.max() + pad - self.y0) / CELL))

    def cells(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip(((np.asarray(x) - self.x0) // CELL).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) // CELL).astype(int), 0, self.ny - 1)
        return iy, ix


def roadway(site: pd.DataFrame, grid: SiteGrid, min_vehicles: int = 3, radius: float = 1.0) -> np.ndarray:
    """Cells within `radius` of the centres of at least `min_vehicles` distinct vehicles (all sessions:
    the road is geometry, not behaviour)."""
    v = site[site["cls"].isin(VEH_CLASSES)]
    seen = np.zeros((grid.ny, grid.nx), dtype=np.int32)
    r = int(np.ceil(radius / CELL))
    for _, g in v.groupby(["recording", "track"]):
        iy, ix = grid.cells(g["x"], g["y"])
        hit = np.zeros_like(seen, dtype=bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    hit[np.clip(iy + dy, 0, grid.ny - 1), np.clip(ix + dx, 0, grid.nx - 1)] = True
        seen += hit
    return binary_closing(seen >= min_vehicles, np.ones((3, 3)))


def road_entries(site: pd.DataFrame, grid: SiteGrid, road: np.ndarray, min_off_s: float = 1.0) -> pd.DataFrame:
    """Pedestrians stepping onto the roadway: off-road for at least min_off_s, then on it."""
    ped = site[site["cls"] == "pedestrian"].sort_values(["recording", "track", "frame"])
    rows = []
    need = int(round(min_off_s * 25 / EVERY))
    for (rec, tid), g in ped.groupby(["recording", "track"]):
        iy, ix = grid.cells(g["x"].to_numpy(), g["y"].to_numpy())
        on = road[iy, ix]
        off_run = 0
        for k in range(len(on)):
            if on[k] and off_run >= need:
                rows.append((rec, tid, float(g["t"].iloc[k]), float(g["x"].iloc[k]), float(g["y"].iloc[k])))
            off_run = 0 if on[k] else off_run + 1
    return pd.DataFrame(rows, columns=["recording", "track", "t", "x", "y"])


def density(entries: pd.DataFrame, grid: SiteGrid, road: np.ndarray, sigma_m: float = 1.5,
            uniform_share: float = 0.1) -> np.ndarray:
    """Memory: smoothed entry counts on the roadway, mixed with a little uniform so nothing is impossible."""
    c = np.zeros((grid.ny, grid.nx))
    if len(entries):
        iy, ix = grid.cells(entries["x"].to_numpy(), entries["y"].to_numpy())
        np.add.at(c, (iy, ix), 1.0)
    s = gaussian_filter(c, sigma_m / CELL) * road
    u = road / road.sum()
    if s.sum() <= 0:
        return u
    return (1 - uniform_share) * s / s.sum() + uniform_share * u


def score(p: np.ndarray, road: np.ndarray, test: pd.DataFrame, grid: SiteGrid, top: float = 0.10) -> dict:
    """Information gain over a uniform prior on the roadway (bits per entry) and the share of entries in
    the top `top` of road area ranked by p."""
    iy, ix = grid.cells(test["x"].to_numpy(), test["y"].to_numpy())
    u = 1.0 / road.sum()
    pv = p[iy, ix]
    ok = road[iy, ix]
    gain = np.log2(np.where(ok, pv, u) / u)
    thr = np.quantile(p[road], 1 - top)
    return {"gain": gain, "in_top": (pv >= thr) & ok}
