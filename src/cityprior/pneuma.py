"""pNEUMA: a swarm of 10 drones over downtown Athens, four weekday mornings (2018).

Barmpounakis & Geroliminis, CC BY 4.0, doi:10.5281/zenodo.10491409,
data source: pNEUMA - open-traffic.epfl.ch. Vehicles only (cars, taxis,
motorcycles, buses, medium and heavy vehicles), 25 Hz.

Each CSV row is one vehicle: track id; type; travelled distance; average speed;
then repeated groups of (lat; lon; speed km/h; lon. acc; lat. acc; time s).
Members are read straight out of the 15.8 GB Zenodo zip with HTTP range
requests, parsed, downsampled to 5 Hz and cached as parquet; raw CSVs never hit
the disk.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from . import pipeline as PL

ZIP_URL = "https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip?download=1"
CACHE_DIR = PL.PROC / "pneuma"
DAYS = ("20181024", "20181029", "20181030", "20181101")
SLOTS = ("0830_0900", "0900_0930", "0930_1000", "1000_1030")
DRONES = ("d4", "d5")
LAT0, LON0 = 37.98, 23.73                     # local tangent plane origin
EVERY = 5                                     # 25 Hz -> 5 Hz


def member(day: str, drone: str, slot: str) -> str:
    return f"pNEUMA_dataset/{day}_{drone}_{slot}.csv"


def parse(text: str, file_id: int) -> pd.DataFrame:
    rows = []
    for line in text.splitlines()[1:]:
        f = [p.strip() for p in line.split(";")]
        if len(f) < 10 or not f[0].isdigit():
            continue
        vals = pd.to_numeric(pd.Series([v for v in f[4:] if v != ""]), errors="coerce").to_numpy()
        bad = np.flatnonzero(np.isnan(vals))
        if len(bad):                                 # a few rows contain corrupted tokens: keep the valid prefix
            vals = vals[: bad[0]]
        vals = vals[: len(vals) // 6 * 6].reshape(-1, 6)[::EVERY]
        if len(vals) == 0:
            continue
        n = len(vals)
        rows.append(pd.DataFrame({
            "track": np.full(n, file_id * 100_000 + int(f[0]), dtype=np.int64), "type_name": f[1],
            "lat": vals[:, 0], "lon": vals[:, 1], "speed": vals[:, 2] / 3.6,
            "lon_acc": vals[:, 3].astype(np.float32), "t": vals[:, 5],
        }))
    df = pd.concat(rows, ignore_index=True)
    df["x"] = ((df["lon"] - LON0) * np.cos(np.deg2rad(LAT0)) * 111_320.0).astype(np.float32)
    df["y"] = ((df["lat"] - LAT0) * 110_540.0).astype(np.float32)
    df["speed"] = df["speed"].astype(np.float32)
    # heading from displacement over +-0.4 s (drone data has no orientation)
    g = df.groupby("track", sort=False)
    dx = g["x"].shift(-2) - g["x"].shift(2)
    dy = g["y"].shift(-2) - g["y"].shift(2)
    df["heading"] = np.arctan2(dy, dx).astype(np.float32)
    df["heading"] = df.groupby("track", sort=False)["heading"].transform(lambda s: s.ffill().bfill())
    return df.drop(columns=["lat", "lon"])


def fetch(days=DAYS, drones=DRONES, slots=SLOTS, log=print) -> pd.DataFrame:
    """All requested (day, drone, slot) members as one table with day/slot columns."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(d, dr, s) for d in days for dr in drones for s in slots
            if not (CACHE_DIR / f"{d}_{dr}_{s}.parquet").exists()]
    if todo:
        import fsspec
        with fsspec.open(ZIP_URL, block_size=2 ** 22).open() as fh:
            z = zipfile.ZipFile(fh)
            for d, dr, s in todo:
                file_id = DAYS.index(d) * 100 + int(dr[1:]) * 10 + SLOTS.index(s)
                with z.open(member(d, dr, s)) as m:
                    text = io.TextIOWrapper(m, encoding="utf-8").read()
                parse(text, file_id).to_parquet(CACHE_DIR / f"{d}_{dr}_{s}.parquet")
                log(f"pNEUMA {d} {dr} {s}: cached")
    parts = []
    for d in days:
        for dr in drones:
            for s in slots:
                p = pd.read_parquet(CACHE_DIR / f"{d}_{dr}_{s}.parquet")
                parts.append(p.assign(day=DAYS.index(d), drone=dr, slot=SLOTS.index(s)))
    return pd.concat(parts, ignore_index=True)
