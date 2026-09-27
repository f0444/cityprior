"""Safety-relevant events mined from infrastructure trajectories.

These are the raw material of the location memory:
  * hard braking of vehicles,
  * pedestrian emergence (where pedestrians first appear on the roadway),
  * vehicle-VRU conflicts measured with time-to-collision (TTC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DT


def hard_braking(
    df: pd.DataFrame, decel: float = -3.5, min_speed: float = 3.0, min_duration: float = 0.5
) -> pd.DataFrame:
    """Onsets of sustained strong deceleration of vehicles.

    Acceleration is the derivative of the 0.5 s-smoothed speed, smoothed again;
    the raw Kalman accelerations contain spikes up to ~28 m/s^2.
    """
    v = df.loc[df["cls"] == "veh", ["track", "t", "x", "y", "region", "speed"]].copy()
    g = v.groupby("track", sort=False)
    v["spd_s"] = g["speed"].transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    v["acc"] = g["spd_s"].transform(
        lambda s: np.gradient(s.to_numpy(), DT) if len(s) > 1 else np.zeros(len(s))
    )
    v["acc"] = v.groupby("track", sort=False)["acc"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).mean()
    )
    braking = v["acc"] <= decel
    run_id = (braking != braking.groupby(v["track"]).shift(fill_value=False)).cumsum()
    runs = v[braking].groupby(run_id[braking])
    ev = runs.agg(
        track=("track", "first"), t=("t", "first"), x=("x", "first"), y=("y", "first"),
        region=("region", "first"), v0=("spd_s", "first"), a_min=("acc", "min"),
        n=("t", "size"),
    )
    ev = ev[(ev["n"] * DT >= min_duration) & (ev["v0"] >= min_speed)]
    return ev.drop(columns="n").reset_index(drop=True)


def pedestrian_emergence(df: pd.DataFrame, t_start: float, margin: float = 1.0) -> pd.DataFrame:
    """First observation of every (stitched) pedestrian track = where a pedestrian
    entered the monitored roadway (crosswalk, lane, between parked cars...).
    Tracks already present when the recording starts are excluded."""
    p = df[df["cls"] == "ped"].sort_values("t")
    first = p.groupby("track").head(1)
    first = first[first["t"] > t_start + margin]
    return first[["track", "t", "x", "y", "region"]].reset_index(drop=True)


def _ttc_disc_model(pairs: pd.DataFrame, taus: np.ndarray, vru_radius: float) -> np.ndarray:
    """First time (s) at which a pedestrian disc touches the vehicle, modelled as
    three discs along its heading, under constant-velocity extrapolation.
    NaN = no contact within the horizon or already overlapping at t=0."""
    ux, uy = np.cos(pairs["heading_v"].to_numpy()), np.sin(pairs["heading_v"].to_numpy())
    L, W = pairs["length"].to_numpy(), pairs["width"].to_numpy()
    offsets = np.stack([-L / 3, np.zeros_like(L), L / 3], axis=1)          # (n, 3)
    r = W / 2 + vru_radius                                                   # (n,)

    t = np.concatenate([[0.0], taus])[None, :, None]                        # (1, T, 1)
    rx = (pairs["x_p"] - pairs["x_v"]).to_numpy()[:, None, None]
    ry = (pairs["y_p"] - pairs["y_v"]).to_numpy()[:, None, None]
    rvx = (pairs["vx_p"] - pairs["vx_v"]).to_numpy()[:, None, None]
    rvy = (pairs["vy_p"] - pairs["vy_v"]).to_numpy()[:, None, None]
    dx = rx + rvx * t - offsets[:, None, :] * ux[:, None, None]
    dy = ry + rvy * t - offsets[:, None, :] * uy[:, None, None]
    hit = (np.hypot(dx, dy) < r[:, None, None]).any(axis=2)                  # (n, T)

    ttc = np.full(len(pairs), np.nan)
    first = hit[:, 1:].argmax(axis=1)
    any_hit = hit[:, 1:].any(axis=1) & ~hit[:, 0]
    ttc[any_hit] = taus[first[any_hit]]
    return ttc


def vru_vehicle_conflicts(
    df: pd.DataFrame,
    ttc_threshold: float = 1.5,
    horizon: float = 3.0,
    max_range: float = 30.0,
    min_vehicle_speed: float = 2.0,
    vru_radius: float = 0.3,
    chunk_frames: int = 1500,
) -> pd.DataFrame:
    """Vehicle-VRU conflict episodes with TTC <= threshold.

    Returns one row per (vehicle, VRU) episode with the minimum TTC, the time and
    VRU position at that moment, and the vehicle speed.
    """
    veh = df.loc[(df["cls"] == "veh") & (df["speed"] >= min_vehicle_speed),
                 ["frame", "track", "x", "y", "vx", "vy", "heading", "length", "width", "speed"]]
    veh = veh.rename(columns={"heading": "heading_v"})
    vru = df.loc[df["cls"].isin(["ped", "micro"]),
                 ["frame", "track", "x", "y", "vx", "vy", "region", "cls"]]
    taus = np.arange(0.1, horizon + 1e-9, 0.1)
    hits = []
    f0, f1 = int(df["frame"].min()), int(df["frame"].max())
    for start in range(f0, f1 + 1, chunk_frames):
        vv = veh[(veh["frame"] >= start) & (veh["frame"] < start + chunk_frames)]
        pp = vru[(vru["frame"] >= start) & (vru["frame"] < start + chunk_frames)]
        pairs = vv.merge(pp, on="frame", suffixes=("_v", "_p"))
        pairs = pairs[np.hypot(pairs["x_p"] - pairs["x_v"], pairs["y_p"] - pairs["y_v"]) < max_range]
        if pairs.empty:
            continue
        ttc = _ttc_disc_model(pairs, taus, vru_radius)
        keep = ttc <= ttc_threshold
        if keep.any():
            h = pairs.loc[keep, ["frame", "track_v", "track_p", "x_p", "y_p", "region", "cls", "speed"]].copy()
            h["ttc"] = ttc[keep]
            hits.append(h)
    if not hits:
        return pd.DataFrame(columns=["track_v", "track_p", "t", "x", "y", "region", "cls", "ttc", "v_speed"])
    h = pd.concat(hits, ignore_index=True).sort_values(["track_v", "track_p", "frame"])
    h = h.reset_index(drop=True)
    new_ep = (h.groupby(["track_v", "track_p"])["frame"].diff().fillna(np.inf) > 3)
    h["episode"] = new_ep.cumsum()
    idx = h.groupby("episode")["ttc"].idxmin()
    ev = h.loc[idx].rename(columns={"x_p": "x", "y_p": "y", "speed": "v_speed"})
    ev["t"] = ev["frame"] * DT
    return ev[["track_v", "track_p", "t", "x", "y", "region", "cls", "ttc", "v_speed"]].reset_index(drop=True)
