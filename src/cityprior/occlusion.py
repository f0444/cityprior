"""'Virtual robotaxi' analysis on real traffic: how often is a pedestrian who is on
or heading into a vehicle's path hidden from that vehicle by other vehicles,
while the infrastructure cameras see it?

Every moving vehicle in the dataset is treated in turn as a virtual ego. Line of
sight is a 2-D ray from the ego's roof-front to the pedestrian, blocked by other
vehicles' footprints. Two occluder sets bound reality:
  * all vehicles     - upper bound (a low car does not fully block a roof sensor),
  * buses and trucks - lower bound (these block a roof sensor too).
Buildings are not modelled (no map), which biases occlusion *down*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DT, TALL_TYPES, stable_heading
from .geometry import segment_hits_boxes


def virtual_ego_occlusion(
    df: pd.DataFrame,
    every_frames: int = 2,
    ego_min_speed: float = 1.5,
    look_ahead_m: float = 40.0,
    path_half_width: float = 2.5,
    path_horizon_s: float = 4.0,
    occluder_range: float = 45.0,
) -> pd.DataFrame:
    """Rows: (ego, pedestrian, frame) where the pedestrian is ahead of the ego and
    on / heading into its straight-line path, with occlusion flags."""
    df = df.assign(heading=stable_heading(df))       # parked / queued cars keep their orientation
    sampled = df[df["frame"] % every_frames == 0]
    ego = sampled[(sampled["cls"] == "veh") & (sampled["speed"] >= ego_min_speed)]
    ego = ego[["frame", "track", "x", "y", "heading", "length", "speed"]]
    ped = sampled.loc[sampled["cls"] == "ped", ["frame", "track", "x", "y", "vx", "vy"]]
    veh = sampled.loc[sampled["cls"] == "veh", ["frame", "track", "x", "y", "heading", "length", "width", "type"]]

    pairs = ego.merge(ped, on="frame", suffixes=("_e", "_p"))
    c, s = np.cos(pairs["heading"]), np.sin(pairs["heading"])
    sx = pairs["x_e"] + c * pairs["length"] / 4          # sensor near the roof front
    sy = pairs["y_e"] + s * pairs["length"] / 4
    dx, dy = pairs["x_p"] - sx, pairs["y_p"] - sy
    lon = dx * c + dy * s
    lat = -dx * s + dy * c
    vlat = -pairs["vx"] * s + pairs["vy"] * c
    # closest lateral offset reached within the horizon under constant velocity
    t_star = np.clip(-lat / vlat.where(vlat.abs() > 1e-6, np.nan), 0, path_horizon_s).fillna(0)
    lat_min = (lat + vlat * t_star).abs()
    keep = (lon > 0) & (lon < look_ahead_m) & (lat_min <= path_half_width)
    pairs = pairs[keep].assign(sx=sx[keep], sy=sy[keep], lon=lon[keep], lat=lat[keep])
    pairs = pairs.reset_index(drop=True).reset_index(names="pair")

    occ = pairs[["pair", "frame", "track_e", "sx", "sy", "x_p", "y_p"]].merge(veh, on="frame")
    occ = occ[(occ["track"] != occ["track_e"])
              & (np.hypot(occ["x"] - occ["sx"], occ["y"] - occ["sy"]) < occluder_range)]
    hit = segment_hits_boxes(
        occ[["sx", "sy"]].to_numpy(), occ[["x_p", "y_p"]].to_numpy(),
        occ[["x", "y"]].to_numpy(), occ["heading"].to_numpy(),
        occ["length"].to_numpy(), occ["width"].to_numpy(),
    )
    occ = occ.assign(hit=hit, hit_tall=hit & occ["type"].isin(TALL_TYPES).to_numpy())
    flags = occ.groupby("pair")[["hit", "hit_tall"]].any()
    pairs["occluded_all"] = pairs["pair"].map(flags["hit"]).fillna(False).astype(bool)
    pairs["occluded_tall"] = pairs["pair"].map(flags["hit_tall"]).fillna(False).astype(bool)
    pairs["t"] = pairs["frame"] * DT
    return pairs[["t", "frame", "track_e", "track_p", "lon", "lat", "speed", "occluded_all", "occluded_tall"]]


def encounter_lead_times(obs: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Per (ego, pedestrian) encounter: how much earlier the infrastructure knew
    about the pedestrian than the ego's own line of sight allowed.

    lead_s = first unoccluded sighting - first moment the pedestrian was
    path-relevant (censored at encounter end if never seen by the ego)."""
    obs = obs.sort_values("t")
    g = obs.groupby(["track_e", "track_p"])
    first = g["t"].min()
    last = g["t"].max()
    seen = obs[~obs[flag]].groupby(["track_e", "track_p"])["t"].min()
    out = pd.DataFrame({"t_first": first, "t_last": last})
    out["t_seen"] = seen.reindex(out.index)
    out["never_seen"] = out["t_seen"].isna()
    out["lead_s"] = out["t_seen"].fillna(out["t_last"]) - out["t_first"]
    out["lon_first"] = g["lon"].first()
    return out.reset_index()
