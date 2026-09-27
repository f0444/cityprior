"""Fleet memory vs infrastructure memory, on real traffic.

A fleet memory (crowd-sourced from vehicles, as in Mobileye REM or CHAMP) only
knows what passing cars happened to see: a pedestrian is recorded if some
equipped car is within sensor range with an unobstructed line of sight, and it
is recorded *where and when that car first saw it*, not where it stepped out.
Here every real vehicle in TGSIM is treated as a potential fleet car; a random
share of them (the penetration) is equipped.

    python -m cityprior.fleet      # real-data comparison + simulator (E7); writes results/fleet_*
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from . import pipeline as PL
from .data import DT, stable_heading
from .geometry import Grid, segment_hits_boxes
from .memory import LocationMemory

SENSOR_RANGE = 50.0
PENETRATIONS = (1.0, 0.3, 0.1, 0.03)
N_DRAWS = 5
TOP_STREET = dict(x=(70.0, 160.0), edges_y=(34.5, 41.5))   # mid-block of the top street (sim calibration)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def driving_vehicles(df: pd.DataFrame, min_speed: float = 2.0) -> np.ndarray:
    """Vehicles that actually drive (a parked car is not a data-collecting fleet car)."""
    v = df[df["cls"] == "veh"]
    vmax = v.groupby("track")["speed"].max()
    return vmax[vmax > min_speed].index.to_numpy()


def line_of_sight(observers: pd.DataFrame, targets: pd.DataFrame, occluders: pd.DataFrame,
                  sensor_range: float = SENSOR_RANGE, chunk_frames: int = 1500) -> pd.DataFrame:
    """(frame, observer, target) pairs within range whose 2-D sight line from the
    observer's roof sensor is not blocked by another vehicle's footprint.

    observers: frame, track, x, y, heading, length   (sensor at L/4 ahead of centre)
    targets:   frame, tid, x, y (+ any extra columns, carried through)
    occluders: frame, track, x, y, heading, length, width
    """
    out = []
    frames = np.unique(targets["frame"])
    for start in range(int(frames.min()), int(frames.max()) + 1, chunk_frames):
        sel = lambda d: d[(d["frame"] >= start) & (d["frame"] < start + chunk_frames)]  # noqa: E731
        ob, tg, oc = sel(observers), sel(targets), sel(occluders)
        if ob.empty or tg.empty:
            continue
        ob = ob.assign(sx=ob["x"] + np.cos(ob["heading"]) * ob["length"] / 4,
                       sy=ob["y"] + np.sin(ob["heading"]) * ob["length"] / 4)
        pairs = ob[["frame", "track", "sx", "sy"]].merge(tg, on="frame")
        pairs = pairs[np.hypot(pairs["x"] - pairs["sx"], pairs["y"] - pairs["sy"]) < sensor_range]
        if pairs.empty:
            continue
        pairs = pairs.reset_index(drop=True).reset_index(names="pair")
        occ = pairs[["pair", "frame", "track", "sx", "sy", "x", "y"]].merge(
            oc.rename(columns={"track": "occ", "x": "ox", "y": "oy"}), on="frame")
        # other vehicles near the sight line's bounding box
        lo_x = np.minimum(occ["sx"], occ["x"]) - 8
        hi_x = np.maximum(occ["sx"], occ["x"]) + 8
        lo_y = np.minimum(occ["sy"], occ["y"]) - 8
        hi_y = np.maximum(occ["sy"], occ["y"]) + 8
        occ = occ[(occ["occ"] != occ["track"]) & occ["ox"].between(lo_x, hi_x) & occ["oy"].between(lo_y, hi_y)]
        hit = segment_hits_boxes(occ[["sx", "sy"]].to_numpy(), occ[["x", "y"]].to_numpy(),
                                 occ[["ox", "oy"]].to_numpy(), occ["heading"].to_numpy(),
                                 occ["length"].to_numpy(), occ["width"].to_numpy())
        blocked = pd.Series(hit, index=occ["pair"].to_numpy()).groupby(level=0).any()
        pairs = pairs[~pairs["pair"].map(blocked).fillna(False).astype(bool)]
        out.append(pairs.drop(columns=["pair", "sx", "sy"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fleet_sightings(df: pd.DataFrame, every_frames: int = 5) -> pd.DataFrame:
    """Every (frame, driving vehicle, pedestrian) sighting."""
    df = df.assign(heading=stable_heading(df))
    s = df[df["frame"] % every_frames == 0]
    veh = s[s["cls"] == "veh"]
    observers = veh[veh["track"].isin(driving_vehicles(df))][["frame", "track", "x", "y", "heading", "length"]]
    targets = s.loc[s["cls"] == "ped", ["frame", "track", "x", "y", "region"]].rename(columns={"track": "tid"})
    occluders = veh[["frame", "track", "x", "y", "heading", "length", "width"]]
    vis = line_of_sight(observers, targets, occluders)
    vis = vis.rename(columns={"track": "observer", "tid": "track"})
    vis["t"] = vis["frame"] * DT
    return vis.sort_values("t").reset_index(drop=True)


def fleet_emergences(vis: pd.DataFrame, emergence: pd.DataFrame, observers: np.ndarray | None) -> pd.DataFrame:
    """What a fleet records as 'a pedestrian appeared here': the first sighting of
    each pedestrian by any equipped car. Joined with the true emergence for delay
    and localisation error."""
    v = vis if observers is None else vis[vis["observer"].isin(observers)]
    first = v.groupby("track").head(1)[["track", "t", "x", "y", "region"]]
    out = first.merge(emergence.rename(columns={"t": "t_true", "x": "x_true", "y": "y_true", "region": "region_true"}),
                      on="track", how="inner")
    out["delay"] = out["t"] - out["t_true"]
    out["loc_error"] = np.hypot(out["x"] - out["x_true"], out["y"] - out["y_true"])
    return out


def edge_observability(df: pd.DataFrame, every_frames: int = 20) -> pd.DataFrame:
    """For points along the parking-lane edges of the top street: which driving
    cars could see the point in each sampled frame (for the simulator's q(x))."""
    df = df.assign(heading=stable_heading(df))
    s = df[df["frame"] % every_frames == 0]
    x0, x1 = TOP_STREET["x"]
    xs = np.arange(x0, x1 + 1e-9, 1.0)
    pts = pd.DataFrame([(xv, yv) for yv in TOP_STREET["edges_y"] for xv in xs], columns=["x", "y"])
    pts["tid"] = np.arange(len(pts))
    frames = np.unique(s["frame"])
    targets = pts.assign(key=1).merge(pd.DataFrame({"frame": frames, "key": 1}), on="key").drop(columns="key")
    veh = s[s["cls"] == "veh"]
    near = veh["x"].between(x0 - SENSOR_RANGE, x1 + SENSOR_RANGE) & veh["y"].between(0, 90)
    observers = veh[near & veh["track"].isin(driving_vehicles(df))][["frame", "track", "x", "y", "heading", "length"]]
    occluders = veh[near][["frame", "track", "x", "y", "heading", "length", "width"]]
    vis = line_of_sight(observers, targets, occluders)
    vis.attrs["n_frames"] = len(frames)
    vis.attrs["points"] = pts
    return vis


def observability_profile(vis: pd.DataFrame, observers: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Share of time each metre of the block is seen by at least one equipped car
    (mean of both kerbs); x mapped to the simulator frame [0, 90]."""
    pts, n = vis.attrs["points"], vis.attrs["n_frames"]
    v = vis if observers is None else vis[vis["track"].isin(observers)]
    seen = v.drop_duplicates(["frame", "tid"]).groupby("tid").size().reindex(pts["tid"], fill_value=0) / n
    q = pts.assign(q=seen.to_numpy()).groupby("x")["q"].mean()
    return q.index.to_numpy() - TOP_STREET["x"][0], q.to_numpy()


def _memory_with(df, events, fleet_ev, spans, grid) -> LocationMemory:
    ev = dict(events)
    ev["emergence"] = fleet_ev[["track", "t", "x", "y", "region"]]
    return LocationMemory.build(df, ev, spans, grid)


def real_data_comparison(df, events, vis, grid) -> dict:
    em = events["emergence"]
    rtype = LocationMemory.build(df, events, [(PL.T_START, PL.T_SPLIT)], grid).regions.set_index("region")["type"]
    em = em.assign(kind=em["region"].map(rtype).fillna("roadway"))
    test = em[em["t"] >= PL.T_SPLIT]
    train_spans = [(PL.T_START, PL.T_SPLIT)]
    drivers = driving_vehicles(df)
    rng = np.random.default_rng(0)

    infra_mem = LocationMemory.build(df, events, train_spans, grid)
    res = {"infrastructure": {
        "all": _scores(infra_mem, test), "off_crosswalk": _scores(infra_mem, test[test["kind"] == "roadway"]),
    }}
    res["fleet"] = []
    for pen in PENETRATIONS:
        draws = 1 if pen == 1.0 else N_DRAWS
        rows = []
        for _ in range(draws):
            obs = drivers if pen == 1.0 else rng.choice(drivers, max(1, int(round(pen * len(drivers)))), replace=False)
            fe = fleet_emergences(vis, em, obs)
            fe_train = fe[fe["t"] < PL.T_SPLIT]          # the fleet only knows its own sighting time
            mem = _memory_with(df, events, fe_train, train_spans, grid)
            train = em[em["t"] < PL.T_SPLIT]
            seen = train["track"].isin(fe_train["track"])
            prompt = train["track"].isin(fe_train.loc[fe_train["delay"] <= 2.0, "track"])
            rows.append({
                "share_seen": float(seen.mean()),
                "share_seen_within_2s": float(prompt.mean()),
                "share_seen_off_crosswalk": float(seen[train["kind"] == "roadway"].mean()),
                "share_seen_crosswalk": float(seen[train["kind"] == "crosswalk"].mean()),
                "median_delay_s": float(fe_train["delay"].median()),
                "median_loc_error_m": float(fe_train["loc_error"].median()),
                "all": _scores(mem, test),
                "off_crosswalk": _scores(mem, test[test["kind"] == "roadway"]),
            })
        res["fleet"].append({"penetration": pen, "draws": rows})
        m = rows
        log(f"fleet {pen:>4}: seen {np.mean([r['share_seen'] for r in m]):.2f} "
            f"(off-crosswalk {np.mean([r['share_seen_off_crosswalk'] for r in m]):.2f}), "
            f"delay {np.mean([r['median_delay_s'] for r in m]):.1f}s, err {np.mean([r['median_loc_error_m'] for r in m]):.1f}m, "
            f"gain {np.mean([r['all']['gain'] for r in m]):.2f} bits (infra {res['infrastructure']['all']['gain']:.2f}), "
            f"off-xwalk {np.mean([r['off_crosswalk']['gain'] for r in m]):.2f} (infra {res['infrastructure']['off_crosswalk']['gain']:.2f})")

    # learning curves: memory from the last k minutes before the test window
    res["learning_curve"] = []
    for minutes in (5, 10, 20, 40, 79):
        spans = [(max(PL.T_START, PL.T_SPLIT - 60 * minutes), PL.T_SPLIT)]
        row = {"minutes": minutes,
               "infrastructure": _scores(LocationMemory.build(df, events, spans, grid), test)}
        for pen in (1.0, 0.1):
            gains = []
            for d in range(1 if pen == 1.0 else N_DRAWS):
                obs = drivers if pen == 1.0 else np.random.default_rng(50 + d).choice(
                    drivers, int(round(pen * len(drivers))), replace=False)
                fe = fleet_emergences(vis, em, obs)
                fe = fe[(fe["t"] >= spans[0][0]) & (fe["t"] < PL.T_SPLIT)]
                gains.append(_scores(_memory_with(df, events, fe, spans, grid), test)["gain"])
            row[f"fleet_{pen}"] = float(np.mean(gains))
        res["learning_curve"].append(row)
        log(f"learning curve {minutes} min: infra {row['infrastructure']['gain']:.2f}, "
            f"fleet 100% {row['fleet_1.0']:.2f}, fleet 10% {row['fleet_0.1']:.2f} bits")
    return res


def _scores(mem: LocationMemory, test: pd.DataFrame) -> dict:
    e = E.emergence_eval(mem, test)
    return {"gain": e["info_gain_bits"][0], "gain_ci": e["info_gain_bits"][1:],
            "capture_10pct": e["capture_at_10pct_area"], "n": e["n_test_events"]}


# ----------------------------------------------------------------- simulator
def simulator_comparison(q_profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """E7: collisions at equal trip time vs calendar hours of history, for memory
    from a fixed camera vs from fleets of different penetration."""
    from .sim.experiments import V_NEAR, V_OPERATING, Bench, risk_at_time
    from .sim.planner import believed_intensity, estimate_memory
    from .sim.world import Profile, SimParams, ped_speed_quantiles, profile_from_tgsim

    p = SimParams()
    df, events = PL.prepare()
    truth = profile_from_tgsim(events["emergence"], street_len=p.street_len)
    mid = events["emergence"][events["emergence"]["x"].between(70, 160) & events["emergence"]["y"].between(28, 50)]
    bench = Bench(p, truth, ped_speed_quantiles(df, mid["track"]))
    flat = Profile(truth.x, np.full(len(truth.x), truth.mean))
    ref = bench.run(believed_intensity(flat, p), V_OPERATING)
    oracle = risk_at_time(bench.curve(believed_intensity(truth, p), v_refs=V_NEAR), ref["trip_time"])
    sources = {"infrastructure camera": (0.95, True)}
    for name, (xq, q) in q_profiles.items():
        qx = np.interp(truth.x, xq, q)
        sources[f"{name} (exposure-corrected)"] = (qx, True)
        if name == "fleet 10%":
            sources[f"{name} (naive)"] = (qx, False)
    res = {"reference_risk": ref["collisions_per_10k"], "reference_time": ref["trip_time"],
           "oracle_risk": oracle, "sources": {}}
    for name, (q, corrected) in sources.items():
        rows = []
        for hours in (1.0, 4.0, 16.0, 64.0, 256.0):
            risks = [risk_at_time(bench.curve(believed_intensity(
                estimate_memory(truth, hours, np.random.default_rng(200 + s), q, corrected), p, floor=0.1),
                v_refs=V_NEAR), ref["trip_time"]) for s in range(2)]
            rows.append({"hours": hours, "risk": risks})
        res["sources"][name] = {"mean_observability": float(np.mean(q)), "rows": rows}
        log(f"E7 {name}: " + ", ".join(f"{r['hours']:g}h {np.mean(r['risk']):.2f}" for r in rows)
            + f"  (no memory {ref['collisions_per_10k']:.2f}, perfect {oracle:.2f})")
    return res


def main(out: Path) -> dict:
    df, events = PL.prepare()
    grid = Grid.around(df["x"].to_numpy(), df["y"].to_numpy(), cell=1.0)
    cache = PL.PROC / "fleet_sightings.parquet"
    if cache.exists():
        vis = pd.read_parquet(cache)
    else:
        vis = fleet_sightings(df)
        vis.to_parquet(cache)
    log(f"fleet sightings: {len(vis):,} (car, pedestrian, 0.5 s) sightings of {vis['track'].nunique():,} pedestrians")
    res = {"sensor_range_m": SENSOR_RANGE, "real": real_data_comparison(df, events, vis, grid)}

    edge = edge_observability(df)
    drivers = driving_vehicles(df)
    q_profiles = {}
    for pen, name in ((1.0, "fleet 100%"), (0.1, "fleet 10%"), (0.03, "fleet 3%")):
        draws = [drivers] if pen == 1.0 else [np.random.default_rng(70 + d).choice(
            drivers, int(round(pen * len(drivers))), replace=False) for d in range(N_DRAWS)]
        profiles = [observability_profile(edge, obs) for obs in draws]
        q_profiles[name] = (profiles[0][0], np.mean([q for _, q in profiles], axis=0))
        log(f"observability {name}: mean {q_profiles[name][1].mean():.2f}, "
            f"min {q_profiles[name][1].min():.2f}, max {q_profiles[name][1].max():.2f}")
    res["observability"] = {k: {"x": v[0].tolist(), "q": v[1].tolist()} for k, v in q_profiles.items()}
    res["sim"] = simulator_comparison(q_profiles)

    out.mkdir(parents=True, exist_ok=True)
    (out / "fleet_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    from .fleet_figures import make_all
    make_all(out / "figures", res)
    log(f"done -> {out}")
    return res


if __name__ == "__main__":
    main(PL.ROOT / "results")
