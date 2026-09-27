"""End-to-end run on TGSIM Foggy Bottom.

    python -m cityprior.pipeline            # everything, writes results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E
from . import events as EV
from . import message as M
from . import occlusion as O
from . import prediction as P
from .geometry import Grid
from .memory import LocationMemory

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/tgsim_foggy_bottom.csv"
PROC = ROOT / "data/processed"

T_START, T_END = 60.2, 7200.0
T_SPLIT = 4800.0                     # memory: first ~79 min, test: last 40 min
HISTORY_MINUTES = (5, 10, 20, 40, 79)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    clean = PROC / "trajectories_clean.parquet"
    if clean.exists():
        df = pd.read_parquet(clean)
    else:
        D.download(RAW)
        df = D.load(RAW)
        n_ped = df.loc[df["cls"] == "ped", "id"].nunique()
        df = D.stitch_fragments(df, "ped")
        df = D.drop_flicker(df, 1.0)
        log(f"pedestrian tracks {n_ped} -> {df.loc[df['cls'] == 'ped', 'track'].nunique()} after stitching + flicker filter")
        PROC.mkdir(parents=True, exist_ok=True)
        df.to_parquet(clean)

    events = {}
    for name, fn in (
        ("hard_brake", lambda: EV.hard_braking(df)),
        ("emergence", lambda: EV.pedestrian_emergence(df, T_START)),
        ("conflict", lambda: EV.vru_vehicle_conflicts(df)),
    ):
        path = PROC / f"events_{name}.parquet"
        if path.exists():
            events[name] = pd.read_parquet(path)
        else:
            events[name] = fn()
            events[name].to_parquet(path)
    return df, events


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    df, events = prepare()
    grid = Grid.around(df["x"].to_numpy(), df["y"].to_numpy(), cell=1.0)
    res: dict = {"dataset": {
        "rows": int(len(df)), "hours": (T_END - T_START) / 3600,
        "tracks": {c: int(df.loc[df["cls"] == c, "track"].nunique()) for c in ("ped", "micro", "veh")},
        "events": {k: int(len(v)) for k, v in events.items()},
    }}
    log(f"dataset {res['dataset']}")

    # ---------------------------------------------------------- memory + H2
    mem = LocationMemory.build(df, events, [(T_START, T_SPLIT)], grid)
    em_test = events["emergence"][events["emergence"]["t"] >= T_SPLIT]
    rtype = mem.regions.set_index("region")["type"]
    em_test = em_test.assign(kind=em_test["region"].map(rtype).fillna("roadway"))
    res["emergence"] = {
        "all": E.emergence_eval(mem, em_test),
        "off_crosswalk": E.emergence_eval(mem, em_test[em_test["kind"] == "roadway"]),
    }
    hb_out = E.passage_outcomes(df, events["hard_brake"], "veh", "track", T_SPLIT, T_END)
    pc_out = E.passage_outcomes(df, events["conflict"][events["conflict"]["cls"] == "ped"],
                                "ped", "track_p", T_SPLIT, T_END)
    res["hard_brake"] = E.rate_eval(mem, "hard_brake", hb_out)
    res["ped_conflict"] = E.rate_eval(mem, "ped_conflict", pc_out)
    log(f"emergence gain {res['emergence']['all']['info_gain_bits']}, "
        f"hard-brake BSS {res['hard_brake']['brier_skill']:.3f}, ped-conflict BSS {res['ped_conflict']['brier_skill']:.3f}")

    # blocked 3-fold CV: every 40-min block is the test period once
    edges = np.linspace(T_START, T_END, 4)
    folds = []
    for k in range(3):
        spans = [(edges[j], edges[j + 1]) for j in range(3) if j != k]
        m = LocationMemory.build(df, events, spans, grid)
        te = events["emergence"]
        te = te[(te["t"] >= edges[k]) & (te["t"] < edges[k + 1])]
        ee = E.emergence_eval(m, te)
        f = {"test_block": k, "emergence_gain_bits": ee["info_gain_bits"][0],
             "emergence_capture_10pct": ee["capture_at_10pct_area"]}
        for what, cls_, evn, col in (("hard_brake", "veh", "hard_brake", "track"),
                                     ("ped_conflict", "ped", "conflict", "track_p")):
            evs = events[evn] if what == "hard_brake" else events[evn][events[evn]["cls"] == "ped"]
            o = E.passage_outcomes(df, evs, cls_, col, edges[k], edges[k + 1])
            r = E.rate_eval(m, what, o)
            f[f"{what}_brier_skill"] = r["brier_skill"]
            f[f"{what}_auc"] = r["auc"]
        folds.append(f)
    res["cv_folds"] = folds
    log(f"cv folds {pd.DataFrame(folds).round(3).to_dict('records')}")

    # learning curve: how much history does the memory need?
    lc = []
    for minutes in HISTORY_MINUTES:
        t0 = max(T_START, T_SPLIT - 60 * minutes)
        m = LocationMemory.build(df, events, [(t0, T_SPLIT)], grid)
        e = E.emergence_eval(m, em_test)
        lc.append({"minutes": minutes, "gain_bits": e["info_gain_bits"],
                   "capture_10pct": e["capture_at_10pct_area"],
                   "hard_brake_bss": E.rate_eval(m, "hard_brake", hb_out)["brier_skill"],
                   "ped_conflict_bss": E.rate_eval(m, "ped_conflict", pc_out)["brier_skill"]})
    res["learning_curve"] = lc
    log(f"learning curve {[(r['minutes'], round(r['gain_bits'][0], 2)) for r in lc]}")

    # ----------------------------------------------------------- prediction
    # (1) probabilistic: which way will the road user head in 3 s?
    test_df = df[df["t"] >= T_SPLIT]
    res["manoeuvre"] = {c: E.manoeuvre_eval(mem, test_df, c) for c in ("veh", "ped")}
    log(f"manoeuvre {res['manoeuvre']}")

    # (2) deterministic 3 s rollouts: constant velocity vs memory-steered CV.
    # Parameters are tuned on a validation split (memory 0-60 min, samples 60-80 min)
    # twice: on all samples, and on curved-motion samples only.
    mem_val = LocationMemory.build(df, events, [(T_START, 3600.0)], grid)
    res["prediction"] = {}
    pred_store = {}
    for cls_, vmin in (("veh", 1.0), ("ped", 0.3)):
        val = P.make_samples(df, cls_, 3600.0, T_SPLIT, vmin)
        test = P.make_samples(df, cls_, T_SPLIT, T_END, vmin)
        turning = test["turn_rad"] > np.deg2rad(30)
        e_cv = P.displacement_errors(P.predict_cv(test["state"]), test["future"])
        r = {"n": int(len(test["state"])), "n_turning": int(turning.sum())}
        for variant, subset in (("tuned_all", None), ("tuned_turning", val["turn_rad"] > np.deg2rad(30))):
            params, _ = P.tune_mod(val, mem_val, cls_, subset)
            e_mod = P.displacement_errors(P.predict_mod(test["state"], mem, cls_, **params), test["future"])
            v = {"params": params}
            for key in ("ade", "fde_1s", "fde_2s", "fde_3s"):
                for name, sel in (("all", slice(None)), ("turning", turning), ("straight", ~turning)):
                    v[f"{key}_{name}"] = {"cv": float(e_cv[key][sel].mean()), "mod": float(e_mod[key][sel].mean())}
            v["fde_3s_turning_improvement_pct"] = E.bootstrap_ci(
                np.stack([e_cv["fde_3s"][turning], e_mod["fde_3s"][turning]], 1),
                stat=lambda a: 100 * (1 - a[:, 1].mean() / a[:, 0].mean()), n=500)
            r[variant] = v
            log(f"prediction {cls_} {variant}: {params} FDE3 all {v['fde_3s_all']} "
                f"turning {v['fde_3s_turning']} straight {v['fde_3s_straight']}")
        res["prediction"][cls_] = r
        pred_store[cls_] = (test, e_cv)

    # -------------------------------------------------- occlusion / H1 (real)
    occ_path = PROC / "occlusion_obs.parquet"
    if occ_path.exists():
        obs = pd.read_parquet(occ_path)
    else:
        obs = O.virtual_ego_occlusion(df)
        obs.to_parquet(occ_path)
    res["occlusion"] = {"n_obs": int(len(obs)),
                        "n_encounters": int(obs[["track_e", "track_p"]].drop_duplicates().shape[0])}
    for flag in ("occluded_all", "occluded_tall"):
        enc = O.encounter_lead_times(obs, flag)
        res["occlusion"][flag] = {
            "share_obs_occluded": float(obs[flag].mean()),
            "share_obs_occluded_within_20m": float(obs.loc[obs["lon"] < 20, flag].mean()),
            "share_encounters_hidden_at_first": float((enc["lead_s"] > 0).mean()),
            "share_encounters_lead_ge_1s": float((enc["lead_s"] >= 1.0).mean()),
            "share_encounters_never_seen_by_ego": float(enc["never_seen"].mean()),
            "median_lead_s_when_hidden": float(enc.loc[enc["lead_s"] > 0, "lead_s"].median()),
        }
    log(f"occlusion {res['occlusion']}")

    # ------------------------------------------------ message for the real AV
    # Prefer the automated test vehicle (type 4) in the held-out period at a moment
    # when a path-relevant pedestrian is hidden from it; otherwise any vehicle.
    test_obs = obs[(obs["t"] >= T_SPLIT) & obs["occluded_all"]]
    av_tracks = set(df.loc[df["type"] == 4, "track"].unique())
    cand = test_obs[test_obs["track_e"].isin(av_tracks)]
    ego_is_av = len(cand) > 0
    cand = (cand if ego_is_av else test_obs).sort_values("lon")
    frame, track = int(cand["frame"].iloc[0]), cand["track_e"].iloc[0]
    ego = df[(df["frame"] == frame) & (df["track"] == track)].iloc[0]
    msg = M.build_message(mem, df[df["frame"] == frame], ego)
    res["message"] = {k: v for k, v in msg.items() if k not in ("mask", "objects")}
    res["message"]["frame"] = frame
    res["message"]["ego_is_automated_test_vehicle"] = bool(ego_is_av)
    res["message"]["memory_full_area"] = mem.footprint_bytes()
    res["message"]["reference"] = M.bandwidth_reference()
    log(f"message {res['message']}")

    _dump(res, out / "metrics.json")
    from . import plots
    plots.make_all(out / "figures", df, events, mem, res, pred_store, obs, msg, ego)
    log(f"done -> {out}")
    return res


def _dump(res: dict, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return None  # curves go to figures, not to json
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict("records")
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(res, default=conv, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    main(ap.parse_args().out)
