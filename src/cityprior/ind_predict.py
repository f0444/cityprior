"""Part 8: does memory of a place make pedestrian and cyclist forecasts better on other days? (inD)

For every pedestrian or cyclist in motion, predict where they will be 1, 2 and 3 s later from the last 1.2 s of
their own track, with
    constant velocity     keep speed and direction
    memory only           the mean displacement of road users who were in the same 1 m cell, heading the same way
    kinematics            gradient boosting on the agent's own recent motion
    kinematics + memory   the same, plus memory features
Evaluation is leave-one-session-out: the test session is a day the model has not seen, and memory is built only
from other days -- also for training rows, whose memory excludes their own session, so the model never learns to
trust a memory that contains its own future.

    python -m cityprior.ind_predict      # writes results/ind_predict_metrics.json + figures/ind2_prediction.png
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import ind as D
from . import pipeline as PL
from .dlr_experiments import cluster_ci

HZ = 5
HIST = (2, 4, 6)                 # past samples used (0.4, 0.8, 1.2 s)
HORIZONS = (5, 10, 15)           # future samples (1, 2, 3 s)
STEP = 2                         # one anchor every 0.4 s
MIN_SPEED = 0.3                  # m/s: standing people start moving at unpredictable times
SITES = (1, 2, 4)                # sites recorded in several sessions
FINE, COARSE = 1.0, 3.0          # memory cells, m
HBINS = 8
MANOEUVRE_DEG = 30.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def anchors(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (track, anchor time): agent-frame history, world position, future displacements."""
    rec_site = dict(zip(meta["recordingId"], meta["locationId"]))
    rec_sess = dict(zip(meta["recordingId"], meta["session"]))
    vru = df[df["cls"].isin(["pedestrian", "bicycle"]) & df["recording"].map(rec_site).isin(SITES)]
    rows = []
    hmax, fmax = max(HIST), max(HORIZONS)
    for (rec, tid), g in vru.groupby(["recording", "track"], sort=False):
        p = g[["x", "y"]].to_numpy(float)
        n = len(p)
        if n < hmax + fmax + 1:
            continue
        idx = np.arange(hmax, n - fmax, STEP)
        v = (p[idx] - p[idx - 2]) / (2 / HZ)
        speed = np.hypot(v[:, 0], v[:, 1])
        keep = speed >= MIN_SPEED
        idx, v, speed = idx[keep], v[keep], speed[keep]
        if len(idx) == 0:
            continue
        th = np.arctan2(v[:, 1], v[:, 0])
        c, s = np.cos(th), np.sin(th)

        def to_agent(d):                      # world displacement -> agent frame (x forward, y left)
            return np.c_[d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c]

        r = {"recording": rec, "track": tid, "site": rec_site[rec], "session": rec_sess[rec],
             "bike": float(g["cls"].iloc[0] == "bicycle"), "x": p[idx, 0], "y": p[idx, 1], "heading": th,
             "speed": speed}
        for k in HIST:
            a = to_agent(p[idx - k] - p[idx])
            r[f"h{k}_x"], r[f"h{k}_y"] = a[:, 0], a[:, 1]
        v_old = (p[idx - 4] - p[idx - 6]) / (2 / HZ)
        r["dspeed"] = speed - np.hypot(v_old[:, 0], v_old[:, 1])
        for h in HORIZONS:
            dw = p[idx + h] - p[idx]
            r[f"w{h}_x"], r[f"w{h}_y"] = dw[:, 0], dw[:, 1]          # world, for memory
            a = to_agent(dw)
            r[f"f{h}_x"], r[f"f{h}_y"] = a[:, 0], a[:, 1]            # agent frame, the target
        end = p[idx + fmax] - p[idx + fmax - 2]
        turn = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(end[:, 1], end[:, 0]) - th)))))
        r["manoeuvre"] = turn >= MANOEUVRE_DEG
        rows.append(pd.DataFrame(r))
    a = pd.concat(rows, ignore_index=True)
    a["hbin"] = ((a["heading"] + np.pi) // (2 * np.pi / HBINS)).astype(int) % HBINS
    return a


def _key(a: pd.DataFrame, cell: float) -> np.ndarray:
    return (a["site"].to_numpy() * 10**9 + (a["x"] // cell).astype(np.int64).to_numpy() * 10**5
            + (a["y"] // cell).astype(np.int64).to_numpy() * 100 + a["hbin"].to_numpy() * 10 + a["bike"].to_numpy().astype(int))


def memory_features(a: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """For rows of `a`, the mean world displacement of `source` rows sharing cell and heading bin, rotated into
    each row's agent frame, minus the constant-velocity displacement; and support counts."""
    out = pd.DataFrame(index=a.index)
    c, s = np.cos(a["heading"].to_numpy()), np.sin(a["heading"].to_numpy())
    for name, cell in (("m1", FINE), ("m3", COARSE)):
        ks, ka = _key(source, cell), _key(a, cell)
        cols = [f"w{h}_{q}" for h in HORIZONS for q in "xy"]
        agg = pd.DataFrame(source[cols].to_numpy(), columns=cols).assign(k=ks).groupby("k").agg(["sum", "count"])
        cnt = pd.Series(agg[(cols[0], "count")].to_numpy(), index=agg.index).reindex(ka).to_numpy()
        out[f"{name}_n"] = np.log1p(np.nan_to_num(cnt))
        for h in HORIZONS:
            mx = pd.Series(agg[(f"w{h}_x", "sum")].to_numpy(), index=agg.index).reindex(ka).to_numpy() / cnt
            my = pd.Series(agg[(f"w{h}_y", "sum")].to_numpy(), index=agg.index).reindex(ka).to_numpy() / cnt
            ax_, ay_ = mx * c + my * s, -mx * s + my * c
            out[f"{name}_{h}_x"] = ax_ - a["speed"].to_numpy() * h / HZ
            out[f"{name}_{h}_y"] = ay_
    return out


KIN = [f"h{k}_{q}" for k in HIST for q in "xy"] + ["speed", "dspeed", "bike"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int = 0) -> dict:
    """Gradient boosting on the residual to constant velocity, one model per horizon and axis."""
    out = {}
    for h in HORIZONS:
        for q in "xy":
            y = train[f"f{h}_{q}"].to_numpy() - (train["speed"].to_numpy() * h / HZ if q == "x" else 0.0)
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=seed)
            m.fit(train[cols].to_numpy(), y)
            pred = m.predict(test[cols].to_numpy())
            out[(h, q)] = pred + (test["speed"].to_numpy() * h / HZ if q == "x" else 0.0)
    return out


def main() -> dict:
    from .ind_figures import fig_prediction
    meta = D.recordings()
    df = D.load()
    a = anchors(df, meta)
    log(f"anchors {len(a):,} ({int(a['bike'].sum()):,} cyclist), tracks {a.groupby(['recording', 'track']).ngroups:,}, "
        f"manoeuvring {100 * a['manoeuvre'].mean():.0f}%")
    sessions = sorted(a[["site", "session"]].drop_duplicates().itertuples(index=False, name=None))
    preds = {m: {(h, q): np.full(len(a), np.nan) for h in HORIZONS for q in "xy"}
             for m in ("constant velocity", "memory only", "kinematics", "kinematics + memory")}
    for site, sess in sessions:
        test_mask = (a["session"] == sess).to_numpy()
        train = a[~test_mask].copy()
        test = a[test_mask].copy()
        # memory from other days only: test rows see every other session of their site; a training row sees the
        # sessions of its site except its own and the test session
        mem_test = memory_features(test, a[(a["site"] == site) & ~test_mask])
        parts = []
        for s2, g in train.groupby("session"):
            src = a[(a["site"] == g["site"].iloc[0]) & (a["session"] != s2) & ~test_mask]
            parts.append(memory_features(g, src))
        mem_train = pd.concat(parts).loc[train.index]
        train = pd.concat([train, mem_train], axis=1)
        test = pd.concat([test, mem_test], axis=1)
        mcols = list(mem_test.columns)
        k = fit_predict(train, test, KIN)
        km = fit_predict(train, test, KIN + mcols)
        for h in HORIZONS:
            cvx = test["speed"].to_numpy() * h / HZ
            preds["constant velocity"][(h, "x")][test_mask] = cvx
            preds["constant velocity"][(h, "y")][test_mask] = 0.0
            for q in "xy":
                fine, coarse = test[f"m1_{h}_{q}"].to_numpy(), test[f"m3_{h}_{q}"].to_numpy()
                n1 = np.expm1(test["m1_n"].to_numpy())
                mem = np.where(n1 >= 3, fine, coarse)
                mem = np.where(np.isfinite(mem), mem, 0.0) + (cvx if q == "x" else 0.0)
                preds["memory only"][(h, q)][test_mask] = mem
                preds["kinematics"][(h, q)][test_mask] = k[(h, q)]
                preds["kinematics + memory"][(h, q)][test_mask] = km[(h, q)]
        e3 = {m: np.hypot(preds[m][(15, "x")][test_mask] - test["f15_x"], preds[m][(15, "y")][test_mask] - test["f15_y"]).mean()
              for m in preds}
        log(f"site {site} session {sess}: {test_mask.sum():,} anchors; 3 s error " + ", ".join(f"{m} {v:.2f} m" for m, v in e3.items()))
    cluster = a["recording"].to_numpy() * 100_000 + a["track"].to_numpy()
    res = {"anchors": int(len(a)), "tracks": int(a.groupby(["recording", "track"]).ngroups),
           "cyclist_share": float(a["bike"].mean()), "manoeuvre_share": float(a["manoeuvre"].mean()),
           "sessions": [list(s) for s in sessions], "errors": {}}
    subsets = {"all": np.ones(len(a), bool), "pedestrians": a["bike"].eq(0).to_numpy(), "cyclists": a["bike"].eq(1).to_numpy(),
               "manoeuvring": a["manoeuvre"].to_numpy(), "straight": ~a["manoeuvre"].to_numpy()}
    for sub, mask in subsets.items():
        res["errors"][sub] = {"n": int(mask.sum())}
        for m in preds:
            res["errors"][sub][m] = {}
            for h in HORIZONS:
                err = np.hypot(preds[m][(h, "x")] - a[f"f{h}_x"], preds[m][(h, "y")] - a[f"f{h}_y"]).to_numpy()
                res["errors"][sub][m][f"{h / HZ:.0f}s"] = cluster_ci(err[mask], cluster[mask], n=200)
        e = res["errors"][sub]
        log(f"{sub}: 3 s error " + ", ".join(f"{m} {e[m]['3s'][0]:.2f}" for m in preds))
    for sub in subsets:
        e = res["errors"][sub]
        res["errors"][sub]["gain_memory_over_kinematics_3s"] = 1 - e["kinematics + memory"]["3s"][0] / e["kinematics"]["3s"][0]
        res["errors"][sub]["gain_over_constant_velocity_3s"] = 1 - e["kinematics + memory"]["3s"][0] / e["constant velocity"]["3s"][0]
    # paired interval for the gain of memory over kinematics at 3 s
    ek = np.hypot(preds["kinematics"][(15, "x")] - a["f15_x"], preds["kinematics"][(15, "y")] - a["f15_y"]).to_numpy()
    em = np.hypot(preds["kinematics + memory"][(15, "x")] - a["f15_x"], preds["kinematics + memory"][(15, "y")] - a["f15_y"]).to_numpy()
    for sub, mask in subsets.items():
        res["errors"][sub]["memory_minus_kinematics_3s_m"] = cluster_ci(em[mask] - ek[mask], cluster[mask], n=300)
    out = PL.ROOT / "results"
    (out / "ind_predict_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_prediction(out / "figures" / "ind2_prediction.png", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
