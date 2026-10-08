"""Part 8: does memory of a place make pedestrian and cyclist forecasts better on other days? (inD)

For every pedestrian or cyclist in motion, predict where they will be 1, 2 and 3 s later from the last 1.2 s of
their own track, with
    constant velocity           keep speed and direction
    memory only                 how road users who were in the same cell heading the same way moved on, scaled
                                to this agent's speed (their slowing, swerving and turning)
    kinematics                  gradient boosting on the agent's own recent motion
    kinematics + context        ... plus the nearest vehicle and nearest other pedestrian/cyclist (relative
                                position and velocity) and how crowded it is: the strong baseline
    kinematics + context + memory
Evaluation is leave-one-session-out: the test session is a day the model has not seen, and memory is built only
from other days -- also for training rows, whose memory excludes their own session, so the model never learns to
trust a memory that contains its own future. A learning curve repeats the full model with memory from a fraction
of the other days' data.

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
NEAR_M = 30.0                    # context: neighbours further than this count as absent
CURVE_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
MODELS = ("constant velocity", "memory only", "kinematics", "kinematics + context", "kinematics + context + memory")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- anchors
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

        r = {"recording": rec, "track": tid, "frame": g["frame"].to_numpy()[idx], "site": rec_site[rec],
             "session": rec_sess[rec], "bike": float(g["cls"].iloc[0] == "bicycle"), "x": p[idx, 0], "y": p[idx, 1],
             "heading": th, "speed": speed}
        for k in HIST:
            a = to_agent(p[idx - k] - p[idx])
            r[f"h{k}_x"], r[f"h{k}_y"] = a[:, 0], a[:, 1]
        v_old = (p[idx - 4] - p[idx - 6]) / (2 / HZ)
        r["dspeed"] = speed - np.hypot(v_old[:, 0], v_old[:, 1])
        for h in HORIZONS:
            a = to_agent(p[idx + h] - p[idx])
            r[f"f{h}_x"], r[f"f{h}_y"] = a[:, 0], a[:, 1]            # agent frame: the target
        end = p[idx + fmax] - p[idx + fmax - 2]
        signed = np.degrees(np.angle(np.exp(1j * (np.arctan2(end[:, 1], end[:, 0]) - th))))
        r["turn"] = signed
        r["manoeuvre"] = np.abs(signed) >= MANOEUVRE_DEG
        rows.append(pd.DataFrame(r))
    a = pd.concat(rows, ignore_index=True)
    a["hbin"] = ((a["heading"] + np.pi) // (2 * np.pi / HBINS)).astype(int) % HBINS
    return a


# ---------------------------------------------------------------------------------------------- context
def context_features(a: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Nearest vehicle and nearest other pedestrian/cyclist at the anchor time, in the agent's frame
    (position and velocity relative to the agent, time and distance of closest approach under constant
    velocities), and the number of pedestrians/cyclists within 5 m. Computed recording by recording."""
    parts = []
    for rec, ar in a.groupby("recording"):
        d = df[df["recording"] == rec].sort_values(["track", "frame"]).copy()
        g = d.groupby("track")
        d["vx"] = (g["x"].diff() * HZ).fillna(0.0)
        d["vy"] = (g["y"].diff() * HZ).fillna(0.0)
        d = d[d["frame"].isin(ar["frame"].unique())]
        other = d[["frame", "track", "x", "y", "vx", "vy", "cls"]]
        q = ar[["frame", "track", "x", "y", "heading"]].reset_index().rename(columns={"index": "row"})
        q = q.merge(other[["frame", "track", "vx", "vy"]], on=["frame", "track"], how="left")
        out = pd.DataFrame(index=ar.index)
        for kind, classes in (("veh", D.VEH_CLASSES), ("vru", ("pedestrian", "bicycle"))):
            m = q.merge(other[other["cls"].isin(classes)], on="frame", suffixes=("", "_o"))
            m = m[m["track_o"] != m["track"]]
            dx, dy = m["x_o"] - m["x"], m["y_o"] - m["y"]
            m["dist"] = np.hypot(dx, dy)
            if kind == "vru":
                out["crowd5"] = m[m["dist"] < 5].groupby("row").size().reindex(ar.index).fillna(0).to_numpy()
            m = m[m["dist"] < NEAR_M].sort_values("dist").drop_duplicates("row").set_index("row")
            c, s = np.cos(m["heading"]), np.sin(m["heading"])
            dx, dy = m["x_o"] - m["x"], m["y_o"] - m["y"]
            m["rx"], m["ry"] = dx * c + dy * s, -dx * s + dy * c
            dvx, dvy = m["vx_o"] - m["vx"].fillna(0), m["vy_o"] - m["vy"].fillna(0)
            m["rvx"], m["rvy"] = dvx * c + dvy * s, -dvx * s + dvy * c
            rv2 = m["rvx"] ** 2 + m["rvy"] ** 2
            m["tca"] = (-(m["rx"] * m["rvx"] + m["ry"] * m["rvy"]) / rv2.where(rv2 > 1e-6)).clip(lower=0, upper=10)
            m["dca"] = np.hypot(m["rx"] + m["rvx"] * m["tca"].fillna(0), m["ry"] + m["rvy"] * m["tca"].fillna(0))
            for col in ("dist", "rx", "ry", "rvx", "rvy", "tca", "dca"):
                out[f"{kind}_{col}"] = m[col].reindex(ar.index).to_numpy()
        parts.append(out)
    return pd.concat(parts).loc[a.index]


# ---------------------------------------------------------------------------------------------- memory
def _key(a: pd.DataFrame, cell: float) -> np.ndarray:
    return (a["site"].to_numpy() * 10**9 + (a["x"] // cell).astype(np.int64).to_numpy() * 10**5
            + (a["y"] // cell).astype(np.int64).to_numpy() * 100 + a["hbin"].to_numpy() * 10 + a["bike"].to_numpy().astype(int))


def memory_features(a: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """For rows of `a`, what road users of `source` did from the same cell and heading: their path in their own
    frame divided by the distance constant velocity would have covered (forward progress and sideways share), the
    shares that turned left and right by more than 30 degrees within 3 s, and support counts; at 1 m and 3 m."""
    out = pd.DataFrame(index=a.index)
    src = pd.DataFrame({"k1": _key(source, FINE), "k3": _key(source, COARSE)})
    for h in HORIZONS:
        cv = source["speed"].to_numpy() * h / HZ
        src[f"px{h}"] = source[f"f{h}_x"].to_numpy() / cv
        src[f"py{h}"] = source[f"f{h}_y"].to_numpy() / cv
    src["left"] = (source["turn"].to_numpy() >= MANOEUVRE_DEG).astype(float)
    src["right"] = (source["turn"].to_numpy() <= -MANOEUVRE_DEG).astype(float)
    cols = [c for c in src.columns if c not in ("k1", "k3")]
    for name, cell, kc in (("m1", FINE, "k1"), ("m3", COARSE, "k3")):
        agg = src.groupby(kc)[cols].mean()
        n = src.groupby(kc).size()
        ka = _key(a, cell)
        out[f"{name}_n"] = np.log1p(n.reindex(ka).fillna(0).to_numpy())
        for c in cols:
            out[f"{name}_{c}"] = agg[c].reindex(ka).to_numpy()
    return out


def memory_prediction(a: pd.DataFrame, mem: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Agent-frame displacement predicted by memory alone: fine cell if it has at least 3 road users, else coarse,
    else constant velocity."""
    cv = a["speed"].to_numpy() * h / HZ
    n1 = np.expm1(mem["m1_n"].to_numpy())
    px = np.where(n1 >= 3, mem[f"m1_px{h}"], mem[f"m3_px{h}"])
    py = np.where(n1 >= 3, mem[f"m1_py{h}"], mem[f"m3_py{h}"])
    px = np.where(np.isfinite(px), px, 1.0)
    py = np.where(np.isfinite(py), py, 0.0)
    return px * cv, py * cv


# ---------------------------------------------------------------------------------------------- models
KIN = [f"h{k}_{q}" for k in HIST for q in "xy"] + ["speed", "dspeed", "bike"]
CTX = [f"{k}_{c}" for k in ("veh", "vru") for c in ("dist", "rx", "ry", "rvx", "rvy", "tca", "dca")] + ["crowd5"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], horizons=HORIZONS, seed: int = 0) -> dict:
    """Gradient boosting on the residual to constant velocity, one model per horizon and axis."""
    out = {}
    for h in horizons:
        for q in "xy":
            y = train[f"f{h}_{q}"].to_numpy() - (train["speed"].to_numpy() * h / HZ if q == "x" else 0.0)
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=seed)
            m.fit(train[cols].to_numpy(), y)
            out[(h, q)] = m.predict(test[cols].to_numpy()) + (test["speed"].to_numpy() * h / HZ if q == "x" else 0.0)
    return out


def with_memory(a: pd.DataFrame, rows: pd.Index, test_mask: np.ndarray, frac: float, rng) -> pd.DataFrame:
    """Memory features for `rows`, each from the other sessions of its site (never the test session), keeping a
    random `frac` of the source recordings."""
    keep_rec = None
    if frac < 1.0:
        recs = a["recording"].unique()
        keep_rec = set(rng.choice(recs, max(1, int(round(frac * len(recs)))), replace=False))
    parts = []
    sub = a.loc[rows]
    for s2, g in sub.groupby("session"):
        src = a[(a["site"] == g["site"].iloc[0]) & (a["session"] != s2) & ~test_mask]
        if g.index.isin(np.flatnonzero(test_mask)).all():           # test rows: every other session of the site
            src = a[(a["site"] == g["site"].iloc[0]) & (a["session"] != s2)]
        if keep_rec is not None:
            src = src[src["recording"].isin(keep_rec)]
        parts.append(memory_features(g, src))
    return pd.concat(parts).loc[rows]


def main() -> dict:
    from .ind_figures import fig_prediction
    meta = D.recordings()
    df = D.load()
    a = anchors(df, meta)
    log(f"anchors {len(a):,} ({int(a['bike'].sum()):,} cyclist), tracks {a.groupby(['recording', 'track']).ngroups:,}, "
        f"manoeuvring {100 * a['manoeuvre'].mean():.0f}%")
    a = pd.concat([a, context_features(a, df)], axis=1)
    log(f"context: nearest vehicle within {NEAR_M:.0f} m for {100 * a['veh_dist'].notna().mean():.0f}% of anchors, "
        f"another pedestrian/cyclist for {100 * a['vru_dist'].notna().mean():.0f}%")
    sessions = sorted(a[["site", "session"]].drop_duplicates().itertuples(index=False, name=None))
    preds = {m: {(h, q): np.full(len(a), np.nan) for h in HORIZONS for q in "xy"} for m in MODELS}
    curve = {f: np.full(len(a), np.nan) for f in CURVE_FRACTIONS}
    curve_base = np.full(len(a), np.nan)
    per_session = []
    rng = np.random.default_rng(0)
    for site, sess in sessions:
        test_mask = (a["session"] == sess).to_numpy()
        tr_idx, te_idx = a.index[~test_mask], a.index[test_mask]
        mem = with_memory(a, a.index, test_mask, 1.0, rng)
        mcols = list(mem.columns)
        full = pd.concat([a, mem], axis=1)
        train, test = full.loc[tr_idx], full.loc[te_idx]
        k = fit_predict(train, test, KIN)
        kc = fit_predict(train, test, KIN + CTX)
        kcm = fit_predict(train, test, KIN + CTX + mcols)
        for h in HORIZONS:
            preds["constant velocity"][(h, "x")][test_mask] = test["speed"].to_numpy() * h / HZ
            preds["constant velocity"][(h, "y")][test_mask] = 0.0
            mx, my = memory_prediction(test, test[mcols], h)
            preds["memory only"][(h, "x")][test_mask], preds["memory only"][(h, "y")][test_mask] = mx, my
            for name, p in (("kinematics", k), ("kinematics + context", kc), ("kinematics + context + memory", kcm)):
                for q in "xy":
                    preds[name][(h, q)][test_mask] = p[(h, q)]
        e3 = {m: float(np.hypot(preds[m][(15, "x")][test_mask] - test["f15_x"], preds[m][(15, "y")][test_mask] - test["f15_y"]).mean())
              for m in MODELS}
        per_session.append({"site": int(site), "session": int(sess), "anchors": int(test_mask.sum()), "error_3s": e3})
        log(f"site {site} session {sess}: {test_mask.sum():,} anchors; 3 s " + ", ".join(f"{m} {v:.2f}" for m, v in e3.items()))
        # learning curve (3 s only): memory from a fraction of the other days' recordings
        curve_base[test_mask] = np.hypot(kc[(15, "x")] - test["f15_x"], kc[(15, "y")] - test["f15_y"])
        for f in CURVE_FRACTIONS:
            if f == 1.0:
                p = kcm
            else:
                memf = with_memory(a, a.index, test_mask, f, rng)
                fullf = pd.concat([a, memf], axis=1)
                p = fit_predict(fullf.loc[tr_idx], fullf.loc[te_idx], KIN + CTX + list(memf.columns), horizons=(15,))
            curve[f][test_mask] = np.hypot(p[(15, "x")] - test["f15_x"], p[(15, "y")] - test["f15_y"])
    cluster = a["recording"].to_numpy() * 100_000 + a["track"].to_numpy()
    # memory for a session comes from the other sessions of its own site
    site_of = meta.groupby("session")["locationId"].first()
    hours = {int(s): float(meta.loc[(meta["locationId"] == site_of[s]) & (meta["session"] != s), "duration"].sum())
             for s in a["session"].unique()}
    res = {"anchors": int(len(a)), "tracks": int(a.groupby(["recording", "track"]).ngroups),
           "cyclist_share": float(a["bike"].mean()), "manoeuvre_share": float(a["manoeuvre"].mean()),
           "per_session": per_session, "errors": {}}
    subsets = {"all": np.ones(len(a), bool), "pedestrians": a["bike"].eq(0).to_numpy(), "cyclists": a["bike"].eq(1).to_numpy(),
               "manoeuvring": a["manoeuvre"].to_numpy(), "straight": ~a["manoeuvre"].to_numpy(),
               "vehicle within 10 m": (a["veh_dist"] < 10).to_numpy()}
    err = {m: {h: np.hypot(preds[m][(h, "x")] - a[f"f{h}_x"], preds[m][(h, "y")] - a[f"f{h}_y"]).to_numpy() for h in HORIZONS}
           for m in MODELS}
    for sub, mask in subsets.items():
        e = res["errors"][sub] = {"n": int(mask.sum())}
        for m in MODELS:
            e[m] = {f"{h / HZ:.0f}s": cluster_ci(err[m][h][mask], cluster[mask], n=200) for h in HORIZONS}
        base, full = err["kinematics + context"][15], err["kinematics + context + memory"][15]
        e["memory_gain_over_context_3s"] = 1 - e["kinematics + context + memory"]["3s"][0] / e["kinematics + context"]["3s"][0]
        e["context_gain_over_kinematics_3s"] = 1 - e["kinematics + context"]["3s"][0] / e["kinematics"]["3s"][0]
        e["full_gain_over_constant_velocity_3s"] = 1 - e["kinematics + context + memory"]["3s"][0] / e["constant velocity"]["3s"][0]
        e["memory_minus_context_3s_m"] = cluster_ci(full[mask] - base[mask], cluster[mask], n=300)
        log(f"{sub}: 3 s " + ", ".join(f"{m} {e[m]['3s'][0]:.2f}" for m in MODELS)
            + f"; memory {100 * e['memory_gain_over_context_3s']:.1f}% over context")
    res["learning_curve"] = {"fractions": list(CURVE_FRACTIONS),
                             "error_3s": {str(f): float(np.nanmean(curve[f])) for f in CURVE_FRACTIONS},
                             "baseline_error_3s": float(np.nanmean(curve_base)),
                             "memory_hours_full_median": float(np.median(list(hours.values()))) / 3600}
    log(f"learning curve: baseline {res['learning_curve']['baseline_error_3s']:.3f}; " + ", ".join(
        f"{f}: {v:.3f}" for f, v in res["learning_curve"]["error_3s"].items()))
    out = PL.ROOT / "results"
    (out / "ind_predict_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_prediction(out / "figures" / "ind2_prediction.png", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
