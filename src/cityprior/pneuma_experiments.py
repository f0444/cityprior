"""Part 5: does location memory transfer across days? (pNEUMA, Athens)

Four weekday mornings (Wed 24, Mon 29, Tue 30 Oct, Thu 1 Nov 2018), two drone
areas, the same four half-hours (08:30-10:30) each day. For every test
half-hour, memory is built from
    same day      the other three half-hours of that morning           (1.5 h)
    other days    the same half-hour on the three other mornings       (1.5 h)
    other days+   every half-hour of the three other mornings          (6 h)
    all           everything except the test half-hour                 (7.5 h)
and scored against area-wide statistics on two targets: the speed class and the
heading 3 s ahead. If "other days" matches "same day" at equal volume, memory
transfers across days.

    python -m cityprior.pneuma_experiments   # writes results/pneuma_metrics.json + figures/pneuma1_*.png
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import pipeline as PL
from . import pneuma as P
from .dlr_experiments import SPEED_EDGES, cluster_ci

CELL_M = 4.0
ALPHA = 2.0
VARIANTS = ("same day", "other days", "other days+", "all")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Vehicle states once per second with the same vehicle 3 s later."""
    df = df.dropna(subset=["heading"])
    df = df.assign(k=np.round(df["t"] * 5).astype(np.int64))
    a = df[df["k"] % 5 == 0][["track", "k", "x", "y", "speed", "heading", "day", "drone", "slot", "type_name"]]
    f = a[["track", "k", "x", "y", "speed", "heading"]].rename(
        columns={"x": "x_f", "y": "y_f", "speed": "speed_f", "heading": "heading_f"})
    a = a.merge(f.assign(k=f["k"] - 15), on=["track", "k"])
    a["b0"] = np.digitize(a["speed"], SPEED_EDGES)
    a["b3"] = np.digitize(a["speed_f"], SPEED_EDGES)
    a["h0"] = ((a["heading"] + np.pi) // (np.pi / 4)).astype(int) % 8
    a["h3"] = ((a["heading_f"] + np.pi) // (np.pi / 4)).astype(int) % 8
    hb = a["h0"].to_numpy().copy()
    hb[a["speed"].to_numpy() < 1.0] = 8
    drone = a["drone"].str[1:].astype(np.int64).to_numpy()
    a["place"] = (drone * 1_000_000_000 + (a["x"] // CELL_M).astype(np.int64) * 100_000
                  + (a["y"] // CELL_M).astype(np.int64) * 100 + hb)
    c, s = np.cos(a["heading"]), np.sin(a["heading"])
    a["lon3"] = (a["x_f"] - a["x"]) * c + (a["y_f"] - a["y"]) * s
    return a.reset_index(drop=True)


def _counts(key, target, train, n_classes):
    u, inv = np.unique(key, return_inverse=True)
    C = np.zeros((len(u), n_classes))
    np.add.at(C, (inv[train], target[train]), 1)
    return C[inv]


def _scores(P, P0, target, test, track, v0=None, lon=None, mean_speed=None):
    rows = np.flatnonzero(test)
    gain = np.log2(P[rows, target[rows]]) - np.log2(P0[rows, target[rows]])
    out = {"gain": gain, "hit": (P[rows].argmax(1) == target[rows]), "track": track[rows]}
    if v0 is not None:
        out["err"] = np.abs(0.5 * (v0[rows] + P[rows] @ mean_speed) * 3.0 - lon[rows])
    return out


def evaluate(a: pd.DataFrame) -> dict:
    day, slot = a["day"].to_numpy(), a["slot"].to_numpy()
    track, place = a["track"].to_numpy(), a["place"].to_numpy()
    targets = {
        "speed in 3 s": (a["b0"].to_numpy(), a["b3"].to_numpy(), 4, np.ones(len(a), bool)),
        "heading in 3 s": (a["h0"].to_numpy(), a["h3"].to_numpy(), 8,
                           (a["speed"].to_numpy() >= 1.0) & (a["speed_f"].to_numpy() >= 1.0)),
    }
    v0, lon = a["speed"].to_numpy(), a["lon3"].to_numpy()
    res = {}
    for tname, (b0, tgt, nc, valid) in targets.items():
        pooled = {v: {"gain": [], "hit": [], "track": [], "err": []} for v in ("area-wide statistics",) + VARIANTS}
        for d in range(4):
            for s in range(4):
                test = (day == d) & (slot == s) & valid
                rest = ~((day == d) & (slot == s)) & valid
                g0 = np.zeros((nc, nc))
                np.add.at(g0, (b0[rest], tgt[rest]), 1)
                g0 += 0.5
                P0 = (g0 / g0.sum(1, keepdims=True))[b0]
                trains = {"same day": (day == d) & (slot != s) & valid,
                          "other days": (day != d) & (slot == s) & valid,
                          "other days+": (day != d) & valid,
                          "all": rest}
                mean_speed = None
                if tname == "speed in 3 s":
                    mean_speed = np.array([a["speed_f"].to_numpy()[rest & (tgt == k)].mean() for k in range(nc)])
                sc = _scores(P0, P0, tgt, test, track, v0 if mean_speed is not None else None, lon, mean_speed)
                for key, val in sc.items():
                    pooled["area-wide statistics"][key].append(val)
                for vname, tr in trains.items():
                    C = _counts(place * 10 + b0, tgt, tr, nc)
                    Pm = (C + ALPHA * P0) / (C.sum(1, keepdims=True) + ALPHA)
                    sc = _scores(Pm, P0, tgt, test, track, v0 if mean_speed is not None else None, lon, mean_speed)
                    for key, val in sc.items():
                        pooled[vname][key].append(val)
        res[tname] = {}
        for vname, parts in pooled.items():
            gain, hit, trk = (np.concatenate(parts[k]) for k in ("gain", "hit", "track"))
            r = {"n": int(len(gain)), "info_gain_bits": cluster_ci(gain, trk, n=300),
                 "top1": float(hit.mean())}
            if parts["err"]:
                r["dist_error_3s_m"] = float(np.concatenate(parts["err"]).mean())
            res[tname][vname] = r
        log(f"E13 {tname}: " + "  ".join(
            f"{v}: {r['info_gain_bits'][0]:.3f} bits, {100 * r['top1']:.1f}%" for v, r in res[tname].items()))
    return res


def learning_curve(a: pd.DataFrame) -> dict:
    """Information gain versus the number of other mornings in memory (all half-hours)."""
    day, slot = a["day"].to_numpy(), a["slot"].to_numpy()
    track, place = a["track"].to_numpy(), a["place"].to_numpy()
    out = {}
    rng = np.random.default_rng(0)
    for tname, b0c, tc, nc, vmask in (("speed in 3 s", "b0", "b3", 4, None), ("heading in 3 s", "h0", "h3", 8, "moving")):
        b0, tgt = a[b0c].to_numpy(), a[tc].to_numpy()
        valid = np.ones(len(a), bool) if vmask is None else (a["speed"].to_numpy() >= 1) & (a["speed_f"].to_numpy() >= 1)
        rows = {}
        for n_days in (1, 2, 3):
            gains, trks = [], []
            for d in range(4):
                others = [x for x in range(4) if x != d]
                use = rng.choice(others, n_days, replace=False)
                test = (day == d) & valid
                rest = (day != d) & valid
                g0 = np.zeros((nc, nc))
                np.add.at(g0, (b0[rest], tgt[rest]), 1)
                g0 += 0.5
                P0 = (g0 / g0.sum(1, keepdims=True))[b0]
                tr = np.isin(day, use) & valid
                C = _counts(place * 10 + b0, tgt, tr, nc)
                Pm = (C + ALPHA * P0) / (C.sum(1, keepdims=True) + ALPHA)
                sc = _scores(Pm, P0, tgt, test, track)
                gains.append(sc["gain"])
                trks.append(sc["track"])
            rows[str(n_days)] = cluster_ci(np.concatenate(gains), np.concatenate(trks), n=300)
        out[tname] = rows
        log(f"E13 learning curve {tname}: " + ", ".join(f"{k} day(s): {v[0]:.3f}" for k, v in rows.items()))
    return out


def main() -> dict:
    from .pneuma_figures import fig_cross_day
    df = P.fetch(log=log)
    log(f"pNEUMA subset: {len(df):,} samples, {df['track'].nunique():,} vehicles")
    a = anchors(df)
    del df
    log(f"anchors {len(a):,}")
    res = {"n_anchors": int(len(a)), "n_vehicles": int(a["track"].nunique()),
           "vehicles_by_type": a.groupby("type_name")["track"].nunique().to_dict(),
           "cross_day": evaluate(a), "learning_curve": learning_curve(a)}
    out = PL.ROOT / "results"
    (out / "pneuma_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_cross_day(out / "figures" / "pneuma1_cross_day.png", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
