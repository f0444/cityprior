"""Part 4: a full day at an instrumented intersection (DLR UT).

E10  memory + live signal phase: predict a vehicle's speed 3 s ahead from
     (i) area-wide statistics, (ii) the live phase of the governing signal,
     (iii) location memory, (iv) memory + live phase. Which signal governs
     which place is itself learned from memory (mutual information between a
     light being green and vehicles stopping there).
E11  time of day: does a day/night-conditioned memory beat a pooled one?
E12  vehicle-VRU conflicts over 24 h: do conflict hot spots of some hours
     predict those of other hours?

Protocol: train on even hours, test on odd hours (both cover day and night).

    python -m cityprior.dlr_experiments      # writes results/dlr_metrics.json + figures/dlr*.png
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import dlr
from . import evaluate as E
from . import events as EV
from . import pipeline as PL
from .geometry import Grid
from .memory import LocationMemory

SPEED_EDGES = np.array([1.0, 4.0, 8.0])          # m/s: stopped | crawling | slow | moving
N_BINS = 4
CELL_M = 4.0
MI_MIN_BITS = 0.03
MIN_SAMPLES = 40
ALPHA = 2.0
DAY_HOURS = range(6, 20)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ anchors
def anchors() -> pd.DataFrame:
    """Vehicle states once per second with the same vehicle's state 3 s later."""
    tab = pq.read_table(dlr.CACHE, columns=["track", "frame", "t", "x", "y", "speed", "heading", "cls"],
                        filters=[("cls", "==", "veh")])
    v = tab.to_pandas()
    v = v[v["frame"] % 10 == 0].drop(columns="cls")
    fut = v[["track", "frame", "x", "y", "speed"]].rename(columns={"x": "x_f", "y": "y_f", "speed": "speed_f"})
    a = v.merge(fut.assign(frame=fut["frame"] - 30), on=["track", "frame"])
    a["b0"] = np.digitize(a["speed"], SPEED_EDGES)
    a["b3"] = np.digitize(a["speed_f"], SPEED_EDGES)
    hb = ((a["heading"].to_numpy() + np.pi) // (2 * np.pi / 8)).astype(np.int64) % 8
    hb[a["speed"].to_numpy() < 1.0] = 8                       # stopped: heading is noise
    a["place"] = ((a["x"] // CELL_M).astype(np.int64) * 100_000 + (a["y"] // CELL_M).astype(np.int64) * 100 + hb)
    a["hour"] = (a["t"] // 3600).astype(int)
    a["night"] = (~a["hour"].isin(DAY_HOURS)).astype(int)
    c, s = np.cos(a["heading"]), np.sin(a["heading"])
    a["lon3"] = (a["x_f"] - a["x"]) * c + (a["y_f"] - a["y"]) * s  # distance travelled along the heading
    return a.reset_index(drop=True)


def _mi_columns(g: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mutual information (bits) between each boolean column of g and y."""
    out = np.zeros(g.shape[1])
    for xv in (False, True):
        px = (g == xv).mean(0)
        for yv in (False, True):
            py = (y == yv).mean()
            pxy = ((g == xv) & (y == yv)[:, None]).mean(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out += np.where(pxy > 0, pxy * np.log2(pxy / (px * py)), 0.0)
    return out


def governing_signals(place: np.ndarray, green: np.ndarray, stopped: np.ndarray) -> dict[int, tuple[int, float]]:
    """For each place, the traffic light whose green state tells most about
    whether vehicles there are stopped 3 s later (learned from memory)."""
    order = np.argsort(place, kind="stable")
    pl, g, y = place[order], green[order], stopped[order]
    cut = np.flatnonzero(np.diff(pl)) + 1
    gov = {}
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(pl)]):
        if e - s < MIN_SAMPLES or y[s:e].all() or not y[s:e].any():
            continue
        mi = _mi_columns(g[s:e], y[s:e])
        k = int(mi.argmax())
        if mi[k] > MI_MIN_BITS:
            gov[int(pl[s])] = (k, float(mi[k]))
    return gov


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a bootstrap interval that resamples whole vehicles: samples of
    the same vehicle one second apart are not independent."""
    _, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(sums)
    boots = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        boots.append((w @ sums) / (w @ counts))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def _counts(key: np.ndarray, target: np.ndarray, train: np.ndarray) -> np.ndarray:
    u, inv = np.unique(key, return_inverse=True)
    C = np.zeros((len(u), N_BINS))
    np.add.at(C, (inv[train], target[train]), 1)
    return C[inv]


def _shrink(C: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return (C + ALPHA * prior) / (C.sum(1, keepdims=True) + ALPHA)


def signal_memory(a: pd.DataFrame, sig: pd.DataFrame) -> tuple[dict, dict]:
    sec = np.clip(np.floor(a["t"].to_numpy()).astype(np.int64), sig.index.min(), sig.index.max())
    states = sig.reindex(sec).to_numpy()
    ids = sig.columns.to_numpy()
    green = np.isin(states, dlr.GREEN_STATES)
    train = (a["hour"] % 2 == 0).to_numpy()
    test = ~train
    b0, b3 = a["b0"].to_numpy(), a["b3"].to_numpy()
    place = a["place"].to_numpy()

    gov = governing_signals(place[train], green[train], (a["speed_f"].to_numpy() < 1.0)[train])
    col = pd.Series(place).map({k: v[0] for k, v in gov.items()}).to_numpy()
    has = ~np.isnan(col)
    sg = np.full(len(a), 2)                                   # 0 not green, 1 green, 2 no governing light
    idx = np.flatnonzero(has)
    sg[idx] = green[idx, col[idx].astype(int)].astype(int)

    g0 = np.zeros((N_BINS, N_BINS))
    np.add.at(g0, (b0[train], b3[train]), 1)
    g0 += 0.5
    P0 = (g0 / g0.sum(1, keepdims=True))[b0]
    models = {
        "area-wide statistics": P0,
        "live signal only": _shrink(_counts(b0 * 10 + sg, b3, train), P0),
        "location memory only": (P_mem := _shrink(_counts(place * 10 + b0, b3, train), P0)),
        "memory + live signal": _shrink(_counts((place * 10 + b0) * 10 + sg, b3, train), P_mem),
    }
    # expected speed in 3 s -> distance along the heading (trapezoid), vs constant velocity
    mean_speed = np.array([a["speed_f"][train & (b3 == k)].mean() for k in range(N_BINS)])
    v0 = a["speed"].to_numpy()
    track = a["track"].to_numpy()
    lon_true = a["lon3"].to_numpy()
    rows = np.arange(len(a))

    def scores(P, m):
        gain = np.log2(P[rows[m], b3[m]]) - np.log2(P0[rows[m], b3[m]])
        lon_pred = 0.5 * (v0[m] + P[m] @ mean_speed) * 3.0
        return {"info_gain_bits": cluster_ci(gain, track[m]), "top1": float((P[m].argmax(1) == b3[m]).mean()),
                "lon_error_3s_m": float(np.mean(np.abs(lon_pred - lon_true[m])))}

    subsets = {
        "all vehicles": test,
        "near a governing light": test & has,
        "near a light, moving": test & has & (b0 >= 1),
        "near a light, light green": test & has & (sg == 1),
        "near a light, light not green": test & has & (sg == 0),
    }
    res = {"n_places": int(len(np.unique(place[train]))), "n_governed_places": len(gov),
           "share_test_governed": float(has[test].mean()), "subsets": {}}
    for name, m in subsets.items():
        r = {"n": int(m.sum()), "cv_lon_error_3s_m": float(np.mean(np.abs(v0[m] * 3.0 - lon_true[m])))}
        r.update({k: scores(P, m) for k, P in models.items()})
        res["subsets"][name] = r
        log(f"E10 {name:30s} n={r['n']:7d} " + "  ".join(
            f"{k}: {v['info_gain_bits'][0]:.2f} bits / {100 * v['top1']:.0f}% / {v['lon_error_3s_m']:.2f} m"
            for k, v in r.items() if isinstance(v, dict)) + f"  CV {r['cv_lon_error_3s_m']:.2f} m")

    # E11: day/night conditioning of the memory
    night = a["night"].to_numpy()
    P_tod = _shrink(_counts((place * 10 + b0) * 10 + night, b3, train), P_mem)
    P_tod_live = _shrink(_counts(((place * 10 + b0) * 10 + sg) * 10 + night, b3, train), models["memory + live signal"])
    tod = {}
    for name, m in (("day", test & (night == 0)), ("night", test & (night == 1))):
        tod[name] = {"n": int(m.sum()),
                     "memory": scores(P_mem, m)["info_gain_bits"][0],
                     "memory by time of day": scores(P_tod, m)["info_gain_bits"][0],
                     "memory + live": scores(models["memory + live signal"], m)["info_gain_bits"][0],
                     "memory + live by time of day": scores(P_tod_live, m)["info_gain_bits"][0]}
    res["time_of_day"] = tod
    log(f"E11 time of day: {tod}")

    gov_table = {str(k): {"light_id": int(ids[v[0]]), "mi_bits": v[1]} for k, v in gov.items()}
    return res, gov_table


# ---------------------------------------------------------------- conflicts
def conflicts() -> dict:
    tab = pq.read_table(dlr.CACHE, columns=["track", "frame", "t", "x", "y", "vx", "vy", "speed", "heading",
                                            "length", "width", "cls"])
    df = tab.to_pandas()
    df = df[df["frame"] % 2 == 0].copy()                      # 5 Hz is plenty for conflict episodes
    df["region"] = ((df["x"] // 20).astype(int) * 100 + (df["y"] // 20).astype(int))
    cache = PL.PROC / "dlr_ut_conflicts.parquet"
    if cache.exists():
        ev = pd.read_parquet(cache)
    else:
        ev = EV.vru_vehicle_conflicts(df)
        ev.to_parquet(cache)
    ev["hour"] = (ev["t"] // 3600).astype(int)
    log(f"E12 conflicts: {len(ev)} (pedestrians {int((ev['cls'] == 'ped').sum())}, "
        f"cyclists {int((ev['cls'] == 'micro').sum())})")
    grid = Grid.around(df["x"].to_numpy(), df["y"].to_numpy(), cell=1.0)
    hours = df["t"] // 3600
    spans_even = [(h * 3600.0, (h + 1) * 3600.0) for h in range(0, 24, 2)]
    events = {"emergence": ev.rename(columns={"track_p": "track"})[["track", "t", "x", "y", "region"]],
              "hard_brake": ev.iloc[0:0][["t", "x", "y", "region"]].assign(track=[]),
              "conflict": ev}
    mem = LocationMemory.build(df.assign(track=df["track"]), events, spans_even, grid)
    test = events["emergence"][(events["emergence"]["t"] // 3600) % 2 == 1]
    e = E.emergence_eval(mem, test, sigma_m=3.0)
    out = {"n_conflicts": int(len(ev)), "n_ped": int((ev["cls"] == "ped").sum()),
           "n_cyclist": int((ev["cls"] == "micro").sum()), "n_test": int(len(test)),
           "per_hour": ev.groupby("hour").size().reindex(range(24), fill_value=0).tolist(),
           "vehicles_per_hour": df[df["cls"] == "veh"].groupby(hours)["track"].nunique().reindex(range(24), fill_value=0).tolist(),
           "gain_bits": e["info_gain_bits"], "capture_10pct": e["capture_at_10pct_area"],
           "capture_20pct": e["capture_at_20pct_area"],
           "curve_area": e["curve_area"].tolist(), "curve_captured": e["curve_captured"].tolist(),
           "xy": ev[["x", "y", "cls"]].to_dict("list")}
    log(f"E12 held-out conflict hot spots: {e['info_gain_bits']} bits, {100 * e['capture_at_10pct_area']:.0f}% "
        f"in top 10% of area ({len(test)} test conflicts)")
    return out


def main() -> dict:
    from .dlr_figures import make_all
    t0 = time.time()
    a = anchors()
    sig = dlr.load_signals()
    log(f"anchors {len(a):,} vehicle states with a 3 s future; signals {sig.shape}")
    res, gov = signal_memory(a, sig)
    res["conflicts"] = conflicts()
    out = PL.ROOT / "results"
    (out / "dlr_metrics.json").write_text(json.dumps({"E10_E11": res, "E12": res.pop("conflicts"),
                                                      "governing": gov}, indent=1, default=float))
    make_all(out / "figures", json.loads((out / "dlr_metrics.json").read_text()))
    log(f"done in {time.time() - t0:.0f} s")
    return res


if __name__ == "__main__":
    main()
