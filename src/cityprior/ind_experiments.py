"""Part 7: does memory of where pedestrians step onto the road hold on other days? (inD, Aachen)

For every recording at a location with several sessions, memory of pedestrian road entries is built
    same session      the other recordings of the same session
    other sessions    recordings of the other sessions, subsampled to the same number of entries
    other sessions+   all recordings of the other sessions
and scored on the recording's own entries against a uniform prior over the roadway: information gain
(bits per entry) and the share of entries in the 10% of road area the memory ranks highest. If
"other sessions" matches "same session" at equal volume, pedestrian memory transfers across days.

    python -m cityprior.ind_experiments   # writes results/ind_metrics.json + figures/ind1_cross_day.png
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import ind as D
from . import pipeline as PL
from .dlr_experiments import cluster_ci

VARIANTS = ("same session", "other sessions", "other sessions+")
DRAWS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_site(entries: pd.DataFrame, rec_session: dict, grid, road, rng) -> dict:
    out = {v: {"gain": [], "top": [], "cluster": []} for v in VARIANTS}
    n_used = []
    for rec, test in entries.groupby("recording"):
        s = rec_session[rec]
        same = entries[(entries["recording"] != rec) & entries["recording"].map(rec_session).eq(s)]
        other = entries[entries["recording"].map(rec_session).ne(s)]
        if len(test) < 5 or len(same) < 10 or len(other) < 10:
            continue
        n = min(len(same), len(other))
        n_used.append(n)
        clusters = rec * 100_000 + test["track"].to_numpy()
        for v, pool in (("same session", same), ("other sessions", other)):
            g = np.zeros(len(test))
            t = np.zeros(len(test))
            for _ in range(DRAWS):
                sub = pool.iloc[rng.choice(len(pool), n, replace=False)]
                r = D.score(D.density(sub, grid, road), road, test, grid)
                g += r["gain"] / DRAWS
                t += r["in_top"] / DRAWS
            out[v]["gain"].append(g), out[v]["top"].append(t), out[v]["cluster"].append(clusters)
        r = D.score(D.density(other, grid, road), road, test, grid)
        out["other sessions+"]["gain"].append(r["gain"])
        out["other sessions+"]["top"].append(r["in_top"].astype(float))
        out["other sessions+"]["cluster"].append(clusters)
    res = {}
    for v, d in out.items():
        if not d["gain"]:
            continue
        gain, top, cl = (np.concatenate(d[k]) for k in ("gain", "top", "cluster"))
        res[v] = {"entries": int(len(gain)), "info_gain_bits": cluster_ci(gain, cl, n=300),
                  "top10_share": float(top.mean()), "_gain": gain, "_cluster": cl, "_top": top}
    res["memory_entries_equal_volume_median"] = float(np.median(n_used)) if n_used else 0.0
    return res


def main() -> dict:
    from .ind_figures import fig_cross_day
    meta = D.recordings()
    df = D.load()
    log(f"inD: {len(meta)} recordings, {df['recording'].nunique()} loaded, {len(df):,} samples at 5 Hz")
    rng = np.random.default_rng(0)
    res: dict = {"locations": {}}
    sess = {}
    for loc, m in meta.groupby("locationId"):
        sess[int(loc)] = [{"session": int(s), "weekday": g["weekday"].iloc[0], "recordings": g["recordingId"].tolist(),
                           "hours": round(float(g["duration"].sum()) / 3600, 2)} for s, g in m.groupby("session")]
    res["sessions"] = sess
    pooled = {v: {"gain": [], "top": [], "cluster": []} for v in VARIANTS}
    maps = {}
    for loc, m in meta.groupby("locationId"):
        loc = int(loc)
        if m["session"].nunique() < 2:
            log(f"location {loc}: one session only, skipped")
            continue
        site = df[df["recording"].isin(m["recordingId"])]
        grid = D.SiteGrid(site["x"].to_numpy(), site["y"].to_numpy())
        road = D.roadway(site, grid)
        entries = D.road_entries(site, grid, road)
        rec_session = dict(zip(m["recordingId"], m["session"]))
        r = evaluate_site(entries, rec_session, grid, road, rng)
        res["locations"][loc] = {"road_m2": int(road.sum()), "entries": int(len(entries)),
                                 "pedestrians": int(site.loc[site["cls"] == "pedestrian", ["recording", "track"]]
                                                    .drop_duplicates().shape[0]),
                                 **{k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
                                    for k, v in r.items()}}
        for v in VARIANTS:
            if v in r:
                for k in ("gain", "top", "cluster"):
                    pooled[v][k].append(r[v]["_" + k] if k != "cluster" else r[v]["_cluster"] + loc * 10_000_000)
        maps[loc] = (grid, road, entries, rec_session)
        log(f"location {loc}: {len(entries)} road entries, road {road.sum()} m2; " + "; ".join(
            f"{v}: {r[v]['info_gain_bits'][0]:.2f} bits, top10 {100 * r[v]['top10_share']:.0f}%" for v in VARIANTS if v in r))
    res["pooled"] = {}
    for v, d in pooled.items():
        gain, top, cl = (np.concatenate(d[k]) for k in ("gain", "top", "cluster"))
        res["pooled"][v] = {"entries": int(len(gain)), "info_gain_bits": cluster_ci(gain, cl, n=300),
                            "top10_share": float(top.mean())}
    same, other = res["pooled"]["same session"]["info_gain_bits"][0], res["pooled"]["other sessions"]["info_gain_bits"][0]
    res["pooled"]["other_over_same"] = other / same
    log("pooled: " + "; ".join(f"{v}: {d['info_gain_bits'][0]:.2f} bits, top10 {100 * d['top10_share']:.0f}%"
                                for v, d in res["pooled"].items() if isinstance(d, dict)) + f"; other/same {other / same:.2f}")
    out = PL.ROOT / "results"
    (out / "ind_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_cross_day(out / "figures" / "ind1_cross_day.png", res, maps)
    log("done")
    return res


if __name__ == "__main__":
    main()
