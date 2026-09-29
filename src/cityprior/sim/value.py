"""E14: what one traversal gains from memory and from a live camera, by how uneven the street is.

Input for the cost-benefit model (cityprior.econ). For each concentration kappa of
pedestrian activity (as in E6) and each information source, the planner's
risk <-> trip-time curve is compared with the no-memory planner at the operating
point:
    seconds saved at equal risk      (the operator takes the gain as time)
    collision reduction at equal time (the operator takes it as safety)

Sources: memory (trusted both ways), memory that may only add caution, live camera
on the whole block with memory (trusted to say "clear"), live camera whose reports
may only add caution.

    python -m cityprior.sim.value     # writes results/value_metrics.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from .. import pipeline as PL
from .engine import Infra
from .experiments import HISTORY_HOURS, V_OPERATING, V_REFS, Bench, risk_at_time
from .planner import believed_intensity, camera_coverage, estimate_memory
from .world import Profile, SimParams, ped_speed_quantiles, profile_from_tgsim

KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seconds_saved(curve: list[dict], ref: dict) -> float:
    """Trip time saved at the reference collision rate. If every point of the curve
    is safer than the reference, the fastest point is taken (the speed limit binds)."""
    c = np.array([r["collisions_per_10k"] for r in curve])
    t = np.array([r["trip_time"] for r in curve])
    ok = c <= ref["collisions_per_10k"]
    if ok.all():
        return float(ref["trip_time"] - t.min())
    order = np.argsort(c)
    return float(ref["trip_time"] - np.interp(ref["collisions_per_10k"], c[order], t[order]))


def main() -> dict:
    p = SimParams()
    df, events = PL.prepare()
    truth = profile_from_tgsim(events["emergence"], street_len=p.street_len)
    mid = events["emergence"][events["emergence"]["x"].between(70, 160) & events["emergence"]["y"].between(28, 50)]
    bench = Bench(p, truth, ped_speed_quantiles(df, mid["track"]))
    flat = Profile(truth.x, np.full(len(truth.x), truth.mean), {"what": "street mean"})
    cov = camera_coverage(1.0, p)
    rows = []
    for kappa in KAPPAS:
        t_k = truth.concentrated(kappa)
        mem = estimate_memory(t_k, HISTORY_HOURS, np.random.default_rng(5))
        lam_flat = believed_intensity(flat, p)
        lam_mem = believed_intensity(mem, p, floor=0.1)
        curves = {
            "no memory": bench.curve(lam_flat, truth=t_k),
            "memory": bench.curve(lam_mem, truth=t_k),
            "memory, adds caution only": bench.curve(lam_mem, truth=t_k, never_faster_than=lam_flat),
            "live camera, trusted": bench.curve(believed_intensity(mem, p, cov, camera_trusted=True, floor=0.1),
                                                Infra(cov, True), truth=t_k),
            "live camera, adds caution only": bench.curve(lam_mem, Infra(cov, True), truth=t_k),
        }
        ref = next(r for r in curves["no memory"] if r["v_ref"] == V_OPERATING)
        row = {"kappa": kappa, "ceiling": t_k.ceiling(), "peak_to_mean": float(t_k.lam.max() / t_k.mean),
               "reference": {"trip_time": ref["trip_time"], "collisions_per_10k": ref["collisions_per_10k"]},
               "sources": {}}
        for name, c in curves.items():
            if name == "no memory":
                continue
            op = next(r for r in c if r["v_ref"] == V_OPERATING)
            row["sources"][name] = {
                "seconds_saved_at_equal_risk": seconds_saved(c, ref),
                "reduction_at_equal_time": 1 - risk_at_time(c, ref["trip_time"]) / ref["collisions_per_10k"],
                "operating_point": {"trip_time": op["trip_time"], "collisions_per_10k": op["collisions_per_10k"]},
                "curve": [{"v_ref": r["v_ref"], "trip_time": r["trip_time"],
                           "collisions_per_10k": r["collisions_per_10k"]} for r in c],
            }
        row["no_memory_curve"] = [{"v_ref": r["v_ref"], "trip_time": r["trip_time"],
                                   "collisions_per_10k": r["collisions_per_10k"]} for r in curves["no memory"]]
        rows.append(row)
        log(f"E14 kappa {kappa} (ceiling {row['ceiling']:.3f}): " + "; ".join(
            f"{n}: {s['seconds_saved_at_equal_risk']:.2f} s / {100 * s['reduction_at_equal_time']:.0f}%"
            for n, s in row["sources"].items()))
    res = {"street_len_m": p.street_len, "operating_v_ref": V_OPERATING, "memory_hours": HISTORY_HOURS, "rows": rows}
    out = PL.ROOT / "results"
    (out / "value_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    log("done")
    return res


if __name__ == "__main__":
    main()
