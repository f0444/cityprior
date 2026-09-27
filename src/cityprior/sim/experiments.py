"""Closed-loop experiments: does location memory help a robotaxi drive?

    python -m cityprior.sim.experiments      # writes results/sim_metrics.json + figures

E1  safety <-> trip-time Pareto: worst case / no memory / memory / oracle memory
E2  value of memory vs live-camera coverage of the block        (H3)
E3  value of memory vs hours of history                          (learning curve)
E4  stale memory: the hotspot moved; effect of a caution floor   (H6)
E5  silent camera outage vs heartbeat-aware fallback             (H6)
E6  value of memory vs how concentrated pedestrian activity is   (generalisation beyond this street)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import pipeline as PL
from .engine import Infra, sample_encounters, simulate, summarize
from .planner import (believed_intensity, camera_coverage, estimate_memory, expected_risk,
                      speed_profile, weight_for_reference_speed, worst_case_speed)
from .world import Profile, SimParams, ped_speed_quantiles, profile_from_tgsim

V_REFS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.2)
V_NEAR = (6.0, 7.0, 8.0, 9.0, 10.0)   # enough to interpolate around the operating point
V_OPERATING = 8.0          # "human-like" operating point used for single-number comparisons
N_INATTENTIVE, N_ATTENTIVE = 30000, 3000
N_BOOT = 500
HISTORY_HOURS = 10.0       # default memory: ~ one week of afternoon peaks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Bench:
    """Shared world, encounter samples (common random numbers) and runner."""

    def __init__(self, p: SimParams, truth: Profile, speed_q: np.ndarray, seed: int = 7):
        self.p, self.truth = p, truth
        rng = np.random.default_rng(seed)
        self.enc_i = sample_encounters(rng, N_INATTENTIVE, p, speed_q, attentive=False)
        self.enc_a = sample_encounters(rng, N_ATTENTIVE, p, speed_q, attentive=True)

    def run(self, lam_believed: np.ndarray | None, v_ref: float | None, infra: Infra | None = None,
            truth: Profile | None = None, never_faster_than: np.ndarray | None = None) -> dict:
        """`never_faster_than`: an intensity map whose speed profile caps this one,
        i.e. the prior may only add caution (asymmetric trust)."""
        truth = truth or self.truth
        w = None if v_ref is None else weight_for_reference_speed(v_ref, self.truth.mean, self.p)
        prof = speed_profile(lam_believed, w, self.p)
        if never_faster_than is not None:
            prof = np.minimum(prof, speed_profile(never_faster_than, w, self.p))
        res_i = simulate(self.enc_i, truth, prof, self.p, infra)
        res_a = simulate(self.enc_a, truth, prof, self.p, infra)
        out = summarize(res_i, res_a, truth, self.p)
        out["v_ref"] = v_ref
        out["model_risk_per_10k"] = 1e4 * self.p.p_inattentive * expected_risk(prof, truth.on(self.p.grid), self.p)
        out["share_known_by_infra_first"] = float(res_i["known_by_infra_first"].mean())
        out["_coll_i"], out["_coll_a"] = res_i["collided"], res_a["collided"]
        return out

    def curve(self, lam_believed, infra=None, truth=None, v_refs=V_REFS, never_faster_than=None) -> list[dict]:
        return [self.run(lam_believed, v, infra, truth, never_faster_than) for v in v_refs]


def paired_bootstrap_reduction(curve: list[dict], ref: dict, p: SimParams, n_boot: int = N_BOOT) -> tuple:
    """Relative collision reduction at the reference trip time, with a paired
    bootstrap over episodes (all runs share the same sampled episodes)."""
    rng = np.random.default_rng(3)
    n_i, n_a = len(ref["_coll_i"]), len(ref["_coll_a"])
    per = 1e4 * ref["exposure_per_traversal"]
    w_i, w_a = p.p_inattentive, 1 - p.p_inattentive
    times = np.array([r["trip_time"] for r in curve])
    order = np.argsort(times)

    def risk(run, ii, ia):
        return per * (w_i * run["_coll_i"][ii].mean() + w_a * run["_coll_a"][ia].mean())

    def reduction(ii, ia):
        c = np.array([risk(r, ii, ia) for r in curve])
        base = risk(ref, ii, ia)
        return 1 - np.interp(ref["trip_time"], times[order], c[order]) / base if base > 0 else np.nan

    point = reduction(np.arange(n_i), np.arange(n_a))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = np.array([reduction(rng.integers(0, n_i, n_i), rng.integers(0, n_a, n_a)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def time_at_risk(curve: list[dict], risk: float) -> float:
    """Trip time a planner family needs to reach a given collision rate."""
    c = np.array([r["collisions_per_10k"] for r in curve])
    t = np.array([r["trip_time"] for r in curve])
    order = np.argsort(c)
    c, t = c[order], t[order]
    if risk < c[0] or risk > c[-1]:
        return float("nan")
    return float(np.interp(risk, c, t))


def risk_at_time(curve: list[dict], trip_time: float) -> float:
    c = np.array([r["collisions_per_10k"] for r in curve])
    t = np.array([r["trip_time"] for r in curve])
    order = np.argsort(t)
    return float(np.interp(trip_time, t[order], c[order]))


def main(out: Path) -> dict:
    p = SimParams()
    df, events = PL.prepare()
    truth = profile_from_tgsim(events["emergence"], street_len=p.street_len)
    mid_block = events["emergence"][(events["emergence"]["x"].between(70, 160))
                                    & (events["emergence"]["y"].between(28, 50))]
    speed_q = ped_speed_quantiles(df, mid_block["track"])
    bench = Bench(p, truth, speed_q)
    rng = np.random.default_rng(11)
    memory = estimate_memory(truth, HISTORY_HOURS, rng)
    flat = Profile(truth.x, np.full(len(truth.x), truth.mean), {"what": "street mean"})
    res: dict = {"world": {
        "emergences_per_hour": truth.total_rate * 3600, "n_real_events": truth.meta["n_events"],
        "ped_speed_median": float(speed_q[50]), "ped_speed_p90": float(speed_q[90]),
        "worst_case_speed": worst_case_speed(p), "memory_hours": HISTORY_HOURS,
        "memory_events_observed": memory.meta["events_observed"],
        "n_inattentive": N_INATTENTIVE, "n_attentive": N_ATTENTIVE, "v_refs": V_REFS,
    }}
    log(f"world {res['world']}")

    # ------------------------------------------------------------------- E1
    e1 = {
        "worst_case": [bench.run(None, None)],
        "no_memory": bench.curve(believed_intensity(flat, p)),
        "memory": bench.curve(believed_intensity(memory, p, floor=0.1)),
        "memory_adds_caution_only": bench.curve(believed_intensity(memory, p, floor=0.1),
                                                never_faster_than=believed_intensity(flat, p)),
        "oracle": bench.curve(believed_intensity(truth, p)),
    }
    ref = next(r for r in e1["no_memory"] if r["v_ref"] == V_OPERATING)
    e1["summary"] = {
        "reference": {"v_ref": V_OPERATING, "trip_time": ref["trip_time"], "collisions_per_10k": ref["collisions_per_10k"]},
        "memory_time_at_reference_risk": time_at_risk(e1["memory"], ref["collisions_per_10k"]),
        "oracle_time_at_reference_risk": time_at_risk(e1["oracle"], ref["collisions_per_10k"]),
        "memory_risk_at_reference_time": risk_at_time(e1["memory"], ref["trip_time"]),
        "oracle_risk_at_reference_time": risk_at_time(e1["oracle"], ref["trip_time"]),
        "memory_reduction_at_reference_time_ci": paired_bootstrap_reduction(e1["memory"], ref, p),
        "oracle_reduction_at_reference_time_ci": paired_bootstrap_reduction(e1["oracle"], ref, p),
        "adds_caution_only_reduction_at_reference_time_ci": paired_bootstrap_reduction(
            e1["memory_adds_caution_only"], ref, p),
        "heterogeneity_ceiling": truth.ceiling(),
    }
    res["E1"] = e1
    log(f"E1 {e1['summary']}  worst case {e1['worst_case'][0]['trip_time']:.1f}s "
        f"{e1['worst_case'][0]['collisions_per_10k']:.3f}/10k")
    res["profiles"] = {
        "x": p.grid.tolist(),
        "lam_true": truth.on(p.grid).tolist(), "lam_memory": memory.on(p.grid).tolist(),
        "v_no_memory": speed_profile(believed_intensity(flat, p),
                                     weight_for_reference_speed(V_OPERATING, truth.mean, p), p).tolist(),
        "v_memory": speed_profile(believed_intensity(memory, p, floor=0.1),
                                  weight_for_reference_speed(V_OPERATING, truth.mean, p), p).tolist(),
        "v_worst": speed_profile(None, None, p).tolist(),
    }

    # ------------------------------------------------------------------- E2
    e2 = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cov = camera_coverage(frac, p)
        infra = Infra(cov, up=frac > 0)
        live_only = bench.curve(believed_intensity(flat, p, cov, camera_trusted=frac > 0), infra, v_refs=V_NEAR)
        live_mem = bench.curve(believed_intensity(memory, p, cov, camera_trusted=frac > 0, floor=0.1), infra,
                               v_refs=V_NEAR)
        r_ref = next(r for r in live_only if r["v_ref"] == V_OPERATING)
        e2.append({
            "coverage": frac, "live_only": live_only, "live_memory": live_mem,
            "reference_risk": r_ref["collisions_per_10k"], "reference_time": r_ref["trip_time"],
            "memory_time_at_reference_risk": time_at_risk(live_mem, r_ref["collisions_per_10k"]),
            "memory_risk_at_reference_time": risk_at_time(live_mem, r_ref["trip_time"]),
            "memory_reduction_ci": paired_bootstrap_reduction(live_mem, r_ref, p),
        })
        log(f"E2 coverage {frac}: live-only {r_ref['trip_time']:.1f}s {r_ref['collisions_per_10k']:.3f}/10k -> "
            f"with memory risk {e2[-1]['memory_risk_at_reference_time']:.3f}/10k at equal time")
    res["E2"] = e2

    # ------------------------------------------------------------------- E3
    e3 = []
    ref_t = e1["summary"]["reference"]["trip_time"]
    for hours in (0.25, 1.0, 4.0, 16.0, 64.0):
        risks = []
        for seed in range(3):
            m = estimate_memory(truth, hours, np.random.default_rng(100 + seed))
            risks.append(risk_at_time(bench.curve(believed_intensity(m, p, floor=0.1), v_refs=V_NEAR), ref_t))
        e3.append({"hours": hours, "risk_at_reference_time": risks})
        log(f"E3 {hours} h: risk at equal time {np.round(risks, 3)} (no memory {e1['summary']['reference']['collisions_per_10k']:.3f})")
    res["E3"] = e3

    # ------------------------------------------------------------------- E4
    # the hotspot moves to the quietest part of the block; memory is from before
    inner = (truth.x >= 5) & (truth.x <= truth.x[-1] - 5)
    shift = float(truth.x[inner][np.argmin(truth.lam[inner])] - truth.x[np.argmax(truth.lam)])
    moved = truth.shifted(shift)
    res["stale_shift_m"] = shift
    e4 = {"no_memory": bench.run(believed_intensity(flat, p), V_OPERATING, truth=moved)}
    for floor in (0.0, 0.1, 0.3, 0.6):
        e4[f"stale_memory_floor_{floor}"] = bench.run(believed_intensity(memory, p, floor=floor),
                                                      V_OPERATING, truth=moved)
    e4["stale_memory_adds_caution_only"] = bench.run(believed_intensity(memory, p, floor=0.1), V_OPERATING,
                                                     truth=moved, never_faster_than=believed_intensity(flat, p))
    fresh = believed_intensity(estimate_memory(moved, HISTORY_HOURS, rng), p, floor=0.1)
    e4["fresh_memory"] = bench.run(fresh, V_OPERATING, truth=moved)
    e4["fresh_memory_adds_caution_only"] = bench.run(fresh, V_OPERATING, truth=moved,
                                                     never_faster_than=believed_intensity(flat, p))
    cov = camera_coverage(0.5, p)
    e4["stale_memory_plus_camera_50pct"] = bench.run(
        believed_intensity(memory, p, cov, camera_trusted=True, floor=0.1), V_OPERATING, Infra(cov, True), moved)
    res["E4"] = e4
    log("E4 " + ", ".join(f"{k}: {v['collisions_per_10k']:.3f}/10k {v['trip_time']:.1f}s" for k, v in e4.items()))

    # ------------------------------------------------------------------- E5
    cov = camera_coverage(1.0, p)
    e5 = {
        "camera_up": bench.run(believed_intensity(memory, p, cov, True, floor=0.1), V_OPERATING, Infra(cov, True)),
        "silent_outage": bench.run(believed_intensity(memory, p, cov, True, floor=0.1), V_OPERATING, Infra(cov, False)),
        "outage_with_heartbeat": bench.run(believed_intensity(memory, p, floor=0.1), V_OPERATING, Infra(cov, False)),
    }
    res["E5"] = e5
    log("E5 " + ", ".join(f"{k}: {v['collisions_per_10k']:.3f}/10k {v['trip_time']:.1f}s" for k, v in e5.items()))

    # ------------------------------------------------------------------- E6
    e6 = []
    for kappa in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        t_k = truth.concentrated(kappa)
        base = bench.curve(believed_intensity(flat, p), truth=t_k, v_refs=V_NEAR)
        r_ref = next(r for r in base if r["v_ref"] == V_OPERATING)
        row = {"kappa": kappa, "ceiling": t_k.ceiling(), "peak_to_mean": float(t_k.lam.max() / t_k.mean),
               "reference_risk": r_ref["collisions_per_10k"]}
        for name, prof in (("oracle", t_k), ("memory", estimate_memory(t_k, HISTORY_HOURS, np.random.default_rng(5)))):
            c = bench.curve(believed_intensity(prof, p, floor=0.1 if name == "memory" else 0.0), truth=t_k,
                            v_refs=V_NEAR)
            row[f"{name}_reduction_ci"] = paired_bootstrap_reduction(c, r_ref, p)
        e6.append(row)
        log(f"E6 kappa {kappa}: ceiling {row['ceiling']:.3f} peak/mean {row['peak_to_mean']:.1f} "
            f"oracle {np.round(row['oracle_reduction_ci'], 3)} memory {np.round(row['memory_reduction_ci'], 3)}")
    res["E6"] = e6

    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_metrics.json").write_text(json.dumps(_public(res), indent=1, default=float))
    from .figures import make_all
    make_all(out / "figures", res)
    log(f"done -> {out}")
    return res


def _public(o):
    """Drop per-episode arrays (keys starting with '_') before writing JSON."""
    if isinstance(o, dict):
        return {k: _public(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [_public(v) for v in o]
    return o


if __name__ == "__main__":
    main(PL.ROOT / "results")
