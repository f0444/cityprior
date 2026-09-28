"""E9: false alarms and spoofing of the live infrastructure camera.

Operating point: location memory + a camera covering the whole block, trusted
to declare covered sections clear (the fast configuration of E5).

  phantoms   the camera reports pedestrians who do not exist: benign ghosts
             (e.g. sidewalk walkers taken for crossers) or targeted pop-ups
             injected just before the car arrives
  deletion   the camera silently drops a share of real pedestrians while its
             heartbeat stays healthy
Defences
  onboard priority   where the car can see for itself, camera reports are
                     ignored; the camera only fills the car's blind spots
  add caution only   camera reports may slow the car, but "clear" is not trusted
  fleet audit        cars report pedestrians they saw that the covering camera
                     had not reported; a CUSUM on these reports revokes trust

    python -m cityprior.sim.spoof      # writes results/spoof_metrics.json + figures/sim5_*.png
"""

from __future__ import annotations

import json
import time

import numpy as np

from .. import pipeline as PL
from .engine import WINDOW, Encounters, Infra, simulate
from .planner import believed_intensity, camera_coverage, estimate_memory, speed_profile, weight_for_reference_speed
from .world import SimParams, ped_speed_quantiles, profile_from_tgsim

DELETE_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
GHOST_RATES = (0, 25, 50, 100, 200)          # false reports per hour on the block
PASSES_PER_HOUR = (2, 5, 10, 20, 50)         # robotaxi traversals of the block
FALSE_REVOCATIONS_PER_YEAR = 1.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _with_tau(enc: Encounters, lo: float, hi: float, seed: int) -> Encounters:
    rng = np.random.default_rng(seed)
    return Encounters(enc.x0, enc.x1, enc.u, rng.uniform(lo, hi, len(enc)), enc.v_p, enc.attentive, enc.seed)


def cusum_delay(p0: float, p1: float, h: float, n_runs: int = 4000, max_steps: int = 200_000,
                seed: int = 0) -> np.ndarray:
    """Traversals until a Bernoulli CUSUM (evidence rate p0 -> p1) crosses h."""
    a, b = np.log(p1 / p0), np.log((1 - p1) / (1 - p0))
    rng = np.random.default_rng(seed)
    s = np.zeros(n_runs)
    delay = np.full(n_runs, np.inf)
    for k in range(1, max_steps + 1):
        alive = np.isinf(delay)
        if not alive.any():
            break
        x = rng.random(alive.sum()) < p1
        s[alive] = np.maximum(0.0, s[alive] + np.where(x, a, b))
        hit = np.flatnonzero(alive)[s[alive] >= h]
        delay[hit] = k
    return delay


def cusum_threshold(p0: float, p1: float, arl0: float) -> float:
    """h with in-control average run length ~ arl0 (Siegmund's approximation,
    ARL0 ~ (e^h - h - 1) / KL(p0 || p1))."""
    kl = p0 * np.log(p0 / p1) + (1 - p0) * np.log((1 - p0) / (1 - p1))
    hs = np.linspace(0.5, 40, 4000)
    arl = (np.exp(hs) - hs - 1) / kl
    return float(hs[np.searchsorted(arl, arl0)])


def main() -> dict:
    from .experiments import V_OPERATING, Bench
    from .figures import fig_spoof

    p = SimParams()
    df, events = PL.prepare()
    truth = profile_from_tgsim(events["emergence"], street_len=p.street_len)
    mid = events["emergence"][events["emergence"]["x"].between(70, 160) & events["emergence"]["y"].between(28, 50)]
    bench = Bench(p, truth, ped_speed_quantiles(df, mid["track"]))
    memory = estimate_memory(truth, 10.0, np.random.default_rng(11))
    cov = camera_coverage(1.0, p)
    trusted = believed_intensity(memory, p, cov, camera_trusted=True, floor=0.1)
    cautious = believed_intensity(memory, p, floor=0.1)
    w = weight_for_reference_speed(V_OPERATING, truth.mean, p)
    profiles = {"trusted": speed_profile(trusted, w, p), "cautious": speed_profile(cautious, w, p)}
    exposure = truth.total_rate * WINDOW
    w_i, w_a = p.p_inattentive, 1 - p.p_inattentive

    def run(lam, infra):
        r = bench.run(lam, V_OPERATING, infra)
        return {k: r[k] for k in ("collisions_per_10k", "trip_time", "hard_brakes_per_1k", "near_misses_per_10k")}

    def evidence_share(infra) -> float:
        """Per-traversal probability that some car sees an unreported pedestrian."""
        e = [simulate(enc, truth, profiles["trusted"], p, infra)["unreported_seen"].mean()
             for enc in (bench.enc_i, bench.enc_a)]
        return exposure * (w_i * e[0] + w_a * e[1])

    res: dict = {"operating_v_ref": V_OPERATING}
    res["honest"] = {
        "camera trusted": run(trusted, Infra(cov, True)),
        "camera trusted + onboard priority": run(trusted, Infra(cov, True, onboard_priority=True)),
        "camera may only add caution": run(cautious, Infra(cov, True)),
        "no camera (memory only)": run(cautious, Infra(cov, False)),
    }
    log(f"honest camera: {res['honest']}")

    # ------------------------------------------------------------ deletion
    res["deletion"] = {"shares": DELETE_SHARES, "camera trusted": [], "camera may only add caution": [],
                       "evidence_per_traversal": []}
    for d in DELETE_SHARES:
        res["deletion"]["camera trusted"].append(run(trusted, Infra(cov, True, delete_share=d)))
        res["deletion"]["camera may only add caution"].append(run(cautious, Infra(cov, True, delete_share=d)))
        res["deletion"]["evidence_per_traversal"].append(evidence_share(Infra(cov, True, delete_share=d)))
        log(f"deletion {d:.2f}: trusted {res['deletion']['camera trusted'][-1]['collisions_per_10k']:.2f}, "
            f"add-only {res['deletion']['camera may only add caution'][-1]['collisions_per_10k']:.2f}, "
            f"evidence/traversal {res['deletion']['evidence_per_traversal'][-1]:.4f}")

    # ---------------------------------------------------------- phantoms
    def phantom_stats(enc, infra, profile) -> dict:
        r = simulate(enc, truth, profile, p, infra)
        ok = ~np.isnan(r["t_finish"])
        return {"mean_delay_s": float(np.mean(r["t_finish"][ok] - r["t_nominal"][ok])),
                "hard_brakes_per_phantom": float(r["hard_brakes"].mean()),
                "share_emergency_braking": float((r["max_decel"] >= 6.0).mean()),
                "mean_max_decel": float(r["max_decel"].mean()),
                "share_refuted": float(r["refuted"].mean())}

    ghosts = bench.enc_i                                      # reported as crossing, anywhere in the window
    targeted = _with_tau(bench.enc_i, -2.5, 0.0, seed=91)     # timed to step out just before the car
    popup = dict(ghost=True, report_from_y=p.occlusion_edge - 0.3)
    res["phantoms"] = {}
    for name, prio in (("naive", False), ("onboard priority", True)):
        res["phantoms"][name] = {
            "benign_ghost": phantom_stats(ghosts, Infra(cov, True, ghost=True, onboard_priority=prio), profiles["trusted"]),
            "targeted_from_sidewalk": phantom_stats(targeted, Infra(cov, True, ghost=True, onboard_priority=prio),
                                                    profiles["trusted"]),
            "targeted_popup": phantom_stats(targeted, Infra(cov, True, onboard_priority=prio, **popup),
                                            profiles["trusted"]),
        }
        base = res["honest"]["camera trusted + onboard priority" if prio else "camera trusted"]
        g = res["phantoms"][name]["benign_ghost"]
        res["phantoms"][name]["per_traversal"] = [
            {"ghosts_per_hour": rate,
             "trip_time": base["trip_time"] + rate / 3600 * WINDOW * g["mean_delay_s"],
             "hard_brakes_per_1k": base["hard_brakes_per_1k"] + 1e3 * rate / 3600 * WINDOW * g["hard_brakes_per_phantom"]}
            for rate in GHOST_RATES]
        log(f"phantoms {name}: {res['phantoms'][name]['benign_ghost']} | targeted pop-up "
            f"{res['phantoms'][name]['targeted_popup']}")

    # -------------------------------------------------------------- audit
    p0 = res["deletion"]["evidence_per_traversal"][0]
    res["audit"] = {"p0": p0, "rows": []}
    for d, p1 in zip(DELETE_SHARES[1:], res["deletion"]["evidence_per_traversal"][1:]):
        for passes in PASSES_PER_HOUR:
            h = cusum_threshold(p0, p1, arl0=passes * 24 * 365 / FALSE_REVOCATIONS_PER_YEAR)
            delay = cusum_delay(p0, p1, h, seed=int(1000 * d) + passes)
            rate_attack = res["deletion"]["camera trusted"][DELETE_SHARES.index(d)]["collisions_per_10k"] / 1e4
            rate_honest = res["honest"]["camera trusted"]["collisions_per_10k"] / 1e4
            res["audit"]["rows"].append({
                "delete_share": d, "passes_per_hour": passes, "threshold": h,
                "median_traversals": float(np.median(delay)), "median_hours": float(np.median(delay)) / passes,
                "p90_hours": float(np.percentile(delay, 90)) / passes,
                "expected_extra_collisions": float(np.mean(delay)) * (rate_attack - rate_honest),
            })
        log("audit delete " + f"{d}: " + ", ".join(
            f"{r['passes_per_hour']}/h -> {r['median_hours']:.1f} h" for r in res["audit"]["rows"] if r["delete_share"] == d))

    out = PL.ROOT / "results"
    (out / "spoof_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_spoof(out / "figures" / "sim5_spoofing.png", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
