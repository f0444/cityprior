"""E8: keeping location memory fresh.

The pedestrian hotspot moves (as in E4) while memory keeps receiving
observations every 15 minutes, from a fixed camera or from a fleet. Update
strategies:

  static        never updated (E4)
  cumulative    adds every observation; old data keeps pulling it back
  forgetting    observations lose half their weight every `half_life` hours
  detect+reset  a sequential G-test compares the last hour of observations
                with what memory expects, per 10 m segment; on a significant
                mismatch memory restarts from the data that triggered it

Each memory state is scored in the closed-loop simulator at the operating
caution level: collisions per 10,000 traversals and trip time.

    python -m cityprior.sim.change      # writes results/change_metrics.json + figures/sim4_*.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .. import pipeline as PL
from .planner import believed_intensity, memory_from_counts
from .world import Profile, SimParams, ped_speed_quantiles, profile_from_tgsim

STEP_H = 0.25            # observations arrive every 15 minutes
PRE_HOURS = 10.0         # memory before the change (as in E1-E7)
WINDOW_H = 1.0           # detector window
MIN_REF_H = 2.0          # detector needs this much reference data after a reset
SEGMENT_M = 10.0
ALPHA = 1e-4             # per check; 96 checks a day -> about one false alarm per 100 days
HALF_LIFE_H = 4.0
EVAL_HOURS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 48.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Stream:
    """Observed emergence counts per metre, one row per 15-minute step."""

    x: np.ndarray
    t0: np.ndarray            # step start, hours relative to the change
    counts: np.ndarray        # (n_steps, n_x)
    q: np.ndarray             # observability profile used when observing

    @property
    def t1(self) -> np.ndarray:
        return self.t0 + STEP_H


def observe(rng: np.random.Generator, before: Profile, after: Profile | None, q: np.ndarray,
            pre_h: float = PRE_HOURS, post_h: float = 48.0) -> Stream:
    t0 = np.arange(-pre_h, post_h - 1e-9, STEP_H)
    lam = np.where((t0 < 0)[:, None] | (after is None), before.lam[None, :],
                   (after.lam if after is not None else before.lam)[None, :])
    counts = rng.poisson(lam * STEP_H * 3600.0 * q[None, :]).astype(float)
    return Stream(before.x, t0, counts, q)


def _memory(stream: Stream, weights: np.ndarray, fallback_mean: float) -> Profile:
    seconds = float(weights.sum() * STEP_H * 3600.0)
    return memory_from_counts(stream.x, weights @ stream.counts, seconds, stream.q, fallback_mean=fallback_mean)


def g_statistic(k: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance of observed segment counts k against expectations mu."""
    mu = np.maximum(mu, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(k > 0, k * np.log(k / mu), 0.0)
    return float(2.0 * np.sum(term - (k - mu)))


def segment_index(x: np.ndarray) -> np.ndarray:
    return np.minimum((x // SEGMENT_M).astype(int), int(x.max() // SEGMENT_M) - 1)


def detect_changes(stream: Stream, fallback_mean: float, alpha: float = ALPHA, start_h: float = 0.0,
                   stop_h: float | None = None) -> list[float]:
    """Sequential windowed G-test, one check per step from `start_h`. After each
    detection, memory (and the detector's reference) restarts at the window
    that triggered it. Returns detection times (hours)."""
    seg = segment_index(stream.x)
    n_seg = seg.max() + 1
    threshold = chi2.ppf(1 - alpha, df=n_seg)
    reset_at = -np.inf
    detections = []
    stop_h = stream.t1[-1] if stop_h is None else stop_h
    for t in np.arange(start_h + STEP_H, stop_h + 1e-9, STEP_H):
        in_window = (stream.t0 >= t - WINDOW_H - 1e-9) & (stream.t1 <= t + 1e-9)
        in_ref = (stream.t0 >= reset_at - 1e-9) & (stream.t1 <= t - WINDOW_H + 1e-9)
        if in_ref.sum() * STEP_H < MIN_REF_H - 1e-9:
            continue
        ref = _memory(stream, in_ref.astype(float), fallback_mean)
        expected = ref.lam * WINDOW_H * 3600.0 * stream.q
        k = np.bincount(seg, weights=stream.counts[in_window].sum(axis=0), minlength=n_seg)
        mu = np.bincount(seg, weights=expected, minlength=n_seg)
        if g_statistic(k, mu) > threshold:
            detections.append(float(t))
            reset_at = t - WINDOW_H
    return detections


def memory_at(stream: Stream, t: float, strategy: str, fallback_mean: float,
              detections: list[float] | None = None) -> Profile:
    done = stream.t1 <= t + 1e-9
    if strategy == "static":
        w = (stream.t1 <= 1e-9).astype(float)
    elif strategy == "cumulative":
        w = done.astype(float)
    elif strategy == "forgetting":
        w = np.where(done, 0.5 ** ((t - stream.t1) / HALF_LIFE_H), 0.0)
    elif strategy == "detect":
        last = max([d for d in (detections or []) if d <= t + 1e-9], default=None)
        start = -np.inf if last is None else last - WINDOW_H
        w = (done & (stream.t0 >= start - 1e-9)).astype(float)
    else:
        raise ValueError(strategy)
    return _memory(stream, w, fallback_mean)


def _quantile(values: np.ndarray, q: float) -> float | None:
    """Quantile that treats undetected changes (inf) as the largest values;
    None if the quantile itself is undetected."""
    v = np.sort(np.asarray(values, float))
    x = v[int(np.ceil(q * len(v))) - 1]
    return float(x) if np.isfinite(x) else None


def main() -> dict:
    from .experiments import V_OPERATING, Bench
    from .figures import fig_change

    p = SimParams()
    df, events = PL.prepare()
    truth = profile_from_tgsim(events["emergence"], street_len=p.street_len)
    mid = events["emergence"][events["emergence"]["x"].between(70, 160) & events["emergence"]["y"].between(28, 50)]
    bench = Bench(p, truth, ped_speed_quantiles(df, mid["track"]))
    inner = (truth.x >= 5) & (truth.x <= truth.x[-1] - 5)
    shift = float(truth.x[inner][np.argmin(truth.lam[inner])] - truth.x[np.argmax(truth.lam)])
    moved = truth.shifted(shift)

    fleet = json.loads((PL.ROOT / "results/fleet_metrics.json").read_text())["observability"]
    q_cam = np.full(len(truth.x), 0.95)
    q_fleet = {name: np.interp(truth.x, v["x"], v["q"]) for name, v in fleet.items()}

    def score(mem: Profile, world: Profile) -> dict:
        r = bench.run(believed_intensity(mem, p, floor=0.1), V_OPERATING, truth=world)
        return {"collisions_per_10k": r["collisions_per_10k"], "trip_time": r["trip_time"]}

    flat = Profile(truth.x, np.full(len(truth.x), truth.mean))
    res: dict = {"shift_m": shift, "alpha": ALPHA, "window_h": WINDOW_H, "half_life_h": HALF_LIFE_H,
                 "references": {"no_memory": score(flat, moved), "perfect_memory": score(moved, moved)}}
    log(f"references after the change: {res['references']}")

    # ---------------------------------------------------------------- recovery
    strategies = [("static", "camera", q_cam), ("cumulative", "camera", q_cam), ("forgetting", "camera", q_cam),
                  ("detect", "camera", q_cam), ("detect", "fleet 10%", q_fleet["fleet 10%"])]
    curves = {f"{s} · {src}": [] for s, src, _ in strategies}
    detections = {f"{s} · {src}": [] for s, src, _ in strategies if s == "detect"}
    for seed in range(3):
        streams = {"camera": observe(np.random.default_rng(300 + seed), truth, moved, q_cam),
                   "fleet 10%": observe(np.random.default_rng(400 + seed), truth, moved, q_fleet["fleet 10%"])}
        for strat, src, _ in strategies:
            key = f"{strat} · {src}"
            det = detect_changes(streams[src], truth.mean) if strat == "detect" else None
            if det is not None:
                detections[key].append(det)
            row, cache = [], {}
            for t in EVAL_HOURS:
                if strat == "static" and "static" in cache:
                    row.append(cache["static"])
                    continue
                s = score(memory_at(streams[src], t, strat, truth.mean, det), moved)
                cache["static"] = s
                row.append(s)
            curves[key].append(row)
        log(f"seed {seed}: " + ", ".join(f"{k}: {np.round([r['collisions_per_10k'] for r in v[-1]], 2).tolist()}"
                                         for k, v in curves.items()))
    res["eval_hours"] = EVAL_HOURS
    res["curves"] = curves
    res["detections_in_runs"] = detections

    # ------------------------------------------- the price of vigilance (no change)
    steady = {}
    for strat, src, q in strategies:
        key = f"{strat} · {src}"
        vals = []
        for seed in range(3):
            st = observe(np.random.default_rng(500 + seed), truth, None, q)
            det = detect_changes(st, truth.mean) if strat == "detect" else None
            vals.append(score(memory_at(st, 24.0, strat, truth.mean, det), truth))
        steady[key] = vals
    res["steady_state_no_change"] = steady
    res["steady_references"] = {"no_memory": score(flat, truth), "perfect_memory": score(truth, truth)}
    log("steady state: " + ", ".join(f"{k}: {np.mean([v['collisions_per_10k'] for v in vals]):.2f}"
                                     for k, vals in steady.items()))

    # ------------------------------------------------ detector delay and false alarms
    det_stats = {}
    for name, q in (("camera", q_cam), ("fleet 10%", q_fleet["fleet 10%"]), ("fleet 3%", q_fleet["fleet 3%"])):
        delays = []
        for seed in range(200):
            st = observe(np.random.default_rng(10_000 + seed), truth, moved, q, post_h=24.0)
            d = detect_changes(st, truth.mean)
            delays.append(d[0] if d else np.inf)
        false_alarms, days = 0, 0.0
        for seed in range(40):
            st = observe(np.random.default_rng(20_000 + seed), truth, None, q, post_h=30 * 24.0)
            false_alarms += len(detect_changes(st, truth.mean))
            days += 30.0
        delays = np.array(delays)
        det_stats[name] = {
            "median_delay_h": _quantile(delays, 0.5), "p90_delay_h": _quantile(delays, 0.9),
            "share_detected_24h": float(np.isfinite(delays).mean()),
            "false_alarms_per_30_days": 30.0 * false_alarms / days,
            "delays": [float(d) if np.isfinite(d) else None for d in delays],
        }
        log(f"detector {name}: median delay {det_stats[name]['median_delay_h']} h, p90 "
            f"{det_stats[name]['p90_delay_h']} h, detected within 24 h {det_stats[name]['share_detected_24h']:.2f}, "
            f"false alarms / 30 days {det_stats[name]['false_alarms_per_30_days']:.2f}")
    res["detector"] = det_stats

    out = PL.ROOT / "results"
    (out / "change_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    fig_change(out / "figures" / "sim4_change_detection.png", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
