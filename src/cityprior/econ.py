"""Part 6: where does city infrastructure pay off? A cost-benefit model per block face.

The unit is one block face like the simulated street: 90 m of kerb with parked cars
and pedestrians stepping out between them. For a robotaxi volume (traversals per
day in the active hours) and the unevenness of pedestrian activity (the ceiling c,
measurable from camera data), the model values five options against the baseline
"robotaxis already use a fleet-learned memory":

    cctv memory     location memory from existing city cameras (software only)
    new memory      new cameras, memory only
    live trusted    new cameras + roadside unit, the car trusts "clear" (heartbeat,
                    onboard priority, fleet audit as in E9)
    live guaranteed same hardware, but the car keeps its old speed so that it is never
                    worse than without the camera: the benefit is taken as safety only

Per-traversal effects come from the simulator (E14: seconds saved at equal risk,
collision reduction at equal time), staleness after a change from E8, fleet
observability from E7. Money comes from published unit values (see PARAMS).
All values are 2023 US dollars.

    python -m cityprior.econ     # writes results/econ_metrics.json + figures/econ*.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from . import pipeline as PL

RESULTS = PL.ROOT / "results"
MILE_M = 1609.344
DAYS = 365.0
OPTIONS = ("cctv memory", "new memory", "live trusted", "live guaranteed")


@dataclass(frozen=True)
class Param:
    base: float
    low: float
    high: float
    unit: str
    source: str


# "sourced" values quote a publication; "assumption" values are stated ranges the
# sensitivity analysis varies.
PARAMS: dict[str, Param] = {
    "discount_rate": Param(0.07, 0.031, 0.07, "per year",
                           "USDOT BCA Guidance 2025 Update II (May 2025): 7%; 2025 Update (Nov 2024): 3.1%"),
    "horizon_years": Param(10, 7, 15, "years", "assumption: service life of roadside cameras and units"),
    "vtts": Param(21.10, 19.40, 33.50, "$/person-hour",
                  "USDOT BCA Guidance 2025, Table A-2: all purposes $21.10 (personal $19.40, business $33.50)"),
    "riders_per_traversal": Param(0.7, 0.5, 1.0, "persons",
                                  "assumption: occupied share of robotaxi miles x riders per trip"),
    "free_flow_share": Param(0.5, 0.2, 0.8, "share",
                             "assumption: traversals where pedestrian risk, not traffic ahead or signals, "
                             "limits the robotaxi's speed"),
    "vehicle_hour": Param(10.0, 5.0, 30.0, "$/vehicle-hour",
                          "assumption: operator cost of a robotaxi hour ($0.3-2 per mile at 12-15 mph)"),
    "cost_killed": Param(13.2e6, 13.2e6, 13.2e6, "$", "USDOT BCA Guidance 2025, Table A-1: K = $13,200,000"),
    "cost_injured": Param(229_800, 118_000, 1_254_700, "$",
                          "USDOT BCA Guidance 2025, Table A-1: injured, severity unknown (C $118,000 .. A $1,254,700)"),
    "fatal_share": Param(0.03, 0.01, 0.097, "share",
                         "assumption for low-speed urban crashes; 0.097 = US 2023 (NHTSA: 7,314 killed, 68,244 injured)"),
    "ped_crash_rate": Param(7 / 271.3, 2.81 / 271.3, 14.42 / 271.3, "per million miles",
                            "Waymo Safety Impact hub, through June 2026: 7 pedestrian injury crashes in 271.3M "
                            "rider-only miles (exact Poisson 95% interval 2.81-14.42; human benchmark: 93)"),
    "site_exposure": Param(3.0, 1.0, 10.0, "x average",
                           "assumption: pedestrian activity of a busy block relative to the operating area"),
    "addressable_share": Param(0.3, 0.1, 0.5, "share",
                               "assumption: pedestrian crashes of the occluded step-out type the prior affects"),
    "cameras_per_site": Param(2, 1, 3, "cameras", "assumption; TGSIM: 12 cameras for 4 intersections"),
    "camera_unit": Param(6220, 4575, 8018, "$/camera",
                         "ITS JPO Sample Unit Cost Database, CCTV video camera furnished and installed, "
                         "2017-2023 entries (n = 46), 2023$: median, interquartile range"),
    "site_works": Param(10_000, 5_000, 25_000, "$", "assumption: mounting, power and backhaul per site"),
    "rsu": Param(11_000, 7_000, 50_000, "$",
                 "ITS America National V2X Deployment Plan (2023) via ITS JPO 2026-SC00587: $7,000-15,000 per "
                 "RSU-ready intersection, up to $50,000 with planning and state requirements"),
    "edge_compute": Param(5_000, 2_000, 15_000, "$", "assumption: real-time perception hardware per site"),
    "integration": Param(2_000, 1_000, 5_000, "$", "assumption: connecting an existing camera to the memory service"),
    "analytics_per_camera": Param(600, 300, 1_500, "$/camera-year", "assumption: video analytics compute/licence"),
    "maintenance_share": Param(0.10, 0.05, 0.20, "of capex per year", "assumption"),
    "changes_per_year": Param(4, 1, 12, "per site", "assumption: works, new shops, school terms, events"),
    "operators": Param(1, 1, 3, "fleets not sharing data", "assumption"),
    "active_hours": Param(16, 12, 18, "hours/day", "assumption: hours with pedestrian activity"),
}


def base_params() -> dict[str, float]:
    return {k: v.base for k, v in PARAMS.items()}


# ----------------------------------------------------------------- simulator inputs
class Effects:
    """Per-traversal effects of each information source vs no memory, interpolated in c."""

    SOURCES = {"memory": "memory", "live trusted": "live camera, trusted",
               "live guaranteed": "live camera, adds caution only"}

    def __init__(self, value: dict | None = None):
        value = value or json.loads((RESULTS / "value_metrics.json").read_text())
        rows = sorted(value["rows"], key=lambda r: r["ceiling"])
        self.c = np.array([r["ceiling"] for r in rows])
        self.street_len = value["street_len_m"]
        self.ref_risk = np.array([r["reference"]["collisions_per_10k"] for r in rows])
        self.dt = {k: np.array([r["sources"][s]["seconds_saved_at_equal_risk"] for r in rows])
                   for k, s in self.SOURCES.items()}
        self.red = {k: np.array([r["sources"][s]["reduction_at_equal_time"] for r in rows])
                    for k, s in self.SOURCES.items()}

    def seconds(self, source: str, c: float) -> float:
        return float(np.interp(c, self.c, self.dt[source]))

    def reduction(self, source: str, c: float) -> float:
        return float(np.interp(c, self.c, self.red[source]))


# ------------------------------------------------------------------- fleet learning
N0_OBSERVABILITY = 61.0       # passes/h; fits E7: 5.5/h -> 0.084, 18.4/h -> 0.26, 184/h -> 0.997
CAMERA_OBSERVABILITY = 0.95
STALE_PENALTY = 1.7           # E8: stale memory costs 1.7x the benefit of fresh memory (worse than none)
# hours of fully stale memory per change (E8 lost benefit-hours / 1.7):
#   camera 0.83 h, fleet 10% (18.4 passes/h) 0.92 h, fleet 3% (5.5/h, detector mostly silent) 13.9 h
STALE_ANCHORS = ((5.5, 13.9), (18.4, 0.92), (184.0, 0.83))
TAU_CAMERA = 0.83


def observability(passes_per_hour: float) -> float:
    return float(1 - np.exp(-passes_per_hour / N0_OBSERVABILITY))


def stale_hours(passes_per_hour: float, max_hours: float = 8760.0) -> float:
    """Hours of fully stale fleet memory after the place changes, by fleet passes per hour.
    Log-log interpolation between the E8 anchors; below 5.5/h evidence-limited (~1/n)."""
    n = max(passes_per_hour, 1e-6)
    (n1, t1), (n2, t2), (n3, t3) = STALE_ANCHORS
    if n <= n1:
        tau = t1 * n1 / n
    elif n >= n3:
        tau = t3
    else:
        xs, ys = np.log([n1, n2, n3]), np.log([t1, t2, t3])
        tau = float(np.exp(np.interp(np.log(n), xs, ys)))
    return float(min(tau, max_hours))


# ----------------------------------------------------------------------- money
def value_per_second(P: dict) -> float:
    """$ per second the simulator saves on one traversal: riders' time plus the vehicle's time,
    counted only on traversals where the robotaxi is free to use it."""
    return P["free_flow_share"] * (P["vtts"] * P["riders_per_traversal"] + P["vehicle_hour"]) / 3600.0


def crash_cost(P: dict) -> float:
    return P["fatal_share"] * P["cost_killed"] + (1 - P["fatal_share"]) * P["cost_injured"]


def base_crash_risk(P: dict, street_len: float) -> float:
    """Addressable pedestrian injury crashes per traversal of the block, without memory."""
    return (P["ped_crash_rate"] * 1e-6 * street_len / MILE_M * P["site_exposure"] * P["addressable_share"])


def per_traversal(option: str, c: float, P: dict, E: Effects, channel: str = "time") -> float:
    """$ per traversal of the option beyond fresh (fleet) memory."""
    if option in ("cctv memory", "new memory"):
        return 0.0                                    # memory value itself is already in the baseline
    src = option
    if channel == "time" and option == "live trusted":
        return max(E.seconds(src, c) - E.seconds("memory", c), 0.0) * value_per_second(P)
    # safety: the old speed is kept, collisions fall
    risk0 = base_crash_risk(P, E.street_len)
    return max(E.reduction(src, c) - E.reduction("memory", c), 0.0) * risk0 * crash_cost(P)


def memory_value_per_traversal(c: float, P: dict, E: Effects) -> float:
    """$ per traversal of fresh memory vs none (taken as time)."""
    return max(E.seconds("memory", c), 0.0) * value_per_second(P)


def camera_memory_increment(n_day: float, c: float, P: dict, E: Effects) -> float:
    """$ per year that camera memory adds to fleet memory: it is stale for less time after a change.
    With several operators not sharing data, each fleet learns from its own share of the passes."""
    ops = max(int(round(P["operators"])), 1)
    per_hour = n_day / P["active_hours"]
    b = memory_value_per_traversal(c, P, E)
    gap = stale_hours(per_hour / ops, 8760.0 / P["changes_per_year"]) - TAU_CAMERA
    return P["changes_per_year"] * STALE_PENALTY * max(gap, 0.0) * per_hour * b


def costs(option: str, P: dict) -> tuple[float, float]:
    """(capex, opex per year) for one site."""
    n_cam = P["cameras_per_site"]
    if option == "cctv memory":
        return P["integration"] * n_cam, P["analytics_per_camera"] * n_cam
    if option == "new memory":
        capex = n_cam * P["camera_unit"] + P["site_works"]
        return capex, P["maintenance_share"] * capex + P["analytics_per_camera"] * n_cam
    capex = n_cam * P["camera_unit"] + P["site_works"] + P["rsu"] + P["edge_compute"]
    return capex, P["maintenance_share"] * capex + P["analytics_per_camera"] * n_cam


def annuity(P: dict) -> float:
    r, T = P["discount_rate"], int(round(P["horizon_years"]))
    return float(sum((1 + r) ** -t for t in range(1, T + 1)))


def annual_benefit(option: str, n_day: float, c: float, P: dict, E: Effects, channel: str = "time") -> float:
    if option in ("cctv memory", "new memory"):
        return camera_memory_increment(n_day, c, P, E)
    return n_day * DAYS * per_traversal(option, c, P, E, "safety" if option == "live guaranteed" else channel)


def npv(option: str, n_day: float, c: float, P: dict, E: Effects, channel: str = "time") -> float:
    capex, opex = costs(option, P)
    return (annual_benefit(option, n_day, c, P, E, channel) - opex) * annuity(P) - capex


def break_even(option: str, c: float, P: dict, E: Effects, n_max: float = 1e6, channel: str = "time") -> float:
    """Smallest robotaxi traversals per day with NPV >= 0 (inf if none up to n_max)."""
    grid = np.geomspace(1, n_max, 241)
    vals = np.array([npv(option, n, c, P, E, channel) for n in grid])
    ok = np.flatnonzero(vals >= 0)
    if len(ok) == 0:
        return float("inf")
    i = ok[0]
    if i == 0:
        return float(grid[0])
    lo, hi = grid[i - 1], grid[i]
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if npv(option, mid, c, P, E, channel) < 0 else (lo, mid)
    return float(hi)


# ---------------------------------------------------------------- uncertainty
def sample_params(rng: np.random.Generator, n: int) -> list[dict]:
    """Independent triangular draws on (low, base, high) for every parameter with a range."""
    draws = []
    for _ in range(n):
        P = {}
        for k, v in PARAMS.items():
            if v.low == v.high:
                P[k] = v.base
            elif k in ("operators", "cameras_per_site", "horizon_years"):
                P[k] = v.base
            else:
                P[k] = float(rng.triangular(v.low, v.base, v.high)) if v.low < v.base < v.high else \
                    float(rng.uniform(min(v.low, v.high), max(v.low, v.high)))
        draws.append(P)
    return draws


def tornado(option: str, c: float, E: Effects) -> list[dict]:
    """Break-even volume when one parameter at a time is set to its low / high value."""
    P0 = base_params()
    base = break_even(option, c, P0, E)
    rows = []
    for k, v in PARAMS.items():
        if v.low == v.high:
            continue
        lo = break_even(option, c, {**P0, k: v.low}, E)
        hi = break_even(option, c, {**P0, k: v.high}, E)
        rows.append({"param": k, "low_value": v.low, "high_value": v.high, "break_even_low": lo,
                     "break_even_high": hi, "swing": abs(np.log(max(lo, 1)) - np.log(max(hi, 1)))
                     if np.isfinite(lo) and np.isfinite(hi) else float("inf")})
    rows.sort(key=lambda r: -r["swing"] if np.isfinite(r["swing"]) else -1e9)
    return [{"base": base}] + rows


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> dict:
    from .econ_figures import make_all
    E = Effects()
    P = base_params()
    c_tgsim = float(json.loads((RESULTS / "sim_metrics.json").read_text())["E1"]["summary"]["heterogeneity_ceiling"])
    res: dict = {"params": {k: vars(v) for k, v in PARAMS.items()}, "c_tgsim": c_tgsim,
                 "value_per_second": value_per_second(P), "crash_cost": crash_cost(P),
                 "base_crash_risk_per_traversal": base_crash_risk(P, E.street_len)}

    # per-traversal values by unevenness
    cs = np.linspace(E.c.min(), 1.0, 25)
    res["per_traversal"] = {
        "c": cs.tolist(),
        "memory_seconds": [E.seconds("memory", c) for c in cs],
        "live_seconds": [E.seconds("live trusted", c) for c in cs],
        "memory_$": [memory_value_per_traversal(c, P, E) for c in cs],
        "live_trusted_$": [per_traversal("live trusted", c, P, E) for c in cs],
        "live_guaranteed_$": [per_traversal("live guaranteed", c, P, E) for c in cs],
        "memory_as_safety_$": [E.reduction("memory", c) * res["base_crash_risk_per_traversal"] * crash_cost(P)
                               for c in cs],
    }
    log(f"value of a second ${value_per_second(P):.4f}; crash ${crash_cost(P):,.0f}; "
        f"risk per traversal {res['base_crash_risk_per_traversal']:.2e}")

    # NPV curves at this street's unevenness
    ns = np.geomspace(10, 1e5, 61)
    res["npv_curves"] = {"n_day": ns.tolist(),
                         **{o: [npv(o, n, c_tgsim, P, E) for n in ns] for o in OPTIONS}}
    res["break_even_base"] = {o: break_even(o, c_tgsim, P, E) for o in OPTIONS}
    log("break-even (traversals/day, base): " + ", ".join(f"{o}: {v:,.0f}" for o, v in res["break_even_base"].items()))

    # side numbers for the text
    capex, opex = costs("live trusted", P)
    annual_cost = capex / annuity(P) + opex
    cm_capex, cm_opex = costs("cctv memory", P)
    mem_b = memory_value_per_traversal(c_tgsim, P, E)
    res["side"] = {
        "live_seconds_beyond_memory": E.seconds("live trusted", c_tgsim) - E.seconds("memory", c_tgsim),
        "live_value_per_traversal": per_traversal("live trusted", c_tgsim, P, E),
        "live_capex": capex, "live_opex": opex, "live_annual_cost": annual_cost,
        "live_cost_per_traversal": {int(n): annual_cost / (n * DAYS) for n in (3000, 10000, 30000)},
        "memory_seconds": E.seconds("memory", c_tgsim), "memory_value_per_traversal": mem_b,
        "memory_as_safety_per_traversal": E.reduction("memory", c_tgsim) * base_crash_risk(P, E.street_len)
        * crash_cost(P),
        "memory_as_safety_human_rate": E.reduction("memory", c_tgsim) * base_crash_risk(
            {**P, "ped_crash_rate": 93 / 271.3}, E.street_len) * crash_cost(P),
        "cctv_memory_annual_cost": cm_capex / annuity(P) + cm_opex,
        "cctv_break_even_if_fleets_have_no_memory": (cm_capex / annuity(P) + cm_opex) / (DAYS * mem_b),
        "camera_memory_increment_max": max(camera_memory_increment(n, c_tgsim, {**P, "operators": 3}, E)
                                           for n in np.geomspace(1, 1e5, 200)),
        "break_even_live_trusted_3pct": break_even("live trusted", c_tgsim, {**P, "discount_rate": 0.031}, E),
    }
    log("side: " + json.dumps(res["side"], default=float))

    # camera memory over fleet memory, one vs three operators
    res["memory_increment"] = {"n_day": ns.tolist(), "memory_full_$": [n * DAYS * memory_value_per_traversal(
        c_tgsim, P, E) for n in ns]}
    for ops in (1, 3):
        res["memory_increment"][f"operators_{ops}"] = [camera_memory_increment(n, c_tgsim, {**P, "operators": ops}, E)
                                                       for n in ns]
    res["memory_increment"]["cctv_opex"] = costs("cctv memory", P)[1]

    # Monte Carlo over parameters
    rng = np.random.default_rng(0)
    draws = sample_params(rng, 2000)
    mc = {}
    for o in ("live trusted", "cctv memory"):
        be = np.array([break_even(o, c_tgsim, d, E) for d in draws[:400]])
        fin = be[np.isfinite(be)]
        mc[o] = {"share_finite": float(np.isfinite(be).mean()),
                 "quantiles": np.percentile(fin, [5, 25, 50, 75, 95]).tolist() if len(fin) else None}
    grid_n = np.array([100, 300, 1000, 3000, 10000, 30000])
    mc["p_positive_live_trusted"] = {int(n): float(np.mean([npv("live trusted", n, c_tgsim, d, E) > 0
                                                            for d in draws])) for n in grid_n}
    res["monte_carlo"] = mc
    log(f"Monte Carlo: {mc}")

    # decision map: best option by volume x unevenness
    cm = np.linspace(E.c.min(), 1.0, 31)
    nm = np.geomspace(10, 1e5, 41)
    best = [[max(("none",) + OPTIONS, key=lambda o: 0.0 if o == "none" else npv(o, n, c, P, E)) for n in nm]
            for c in cm]
    res["decision_map"] = {"c": cm.tolist(), "n_day": nm.tolist(), "best": best}
    res["tornado_live_trusted"] = tornado("live trusted", c_tgsim, E)
    res["break_even_by_c"] = {"c": cm.tolist(), "live trusted": [break_even("live trusted", c, P, E) for c in cm]}

    (RESULTS / "econ_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    make_all(RESULTS / "figures", res)
    log("done")
    return res


if __name__ == "__main__":
    main()
