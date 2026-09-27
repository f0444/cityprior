"""Short-term motion prediction with and without location memory.

CV      : constant velocity from the current (Kalman) state.
CV+MoD  : constant velocity whose heading/speed are pulled towards the dominant
          locally observed motion (maps-of-dynamics idea, cf. CLiFF-LHMP) that is
          compatible with the agent's current heading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DT
from .geometry import wrap_angle
from .memory import LocationMemory

HORIZON_STEPS = 30  # 3 s at 10 Hz


def make_samples(df: pd.DataFrame, cls: str, t0: float, t1: float, min_speed: float,
                 hist_steps: int = 10, stride_steps: int = 10) -> dict[str, np.ndarray]:
    """Anchor states with a fully observed 3 s future, one every `stride_steps` frames."""
    d = df[(df["cls"] == cls) & (df["t"] >= t0) & (df["t"] < t1)].sort_values(["track", "frame"])
    states, futures, track_ids = [], [], []
    for tid, g in d.groupby("track", sort=False):
        f = g["frame"].to_numpy()
        if len(f) < hist_steps + HORIZON_STEPS + 1:
            continue
        # contiguous runs of frames
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        xy = g[["x", "y"]].to_numpy()
        st = g[["x", "y", "vx", "vy"]].to_numpy()
        for seg in np.split(np.arange(len(f)), breaks):
            stop = len(seg) - HORIZON_STEPS
            if stop <= hist_steps:
                continue
            for a in seg[hist_steps:stop:stride_steps]:
                if np.hypot(*st[a, 2:]) < min_speed:
                    continue
                states.append(st[a])
                futures.append(xy[a + 1: a + 1 + HORIZON_STEPS])
                track_ids.append(tid)
    fut = np.asarray(futures).reshape(-1, HORIZON_STEPS, 2)
    s = np.asarray(states).reshape(-1, 4)
    h0 = np.arctan2(s[:, 3], s[:, 2])
    tail = fut[:, -1] - fut[:, -11]
    turn = np.abs(wrap_angle(np.arctan2(tail[:, 1], tail[:, 0]) - h0))
    return {"state": s, "future": fut, "track": np.asarray(track_ids), "turn_rad": turn}


def predict_cv(state: np.ndarray, steps: int = HORIZON_STEPS) -> np.ndarray:
    t = DT * np.arange(1, steps + 1)[None, :, None]
    return state[:, None, :2] + state[:, None, 2:] * t


def predict_mod(state: np.ndarray, mem: LocationMemory, cls: str, beta: float, gamma: float,
                kappa: float = 2.0, deadband_deg: float = 0.0, steps: int = HORIZON_STEPS) -> np.ndarray:
    """Roll out CV while steering towards the memory's local flow.
    beta: heading pull per step, gamma: speed pull per step (both in [0, 1]);
    headings closer than `deadband_deg` to the local flow are left alone, so the
    prior only acts where the place bends motion (turns, crosswalk ends)."""
    pos = state[:, :2].copy()
    speed = np.hypot(state[:, 2], state[:, 3])
    heading = np.arctan2(state[:, 3], state[:, 2])
    out = np.empty((len(state), steps, 2))
    for k in range(steps):
        direction, mode_speed, ok = mem.flow_mode(cls, pos[:, 0], pos[:, 1], heading, kappa=kappa)
        delta = wrap_angle(direction - heading)
        steer = ok & (np.abs(delta) > np.deg2rad(deadband_deg))
        heading = np.where(steer, heading + beta * delta, heading)
        speed = np.where(ok, speed + gamma * (mode_speed - speed), speed)
        pos = pos + DT * speed[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
        out[:, k] = pos
    return out


def displacement_errors(pred: np.ndarray, future: np.ndarray) -> dict[str, np.ndarray]:
    err = np.linalg.norm(pred - future, axis=2)
    return {"ade": err.mean(axis=1), "fde_1s": err[:, 9], "fde_2s": err[:, 19], "fde_3s": err[:, 29]}


def tune_mod(samples: dict, mem: LocationMemory, cls: str, subset: np.ndarray | None = None
             ) -> tuple[dict, float]:
    """Small grid search on a validation split (optionally on a subset of its
    samples); returns best params and ADE."""
    grid = [dict(beta=b, gamma=g, kappa=k, deadband_deg=d)
            for b in (0.0, 0.05, 0.1, 0.2) for g in (0.0, 0.02, 0.05)
            for k in (2.0, 6.0) for d in (0.0, 10.0, 20.0)]
    best, best_ade = grid[0], np.inf
    for params in grid:
        pred = predict_mod(samples["state"], mem, cls, **params)
        ade = displacement_errors(pred, samples["future"])["ade"]
        ade = (ade if subset is None else ade[subset]).mean()
        if ade < best_ade - 1e-4:
            best, best_ade = params, ade
    return best, float(best_ade)
