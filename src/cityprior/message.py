"""What the City Model would send to one robotaxi, and what it costs in bytes.

The message is split by update rate:
  * static tile (location memory for the corridor ahead) - sent once, cached, versioned;
  * live object list from infrastructure - 10 Hz, only objects in the corridor;
  * short-term predictions for vulnerable road users - 10 Hz.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .memory import LocationMemory

OBJECT_BYTES = 16      # id, class, x, y (cm), vx, vy (cm/s), confidence, age, padding
PREDICTION_BYTES = 12  # 3 future points (1, 2, 3 s) x (x, y) int16


def corridor_mask(mem: LocationMemory, x: float, y: float, heading: float,
                  length: float = 80.0, half_width: float = 12.0, back: float = 10.0) -> np.ndarray:
    g = mem.grid
    xs = g.x0 + (np.arange(g.nx) + 0.5) * g.cell
    ys = g.y0 + (np.arange(g.ny) + 0.5) * g.cell
    X, Y = np.meshgrid(xs, ys)
    c, s = np.cos(heading), np.sin(heading)
    lon = (X - x) * c + (Y - y) * s
    lat = -(X - x) * s + (Y - y) * c
    return (lon > -back) & (lon < length) & (np.abs(lat) < half_width)


def build_message(mem: LocationMemory, frame_df: pd.DataFrame, ego: pd.Series,
                  length: float = 80.0, half_width: float = 12.0) -> dict:
    """Assemble the message for `ego` (a row of the trajectory table) at one frame."""
    mask = corridor_mask(mem, ego["x"], ego["y"], ego["heading"], length, half_width)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    layers = {k: np.where(mask, v, 0)[r0:r1, c0:c1] for k, v in mem.layers_uint8().items()}
    tile_raw = sum(a.nbytes for a in layers.values())
    tile_zlib = len(zlib.compress(b"".join(a.tobytes() for a in layers.values()), 9))

    others = frame_df[frame_df["track"] != ego["track"]]
    idx = mem.grid.flat(others["x"].to_numpy(), others["y"].to_numpy())
    inside = (idx >= 0) & mask.ravel()[np.clip(idx, 0, None)]
    objects = others[inside]
    n_vru = int(objects["cls"].isin(["ped", "micro"]).sum())
    live_bytes = len(objects) * OBJECT_BYTES + n_vru * PREDICTION_BYTES
    return {
        "mask": mask,
        "tile_shape": (int(r1 - r0), int(c1 - c0)),
        "tile_layers": list(layers),
        "tile_bytes_raw": int(tile_raw),
        "tile_bytes_zlib": int(tile_zlib),
        "objects": objects,
        "n_objects": int(len(objects)),
        "n_vru": n_vru,
        "live_bytes_per_frame": int(live_bytes),
        "live_kbit_s_at_10hz": live_bytes * 8 * 10 / 1000,
    }


def bandwidth_reference(n_cameras: int = 12, mbps_per_4k_stream: float = 15.0) -> dict:
    """Reference points (orders of magnitude, not measurements)."""
    dense_cells = (200 / 0.5) ** 2
    return {
        "raw_video_all_cameras_mbit_s": n_cameras * mbps_per_4k_stream,
        "raw_video_one_camera_mbit_s": mbps_per_4k_stream,
        "dense_tensor_200m_0.5m_10ch_uint8_10hz_mbit_s": dense_cells * 10 * 8 * 10 / 1e6,
    }
