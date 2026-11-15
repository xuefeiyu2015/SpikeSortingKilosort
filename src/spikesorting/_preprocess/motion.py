"""What motion correction measured, kept for the drift figures.

SpikeInterface's ``correct_motion`` detects and localises every peak and
estimates the probe's displacement, uses that to interpolate the traces, and
throws the rest away. This keeps the part worth looking at -- the motion
itself, and a subsample of the peaks before and after correction -- in
``drift.npz``, a few tens of MB, beside the binary it describes
(``preprocessed/motion_<preset>/``). Nothing is recomputed to make it: the peaks are the ones the motion
was estimated from, and "after" is those same peaks moved by the estimate, so
the two panels show exactly the same spikes.

Pure NumPy apart from :func:`save_drift`/:func:`load_drift`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DRIFT_PEAKS",
    "DRIFT_FILE",
    "subsample_indices",
    "compute_drift_raster",
    "drift_raster_edges",
    "save_drift",
    "load_drift",
]

#: Peaks kept for the figures. A raster of a million points looks the same as
#: one of thirty million, and costs ~30 MB rather than ~1 GB on the share.
DRIFT_PEAKS = 1_000_000
DRIFT_FILE = "drift.npz"

_ARRAYS = (
    "times_s", "depth_raw_um", "depth_corrected_um", "amplitude",
    "temporal_bins_s", "spatial_bins_um", "displacement_um",
)


def subsample_indices(n: int, n_max: int) -> np.ndarray:
    """Every k-th of ``n`` items, at most ``n_max`` of them; all when ``n <= n_max``.

    A stride, not a random draw: the result is the same on every run, and it
    keeps the full raster's density -- a band that fires twice as hard stays
    twice as dark.
    """
    if n_max < 1:
        raise ValueError(f"n_max must be at least 1, got {n_max}")
    if n <= n_max:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, n_max).round().astype(np.int64))


def drift_raster_edges(
    times_s: np.ndarray,
    depths_um: list[np.ndarray] | tuple[np.ndarray, ...],
    time_bin_s: float = 10.0,
    depth_bin_um: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges shared by every raster drawn from these peaks.

    The depth range covers all of ``depths_um`` (before and after), so the two
    panels are on one axis and a band that moved is seen to move.
    """
    times_s = np.asarray(times_s, dtype=float)
    depth = np.concatenate([np.asarray(d, dtype=float) for d in depths_um])
    if times_s.size == 0 or depth.size == 0:
        raise ValueError("no peaks to draw")
    t0, t1 = float(np.min(times_s)), float(np.max(times_s))
    d0, d1 = float(np.min(depth)), float(np.max(depth))
    t_edges = np.arange(t0, t1 + time_bin_s, time_bin_s)
    d_edges = np.arange(np.floor(d0), d1 + depth_bin_um, depth_bin_um)
    if t_edges.size < 2:
        t_edges = np.array([t0, t0 + time_bin_s])
    if d_edges.size < 2:
        d_edges = np.array([d0, d0 + depth_bin_um])
    return t_edges, d_edges


def compute_drift_raster(
    times_s: np.ndarray,
    depths_um: np.ndarray,
    amplitudes: np.ndarray,
    t_edges: np.ndarray,
    d_edges: np.ndarray,
) -> np.ndarray:
    """Summed absolute amplitude per (depth, time) bin: Kilosort's drift scatter
    as an image.

    Shape ``(len(d_edges) - 1, len(t_edges) - 1)``, depth on rows. An image
    rather than a scatter, so it draws in a second however many peaks there
    are; a unit sitting at one depth is a dark line, a unit drifting is a
    line that bends.
    """
    image, _, _ = np.histogram2d(
        np.asarray(depths_um, dtype=float),
        np.asarray(times_s, dtype=float),
        bins=(d_edges, t_edges),
        weights=np.abs(np.asarray(amplitudes, dtype=float)),
    )
    return image


def save_drift(folder: str | Path, drift: dict[str, Any]) -> Path:
    """Write ``drift`` (the arrays in ``_ARRAYS`` plus ``preset``/``n_peaks``)."""
    missing = [k for k in _ARRAYS if k not in drift]
    if missing:
        raise ValueError(f"drift record is missing {missing}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / DRIFT_FILE
    np.savez(
        path,
        **{k: np.asarray(drift[k]) for k in _ARRAYS},
        preset=np.asarray(str(drift.get("preset", ""))),
        n_peaks=np.asarray(int(drift.get("n_peaks", len(drift["times_s"])))),
    )
    return path


def load_drift(folder: str | Path) -> dict[str, Any] | None:
    """What :func:`save_drift` wrote, or None when there is none (or it is torn)."""
    path = Path(folder) / DRIFT_FILE
    try:
        with np.load(path, allow_pickle=False) as data:
            drift: dict[str, Any] = {k: data[k] for k in _ARRAYS}
            drift["preset"] = str(data["preset"])
            drift["n_peaks"] = int(data["n_peaks"])
    except (OSError, ValueError, KeyError):
        return None
    return drift
