"""Drift figures for motion correction done before Kilosort.

Drawing only: the rasters come from :func:`.._preprocess.motion.compute_drift_raster`,
the motion from what ``correct_motion`` estimated. Kilosort draws its own
``drift_scatter.png``/``drift_amount.png`` only when *it* corrects drift
(``nblocks > 0``); these are the same two pictures for a SpikeInterface preset,
with "before" beside "after".
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["plot_drift_rasters", "plot_motion_traces"]


def plot_drift_rasters(
    before: np.ndarray,
    after: np.ndarray,
    t_edges: np.ndarray,
    d_edges: np.ndarray,
    title: str = "",
) -> Any:
    """Peak amplitude across time and depth, before and after correction.

    ``before``/``after`` are ``(depth, time)`` images on the same edges. One
    grey scale for both -- clipped at the 99th percentile of the occupied bins
    -- so the panels differ only where the correction moved something.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    occupied = np.concatenate([before[before > 0], after[after > 0]])
    vmax = float(np.percentile(occupied, 99)) if occupied.size else 1.0
    extent = (t_edges[0], t_edges[-1], d_edges[0], d_edges[-1])
    for ax, image, name in ((axes[0], before, "before correction"),
                            (axes[1], after, "after correction")):
        ax.imshow(image, origin="lower", aspect="auto", extent=extent,
                  cmap="Greys", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("depth (um)")
    if title:
        fig.suptitle(title)
    return fig


def plot_motion_traces(
    temporal_bins_s: np.ndarray,
    spatial_bins_um: np.ndarray,
    displacement_um: np.ndarray,
    title: str = "",
) -> Any:
    """The estimated displacement over time, one line per depth window.

    ``displacement_um`` is ``(time, depth window)``. A rigid estimate has a
    single window; a non-rigid one (DREDge, MEDiCINe) one per window, coloured
    by depth, so a probe that bends rather than slides shows as lines
    fanning apart.
    """
    import matplotlib.pyplot as plt

    displacement_um = np.asarray(displacement_um, dtype=float)
    if displacement_um.ndim == 1:
        displacement_um = displacement_um[:, None]
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    n = displacement_um.shape[1]
    colours = plt.cm.viridis(np.linspace(0, 1, max(n, 1)))
    for j in range(n):
        label = f"{spatial_bins_um[j]:.0f} um" if j < len(spatial_bins_um) else None
        ax.plot(temporal_bins_s, displacement_um[:, j], color=colours[j], lw=1, label=label)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("estimated displacement (um)")
    if 1 < n <= 12:
        ax.legend(title="depth window", fontsize=8, ncol=2, loc="best")
    elif n > 12:
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(float(np.min(spatial_bins_um)), float(np.max(spatial_bins_um))),
        )
        fig.colorbar(sm, ax=ax, label="depth window (um)")
    if title:
        ax.set_title(title)
    return fig
