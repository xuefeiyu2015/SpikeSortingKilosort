"""Probe geometry figures.

Rendering only -- takes a probe dict and draws it.

Worth actually looking at. No code can tell you a channel map is wrong, but a
plot usually can: a Utah array that should be a 10x10 grid but comes out as a
line, or a Neuropixels shank with sites in the wrong column, is obvious on sight
and invisible in the numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["plot_probe_geometry", "plot_probe_channel_order"]


def _axes(ax: Any = None, **kwargs: Any) -> Any:
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax
    _, ax = plt.subplots(**kwargs)
    return ax


def plot_probe_geometry(
    probe: dict,
    ax: Any = None,
    annotate: bool | None = None,
    max_annotations: int = 128,
    title: str | None = None,
) -> Any:
    """Scatter the contacts in physical space, coloured by shank/group.

    ``annotate`` labels each contact with its channel index; by default it turns
    itself on only when there are few enough contacts to stay readable.
    """
    ax = _axes(ax, figsize=(4, 8))

    xc = np.asarray(probe["xc"], dtype=float)
    yc = np.asarray(probe["yc"], dtype=float)
    kcoords = np.asarray(probe.get("kcoords", np.zeros_like(xc)), dtype=float)
    chan_map = np.asarray(probe.get("chanMap", np.arange(xc.size)))

    groups = np.unique(kcoords)
    # One group per channel means "independent electrodes" (a Utah array), where
    # colouring by group is noise rather than information.
    colour_by_group = 1 < groups.size < xc.size

    if colour_by_group:
        ax.scatter(xc, yc, c=kcoords, s=24, cmap="viridis", edgecolors="none")
    else:
        ax.scatter(xc, yc, s=24, color="0.25", edgecolors="none")

    if annotate is None:
        annotate = xc.size <= max_annotations
    if annotate:
        for x, y, channel in zip(xc, yc, chan_map):
            ax.annotate(str(int(channel)), (x, y), fontsize=6, xytext=(3, 0),
                        textcoords="offset points", va="center")

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="datalim")

    if title is None:
        title = f"{probe.get('n_chan', xc.size)} channels"
        if colour_by_group:
            title += f", {groups.size} shanks"
        if probe.get("_placeholder"):
            title += "  [PLACEHOLDER GEOMETRY]"
    ax.set_title(title, fontsize=10)
    return ax


def plot_probe_channel_order(probe: dict, ax: Any = None) -> Any:
    """Depth against channel index -- reveals a scrambled or shifted map.

    On a linear probe the correct map is monotonic. A jump or a reversal here is
    the signature of an off-by-one index convention or the wrong ``.cmp``.
    """
    ax = _axes(ax, figsize=(6, 3))
    yc = np.asarray(probe["yc"], dtype=float)
    chan_map = np.asarray(probe.get("chanMap", np.arange(yc.size)))

    ax.plot(chan_map, yc, ".-", linewidth=0.6, markersize=3, color="0.3")
    ax.set_xlabel("channel index")
    ax.set_ylabel("y (um)")
    ax.set_title("depth vs channel order", fontsize=10)
    return ax
