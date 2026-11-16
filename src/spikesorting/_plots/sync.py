"""``sync_overview.png``: what synchronization did, at a glance.

Drawing only. The traces and residuals come from ``pipeline._sync_figure_data``
(computed with ``_sync/overlay.py`` and ``_sync/compare.py``); nothing here
reads a file or computes a number.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

__all__ = ["plot_sync_overview"]

_STYLE = {"blackrock": dict(color="black", lw=0.9, label="Blackrock"),
          "onebox": dict(color="tab:orange", lw=0.9, ls="--", label="OneBox")}


def _draw(ax: Any, sides: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for side in ("blackrock", "onebox"):
        if side in sides:
            x, y = sides[side]
            ax.plot(x, y, **_STYLE[side])


def plot_sync_overview(data: dict[str, Any]) -> Any:
    """Before | after | zoom, one row per shared channel, then every 1 Hz edge.

    ``data`` is ``pipeline._sync_figure_data``'s: ``channels`` (each with
    ``before``/``after``/``zoom`` -> side -> ``(x, y)``), ``edges`` (or None),
    ``window_s``, ``zoom_ms``, ``title``, ``verdict``, ``note``.
    """
    import matplotlib.pyplot as plt

    channels = data["channels"]
    edges = data.get("edges")
    n_rows = len(channels) + (1 if edges is not None else 0)
    fig = plt.figure(figsize=(18, 1.6 * len(channels) + (3.2 if edges is not None else 0) + 1.6),
                     constrained_layout=True)
    heights = [1.0] * len(channels) + ([1.9] if edges is not None else [])
    grid = fig.add_gridspec(n_rows, 3, width_ratios=[3, 3, 1.3], height_ratios=heights)

    width = data["window_s"][1] - data["window_s"][0]
    for row, channel in enumerate(channels):
        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        for ax, view in zip(axes, ("before", "after", "zoom")):
            _draw(ax, channel.get(view, {}))
            ax.set_yticks([])
            if row < len(channels) - 1:
                ax.tick_params(labelbottom=False)
        axes[0].set_ylabel(channel["name"], rotation=0, ha="right", va="center")
        axes[0].set_xlim(0, width)
        axes[1].set_xlim(0, width)
        axes[2].set_xlim(-data["zoom_ms"], data["zoom_ms"])
        axes[2].axvline(0, color="0.7", lw=0.6)
        if row == 0:
            axes[0].set_title("before: each system on its own clock")
            axes[1].set_title("after: OneBox on Blackrock time")
            axes[2].set_title(f"after, +/-{data['zoom_ms']:g} ms around an edge")
            axes[1].legend(loc="upper right", fontsize=8, ncol=2)
        if row == len(channels) - 1:
            axes[0].set_xlabel("s from the window start (each recording's own)")
            axes[1].set_xlabel(f"s from {data['window_s'][0]:.3f} s (Blackrock time)")
            axes[2].set_xlabel("ms from the channel's own edge")

    if edges is not None:
        ax_t = fig.add_subplot(grid[n_rows - 1, 0:2])
        ax_h = fig.add_subplot(grid[n_rows - 1, 2], sharey=ax_t)
        ax_t.plot(edges["t_min"], edges["residual_ms"], ".", ms=1.5, color="tab:blue")
        ax_t.axhline(0, color="k", lw=0.5)
        ax_t.set_xlabel("minutes (Blackrock time, from the first paired edge)")
        ax_t.set_ylabel("OneBox - Blackrock (ms)")
        ax_t.set_title(f"every 1 Hz edge under the map: {edges['n_paired']:,} paired "
                       f"(OneBox {edges['n_onebox']:,}, Blackrock {edges['n_blackrock']:,}); "
                       f"median {edges['median_ms']:+.3f} ms, max |dev| {edges['max_dev_ms']:.3f} ms")
        ax_h.hist(edges["residual_ms"], bins=100, orientation="horizontal", color="tab:blue")
        ax_h.set_xlabel("edges")
        ax_h.tick_params(labelleft=False)

    note = "\n".join(textwrap.wrap(data.get("note") or "", 190))
    fig.suptitle(f"{data['title']}  |  {data['verdict']}" + (f"\n{note}" if note else ""),
                 fontsize=11)
    return fig
