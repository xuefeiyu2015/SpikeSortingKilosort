"""Figures for OneBox vs Blackrock synchrony. Drawing only.

Every function takes data already computed by ``_sync/compare.py`` or read by
``_io/analog.py``, and returns the figure it drew.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "plot_asynchrony_before_after",
    "plot_window_traces",
    "plot_window_before_after",
]

BLACKROCK_STYLE = dict(color="black", linewidth=0.9, label="Blackrock")
ONEBOX_STYLE = dict(color="tab:orange", linewidth=0.9, linestyle="--", label="OneBox")


def plot_asynchrony_before_after(
    comparisons: Sequence[Any], summaries: Sequence[dict[str, float]], fig: Any = None
) -> Any:
    """Error over the session, unsynchronized (left) and synchronized (right).

    One colour per event set. Each panel has its own y-axis -- the two differ by
    orders of magnitude, which is the point -- and the worst pair is ringed.
    """
    import matplotlib.pyplot as plt

    fig = fig or plt.figure(figsize=(13, 4.2))
    axes = fig.subplots(1, 2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for which, ax in zip(("before", "after"), axes):
        for color, comp, summary in zip(colors, comparisons, summaries):
            err_ms = getattr(comp, f"{which}_s") * 1e3
            t_min = (comp.blackrock_s - comp.blackrock_s[0]) / 60.0
            ax.scatter(
                t_min, err_ms, s=6, color=color, linewidths=0,
                label=f"{comp.label}: max {summary[f'{which}_max_ms']:.3g} ms, "
                f"median {summary[f'{which}_median_ms']:.3g} ms",
            )
            worst = int(np.argmax(np.abs(err_ms)))
            ax.scatter(t_min[worst], err_ms[worst], s=90, facecolors="none",
                       edgecolors=color, linewidths=1.5)
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_xlabel("time into the session (min)")
        ax.set_ylabel("OneBox - Blackrock (ms)")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title("before: OneBox seconds + one start offset")
    axes[1].set_title("after: onebox_to_blackrock map")
    fig.tight_layout()
    return fig


def _draw_traces(axes: Sequence[Any], traces: Sequence[Any], marker_s: float | None) -> None:
    for ax, trace in zip(axes, traces):
        if trace.t_blackrock is not None:
            ax.plot(trace.t_blackrock, trace.y_blackrock, **BLACKROCK_STYLE)
        if trace.t_onebox is not None:
            ax.plot(trace.t_onebox, trace.y_onebox, **ONEBOX_STYLE)
        if marker_s is not None:
            ax.axvline(marker_s, color="crimson", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(trace.name, rotation=0, ha="right", va="center")
        ax.set_yticks([])


def plot_window_traces(
    traces: Sequence[Any], title: str = "", marker_s: float | None = None,
    time_origin_s: float = 0.0, fig: Any = None,
) -> Any:
    """Every channel stacked, Blackrock solid and OneBox dashed, on Blackrock time.

    Traces are normalized 0..1 per system. ``time_origin_s`` is subtracted from
    the x-axis so it reads in seconds from that point (the NSP clock is ~1.5e9 s).
    """
    import matplotlib.pyplot as plt

    shifted = [_shift(t, time_origin_s) for t in traces]
    fig = fig or plt.figure(figsize=(13, 1.3 * len(traces) + 1))
    axes = np.atleast_1d(fig.subplots(len(traces), 1, sharex=True))
    _draw_traces(axes, shifted, None if marker_s is None else marker_s - time_origin_s)
    axes[0].legend(*_legend(axes), loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel(f"Blackrock time - {time_origin_s:.3f} (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_window_before_after(
    before: Sequence[Any], after: Sequence[Any], title: str = "",
    marker_s: float | None = None, time_origin_s: float = 0.0, fig: Any = None,
) -> Any:
    """The same window twice, side by side: unsynchronized left, synchronized right."""
    import matplotlib.pyplot as plt

    fig = fig or plt.figure(figsize=(14, 1.3 * len(before) + 1.2))
    axes = np.atleast_2d(fig.subplots(len(before), 2, sharex=True, squeeze=False))
    marker = None if marker_s is None else marker_s - time_origin_s
    _draw_traces(axes[:, 0], [_shift(t, time_origin_s) for t in before], marker)
    _draw_traces(axes[:, 1], [_shift(t, time_origin_s) for t in after], marker)
    axes[0, 0].set_title("before synchronization")
    axes[0, 1].set_title("after synchronization")
    axes[0, 1].legend(*_legend(axes[:, 1]), loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(f"Blackrock time - {time_origin_s:.3f} (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _shift(trace: Any, origin: float) -> Any:
    from copy import copy

    out = copy(trace)
    if out.t_blackrock is not None:
        out.t_blackrock = out.t_blackrock - origin
    if out.t_onebox is not None:
        out.t_onebox = out.t_onebox - origin
    return out


def _legend(axes: Sequence[Any]) -> tuple[list[Any], list[str]]:
    seen: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    return list(seen.values()), list(seen.keys())
