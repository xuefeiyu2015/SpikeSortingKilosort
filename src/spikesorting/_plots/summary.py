"""Summary figures for sorted units and for alignment quality.

Drawing only. Each function accepts an optional ``ax``/``fig`` so panels compose
into larger figures, and returns the axes it drew on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "plot_isi_histogram",
    "plot_mean_waveform",
    "plot_firing_rate",
    "plot_amplitudes",
    "plot_unit_summary",
    "plot_alignment_residuals",
    "plot_sorting_overview",
    "save_figure",
]


def _axes(ax: Any = None, **kwargs: Any) -> Any:
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax
    _, ax = plt.subplots(**kwargs)
    return ax


def plot_isi_histogram(
    counts: np.ndarray,
    bin_edges_ms: np.ndarray,
    ax: Any = None,
    refractory_ms: float = 1.5,
    color: str = "0.3",
) -> Any:
    """ISI distribution with the refractory period marked."""
    ax = _axes(ax)
    centers = (np.asarray(bin_edges_ms)[:-1] + np.asarray(bin_edges_ms)[1:]) / 2.0
    width = float(np.diff(bin_edges_ms)[0]) if len(bin_edges_ms) > 1 else 1.0
    ax.bar(centers, counts, width=width, color=color, edgecolor="none")
    ax.axvline(refractory_ms, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("count")
    return ax


def plot_mean_waveform(
    waveform: np.ndarray,
    t_ms: np.ndarray | None = None,
    sem: np.ndarray | None = None,
    ax: Any = None,
    color: str = "black",
    label: str | None = None,
) -> Any:
    """Mean waveform, optionally with a +/-1 SEM band."""
    ax = _axes(ax)
    waveform = np.asarray(waveform)
    x = np.arange(waveform.size) if t_ms is None else np.asarray(t_ms)
    ax.plot(x, waveform, color=color, linewidth=1.5, label=label)
    if sem is not None:
        sem = np.asarray(sem)
        ax.fill_between(x, waveform - sem, waveform + sem, color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("time (ms)" if t_ms is not None else "sample")
    ax.set_ylabel("amplitude")
    return ax


def plot_firing_rate(
    centers_s: np.ndarray, rate_hz: np.ndarray, ax: Any = None, color: str = "steelblue"
) -> Any:
    """Firing rate across the session."""
    ax = _axes(ax)
    ax.plot(np.asarray(centers_s), np.asarray(rate_hz), color=color, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.set_ylim(bottom=0)
    return ax


def plot_amplitudes(
    times_s: np.ndarray,
    amplitudes: np.ndarray,
    ax: Any = None,
    color: str = "0.4",
    max_points: int = 20000,
) -> Any:
    """Spike amplitude against time -- the clearest view of electrode drift.

    Subsamples above ``max_points`` so a million-spike unit still renders.
    """
    ax = _axes(ax)
    times = np.asarray(times_s)
    amps = np.asarray(amplitudes)
    if times.size > max_points:
        step = int(np.ceil(times.size / max_points))
        times, amps = times[::step], amps[::step]
    ax.scatter(times, amps, s=1, color=color, alpha=0.3, linewidths=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    return ax


def plot_unit_summary(unit: dict[str, Any], fig: Any = None) -> Any:
    """Four-panel summary for one unit.

    ``unit`` carries precomputed arrays: ``waveform``, optional ``waveform_t_ms``
    and ``waveform_sem``, ``isi_counts`` + ``isi_edges_ms``, ``rate_centers_s`` +
    ``rate_hz``, optional ``amp_times_s`` + ``amplitudes``, plus ``unit_id``,
    ``label`` and ``channel`` for the title.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(10, 7), dpi=110)
    axes = fig.subplots(2, 2)

    if unit.get("waveform") is not None and np.size(unit["waveform"]):
        plot_mean_waveform(
            unit["waveform"], unit.get("waveform_t_ms"), unit.get("waveform_sem"), ax=axes[0, 0]
        )
    axes[0, 0].set_title("mean waveform (best channel)")

    if unit.get("isi_counts") is not None:
        plot_isi_histogram(unit["isi_counts"], unit["isi_edges_ms"], ax=axes[0, 1])
    axes[0, 1].set_title("ISI")

    if unit.get("rate_centers_s") is not None:
        plot_firing_rate(unit["rate_centers_s"], unit["rate_hz"], ax=axes[1, 0])
    axes[1, 0].set_title("firing rate")

    if unit.get("amplitudes") is not None and np.size(unit["amplitudes"]):
        plot_amplitudes(unit["amp_times_s"], unit["amplitudes"], ax=axes[1, 1])
    axes[1, 1].set_title("amplitude stability")

    fig.suptitle(
        f"unit {unit.get('unit_id')} | {unit.get('label', 'unsorted')} | "
        f"channel {unit.get('channel')} | {unit.get('n_spikes', '?')} spikes"
    )
    fig.tight_layout()
    return fig


def plot_alignment_residuals(
    times_s: np.ndarray,
    residuals_s: np.ndarray,
    tolerance_s: float = 1e-3,
    ax: Any = None,
) -> Any:
    """Step 9 validation: burst residuals over the session, in milliseconds.

    The tolerance band makes a pass/fail obvious at a glance, and a sloped cloud
    rather than a flat one points at an uncorrected clock-rate difference.
    """
    ax = _axes(ax)
    times = np.asarray(times_s)
    residuals_ms = np.asarray(residuals_s) * 1e3
    tolerance_ms = tolerance_s * 1e3

    ax.axhspan(-tolerance_ms, tolerance_ms, color="seagreen", alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.scatter(times, residuals_ms, s=12, color="crimson", linewidths=0)
    ax.set_xlabel("time (s, Blackrock)")
    ax.set_ylabel("residual (ms)")
    ax.set_title(f"14 s burst alignment residuals (tolerance +/-{tolerance_ms:g} ms)")
    return ax


def plot_sorting_overview(
    firing_rates: np.ndarray,
    amplitudes: np.ndarray | None = None,
    contamination_pct: np.ndarray | None = None,
    fig: Any = None,
) -> Any:
    """Session-level distributions across units."""
    import matplotlib.pyplot as plt

    panels = 1 + (amplitudes is not None) + (contamination_pct is not None)
    if fig is None:
        fig = plt.figure(figsize=(4 * panels, 3.2), dpi=110)
    axes = np.atleast_1d(fig.subplots(1, panels))

    axes[0].hist(np.asarray(firing_rates), bins=20, color="0.5")
    axes[0].set_xlabel("firing rate (Hz)")
    axes[0].set_ylabel("# units")

    index = 1
    if amplitudes is not None:
        axes[index].hist(np.asarray(amplitudes), bins=20, color="0.5")
        axes[index].set_xlabel("amplitude")
        axes[index].set_ylabel("# units")
        index += 1
    if contamination_pct is not None:
        axes[index].hist(np.minimum(100, np.asarray(contamination_pct)), bins=np.arange(0, 105, 5), color="0.5")
        axes[index].axvline(10, color="black", linestyle="--", linewidth=1)
        axes[index].set_xlabel("% contamination")
        axes[index].set_ylabel("# units")

    fig.tight_layout()
    return fig


def save_figure(fig: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save and close a figure."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
