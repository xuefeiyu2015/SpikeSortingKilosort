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
    "plot_unit_page",
    "save_unit_pages",
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
    band: np.ndarray | None = None,
    ax: Any = None,
    color: str = "black",
    label: str | None = None,
) -> Any:
    """Mean waveform, optionally with a +/- ``band`` around it (std, SEM, ...)."""
    ax = _axes(ax)
    waveform = np.asarray(waveform)
    x = np.arange(waveform.size) if t_ms is None else np.asarray(t_ms)
    ax.plot(x, waveform, color=color, linewidth=1.5, label=label)
    if band is not None:
        band = np.asarray(band)
        ax.fill_between(x, waveform - band, waveform + band, color=color, alpha=0.2, linewidth=0)
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


def _no_waveform(ax: Any) -> None:
    """Say so in the panel: there is no MeanWaveform for this unit in the .mat."""
    ax.text(0.5, 0.5, "no measured waveform", ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_unit_summary(unit: dict[str, Any], fig: Any = None) -> Any:
    """Four-panel summary for one unit.

    ``unit`` carries precomputed arrays: ``waveform``, optional ``waveform_t_ms``
    and ``waveform_std``, ``isi_counts`` + ``isi_edges_ms``, ``rate_centers_s`` +
    ``rate_hz``, optional ``amp_times_s`` + ``amplitudes``, plus ``unit_id``,
    ``label`` and ``channel`` for the title.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(10, 7), dpi=110)
    axes = fig.subplots(2, 2)

    if unit.get("waveform") is not None and np.size(unit["waveform"]):
        plot_mean_waveform(
            unit["waveform"], unit.get("waveform_t_ms"), unit.get("waveform_std"), ax=axes[0, 0]
        )
    else:
        _no_waveform(axes[0, 0])
    axes[0, 0].set_title("mean waveform +/- std (best channel)")

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
        f"unit {unit.get('unit_id')} | {_unit_class(unit) or 'no label'} | "
        f"channel {unit.get('channel')} | {unit.get('n_spikes', '?')} spikes"
    )
    fig.tight_layout()
    return fig


#: The columns of a unit page, left to right, and each one's heading.
UNIT_PANELS = ("mean waveform +/- std", "ISI", "firing rate", "amplitude stability")


def _unit_class(unit: dict[str, Any]) -> str:
    """``SU (good)`` / ``MU (mua)``, the label alone, or ``""`` when unlabelled."""
    label, unit_class = unit.get("label") or "", unit.get("unit_class") or ""
    if unit_class and label:
        return f"{unit_class} ({label})"
    return unit_class or label


def _unit_header(unit: dict[str, Any]) -> str:
    """One line naming the unit and its headline numbers, for above its row."""

    def number(key: str, fmt: str, scale: float = 1.0) -> str | None:
        value = unit.get(key)
        if value is None or not np.isfinite(value):
            return None
        return format(value * scale, fmt)

    parts = [
        f"unit {unit.get('unit_id')}",
        *([_unit_class(unit)] if _unit_class(unit) else []),
        f"ch {unit.get('channel')}",
        f"{unit.get('n_spikes', 0):,} spikes",
    ]
    for key, fmt, scale, template in (
        ("firing_rate_hz", ".2f", 1.0, "{} Hz"),
        ("isi_fraction", ".2f", 100.0, "ISI viol {}%"),
        ("presence_ratio", ".2f", 1.0, "presence {}"),
        ("amp_cv", ".2f", 1.0, "amp CV {}"),
        ("peak_to_trough_ms", ".2f", 1.0, "p-t {} ms"),
    ):
        text = number(key, fmt, scale)
        if text is not None:
            parts.append(template.format(text))
    return "  |  ".join(parts)


def plot_unit_page(units: list[dict[str, Any]], rows: int, fig: Any = None) -> Any:
    """One page of units: a row per unit, a column per panel in ``UNIT_PANELS``.

    Each ``unit`` is the dict :func:`plot_unit_summary` takes, plus the header
    fields read by ``_unit_header`` (``unit_class``, ``firing_rate_hz``,
    ``isi_fraction``, ...) and optional ``waveform_units`` for the y label.
    ``rows`` fixes the grid, so a short last page keeps the same row height.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(8.5, 11))
    grid = fig.add_gridspec(rows, len(UNIT_PANELS), hspace=0.95, wspace=0.42,
                            left=0.07, right=0.98, top=0.95, bottom=0.06)

    for row, unit in enumerate(units):
        axes = [fig.add_subplot(grid[row, col]) for col in range(len(UNIT_PANELS))]

        if unit.get("waveform") is not None and np.size(unit["waveform"]):
            plot_mean_waveform(
                unit["waveform"], unit.get("waveform_t_ms"), unit.get("waveform_std"),
                ax=axes[0],
            )
            axes[0].set_ylabel(unit.get("waveform_units", "amplitude"))
        else:
            _no_waveform(axes[0])
        if unit.get("isi_counts") is not None:
            plot_isi_histogram(unit["isi_counts"], unit["isi_edges_ms"], ax=axes[1])
        if unit.get("rate_centers_s") is not None:
            plot_firing_rate(unit["rate_centers_s"], unit["rate_hz"], ax=axes[2])
        if unit.get("amplitudes") is not None and np.size(unit["amplitudes"]):
            plot_amplitudes(unit["amp_times_s"], unit["amplitudes"], ax=axes[3])

        for ax, title in zip(axes, UNIT_PANELS):
            ax.set_title(title, fontsize=7, pad=2)
            ax.tick_params(labelsize=6)
            ax.xaxis.label.set_size(6)
            ax.yaxis.label.set_size(6)

        # The header spans the row: centred over the four panels, just above them.
        left, right = axes[0].get_position().x0, axes[-1].get_position().x1
        top = axes[0].get_position().y1
        fig.text((left + right) / 2, top + 0.022, _unit_header(unit),
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    return fig


def save_unit_pages(
    units: list[dict[str, Any]],
    path: str | Path,
    rows_per_page: int = 6,
    title: str | None = None,
) -> Path:
    """Every unit in one multi-page PDF, ``rows_per_page`` units to a page.

    Pages are rendered and closed one at a time, so a few hundred units never
    hold more than one figure in memory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pages = max(1, int(np.ceil(len(units) / rows_per_page)))
    with PdfPages(path) as pdf:
        for page in range(n_pages):
            chunk = units[page * rows_per_page : (page + 1) * rows_per_page]
            fig = plot_unit_page(chunk, rows_per_page)
            footer = f"page {page + 1} / {n_pages}"
            fig.text(0.98, 0.01, f"{title}  |  {footer}" if title else footer,
                     ha="right", va="bottom", fontsize=6, color="0.4")
            pdf.savefig(fig)
            plt.close(fig)
    return path


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
