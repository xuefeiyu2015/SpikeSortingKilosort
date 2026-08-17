"""Rendering only.

Every function here takes already-computed data and draws it. None of them
compute anything, read files, or modify their inputs -- the numbers come from
:mod:`spikesorting._export.metrics` and :mod:`spikesorting._sync.align`.
"""

from .probe import plot_probe_channel_order, plot_probe_geometry  # noqa: F401
from .summary import (  # noqa: F401
    plot_alignment_residuals,
    plot_amplitudes,
    plot_firing_rate,
    plot_isi_histogram,
    plot_mean_waveform,
    plot_sorting_overview,
    plot_unit_summary,
    save_figure,
)
