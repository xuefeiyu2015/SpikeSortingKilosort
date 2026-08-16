"""Unit quality metrics.

Pure computation: every function takes arrays and returns numbers or arrays. No
plotting, no file access, no printing -- the matching ``plot_*`` functions live in
:mod:`spikesorting._plots.summary` and take these results as input.

Spike times are in **seconds** throughout. Kilosort's own outputs are sample
indices, so convert with
:func:`spikesorting._sync.tprime.spike_times_to_seconds` first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "compute_isi",
    "compute_isi_histogram",
    "compute_isi_violations",
    "compute_firing_rate",
    "compute_presence_ratio",
    "compute_amplitude_stability",
    "compute_mean_waveform",
    "compute_waveform_features",
    "compute_unit_metrics",
]


def compute_isi(spike_times_s: np.ndarray) -> np.ndarray:
    """Inter-spike intervals in seconds."""
    times = np.asarray(spike_times_s, dtype=np.float64).reshape(-1)
    return np.diff(np.sort(times)) if times.size > 1 else np.empty(0, dtype=np.float64)


def compute_isi_histogram(
    isi_s: np.ndarray, bin_ms: float = 1.0, max_ms: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(counts, bin_edges_ms)`` for an ISI distribution."""
    if bin_ms <= 0 or max_ms <= 0:
        raise ValueError("bin_ms and max_ms must be positive")
    edges = np.arange(0.0, max_ms + bin_ms, bin_ms)
    counts, _ = np.histogram(np.asarray(isi_s, dtype=np.float64) * 1000.0, bins=edges)
    return counts, edges


def compute_isi_violations(
    isi_s: np.ndarray, refractory_s: float = 1.5e-3, censored_s: float = 0.0
) -> dict[str, float]:
    """Refractory-period violations.

    ``fraction`` is violations over total intervals -- the number to look at when
    deciding whether a cluster is a single unit. Intervals shorter than
    ``censored_s`` are excluded, for sorters that cannot resolve near-coincident
    spikes at all.
    """
    isi = np.asarray(isi_s, dtype=np.float64).reshape(-1)
    if isi.size == 0:
        return {"n_violations": 0.0, "n_intervals": 0.0, "fraction": float("nan")}
    usable = isi[isi >= censored_s] if censored_s > 0 else isi
    if usable.size == 0:
        return {"n_violations": 0.0, "n_intervals": 0.0, "fraction": float("nan")}
    n_violations = float(np.count_nonzero(usable < refractory_s))
    return {
        "n_violations": n_violations,
        "n_intervals": float(usable.size),
        "fraction": n_violations / usable.size,
    }


def compute_firing_rate(
    spike_times_s: np.ndarray,
    duration_s: float,
    bin_s: float = 10.0,
    start_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(bin_centers_s, rate_hz)`` over the recording."""
    if bin_s <= 0:
        raise ValueError("bin_s must be positive")
    if duration_s <= 0:
        return np.empty(0), np.empty(0)
    edges = np.arange(start_s, start_s + duration_s + bin_s, bin_s)
    counts, _ = np.histogram(np.asarray(spike_times_s, dtype=np.float64), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts / bin_s


def compute_presence_ratio(
    spike_times_s: np.ndarray, duration_s: float, n_bins: int = 100, start_s: float = 0.0
) -> float:
    """Fraction of the recording in which the unit fired at all.

    A unit that drifts out of range mid-session shows a presence ratio well below
    1 even when its ISI stats look immaculate.
    """
    times = np.asarray(spike_times_s, dtype=np.float64).reshape(-1)
    if times.size == 0 or duration_s <= 0 or n_bins <= 0:
        return 0.0
    edges = np.linspace(start_s, start_s + duration_s, n_bins + 1)
    counts, _ = np.histogram(times, bins=edges)
    return float(np.count_nonzero(counts) / n_bins)


def compute_amplitude_stability(
    spike_times_s: np.ndarray, amplitudes: np.ndarray, n_bins: int = 20
) -> dict[str, float]:
    """Drift in spike amplitude over the session.

    ``slope_per_hour`` is a linear fit of amplitude against time, normalised by
    the median amplitude: a large value means the unit was drifting away from (or
    toward) the electrode.
    """
    times = np.asarray(spike_times_s, dtype=np.float64).reshape(-1)
    amps = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if times.size != amps.size:
        raise ValueError(f"spike times and amplitudes differ: {times.size} vs {amps.size}")
    if times.size < 2:
        return {"cv": float("nan"), "slope_per_hour": float("nan"), "median": float("nan")}

    median = float(np.median(amps))
    mean = float(np.mean(amps))
    slope, _ = np.polyfit(times, amps, 1)
    return {
        "cv": float(np.std(amps) / mean) if mean else float("nan"),
        "slope_per_hour": float(slope * 3600.0 / median) if median else float("nan"),
        "median": median,
    }


def compute_mean_waveform(waveforms: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, sem)`` across spikes.

    ``waveforms`` is ``(n_timepoints, n_spikes)`` by default, matching Kilosort's
    ``data_tools.get_spike_waveforms`` layout.
    """
    data = np.asarray(waveforms, dtype=np.float64)
    if data.size == 0:
        return np.empty(0), np.empty(0)
    mean = data.mean(axis=axis)
    n = data.shape[axis]
    sem = data.std(axis=axis) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def compute_waveform_features(mean_waveform: np.ndarray, fs: float) -> dict[str, float]:
    """Shape features of a single-channel mean waveform.

    ``peak_to_trough_ms`` separates fast-spiking from regular-spiking units and is
    the feature most worth carrying into downstream analysis.
    """
    wave = np.asarray(mean_waveform, dtype=np.float64).reshape(-1)
    if wave.size == 0 or fs <= 0:
        return {
            "trough_amplitude": float("nan"),
            "peak_amplitude": float("nan"),
            "peak_to_trough_ms": float("nan"),
            "half_width_ms": float("nan"),
        }

    trough_idx = int(np.argmin(wave))
    after = wave[trough_idx:]
    peak_idx = trough_idx + int(np.argmax(after)) if after.size else trough_idx

    trough = float(wave[trough_idx])
    half = trough / 2.0
    below = np.flatnonzero(wave <= half)
    half_width = (below[-1] - below[0]) / fs * 1000.0 if below.size > 1 else float("nan")

    return {
        "trough_amplitude": trough,
        "peak_amplitude": float(wave[peak_idx]),
        "peak_to_trough_ms": (peak_idx - trough_idx) / fs * 1000.0,
        "half_width_ms": float(half_width),
    }


def compute_unit_metrics(
    spike_times_s: np.ndarray,
    duration_s: float,
    amplitudes: np.ndarray | None = None,
    mean_waveform: np.ndarray | None = None,
    fs: float | None = None,
    refractory_s: float = 1.5e-3,
    start_s: float = 0.0,
) -> dict[str, Any]:
    """All per-unit metrics in one dict, for the export table.

    ``start_s`` is when the recording window begins on the timebase of
    ``spike_times_s``. It matters once times are mapped onto Blackrock time,
    where the recording no longer starts at zero -- assuming zero would put every
    spike outside the presence-ratio window and report 0.0 for healthy units.
    """
    times = np.asarray(spike_times_s, dtype=np.float64).reshape(-1)
    isi = compute_isi(times)

    metrics: dict[str, Any] = {
        "n_spikes": int(times.size),
        "firing_rate_hz": float(times.size / duration_s) if duration_s > 0 else float("nan"),
        "presence_ratio": compute_presence_ratio(times, duration_s, start_s=start_s),
        "isi_median_ms": float(np.median(isi) * 1000.0) if isi.size else float("nan"),
    }
    metrics.update(
        {f"isi_{k}": v for k, v in compute_isi_violations(isi, refractory_s).items()}
    )
    if amplitudes is not None:
        metrics.update(
            {f"amp_{k}": v for k, v in compute_amplitude_stability(times, amplitudes).items()}
        )
    if mean_waveform is not None and fs:
        metrics.update(compute_waveform_features(mean_waveform, fs))
    return metrics
