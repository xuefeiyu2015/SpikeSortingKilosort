"""Final export: aligned spike times plus per-unit information (step 10).

Writes what an analysis copies away, beside the sorting it came from:

==================================== ============================================
``units.csv``                        one row per unit: label, channel, counts,
                                     rate, ISI, presence, amplitude, waveform shape
``<recording>.sorted_spikes.mat``    every spike: time, channel, unit, sample
                                     index, plus the measured mean waveforms
==================================== ============================================

Kilosort's own arrays stay in the sorting folder above; nothing here copies them.

Computation only -- figures are produced separately by
:mod:`spikesorting._plots.summary` from these same arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .curated import PhyResults

__all__ = [
    "export_sorted_spikes_mat",
    "template_for_unit",
    "build_unit_table",
    "export_unit_table",
]


def template_for_unit(results: PhyResults, unit_id: int) -> int | None:
    """The template most of this unit's spikes came from.

    Falls back to treating the cluster id as a template id, which is correct for
    uncurated Kilosort4 output.
    """
    if results.spike_templates is None:
        return int(unit_id)
    mask = results.spike_clusters == unit_id
    if not np.any(mask):
        return None
    template_ids, counts = np.unique(results.spike_templates[mask], return_counts=True)
    return int(template_ids[int(np.argmax(counts))])


def mean_template_waveform(
    results: PhyResults, unit_id: int
) -> tuple[np.ndarray, int | None]:
    """``(waveform_on_best_channel, channel)`` from the template array.

    Templates are what Kilosort fit, not an average of the raw traces, so this is
    cheap and needs no access to the original recording. For a waveform measured
    from the raw data, use ``kilosort.data_tools.mean_waveform`` on the sorting
    machine instead.
    """
    if results.templates is None:
        return np.empty(0), None
    template_id = template_for_unit(results, unit_id)
    if template_id is None or template_id >= results.templates.shape[0]:
        return np.empty(0), None

    template = results.templates[template_id]  # (n_timepoints, n_channels)
    best_local = int((template**2).sum(axis=0).argmax())
    return template[:, best_local].astype(np.float64), best_channel(results, unit_id)


def best_channel(results: PhyResults, unit_id: int) -> int | None:
    """The recording channel a unit is largest on, from its template.

    The peak of the template's per-channel power, mapped through ``channel_map``
    to a row of the binary. Shared by the waveform export, which cuts snippets on
    exactly this channel, and by :func:`mean_template_waveform`, so the two can
    never disagree about which channel a unit belongs to.
    """
    if results.templates is None:
        return None
    template_id = template_for_unit(results, unit_id)
    if template_id is None or template_id >= results.templates.shape[0]:
        return None
    local = int((results.templates[template_id] ** 2).sum(axis=0).argmax())
    return int(results.channel_map[local]) if results.channel_map is not None else local


def recording_window(
    results: PhyResults,
    unit_ids: np.ndarray,
    spike_times_s: dict[int, np.ndarray] | None,
) -> tuple[float, float]:
    """``(start_s, duration_s)`` of the recording, on whichever timebase is in use.

    On the sorter's timebase the recording starts at zero. Once times are mapped
    onto Blackrock time it starts wherever Blackrock's clock happened to be, so
    the window has to be derived from the times themselves -- otherwise every
    time-windowed metric silently measures the wrong interval.
    """
    if spike_times_s is None:
        return 0.0, results.duration_s

    present = [np.asarray(spike_times_s[int(u)]) for u in unit_ids if int(u) in spike_times_s]
    present = [t for t in present if t.size]
    if not present:
        return 0.0, results.duration_s

    start = float(min(t[0] for t in present))
    stop = float(max(t[-1] for t in present))
    return start, stop - start


def build_unit_table(
    results: PhyResults,
    unit_ids: np.ndarray | None = None,
    spike_times_s: dict[int, np.ndarray] | None = None,
    duration_s: float | None = None,
    refractory_s: float = 1.5e-3,
    start_s: float | None = None,
) -> pd.DataFrame:
    """One row of metrics per unit.

    ``spike_times_s`` supplies already-aligned times per unit; without it the
    sorter's own timebase is used. Either way the metrics are computed on whatever
    timebase is passed, and the recording window is derived to match.
    """
    unit_ids = results.unit_ids if unit_ids is None else np.asarray(unit_ids)
    window_start, window_duration = recording_window(results, unit_ids, spike_times_s)
    duration_s = window_duration if duration_s is None else float(duration_s)
    start_s = window_start if start_s is None else float(start_s)

    rows: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        unit_id = int(unit_id)
        times = (
            spike_times_s[unit_id]
            if spike_times_s is not None and unit_id in spike_times_s
            else results.times_for(unit_id)
        )
        waveform, channel = mean_template_waveform(results, unit_id)

        row: dict[str, Any] = {
            "unit_id": unit_id,
            "label": results.labels.get(unit_id, ""),
            "channel": channel,
            "template_id": template_for_unit(results, unit_id),
            "first_spike_s": float(times[0]) if times.size else float("nan"),
            "last_spike_s": float(times[-1]) if times.size else float("nan"),
        }
        row.update(
            metrics.compute_unit_metrics(
                spike_times_s=times,
                duration_s=duration_s,
                amplitudes=results.amplitudes_for(unit_id),
                mean_waveform=waveform if waveform.size else None,
                fs=results.fs,
                refractory_s=refractory_s,
                start_s=start_s,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def export_unit_table(out_dir: str | Path, table: pd.DataFrame) -> Path:
    """Write ``units.csv`` -- the table :func:`build_unit_table` returns."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "units.csv"
    table.to_csv(path, index=False)
    return path


def export_sorted_spikes_mat(
    out_dir: str | Path,
    results: PhyResults,
    unit_ids: np.ndarray,
    spike_times_s: dict[int, np.ndarray] | None,
    timebase: str,
    table: pd.DataFrame,
    waveforms: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    filename: str = "sorted_spikes.mat",
) -> Path:
    """Write the sorted spikes as the online-spike container's twin.

    Field names are the ones ``jlab_loader`` reads off a ``.nev``
    (``TimeStamps``/``Channel``/``Unit``/``Waveforms``), so the same analysis
    code segments this product into trials without a branch. ``Unit`` holds the
    Kilosort cluster id -- the name is for compatibility, not a claim that a
    cluster is a Blackrock unit.

    Times are seconds rather than raw ticks: they come from a fitted map, and
    integer ticks would imply a precision the fit does not have. ``TimeRes`` is
    carried so the tick view can be recovered.
    """
    from .._io.matlab import Chunked, write_mat

    unit_ids = np.asarray(unit_ids)
    times: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    units: list[np.ndarray] = []
    channels: list[np.ndarray] = []
    for unit_id in unit_ids:
        unit_id = int(unit_id)
        mask = results.spike_clusters == unit_id
        t = (
            spike_times_s[unit_id]
            if spike_times_s is not None and unit_id in spike_times_s
            else results.times_for(unit_id)
        )
        times.append(np.asarray(t, dtype=np.float64))
        samples.append(np.asarray(results.spike_samples[mask], dtype=np.int64))
        units.append(np.full(t.size, unit_id, dtype=np.float64))
        channel = best_channel(results, unit_id)
        channels.append(np.full(t.size, np.nan if channel is None else channel))

    def flat(parts, dtype=np.float64):
        return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype=dtype)

    time_s = flat(times)
    order = np.argsort(time_s, kind="stable")     # one train, in time order
    product: dict[str, Any] = {
        "TimeStamps": time_s[order],
        "Channel": flat(channels)[order],
        "Unit": flat(units)[order],
        "spike_sample": flat(samples)[order],
        "TimeRes": float(results.fs),
    }

    info: dict[str, Any] = {
        "Channel_Number": np.array(
            [
                np.nan if best_channel(results, int(u)) is None else best_channel(results, int(u))
                for u in unit_ids
            ],
            dtype=np.float64,
        ),
        "Unit_No": unit_ids.astype(np.float64),
        "Label": " | ".join(str(results.labels.get(int(u), "")) for u in unit_ids),
        "n_spikes": np.array([int((results.spike_clusters == int(u)).sum()) for u in unit_ids],
                             dtype=np.float64),
        "samplingrate": float(results.fs),
        "timebase": timebase,
    }
    if "isi_fraction" in table:
        info["ViolationRate"] = table["isi_fraction"].to_numpy(dtype=np.float64)
    if provenance:
        info.update({k: v for k, v in provenance.items() if isinstance(v, (str, float, int))})

    if waveforms is not None:
        # Cut through Kilosort's own preprocessing and scaled to microvolts, so
        # unlike the Kilosort template these carry an amplitude, and the mean is
        # exactly the mean of the snippets. The mean is always here; the
        # snippets only when the session asked to keep them.
        product["MeanWaveform"] = np.asarray(waveforms["mean"], dtype=np.float32).T
        product["StdWaveform"] = np.asarray(waveforms["std"], dtype=np.float32).T
        product["MeanWaveformUnit"] = str(waveforms.get("units", "ADC"))
        product["window_ms"] = float(waveforms.get("window_ms", float("nan")))
        info["MeanWaveform_Unit_No"] = np.asarray(waveforms["unit_ids"], dtype=np.float64)
        snippets = waveforms.get("snippets")
        if snippets is not None:
            # Given as (nSamp, nSpikes), which is the HDF5 shape; MATLAB reverses
            # it and reads nSpikes x nSamp -- the online container's orientation.
            # Single precision, in MeanWaveformUnit: they are preprocessed and
            # scaled, so int16 counts would be both lossy and the wrong unit.
            snippets = np.asarray(snippets, dtype=np.float32)
            product["Waveforms"] = Chunked(
                shape=snippets.shape,
                dtype=np.float32,
                fill=lambda dataset, s=snippets: dataset.__setitem__(slice(None), s),
                chunks=None,
                compression="gzip",
            )

    product["info"] = info
    return write_mat(Path(out_dir) / filename, {"sorted_spikes": product})
