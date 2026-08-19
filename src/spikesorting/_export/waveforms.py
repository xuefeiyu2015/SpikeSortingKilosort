"""Per-spike waveform snippets, cut from the binary the sorter read.

Kilosort saves templates and per-spike amplitudes, but no snippets: the shape in
``templates.npy`` is what it *fitted*, in whitened units, and it is not an
average of anything. Everything here is measured from the raw samples instead, so
a mean waveform comes out with a real amplitude -- the analogue of the
``_spikes_waveform.mat`` product Blackrock fills from the ``.nev``.

**One sequential pass, never a seek per spike.** A million spikes is a million
scattered reads into a file that may be 100 GB on a share, which is the failure
:mod:`spikesorting._io.spikeglx` was rewritten to avoid. The binary is walked in
order and every snippet landing in the current block is cut from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

__all__ = [
    "SnippetPlan",
    "plan_snippets",
    "accumulate",
    "SnippetResult",
    "highpass",
    "snippet_regions",
]


@dataclass(frozen=True)
class SnippetPlan:
    """Which samples to cut, on which channel, for which unit.

    ``sample``/``channel``/``unit_id`` are parallel and already sorted by sample,
    which is what lets one forward pass serve them all.
    """

    sample: np.ndarray
    channel: np.ndarray
    unit_id: np.ndarray
    before: int
    after: int
    #: Position of each kept spike in the *original* spike list, so anything
    #: parallel to ``spike_times.npy`` -- seconds on the aligned timebase,
    #: amplitudes -- can be cut down to the same subset without repeating the
    #: selection. Parallel to ``sample``, like the other three.
    spike_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    #: Spikes dropped because their window ran off one end of the recording.
    n_dropped: int = 0
    #: Spikes that could have been cut, before ``max_per_unit`` thinned them.
    n_available: int = 0

    @property
    def width(self) -> int:
        return self.before + self.after

    @property
    def n_spikes(self) -> int:
        return int(self.sample.size)


@dataclass
class SnippetResult:
    """Snippets plus the running mean and standard deviation per unit."""

    snippets: np.ndarray                 # (width, n_spikes) int16
    unit_ids: np.ndarray                 # (n_units,)
    mean: np.ndarray                     # (width, n_units) float32
    std: np.ndarray                      # (width, n_units) float32
    n_per_unit: np.ndarray               # (n_units,)
    notes: list[str] = field(default_factory=list)


def _uniform_in_time(
    samples: np.ndarray, clusters: np.ndarray, n_samples: int, max_per_unit: int
) -> np.ndarray:
    """Positions of <= ``max_per_unit`` spikes per unit, spread over the recording.

    **Uniform in time, not in spike count.** Taking every k-th spike would be
    uniform in the spike *list*, so a unit firing at 50 Hz during one task and
    2 Hz during another would be represented almost entirely by the busy task --
    and the subset could then say nothing about the rest of the session. Instead
    the recording is cut into ``max_per_unit`` equal-width bins and the first
    spike in each occupied bin is kept.

    Bins span ``[0, n_samples)`` -- the whole recording rather than each unit's
    own first-to-last span -- so two units' subsets are on the same time grid and
    comparable. Empty bins contribute nothing, so a unit active in only one part
    of the session is represented only there, which is correct.

    No RNG: the choice is a function of the spike times alone, so two runs of the
    export produce identical files.
    """
    span = max(int(n_samples), 1)
    bin_id = (samples.astype(np.int64) * max_per_unit) // span
    np.clip(bin_id, 0, max_per_unit - 1, out=bin_id)
    # One key per (unit, bin). samples is already sorted, and np.unique returns
    # the first occurrence of each key, so "first spike in the bin" falls out.
    key = clusters.astype(np.int64) * max_per_unit + bin_id
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def plan_snippets(
    spike_samples: np.ndarray,
    spike_clusters: np.ndarray,
    channel_for_unit: dict[int, int],
    n_samples: int,
    before: int,
    after: int,
    max_per_unit: int | None = None,
    margin: int = 0,
) -> SnippetPlan:
    """Decide which spikes can be cut, and from which channel. Pure.

    A spike closer to either end of the recording than its window is dropped
    rather than zero-padded: a padded snippet averages into the mean as though it
    were signal, and nothing downstream could tell. The count is carried on the
    plan so the caller can report it.

    ``max_per_unit`` caps how many spikes each unit contributes, chosen uniformly
    over the recording (see :func:`_uniform_in_time`). The mean's standard error
    falls as 1/sqrt(n), so a couple of thousand spikes is already well past the
    point where more changes the answer -- while the reading is what costs. None
    measures every spike.

    ``margin`` widens that boundary rule by the filter pad, so no kept spike ever
    needs signal from before the file starts. Keeping the pad off the ends here
    is what lets :func:`snippet_regions` stay a plain arithmetic function with no
    clamping, and the handful of spikes it costs are counted in ``n_dropped``
    like any other boundary spike.
    """
    spike_samples = np.asarray(spike_samples, dtype=np.int64).reshape(-1)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int64).reshape(-1)
    if spike_samples.size != spike_clusters.size:
        raise ValueError(
            f"{spike_samples.size} spike times but {spike_clusters.size} cluster ids"
        )

    known = np.array([int(c) in channel_for_unit for c in spike_clusters], dtype=bool)
    edge = int(before) + int(margin)
    inside = (spike_samples >= edge) & (
        spike_samples + int(after) + int(margin) <= n_samples
    )
    keep = np.flatnonzero(known & inside)

    order = np.argsort(spike_samples[keep], kind="stable")
    index = keep[order]                        # into the original spike arrays
    n_available = int(index.size)

    if max_per_unit is not None and max_per_unit > 0:
        chosen = _uniform_in_time(
            spike_samples[index], spike_clusters[index], n_samples, int(max_per_unit)
        )
        index = index[chosen]

    clusters = spike_clusters[index]
    return SnippetPlan(
        sample=spike_samples[index],
        channel=np.array([channel_for_unit[int(c)] for c in clusters], dtype=np.int64),
        unit_id=clusters,
        before=int(before),
        after=int(after),
        spike_index=index,
        n_dropped=int((~inside).sum()),
        n_available=n_available,
    )


def highpass(
    samples: np.ndarray, fs: float, cutoff: float, order: int
) -> np.ndarray:
    """Zero-phase Butterworth high-pass along the sample axis. Pure.

    **Zero-phase matters here specifically.** A causal filter delays the signal,
    which would slide every waveform relative to the spike sample it is supposed
    to be centred on -- a shift nothing downstream could detect. ``sosfiltfilt``
    runs the filter forwards and backwards, so the delay cancels exactly.

    ``samples`` is ``(n, n_chan)``; the filter runs down ``axis=0``.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * float(fs)
    if not 0.0 < cutoff < nyquist:
        raise ValueError(f"cutoff {cutoff} Hz is not below Nyquist ({nyquist} Hz)")
    sos = butter(int(order), cutoff / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, np.asarray(samples, dtype=np.float64), axis=0)


def snippet_regions(plan: SnippetPlan, pad: int) -> list[tuple[int, int]]:
    """Merged ``(start, stop)`` sample ranges covering every snippet, plus ``pad``.

    The pad is signal read only to be **thrown away**: a filter applied to a bare
    2 ms window rings at both edges, and those edges are the snippet. Reading a
    few extra milliseconds either side and discarding them after filtering is
    what makes the kept samples free of the transient. Phy and Kilosort both skip
    this and filter the bare window.

    Overlapping ranges are merged, so a burst of spikes costs one read rather
    than one read each. Returned in increasing order, which is the access pattern
    a memmap over a network share needs.
    """
    if plan.n_spikes == 0:
        return []
    starts = plan.sample - plan.before - pad
    stops = plan.sample + plan.after + pad
    merged: list[tuple[int, int]] = []
    for start, stop in zip(starts.tolist(), stops.tolist()):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def accumulate(
    plan: SnippetPlan,
    blocks: Iterator[tuple[int, np.ndarray]],
    uv_per_digit: float | np.ndarray = 1.0,
    highpass_hz: float | None = None,
    fs: float | None = None,
    highpass_order: int = 3,
    pad: int = 0,
) -> SnippetResult:
    """Cut every snippet from ``blocks`` and build the per-unit mean. Pure.

    ``blocks`` yields ``(start_sample, samples)`` with ``samples`` shaped
    ``(n, n_chan)`` -- the same contract :func:`spikesorting._io.spikeglx.
    iter_channel` uses, so a memmap, a file reader or a synthetic array all work
    and the tests need no recording.

    The mean and standard deviation are accumulated in microvolts as the pass
    goes, from sums rather than by keeping the snippets around a second time.

    With ``highpass_hz`` set, each block is filtered *once* on arrival and then
    ``pad`` samples are dropped from each end, so every snippet cut from what is
    left sits well inside the filtered signal rather than in its transient. Pair
    it with the ranges :func:`snippet_regions` produces for the same ``pad`` and
    the arithmetic lines up: a region begins ``pad`` before the first snippet it
    holds, so trimming it leaves that snippet at the block's own edge.
    """
    width = plan.width
    unit_ids = np.unique(plan.unit_id) if plan.n_spikes else np.empty(0, dtype=np.int64)
    index_of = {int(u): i for i, u in enumerate(unit_ids)}

    snippets = np.zeros((width, plan.n_spikes), dtype=np.int16)
    total = np.zeros((width, unit_ids.size), dtype=np.float64)
    total_sq = np.zeros((width, unit_ids.size), dtype=np.float64)
    counts = np.zeros(unit_ids.size, dtype=np.int64)
    filled = np.zeros(plan.n_spikes, dtype=bool)

    if highpass_hz is not None and not fs:
        raise ValueError("highpass_hz needs fs to convert the cutoff")

    scale = np.asarray(uv_per_digit, dtype=np.float64)
    for start, block in blocks:
        if highpass_hz is not None:
            block = highpass(block, fs, highpass_hz, highpass_order)
            if pad:
                # Everything the filter's transient touched leaves here.
                block = block[pad : block.shape[0] - pad]
                start = start + pad
        stop = start + block.shape[0]
        # Spikes whose whole window lies inside this block. The plan is sorted,
        # so this is a pair of binary searches rather than a scan.
        first = int(np.searchsorted(plan.sample, start + plan.before, side="left"))
        last = int(np.searchsorted(plan.sample, stop - plan.after, side="right"))
        for i in range(first, last):
            begin = plan.sample[i] - plan.before - start
            cut = block[begin : begin + width, plan.channel[i]]
            # int16 keeps the snippet array the size the export promises; the
            # mean below uses the unrounded values, so rounding costs it nothing.
            snippets[:, i] = np.rint(cut) if cut.dtype.kind == "f" else cut
            filled[i] = True
            unit = index_of[int(plan.unit_id[i])]
            gain = float(scale[plan.channel[i]]) if scale.ndim else float(scale)
            in_uv = cut.astype(np.float64) * gain
            total[:, unit] += in_uv
            total_sq[:, unit] += in_uv**2
            counts[unit] += 1

    notes: list[str] = []
    if plan.n_dropped:
        notes.append(
            f"{plan.n_dropped} spike(s) dropped: their window, plus any filter "
            "pad, ran past the start or end of the recording"
        )
    missing = int((~filled).sum())
    if missing:
        notes.append(
            f"{missing} spike(s) were not covered by the blocks supplied and are "
            "zero in the output"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / counts
        variance = total_sq / counts - mean**2
    variance[variance < 0] = 0.0                      # rounding, not signal

    return SnippetResult(
        snippets=snippets,
        unit_ids=unit_ids,
        mean=mean.astype(np.float32),
        std=np.sqrt(variance).astype(np.float32),
        n_per_unit=counts,
        notes=notes,
    )
