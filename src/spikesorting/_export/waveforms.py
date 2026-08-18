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

__all__ = ["SnippetPlan", "plan_snippets", "accumulate", "SnippetResult"]


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
    #: Spikes dropped because their window ran off one end of the recording.
    n_dropped: int = 0

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


def plan_snippets(
    spike_samples: np.ndarray,
    spike_clusters: np.ndarray,
    channel_for_unit: dict[int, int],
    n_samples: int,
    before: int,
    after: int,
) -> SnippetPlan:
    """Decide which spikes can be cut, and from which channel. Pure.

    A spike closer to either end of the recording than its window is dropped
    rather than zero-padded: a padded snippet averages into the mean as though it
    were signal, and nothing downstream could tell. The count is carried on the
    plan so the caller can report it.
    """
    spike_samples = np.asarray(spike_samples, dtype=np.int64).reshape(-1)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int64).reshape(-1)
    if spike_samples.size != spike_clusters.size:
        raise ValueError(
            f"{spike_samples.size} spike times but {spike_clusters.size} cluster ids"
        )

    known = np.array([int(c) in channel_for_unit for c in spike_clusters], dtype=bool)
    inside = (spike_samples >= before) & (spike_samples + after <= n_samples)
    keep = known & inside

    order = np.argsort(spike_samples[keep], kind="stable")
    samples = spike_samples[keep][order]
    clusters = spike_clusters[keep][order]
    channels = np.array([channel_for_unit[int(c)] for c in clusters], dtype=np.int64)

    return SnippetPlan(
        sample=samples,
        channel=channels,
        unit_id=clusters,
        before=int(before),
        after=int(after),
        n_dropped=int((~inside).sum()),
    )


def accumulate(
    plan: SnippetPlan,
    blocks: Iterator[tuple[int, np.ndarray]],
    uv_per_digit: float | np.ndarray = 1.0,
) -> SnippetResult:
    """Cut every snippet from ``blocks`` and build the per-unit mean. Pure.

    ``blocks`` yields ``(start_sample, samples)`` with ``samples`` shaped
    ``(n, n_chan)`` -- the same contract :func:`spikesorting._io.spikeglx.
    iter_channel` uses, so a memmap, a file reader or a synthetic array all work
    and the tests need no recording.

    The mean and standard deviation are accumulated in microvolts as the pass
    goes, from sums rather than by keeping the snippets around a second time.
    """
    width = plan.width
    unit_ids = np.unique(plan.unit_id) if plan.n_spikes else np.empty(0, dtype=np.int64)
    index_of = {int(u): i for i, u in enumerate(unit_ids)}

    snippets = np.zeros((width, plan.n_spikes), dtype=np.int16)
    total = np.zeros((width, unit_ids.size), dtype=np.float64)
    total_sq = np.zeros((width, unit_ids.size), dtype=np.float64)
    counts = np.zeros(unit_ids.size, dtype=np.int64)
    filled = np.zeros(plan.n_spikes, dtype=bool)

    scale = np.asarray(uv_per_digit, dtype=np.float64)
    for start, block in blocks:
        stop = start + block.shape[0]
        # Spikes whose whole window lies inside this block. The plan is sorted,
        # so this is a pair of binary searches rather than a scan.
        first = int(np.searchsorted(plan.sample, start + plan.before, side="left"))
        last = int(np.searchsorted(plan.sample, stop - plan.after, side="right"))
        for i in range(first, last):
            begin = plan.sample[i] - plan.before - start
            cut = block[begin : begin + width, plan.channel[i]]
            snippets[:, i] = cut
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
            f"{plan.n_dropped} spike(s) dropped: their window ran past the start "
            "or end of the recording"
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
