"""Test configuration.

Puts ``src/`` and ``tools/`` on the path so the suite runs from a checkout
without installing, and provides the synthetic-recording fixtures the
pure-compute tests use. ``tools/`` is there for ``doctor``, which checks the
environment rather than running the pipeline and so lives outside the package.

Nothing here needs a GPU, Kilosort, SpikeInterface or real recordings -- that is
the point. These tests cover the layer that must be correct before any rig time
is spent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


@pytest.fixture
def fs() -> float:
    return 30000.0


def square_wave(
    duration_s: float,
    fs: float,
    period_s: float = 1.0,
    duty: float = 0.5,
    phase_s: float = 0.0,
) -> np.ndarray:
    """A 0/1 square wave, first rising edge at ``phase_s``.

    Built in integer samples rather than by comparing floating-point times: at
    30 kHz, ``i / fs`` lands just below an exact period boundary often enough to
    shift an edge by one sample, which would make edge tests flaky for reasons
    that have nothing to do with the detector.
    """
    n = int(round(duration_s * fs))
    period = int(round(period_s * fs))
    high = int(round(period * duty))
    phase = int(round(phase_s * fs))
    index = np.arange(n)
    wave = (((index - phase) % period) < high).astype(np.uint8)
    wave[index < phase] = 0
    return wave


def coded_burst_times(
    n_bursts: int,
    interval_s: float = 14.0,
    pulses_per_burst: int = 8,
    intra_gap_s: float = 0.02,
    start_s: float = 0.0,
    coded: bool = True,
) -> np.ndarray:
    """Pulse times for the repeating dense burst, like the Blackrock DO1 signal.

    With ``coded=True`` each burst gets its own intra-burst interval pattern,
    which is what makes the real signal unambiguous. ``coded=False`` produces a
    perfectly periodic train -- useful for testing that the ambiguous case is
    handled honestly rather than guessed at.
    """
    rng = np.random.default_rng(12345)
    times: list[float] = []
    for i in range(n_bursts):
        onset = start_s + i * interval_s
        if coded:
            gaps = intra_gap_s * (1.0 + rng.uniform(0.0, 1.0, pulses_per_burst - 1))
        else:
            gaps = np.full(pulses_per_burst - 1, intra_gap_s)
        times.append(onset)
        times.extend(onset + np.cumsum(gaps))
    return np.sort(np.asarray(times, dtype=np.float64))


@pytest.fixture
def burst_times() -> np.ndarray:
    return coded_burst_times(20)


@pytest.fixture
def kilosort_results(tmp_path) -> Path:
    """A minimal, self-consistent Kilosort4 results folder.

    Three units with known spike counts, plus templates whose peak channel is
    known, so the export layer can be checked against exact expected values.
    """
    results_dir = tmp_path / "kilosort4"
    results_dir.mkdir()

    rng = np.random.default_rng(0)
    fs = 30000.0
    counts = {0: 500, 1: 300, 2: 120}

    samples, clusters = [], []
    for unit_id, count in counts.items():
        # Regular firing plus jitter, so ISIs are well defined and non-degenerate.
        times = np.cumsum(rng.uniform(0.02, 0.08, size=count)) + 0.5 * unit_id
        samples.append(np.rint(times * fs).astype(np.int64))
        clusters.append(np.full(count, unit_id, dtype=np.int32))

    spike_samples = np.concatenate(samples)
    spike_clusters = np.concatenate(clusters)
    order = np.argsort(spike_samples)
    spike_samples, spike_clusters = spike_samples[order], spike_clusters[order]

    n_units, n_time, n_chan = 3, 61, 8
    templates = np.zeros((n_units, n_time, n_chan), dtype=np.float32)
    peak_channels = [2, 5, 7]
    trough = np.zeros(n_time, dtype=np.float32)
    trough[25:35] = -np.hanning(10)
    trough[35:45] = 0.4 * np.hanning(10)
    for unit_id, channel in enumerate(peak_channels):
        templates[unit_id, :, channel] = trough * (unit_id + 1)

    np.save(results_dir / "spike_times.npy", spike_samples)
    np.save(results_dir / "spike_clusters.npy", spike_clusters)
    np.save(results_dir / "spike_templates.npy", spike_clusters.astype(np.int32))
    np.save(results_dir / "amplitudes.npy", rng.normal(50, 5, size=spike_samples.size))
    np.save(results_dir / "templates.npy", templates)
    np.save(results_dir / "channel_map.npy", np.arange(n_chan, dtype=np.int32))

    (results_dir / "params.py").write_text(
        "dat_path = 'recording.bin'\n"
        "n_channels_dat = 8\n"
        "dtype = 'int16'\n"
        "offset = 0\n"
        f"sample_rate = {fs}\n"
        "hp_filtered = False\n",
        encoding="utf-8",
    )
    (results_dir / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n2\tgood\n", encoding="utf-8"
    )
    return results_dir
