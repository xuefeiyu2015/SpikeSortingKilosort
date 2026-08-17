"""Unit quality metrics, checked against hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._export import metrics


def test_isi_from_regular_spikes():
    times = np.arange(0.0, 1.0, 0.01)
    isi = metrics.compute_isi(times)
    assert isi.size == times.size - 1
    assert np.allclose(isi, 0.01)


def test_isi_sorts_unordered_input():
    assert np.allclose(metrics.compute_isi(np.array([2.0, 0.0, 1.0])), [1.0, 1.0])


def test_isi_of_a_single_spike_is_empty():
    assert metrics.compute_isi(np.array([1.0])).size == 0


def test_isi_histogram_bins_in_milliseconds():
    isi = np.array([0.001, 0.0015, 0.05])  # 1 ms, 1.5 ms, 50 ms
    counts, edges = metrics.compute_isi_histogram(isi, bin_ms=1.0, max_ms=100.0)
    assert edges[0] == 0.0 and edges[-1] == 100.0
    assert counts[1] == 2  # both sub-2 ms intervals
    assert counts[50] == 1


def test_isi_violations_counts_refractory_breaches():
    isi = np.array([0.0005, 0.001, 0.002, 0.05])  # two below 1.5 ms
    result = metrics.compute_isi_violations(isi, refractory_s=1.5e-3)
    assert result["n_violations"] == 2
    assert result["fraction"] == pytest.approx(0.5)


def test_isi_violations_on_empty_input_is_nan_not_zero():
    # Reporting 0% contamination for a unit with no data would be a lie.
    result = metrics.compute_isi_violations(np.empty(0))
    assert np.isnan(result["fraction"])


def test_firing_rate_is_counts_per_bin_second():
    times = np.arange(0.0, 10.0, 0.1)  # 10 Hz
    centers, rate = metrics.compute_firing_rate(times, duration_s=10.0, bin_s=1.0)
    assert centers.size == rate.size
    assert np.allclose(rate[:9], 10.0)


def test_firing_rate_rejects_bad_bin():
    with pytest.raises(ValueError):
        metrics.compute_firing_rate(np.array([1.0]), 10.0, bin_s=0.0)


def test_presence_ratio_detects_a_unit_that_vanishes():
    full = np.arange(0.0, 100.0, 0.1)
    assert metrics.compute_presence_ratio(full, 100.0) == pytest.approx(1.0)

    first_half = np.arange(0.0, 50.0, 0.1)
    assert metrics.compute_presence_ratio(first_half, 100.0) == pytest.approx(0.5, abs=0.02)

    assert metrics.compute_presence_ratio(np.empty(0), 100.0) == 0.0


def test_amplitude_stability_detects_drift():
    times = np.arange(0.0, 3600.0, 1.0)
    steady = np.full(times.size, 50.0)
    assert metrics.compute_amplitude_stability(times, steady)["slope_per_hour"] == pytest.approx(0.0)

    halving = np.linspace(50.0, 25.0, times.size)
    drifting = metrics.compute_amplitude_stability(times, halving)
    assert drifting["slope_per_hour"] < -0.5


def test_amplitude_stability_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ"):
        metrics.compute_amplitude_stability(np.arange(5.0), np.arange(3.0))


def test_mean_waveform_and_sem():
    waveforms = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 10))
    mean, sem = metrics.compute_mean_waveform(waveforms)
    assert np.allclose(mean, [1.0, 2.0, 3.0])
    assert np.allclose(sem, 0.0)


def test_waveform_features_measure_trough_to_peak():
    fs = 30000.0
    wave = np.zeros(90)
    wave[30] = -1.0  # trough
    wave[60] = 0.5  # peak, 30 samples = 1 ms later
    features = metrics.compute_waveform_features(wave, fs)
    assert features["trough_amplitude"] == pytest.approx(-1.0)
    assert features["peak_amplitude"] == pytest.approx(0.5)
    assert features["peak_to_trough_ms"] == pytest.approx(1.0)


def test_waveform_features_on_empty_input_are_nan():
    features = metrics.compute_waveform_features(np.empty(0), 30000.0)
    assert all(np.isnan(v) for v in features.values())


def test_compute_unit_metrics_assembles_everything():
    times = np.arange(0.0, 60.0, 0.01)
    result = metrics.compute_unit_metrics(
        times, duration_s=60.0, amplitudes=np.full(times.size, 40.0)
    )
    assert result["n_spikes"] == times.size
    assert result["firing_rate_hz"] == pytest.approx(100.0, rel=1e-3)
    assert result["presence_ratio"] == pytest.approx(1.0)
    assert result["isi_fraction"] == pytest.approx(0.0)
    assert "amp_cv" in result
