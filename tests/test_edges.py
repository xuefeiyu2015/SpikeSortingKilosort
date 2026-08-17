"""Edge detection: the layer everything downstream trusts."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import square_wave

from spikesorting._sync import edges


def test_rising_edges_at_known_times(fs):
    wave = square_wave(10.0, fs, period_s=1.0, phase_s=0.25)
    times = edges.detect_pulse_times(wave, 0.5, fs, duration_ms=500)
    expected = 0.25 + np.arange(10)
    assert np.allclose(times, expected)


def test_falling_edges_complement_rising(fs):
    wave = square_wave(5.0, fs, phase_s=0.1)
    rising = edges.rising_edge_indices(wave, 0.5)
    falling = edges.falling_edge_indices(wave, 0.5)
    assert rising.size == falling.size == 5
    # 50% duty at 1 Hz means each pulse is high for half a second.
    assert np.allclose((falling - rising) / fs, 0.5)


def test_previous_above_prevents_double_detection():
    # A chunk starting already-high must not report an edge at index 0.
    block = np.array([1, 1, 0, 1], dtype=np.uint8)
    assert edges.rising_edge_indices(block, 0.5, previous_above=True).tolist() == [3]
    assert edges.rising_edge_indices(block, 0.5, previous_above=False).tolist() == [0, 3]


def test_empty_input_returns_empty():
    empty = np.empty(0)
    assert edges.rising_edge_indices(empty, 0.5).size == 0
    assert edges.falling_edge_indices(empty, 0.5).size == 0
    assert edges.pulse_widths(empty, empty).size == 0


def test_digital_bit_signal_extracts_the_right_bit():
    words = np.array([0b0000000, 0b1000000, 0b0100000, 0b1100000], dtype=np.uint16)
    assert edges.digital_bit_signal(words, 6).tolist() == [0, 1, 0, 1]
    assert edges.digital_bit_signal(words, 5).tolist() == [0, 0, 1, 1]


def test_digital_bit_rejects_negative_bit():
    with pytest.raises(ValueError):
        edges.digital_bit_signal(np.zeros(4, dtype=np.uint16), -1)


def test_pulse_width_filter_rejects_wrong_duration(fs):
    # 500 ms pulses plus one 50 ms glitch that must be filtered out.
    wave = square_wave(5.0, fs, phase_s=0.1)
    glitch_start = int(4.7 * fs)
    wave[glitch_start : glitch_start + int(0.05 * fs)] = 1

    all_edges = edges.detect_pulse_times(wave, 0.5, fs, duration_ms=0)
    filtered = edges.detect_pulse_times(wave, 0.5, fs, duration_ms=500)
    assert all_edges.size == filtered.size + 1
    assert np.allclose(filtered, 0.1 + np.arange(5))


def test_truncated_final_pulse_is_dropped_when_width_matters(fs):
    # Recording stops mid-pulse: width is unknowable, so a width-filtered
    # extraction must not report it, while an unfiltered one must.
    wave = np.zeros(int(1.2 * fs), dtype=np.uint8)
    wave[int(0.1 * fs) : int(0.6 * fs)] = 1
    wave[int(1.1 * fs) :] = 1

    assert edges.detect_pulse_times(wave, 0.5, fs, duration_ms=0).size == 2
    assert edges.detect_pulse_times(wave, 0.5, fs, duration_ms=500).size == 1


@pytest.mark.parametrize("chunk", [1, 7, 999, 7777, 100000])
def test_streaming_matches_one_shot(fs, chunk):
    wave = square_wave(6.0, fs, phase_s=0.33)
    one_shot = edges.detect_pulse_times(wave, 0.5, fs, duration_ms=500)
    chunks = ((s, wave[s : s + chunk]) for s in range(0, wave.size, chunk))
    streamed = edges.stream_pulse_times(chunks, 0.5, fs, duration_ms=500)
    assert np.array_equal(one_shot, streamed)


def test_indices_to_seconds_applies_chunk_offset(fs):
    indices = np.array([0, 30000, 60000])
    assert np.allclose(edges.indices_to_seconds(indices, fs), [0.0, 1.0, 2.0])
    assert np.allclose(edges.indices_to_seconds(indices, fs, start_sample=15000), [0.5, 1.5, 2.5])


def test_indices_to_seconds_rejects_bad_rate():
    with pytest.raises(ValueError):
        edges.indices_to_seconds(np.array([0]), 0.0)


def test_compare_edge_sets_detects_agreement_and_disagreement():
    a = np.arange(10, dtype=np.float64)
    assert edges.compare_edge_sets(a, a)["agree"] is True

    shifted = a + 1e-3
    result = edges.compare_edge_sets(a, shifted, tolerance_s=1e-4)
    assert result["agree"] is False
    assert result["max_abs_diff_s"] == pytest.approx(1e-3)


def test_compare_edge_sets_survives_a_missing_pulse():
    # One dropped pulse must not misalign every later comparison.
    a = np.arange(10, dtype=np.float64)
    b = np.delete(a, 4)
    result = edges.compare_edge_sets(a, b, tolerance_s=1e-6)
    assert result["agree"] is False
    assert result["n_matched"] == 9


def test_compare_edge_sets_handles_empty():
    empty = np.empty(0)
    assert edges.compare_edge_sets(empty, empty)["agree"] is True
    assert edges.compare_edge_sets(np.array([1.0]), empty)["agree"] is False
