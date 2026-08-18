"""Sample index -> NSP clock, the axis every other Blackrock product uses.

Kilosort writes sample indices. The `.nev` markers, eye traces and online spikes
beside them are timestamped on the NSP's PTP clock, whose origin is ~1.5e9 s and
whose rate is *not* the nominal one. Both facts come out of the file's own
per-sample timestamps, by least squares.

Everything here runs on synthetic timestamp arrays: no neo, no recording, no GPU.
The numbers used are the ones measured on the real files -- 29999.8646 Hz on a
Hub1 `.ns6` (-4.5 ppm) and PTP glitches of +1.033 / -1.867 / +0.933 ms.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._io.blackrock import (
    NspSegment,
    NspTimeMap,
    fit_nsp_segment,
    nsp_time_map,
)

TRUE_RATE = 29_999.8646          # measured on Hub1-Athos_20260717…ns6
T0 = 1_521_182_029.171230        # ...and its PTP start, in seconds
NOMINAL = 30_000.0


def _ptp(n: int, rate: float = TRUE_RATE, t0: float = T0) -> np.ndarray:
    """Nanosecond PTP timestamps for ``n`` samples at ``rate``.

    Built by adding integer offsets to an integer base, because ``(t0 + k/rate) *
    1e9`` is not representable: float64's spacing at 1.5e18 is 256 ns, which
    would quantise the synthetic clock far more coarsely than the real hardware
    does and make this file test its own rounding rather than the fit.
    """
    base = int(round(t0 * 1e9))
    return base + np.rint(np.arange(n) * (1e9 / rate)).astype(np.int64)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_rate_the_clock_actually_ran_at():
    n = 1_000_000
    samples = np.arange(0, n, 50)
    seconds = _ptp(n)[samples] / 1e9

    segment = fit_nsp_segment(samples, seconds)

    assert 1 / segment.slope == pytest.approx(TRUE_RATE, abs=1e-3)
    assert segment.intercept == pytest.approx(T0, abs=1e-6)
    assert segment.residual_max_s < 1e-6


def test_using_the_nominal_rate_would_be_wrong_by_tens_of_milliseconds():
    # The reason this module exists. -4.5 ppm sounds negligible until it is
    # multiplied by three hours, and the error is one-sided against the event
    # markers, which come from a different crystal.
    session_s = 11_280.0
    n = int(session_s * TRUE_RATE)
    error_at_end = n / NOMINAL - n / TRUE_RATE

    assert abs(error_at_end) > 0.05          # 51 ms, measured on the real file


def test_a_corrupted_timestamp_barely_moves_the_fit():
    # PTP files carry occasional bad packet timestamps -- jumps of a millisecond,
    # sometimes *backwards*, which is impossible between sequential samples. This
    # is the property that makes fitting the right choice over reading individual
    # timestamps: a lookup would return the corrupted value verbatim.
    n = 1_000_000
    stamps = _ptp(n)
    glitched = stamps.copy()
    glitched[500_000] += int(1.033e6)        # +1.033 ms
    glitched[500_027] -= int(1.867e6)        # -1.867 ms, backwards
    glitched[500_057] += int(0.933e6)        # +0.933 ms

    samples = np.arange(0, n, 50)
    clean = fit_nsp_segment(samples, stamps[samples] / 1e9)
    dirty = fit_nsp_segment(samples, glitched[samples] / 1e9)

    # Over the whole recording the two maps disagree by well under a sample.
    ends = np.array([0, n // 2, n - 1])
    assert np.abs(clean.apply(ends) - dirty.apply(ends)).max() < 1e-5


def test_two_points_are_the_minimum():
    with pytest.raises(ValueError, match="at least two"):
        fit_nsp_segment(np.array([0]), np.array([T0]))


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------


def _map(*segments: NspSegment, fs: float = NOMINAL) -> NspTimeMap:
    return NspTimeMap(
        segments=segments, fs_nominal=fs, spec="3.0", per_sample_timestamps=True
    )


def test_apply_puts_samples_on_the_nsp_clock():
    time_map = _map(NspSegment(0, 1_000_000, 1 / TRUE_RATE, T0, 1e-7))

    times = time_map.apply(np.array([0, 30_000, 999_999]))

    assert times[0] == pytest.approx(T0)
    assert times[1] - times[0] == pytest.approx(30_000 / TRUE_RATE)
    assert time_map.drift_ppm == pytest.approx(-4.5, abs=0.2)


def test_a_two_segment_map_is_continuous_across_the_boundary():
    # A paused recording: the second segment starts later on the clock than the
    # sample count alone implies, and that pause must not be swallowed.
    pause_s = 0.5
    first = NspSegment(0, 100_000, 1 / TRUE_RATE, T0, 1e-7)
    second_start = T0 + 100_000 / TRUE_RATE + pause_s
    second = NspSegment(100_000, 100_000, 1 / TRUE_RATE, second_start, 1e-7)
    time_map = _map(first, second)

    last_of_first = time_map.apply(np.array([99_999]))[0]
    first_of_second = time_map.apply(np.array([100_000]))[0]

    assert first_of_second - last_of_first == pytest.approx(pause_s + 1 / TRUE_RATE, abs=1e-6)
    assert time_map.residual_max_s == pytest.approx(1e-7)


def test_the_map_round_trips_through_json():
    time_map = _map(NspSegment(0, 1_000, 1 / TRUE_RATE, T0, 1.27e-4))

    restored = NspTimeMap.from_dict(time_map.to_dict())

    assert restored.apply(np.array([500]))[0] == pytest.approx(
        time_map.apply(np.array([500]))[0]
    )
    assert restored.to_dict()["measured_rate_hz"] == pytest.approx(TRUE_RATE, abs=1e-3)
    assert "nsp_seconds = slope" in restored.to_dict()["formula"]


# ---------------------------------------------------------------------------
# Reading it off a file
# ---------------------------------------------------------------------------


class _FakeReader:
    """The handful of BlackrockRawIO calls nsp_time_map makes."""

    def __init__(self, segments, fs=NOMINAL, resolution=1e9, per_sample=True):
        self._segments = segments          # list of timestamp arrays
        self._fs = fs
        self.per_sample = per_sample
        self._nsx_spec = {6: "3.0"}
        self._nsx_basic_header = {
            6: np.array(
                [(resolution,)], dtype=[("timestamp_resolution", "f8")]
            )[0]
        }
        self._nsx_data_header = {
            6: {
                i: {"timestamp": (ts if per_sample else np.array(ts[0]))}
                for i, ts in enumerate(segments)
            }
        }

    def get_signal_sampling_rate(self, stream_index=0):
        return self._fs

    def segment_count(self, block_index=0):
        return len(self._segments)

    def get_signal_size(self, block_index, seg_index, stream_index):
        return self._segments[seg_index].size

    def get_signal_t_start(self, block_index, seg_index, stream_index):
        return float(self._segments[seg_index][0]) / 1e9


def test_a_ptp_file_is_measured_from_its_own_timestamps():
    reader = _FakeReader([_ptp(200_000)])

    time_map = nsp_time_map(reader, 6)

    assert time_map.per_sample_timestamps is True
    assert 1 / time_map.segments[0].slope == pytest.approx(TRUE_RATE, abs=1e-2)
    assert time_map.drift_ppm == pytest.approx(-4.5, abs=0.2)


def test_a_file_with_one_timestamp_per_block_falls_back_to_the_nominal_rate():
    # Older file specs carry a scalar timestamp, so the true rate cannot be
    # measured. The origin still can, and saying which case you are in matters.
    reader = _FakeReader([_ptp(200_000)], per_sample=False)

    time_map = nsp_time_map(reader, 6)

    assert time_map.per_sample_timestamps is False
    assert time_map.segments[0].slope == pytest.approx(1 / NOMINAL)
    assert np.isnan(time_map.segments[0].residual_max_s)
    assert time_map.segments[0].intercept == pytest.approx(T0, abs=1e-6)


def test_a_stream_that_is_not_linear_in_time_refuses_to_be_mapped():
    # Exporting spike times against a map that does not fit is worse than not
    # exporting them: the numbers look fine and are wrong.
    stamps = _ptp(200_000)
    stamps[100_000:] += int(50e6)          # a 50 ms step nothing accounts for
    reader = _FakeReader([stamps])

    with pytest.raises(ValueError, match="not linear"):
        nsp_time_map(reader, 6, tolerance_s=1e-3)

    # ...but it can still be measured when the caller wants to look.
    assert nsp_time_map(reader, 6, tolerance_s=None).residual_max_s > 1e-3


def test_each_segment_of_a_paused_recording_is_fitted_on_its_own():
    first = _ptp(100_000)
    second = _ptp(100_000, t0=T0 + 100_000 / TRUE_RATE + 0.5)   # half-second pause
    reader = _FakeReader([first, second])

    time_map = nsp_time_map(reader, 6)

    assert len(time_map.segments) == 2
    assert time_map.segments[1].first_sample == 100_000
    gap = time_map.apply(np.array([100_000]))[0] - time_map.apply(np.array([99_999]))[0]
    assert gap == pytest.approx(0.5 + 1 / TRUE_RATE, abs=1e-5)
