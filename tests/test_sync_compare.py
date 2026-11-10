"""OneBox vs Blackrock synchrony: the comparison layer, and the readers it feeds."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import coded_burst_times

from spikesorting._sync import compare

# A OneBox 40 ppm fast against the NSP, recording started 12.5 s after it.
SLOPE = 1.0 + 40e-6
INTERCEPT = 12.5


def _onebox_clock(blackrock_s):
    return (np.asarray(blackrock_s) - INTERCEPT) / SLOPE


def test_synchronization_removes_the_drift_a_start_offset_leaves():
    # Three hours of coded bursts: 40 ppm is 432 ms by the end -- far past the
    # gap between two pulses, so pairing must go through the map.
    br = coded_burst_times(770, interval_s=14.0, start_s=100.0) + INTERCEPT
    comp = compare.compare_event_times(br, _onebox_clock(br), SLOPE, INTERCEPT, label="burst")
    summary = compare.summarize_asynchrony(comp)

    assert comp.n_pairs == br.size                       # every pulse, the right one
    assert summary["after_max_ms"] < 1e-6
    assert summary["before_max_ms"] == pytest.approx(40e-6 * 10780 * 1e3, rel=0.02)
    assert abs(comp.before_s[0]) < 1e-3                  # "before" is start-aligned
    # The unsynchronized error is worst at the end, as drift is.
    assert compare.largest_asynchrony(comp, "before") == comp.n_pairs - 1


def test_unpaired_events_are_counted_not_paired():
    br = np.arange(0.0, 600.0, 2.0) + INTERCEPT
    obx = _onebox_clock(br)
    comp = compare.compare_event_times(br, np.delete(obx, [5, 50]), SLOPE, INTERCEPT)

    assert comp.n_pairs == br.size - 2
    assert comp.n_blackrock == br.size and comp.n_onebox == br.size - 2


def test_events_that_never_pair_raise_rather_than_report_nothing():
    with pytest.raises(ValueError, match="pair"):
        compare.compare_event_times(np.array([1.0, 2.0]), np.array([500.0]), 1.0, 0.0)


def test_largest_asynchrony_finds_the_worst_pair_after_sync():
    br = np.arange(0.0, 100.0, 1.0)
    obx = br.copy()
    obx[37] += 0.8e-3                                     # one late OneBox edge
    comp = compare.compare_event_times(br, obx, 1.0, 0.0)
    assert compare.largest_asynchrony(comp, "after") == 37
    with pytest.raises(ValueError):
        compare.largest_asynchrony(comp, "during")


def test_sample_ranges_invert_the_maps():
    fs = 30000.0
    k0, k1 = compare.onebox_sample_range(112.5, 113.5, fs, SLOPE, INTERCEPT, 10**9)
    t = SLOPE * (np.array([k0, k1]) / fs) + INTERCEPT
    assert t[0] <= 112.5 + 1 / fs and t[1] >= 113.5 - 1 / fs
    assert compare.onebox_sample_range(0.0, 5.0, fs, SLOPE, INTERCEPT, 100) == (0, 0)

    # Two segments: a window in the second starts from its own first sample.
    segments = [(0, 1000, 1e-3, 100.0), (1000, 1000, 1e-3, 200.0)]
    s0, s1 = compare.blackrock_sample_range(100.1, 100.2, segments)
    assert s0 in (99, 100) and s1 in (200, 201)            # covers the window
    # Clipped to the segment it starts in, never into the next run of samples.
    s0, s1 = compare.blackrock_sample_range(200.5, 201.5, segments)
    assert s0 in (1499, 1500) and s1 == 2000


def test_display_helpers():
    assert compare.display_stride(100, 1000) == 1
    assert compare.display_stride(10_001, 1000) == 11
    square = np.r_[np.zeros(500), np.full(500, 4.0)]
    assert compare.auto_threshold(square) == pytest.approx(2.0)
    # A 1-sample spike survives a 100x reduction, where a stride would drop it.
    spiky = np.zeros(100_000)
    spiky[50_001] = 1.0
    t, y = compare.display_envelope(np.arange(spiky.size), spiky, 1000)
    assert y.size <= 1000 and y.max() == 1.0 and t.size == y.size
    t, y = compare.display_envelope(np.arange(10), np.arange(10.0), 1000)
    assert y.size == 10                                   # short: unchanged
    # A sparse pulse train has both percentiles on the baseline; it must still
    # scale to 0..1 and get a threshold between its levels.
    sparse = np.zeros(10_000)
    sparse[::1000] = 3.0
    assert compare.normalize_trace(sparse).max() == pytest.approx(1.0)
    assert compare.auto_threshold(sparse) == pytest.approx(1.5)
    scaled = compare.normalize_trace(square)
    assert scaled.min() == pytest.approx(0.0) and scaled.max() == pytest.approx(1.0)


def test_window_traces_put_the_onebox_on_blackrock_time_and_plot(tmp_path):
    # End to end on a real export: the stamped map is what places the samples.
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    from test_spikeglx import _onebox_file

    from spikesorting._io import analog, matlab, spikeglx
    from spikesorting._plots import sync as plots

    n = 30000 * 4
    data = np.zeros((n, 4), dtype=np.int16)
    data[30000:45000, 0] = 20000                           # a 0.5 s pulse at 1.0 s
    obx = _onebox_file(tmp_path, data)
    path = spikeglx.export_onebox_analog(obx, tmp_path / "out", named={"photodiode": 1})
    with pytest.raises(ValueError, match="time_remapping"):
        analog.load_onebox_analog(path)
    matlab.update_mat(path, {"analog": {
        "timebase": "blackrock", "blackrock_slope": SLOPE, "blackrock_intercept": INTERCEPT,
    }})

    info = analog.load_onebox_analog(path)
    assert info.names == ("XA0", "photodiode")
    traces = analog.get_window_traces(info, None, INTERCEPT + 0.5, INTERCEPT + 2.0)

    pulse = traces[0]
    events, threshold = analog.detect_onebox_events(info, "XA0")
    assert events.tolist() == pytest.approx([1.0])        # OneBox seconds
    assert analog.get_recording_overlap(info, None) == pytest.approx(
        (INTERCEPT, SLOPE * 4.0 + INTERCEPT))
    rising = pulse.t_onebox[np.argmax(pulse.y_onebox > 0.5)]
    assert rising == pytest.approx(SLOPE * 1.0 + INTERCEPT, abs=1e-4)
    assert pulse.t_blackrock is None                       # no Blackrock side given

    unsynced = analog.get_window_traces(info, None, INTERCEPT + 0.5, INTERCEPT + 2.0,
                                        onebox_offset_s=INTERCEPT)
    br = np.array([1.0, 2.0, 3.0]) * SLOPE + INTERCEPT
    comp = compare.compare_event_times(br, _onebox_clock(br), SLOPE, INTERCEPT, label="x")
    summary = compare.summarize_asynchrony(comp)
    for fig in (
        plots.plot_asynchrony_before_after([comp], [summary]),
        plots.plot_window_traces(traces, "t", marker_s=INTERCEPT + 1.0, time_origin_s=INTERCEPT),
        plots.plot_window_before_after(unsynced, traces, "t", time_origin_s=INTERCEPT),
    ):
        assert fig.axes
    with h5py.File(path, "r") as handle:                   # samples untouched
        assert handle["analog/data"].shape == (n, 2)
