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

    assert comp.n_pairs == br.size                       # every pulse, the right one
    assert np.abs(comp.after_s).max() < 1e-9
    assert np.abs(comp.before_s).max() == pytest.approx(40e-6 * 10780, rel=0.02)
    assert abs(comp.before_s[0]) < 1e-3                  # "before" is start-aligned
    # The unsynchronized error is worst at the end, as drift is.
    assert int(np.argmax(np.abs(comp.before_s))) == comp.n_pairs - 1


def test_unpaired_events_are_counted_not_paired():
    br = np.arange(0.0, 600.0, 2.0) + INTERCEPT
    obx = _onebox_clock(br)
    comp = compare.compare_event_times(br, np.delete(obx, [5, 50]), SLOPE, INTERCEPT)

    assert comp.n_pairs == br.size - 2
    assert comp.n_blackrock == br.size and comp.n_onebox == br.size - 2


def test_events_that_never_pair_raise_rather_than_report_nothing():
    with pytest.raises(ValueError, match="pair"):
        compare.compare_event_times(np.array([1.0, 2.0]), np.array([500.0]), 1.0, 0.0)


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


def test_the_analog_export_reads_back_on_blackrock_time(tmp_path):
    # End to end on a real export: the stamped map is what places the samples.
    import h5py
    from test_spikeglx import _onebox_file

    from spikesorting._io import analog, matlab, spikeglx

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
    events, threshold = analog.detect_onebox_events(info, "XA0")
    assert events.tolist() == pytest.approx([1.0])        # OneBox seconds
    assert analog.get_recording_overlap(info, None) == pytest.approx(
        (INTERCEPT, SLOPE * 4.0 + INTERCEPT))
    assert info.blackrock_time(30000, 30001)[0] == pytest.approx(SLOPE * 1.0 + INTERCEPT)
    assert analog.read_onebox(info, "XA0", 29999, 30001)[1] > 0
    with h5py.File(path, "r") as handle:                   # samples untouched
        assert handle["analog/data"].shape == (n, 2)


# --- event onsets: leaving the baseline, whichever way --------------------------

FS = 1000.0


def _photodiode(onsets_s, dur_s=0.2, total_s=60.0, rest=4.0, lit=1.0, noise=0.02, seed=0):
    """A photodiode resting at ``rest`` that goes to ``lit`` while the target shows."""
    rng = np.random.default_rng(seed)
    x = np.full(int(total_s * FS), rest) + rng.normal(0, noise, int(total_s * FS))
    for t in onsets_s:
        x[int(t * FS):int((t + dur_s) * FS)] += lit - rest
    return x


def _chunks(x, size=7_919):                         # a prime: edges land on boundaries
    return ((k, x[k:k + size]) for k in range(0, x.size, size))


def _onsets(x, **kw):
    levels = compare.event_levels(x)
    return compare.stream_event_onsets(_chunks(x), levels, FS, **kw), levels


ONSETS = [5.0, 9.3, 17.25, 30.0, 41.7, 55.5]


def test_a_photodiode_that_dips_gives_the_moment_the_target_appears():
    times, levels = _onsets(_photodiode(ONSETS, rest=4.0, lit=1.0))
    assert levels.polarity == -1
    assert times == pytest.approx(ONSETS, abs=2e-3)


def test_an_inverted_copy_gives_the_same_onsets():
    # Each system gets its own levels, so a copy recorded the other way up
    # still pairs with the first instead of being off by every flash's length.
    up, _ = _onsets(_photodiode(ONSETS, rest=0.0, lit=3.0, seed=1))
    down, _ = _onsets(_photodiode(ONSETS, rest=4.0, lit=1.0, seed=2))
    assert up == pytest.approx(down, abs=2e-3)


def test_levels_come_from_the_whole_session_not_its_first_minutes():
    # Nothing on screen for the first 5 minutes: the old threshold, set from
    # them, sat in the noise and found ~44 M "edges" on the rig.
    late = [320.0, 333.3, 351.0, 380.2]
    times, _ = _onsets(_photodiode(late, total_s=400.0))
    assert times == pytest.approx(late, abs=2e-3)


def test_noise_on_a_slow_edge_counts_once():
    x = _photodiode([], total_s=10.0, rest=0.0, noise=0.0)
    ramp = np.linspace(0.0, 1.0, 200)                # a slow 200 ms rise...
    x[2000:2200] = ramp + 0.08 * np.sin(np.arange(200) * 2.5)   # ...with ripple on it
    x[2200:4000] = 1.0
    times, _ = _onsets(x)
    assert times.size == 1


def test_monitor_flicker_inside_one_flash_is_one_event():
    x = _photodiode([2.0], dur_s=0.5, total_s=10.0, rest=0.0, lit=1.0, noise=0.0)
    x[2000:2500][(np.arange(500) % 8) < 2] = 0.0      # 125 Hz refresh dropouts
    times, _ = _onsets(x)
    assert times == pytest.approx([2.0], abs=2e-3)


def test_a_channel_with_no_events_says_so():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="never leaves its baseline"):
        compare.event_levels(rng.normal(2.0, 0.05, 60_000))


def test_noise_crossing_a_threshold_is_refused_not_counted():
    x = np.tile([0.0] * 3 + [1.0] * 3, 20_000).astype(float)   # 167 "events" a second
    levels = compare.event_levels(x)
    with pytest.raises(ValueError, match="not events"):
        compare.stream_event_onsets(_chunks(x), levels, FS, min_gap_s=0.0)
