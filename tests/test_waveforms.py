"""Per-spike snippets and the mean waveform, on synthetic traces.

The distinction this whole module exists for: Kilosort's ``templates.npy`` is a
fitted shape in whitened units, not an average and not in microvolts. What is
tested here is measured from raw samples, so a planted spike of known amplitude
must come back at that amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._export.waveforms import (
    accumulate,
    highpass,
    plan_snippets,
    snippet_regions,
)

WIDTH_BEFORE, WIDTH_AFTER = 20, 41           # 61 samples, Kilosort's template width


def _trace(n_samples: int, n_chan: int, spikes, shape, channel: int) -> np.ndarray:
    """A flat recording with ``shape`` planted at each sample in ``spikes``."""
    data = np.zeros((n_samples, n_chan), dtype=np.int16)
    for s in spikes:
        data[s - WIDTH_BEFORE : s - WIDTH_BEFORE + shape.size, channel] += shape
    return data


def _one_block(data):
    yield 0, data


def test_the_mean_recovers_a_planted_amplitude_in_microvolts():
    # The point of reading the raw binary at all. A template would give the
    # shape but not the amplitude, and not in physical units.
    peak_uv, uv_per_digit = -180.0, 0.195
    shape = np.zeros(WIDTH_BEFORE + WIDTH_AFTER, dtype=np.int16)
    shape[WIDTH_BEFORE] = int(round(peak_uv / uv_per_digit))

    spikes = np.array([500, 1500, 2500, 3500])
    data = _trace(6000, 4, spikes, shape, channel=2)
    plan = plan_snippets(spikes, np.zeros(4, dtype=np.int64), {0: 2}, 6000,
                         WIDTH_BEFORE, WIDTH_AFTER)

    result = accumulate(plan, _one_block(data), uv_per_digit=uv_per_digit)

    assert result.mean.shape == (WIDTH_BEFORE + WIDTH_AFTER, 1)
    assert result.mean[WIDTH_BEFORE, 0] == pytest.approx(peak_uv, abs=0.1)
    assert result.std[WIDTH_BEFORE, 0] == pytest.approx(0.0, abs=1e-6)   # identical spikes
    assert result.n_per_unit.tolist() == [4]


def test_snippets_are_cut_at_the_spike_sample():
    shape = np.arange(1, WIDTH_BEFORE + WIDTH_AFTER + 1, dtype=np.int16)
    spikes = np.array([1000, 2000])
    data = _trace(4000, 3, spikes, shape, channel=1)

    plan = plan_snippets(spikes, np.zeros(2, dtype=np.int64), {0: 1}, 4000,
                         WIDTH_BEFORE, WIDTH_AFTER)
    result = accumulate(plan, _one_block(data))

    assert result.snippets.shape == (WIDTH_BEFORE + WIDTH_AFTER, 2)
    assert np.array_equal(result.snippets[:, 0], shape)
    assert np.array_equal(result.snippets[:, 1], shape)


def test_a_spike_too_close_to_either_end_is_dropped_not_padded():
    # A zero-padded snippet averages into the mean as though it were signal, and
    # nothing downstream could tell it apart from a quiet channel.
    spikes = np.array([5, 1000, 3995])           # first and last have no room
    plan = plan_snippets(spikes, np.zeros(3, dtype=np.int64), {0: 0}, 4000,
                         WIDTH_BEFORE, WIDTH_AFTER)

    assert plan.n_spikes == 1
    assert plan.sample.tolist() == [1000]
    assert plan.n_dropped == 2

    result = accumulate(plan, _one_block(np.zeros((4000, 2), dtype=np.int16)))
    assert "dropped" in result.notes[0]


def test_each_unit_is_cut_from_its_own_channel():
    shape = np.full(WIDTH_BEFORE + WIDTH_AFTER, 10, dtype=np.int16)
    data = np.zeros((4000, 4), dtype=np.int16)
    data[980:1041, 1] = shape                    # unit 0 on channel 1
    data[1980:2041, 3] = shape * 2               # unit 1 on channel 3

    plan = plan_snippets(
        np.array([1000, 2000]), np.array([0, 1]), {0: 1, 1: 3}, 4000,
        WIDTH_BEFORE, WIDTH_AFTER,
    )
    result = accumulate(plan, _one_block(data))

    assert plan.channel.tolist() == [1, 3]
    assert result.unit_ids.tolist() == [0, 1]
    assert result.mean[0, 0] == pytest.approx(10.0)
    assert result.mean[0, 1] == pytest.approx(20.0)


def test_one_forward_pass_over_chunks_gives_the_same_answer():
    # The pass has to work chunked -- a real binary is read in blocks, and a
    # spike must be cut once, from whichever block wholly contains its window.
    rng = np.random.default_rng(0)
    data = rng.integers(-100, 100, size=(5000, 3), dtype=np.int16)
    spikes = np.array([200, 900, 1500, 2400, 3300, 4700])
    plan = plan_snippets(spikes, np.zeros(spikes.size, dtype=np.int64), {0: 2}, 5000,
                         WIDTH_BEFORE, WIDTH_AFTER)

    whole = accumulate(plan, _one_block(data))

    def blocks(size):
        for start in range(0, 5000, size):
            yield start, data[start : start + size + WIDTH_BEFORE + WIDTH_AFTER]

    for size in (512, 1000, 4096):
        chunked = accumulate(plan, blocks(size))
        assert np.array_equal(chunked.snippets, whole.snippets), size
        assert np.allclose(chunked.mean, whole.mean), size
        assert not chunked.notes, size            # nothing uncovered


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="cluster ids"):
        plan_snippets(np.arange(5), np.zeros(3, dtype=np.int64), {0: 0}, 100, 10, 10)


# ---------------------------------------------------------------------------
# High-pass: the export must measure the band the sorter sorted
# ---------------------------------------------------------------------------

FS = 30000.0


def _spike_shape(peak: int) -> np.ndarray:
    """A biphasic trough-then-rebound, not a delta -- a delta survives any filter."""
    t = np.arange(WIDTH_BEFORE + WIDTH_AFTER) - WIDTH_BEFORE
    shape = -np.exp(-((t / 4.0) ** 2)) + 0.35 * np.exp(-(((t - 9) / 7.0) ** 2))
    return np.round(shape / np.abs(shape).max() * abs(peak)).astype(np.int16)


def _broadband(n_samples: int, n_chan: int, spikes, shape, channel: int) -> np.ndarray:
    """Spikes riding on the LFP a broadband recording actually carries."""
    t = np.arange(n_samples) / FS
    pedestal = 600.0 + 900.0 * np.sin(2 * np.pi * 3.0 * t) + 400.0 * np.sin(2 * np.pi * 11.0 * t)
    data = np.repeat(pedestal[:, None], n_chan, axis=1).astype(np.int16)
    for s in spikes:
        data[s - WIDTH_BEFORE : s - WIDTH_BEFORE + shape.size, channel] += shape
    return data


def _plan_for(spikes, n_samples, margin=0, **kw):
    return plan_snippets(
        spikes, np.zeros(len(spikes), dtype=np.int64), {0: 1}, n_samples,
        WIDTH_BEFORE, WIDTH_AFTER, margin=margin, **kw,
    )


def test_the_highpass_removes_the_lfp_a_broadband_recording_carries():
    # The reason this exists. Kilosort sorted the 300 Hz-filtered signal and Phy
    # displayed a filtered trace, so a mean cut from raw .ns6 is neither: every
    # snippet rides on the LFP, and averaging does not remove it.
    n = 200_000
    rng = np.random.default_rng(0)
    spikes = np.sort(rng.choice(np.arange(2000, n - 2000), size=300, replace=False))
    data = _broadband(n, 3, spikes, _spike_shape(-500), channel=1)

    # The same spikes with no LFP under them: what the mean *should* converge to.
    clean = np.zeros_like(data)
    for s in spikes:
        clean[s - WIDTH_BEFORE : s + WIDTH_AFTER, 1] = _spike_shape(-500)

    plan = _plan_for(spikes, n)
    raw = accumulate(plan, _one_block(data))
    filtered = accumulate(plan, _one_block(data), highpass_hz=300.0, fs=FS)
    ideal = accumulate(plan, _one_block(clean), highpass_hz=300.0, fs=FS)

    # The trough survives either way -- it is the baseline that differs.
    assert int(np.argmin(raw.mean[:, 0])) == WIDTH_BEFORE
    assert int(np.argmin(filtered.mean[:, 0])) == WIDTH_BEFORE

    # Raw, the pedestal is simply present in the mean: it sits ~600 off the
    # waveform it should have converged to, on a spike only 500 deep.
    assert np.abs(raw.mean[:, 0] - ideal.mean[:, 0]).max() > 400

    # Filtered, it lands within a few percent of the no-LFP recording -- so the
    # LFP is gone rather than merely reduced. It does not land exactly there, and
    # should not: zero-phase filtering a pulse with net area puts symmetric lobes
    # either side of it, and both means carry them.
    assert np.abs(filtered.mean[:, 0] - ideal.mean[:, 0]).max() < 15


def test_filtering_padded_regions_equals_filtering_the_whole_recording():
    # The claim the pad rests on. Phy and Kilosort both filter the bare 2 ms
    # window, where a filter rings at exactly the samples being kept.
    n = 120_000
    rng = np.random.default_rng(1)
    spikes = np.sort(rng.choice(np.arange(3000, n - 3000), size=60, replace=False))
    data = _broadband(n, 3, spikes, _spike_shape(-500), channel=1)
    pad = int(0.01 * FS)

    plan = _plan_for(spikes, n, margin=pad)
    whole = accumulate(plan, _one_block(highpass(data, FS, 300.0, 3)))
    regions = snippet_regions(plan, pad)
    piecewise = accumulate(
        plan, ((a, data[a:b]) for a, b in regions),
        highpass_hz=300.0, fs=FS, pad=pad,
    )

    assert len(regions) <= plan.n_spikes            # overlapping windows merged
    assert np.abs(piecewise.mean - whole.mean).max() < 1e-3


def test_the_pad_is_what_makes_that_true():
    # If a two-sample pad agreed as well, the pad would be decoration.
    n = 120_000
    rng = np.random.default_rng(1)
    spikes = np.sort(rng.choice(np.arange(3000, n - 3000), size=60, replace=False))
    data = _broadband(n, 3, spikes, _spike_shape(-500), channel=1)

    plan = _plan_for(spikes, n, margin=2)
    whole = accumulate(plan, _one_block(highpass(data, FS, 300.0, 3)))
    starved = accumulate(
        plan, ((a, data[a:b]) for a, b in snippet_regions(plan, 2)),
        highpass_hz=300.0, fs=FS, pad=2,
    )

    assert np.abs(starved.mean - whole.mean).max() > 1.0


def test_a_spike_within_the_filter_pad_of_an_end_is_dropped():
    # The pad must never be filled with signal from before the file began.
    n = 20_000
    pad = 300
    spikes = np.array([WIDTH_BEFORE + 5, n // 2, n - WIDTH_AFTER - 5])
    plan = _plan_for(spikes, n, margin=pad)

    assert plan.n_spikes == 1 and plan.sample.tolist() == [n // 2]
    assert plan.n_dropped == 2
    assert min(start for start, _ in snippet_regions(plan, pad)) >= 0


# ---------------------------------------------------------------------------
# Subsampling: uniform over the recording, not over the spike list
# ---------------------------------------------------------------------------


def test_the_subset_is_uniform_in_time_not_in_spike_count():
    # The distinction the whole selection rule turns on. Every-k-th-spike is
    # uniform in the spike *list*, so a unit that fires hard during one task
    # would be represented almost entirely by that task -- and the subset could
    # say nothing about the rest of the session.
    n = 30_000 * 300
    busy = np.arange(1_000, n // 3, 600)                 # 50 Hz for one third
    quiet = np.arange(n // 3, n - 1_000, 15_000)         # 2 Hz for the rest
    spikes = np.concatenate([busy, quiet])

    plan = _plan_for(spikes, n, max_per_unit=60)

    assert plan.n_available == spikes.size
    assert busy.size > 12 * quiet.size                   # heavily lopsided input
    per_third = np.histogram(plan.sample, bins=3, range=(0, n))[0]
    assert per_third.min() >= 15, per_third              # ...and an even output


def test_the_cap_is_a_ceiling_and_a_short_unit_keeps_everything():
    n = 100_000
    many = np.arange(1_000, n - 1_000, 50)
    assert _plan_for(many, n, max_per_unit=40).n_spikes <= 40

    few = np.arange(1_000, n - 1_000, 9_000)
    plan = _plan_for(few, n, max_per_unit=500)
    assert plan.sample.tolist() == few.tolist()


def test_the_same_session_selects_the_same_spikes_twice():
    # No RNG anywhere: re-running the export must not produce a different file.
    n = 400_000
    spikes = np.arange(1_000, n - 1_000, 137)
    first, second = (_plan_for(spikes, n, max_per_unit=100) for _ in range(2))
    assert first.sample.tolist() == second.sample.tolist()


def test_each_kept_spike_knows_where_it_came_from():
    # spike_index is what lets the exported spike_time_s be the *aligned* time of
    # exactly these spikes, without re-deriving the selection.
    n = 200_000
    spikes = np.arange(1_000, n - 1_000, 311)
    clusters = np.arange(spikes.size) % 3
    channels = {0: 0, 1: 1, 2: 2}

    plan = plan_snippets(
        spikes, clusters, channels, n, WIDTH_BEFORE, WIDTH_AFTER, max_per_unit=25
    )

    assert plan.spike_index.size == plan.n_spikes
    assert spikes[plan.spike_index].tolist() == plan.sample.tolist()
    assert clusters[plan.spike_index].tolist() == plan.unit_id.tolist()


def test_no_cap_and_no_filter_is_exactly_what_it_was_before():
    # The defaults-off path must be untouched, so the Neuropixels export and any
    # every-spike run come out of this change bit for bit.
    n = 60_000
    spikes = np.arange(1_000, n - 1_000, 700)
    data = _broadband(n, 3, spikes, _spike_shape(-500), channel=1)

    plan = _plan_for(spikes, n, max_per_unit=None)
    result = accumulate(plan, _one_block(data), highpass_hz=None)

    assert plan.n_spikes == plan.n_available == spikes.size
    reference = accumulate(_plan_for(spikes, n), _one_block(data))
    assert np.array_equal(result.mean, reference.mean)
    assert np.array_equal(result.snippets, reference.snippets)
