"""Per-spike snippets and the mean waveform, on synthetic traces.

The distinction this whole module exists for: Kilosort's ``templates.npy`` is a
fitted shape in whitened units, not an average and not in microvolts. What is
tested here is measured from raw samples, so a planted spike of known amplitude
must come back at that amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._export.waveforms import accumulate, plan_snippets

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
