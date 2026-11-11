"""Picking the 1 Hz cycle without the burst: the vote is whole periods, nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._sync import cycles


def _flashes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return 1000.0 + np.cumsum(rng.uniform(1.0, 2.5, n))


@pytest.mark.parametrize("k", [-7, 0, 3, 42])
def test_irregular_events_pick_the_whole_cycle(k):
    br = _flashes()
    # The OneBox copy through a map that is k cycles off, plus the .ns2's few
    # ms of filter delay and some jitter -- neither may move the vote.
    rng = np.random.default_rng(1)
    mapped = br - k * 1.0 - 0.004 + rng.normal(0, 0.001, br.size)
    vote = cycles.cycle_shift_from_events(br, mapped)
    assert vote is not None
    assert vote.n_periods == k and vote.shift_s == pytest.approx(k)
    assert vote.n_support > 0.9 * br.size


def test_events_missing_on_one_side_still_vote():
    br = _flashes()
    mapped = np.delete(br, np.arange(0, br.size, 3)) + 5.0      # a third undetected
    assert cycles.cycle_shift_from_events(br, mapped).n_periods == -5


def test_a_periodic_train_cannot_pick_a_cycle():
    # Every whole shift lines a 1 Hz train up equally: that is the ambiguity
    # itself, so it must abstain rather than pick one.
    br = 1000.0 + np.arange(300.0)
    assert cycles.cycle_shift_from_events(br, br + 2.0) is None


def test_too_few_events_abstain():
    br = _flashes(5)
    assert cycles.cycle_shift_from_events(br, br) is None
    assert cycles.cycle_shift_from_events(np.empty(0), br) is None


def test_lags_vote_for_minus_the_lag():
    # A OneBox copy 3 s late (positive lag) is moved 3 s earlier.
    lags = np.array([3.002, 2.998, 3.003, 3.001, 0.47, np.nan])
    vote = cycles.cycle_shift_from_lags(lags)
    assert vote.n_periods == -3 and vote.n_support == 4
    assert cycles.cycle_shift_from_lags(np.array([3.0, 3.0])) is None   # too few
    assert cycles.cycle_shift_from_lags(np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])) is None


def test_a_blink_cannot_own_the_eye_correlation():
    from spikesorting._sync import compare

    rng = np.random.default_rng(2)
    eye = np.cumsum(rng.normal(size=20_000))
    later = np.roll(eye, 300)
    later[5000:5050] += 1e4                                  # a blink on one copy
    lag, corr = compare.xcorr_lag(
        compare.clipped_diff(eye[400:]), compare.clipped_diff(later[400:]), 1000.0, 1.0
    )
    assert lag == pytest.approx(0.3, abs=0.002) and corr > 0.9
