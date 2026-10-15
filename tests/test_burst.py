"""14 s coded burst matching -- the coarse alignment step."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import coded_burst_times

from spikesorting._sync import burst


def test_group_burst_onsets_collapses_dense_pulses():
    times = coded_burst_times(n_bursts=5, interval_s=14.0, pulses_per_burst=8)
    onsets = burst.group_burst_onsets(times, min_gap_s=7.0)
    assert onsets.size == 5
    assert np.allclose(onsets, np.arange(5) * 14.0)


def test_group_burst_onsets_on_empty_input():
    assert burst.group_burst_onsets(np.empty(0)).size == 0


def test_burst_patterns_return_intra_burst_intervals():
    times = coded_burst_times(n_bursts=3, pulses_per_burst=4, intra_gap_s=0.02, coded=False)
    patterns = burst.burst_patterns(times)
    assert len(patterns) == 3
    assert all(np.allclose(p, 0.02) for p in patterns)


def test_coded_bursts_have_distinct_patterns():
    # Guards the fixture itself: if the "coded" bursts were identical, the
    # disambiguation tests below would pass for the wrong reason.
    patterns = burst.burst_patterns(coded_burst_times(5, pulses_per_burst=8))
    assert not np.allclose(patterns[0], patterns[1])


def test_estimate_offset_recovers_a_known_shift():
    reference = coded_burst_times(20, pulses_per_burst=1)
    offset = 123.456
    other = reference - offset
    assert burst.estimate_offset(reference, other) == pytest.approx(offset, abs=1e-9)


def test_estimate_offset_rejects_empty_input():
    with pytest.raises(ValueError):
        burst.estimate_offset(np.empty(0), np.array([1.0]))


def test_estimate_offset_on_a_periodic_train_is_ambiguous():
    # Documents the limitation the coded burst exists to solve: with a purely
    # periodic train, every whole-cycle shift fits equally well and the densest
    # cluster simply maximises overlap. Here the true shift is -50 s but maximum
    # overlap is at +20 s, and onsets alone cannot tell them apart.
    all_bursts = coded_burst_times(30, pulses_per_burst=1, coded=False)
    reference = all_bursts[5:]
    other = all_bursts[:25] + 50.0
    assert burst.estimate_offset(reference, other) == pytest.approx(20.0, abs=1e-6)


def test_candidate_offsets_are_one_burst_interval_apart():
    reference = coded_burst_times(20, pulses_per_burst=1, coded=False)
    other = reference.copy()
    candidates = np.sort(burst.candidate_offsets(reference, other, max_candidates=5))
    assert np.allclose(np.diff(candidates), 14.0, atol=1e-6)


def test_codes_pair_each_burst_with_itself_and_nothing_else():
    # A one-cycle error lines the onsets up perfectly -- the ambiguity onsets
    # cannot resolve. The intervals inside each burst can: every burst's code
    # matches its own and no other.
    reference = coded_burst_times(20, pulses_per_burst=8)

    _, _, ref_idx, other_idx = burst.match_burst_codes(reference, reference - 3.0)

    assert ref_idx.tolist() == other_idx.tolist() == list(range(20))


def test_match_times_is_one_to_one():
    reference = np.array([0.0, 1.0, 2.0])
    # Two candidates crowd around 1.0; only the nearest may match.
    other = np.array([0.001, 0.999, 1.05, 2.002])
    ref_idx, other_idx = burst.match_times(reference, other, 0.0, tolerance_s=0.01)
    assert ref_idx.tolist() == [0, 1, 2]
    assert other_idx.tolist() == [0, 1, 3]
    assert len(set(other_idx.tolist())) == other_idx.size


def test_match_bursts_end_to_end():
    reference = coded_burst_times(20, pulses_per_burst=6)
    offset = 37.5
    other = reference - offset

    match = burst.match_bursts(reference, other)
    assert match.offset_s == pytest.approx(offset, abs=1e-9)
    assert match.n_matched == 20
    assert match.max_residual_s < 1e-9
    assert match.slope == pytest.approx(1.0, abs=1e-12)
    assert match.code_matches == 20


def test_match_bursts_uses_the_code_when_starts_do_not_overlap():
    # The case that matters: each system missed bursts the other caught, so the
    # maximum-overlap alignment is off by five cycles. Only the intra-burst code
    # picks the right one.
    full = coded_burst_times(30, pulses_per_burst=8)
    onsets = np.arange(30) * 14.0
    reference = full[full >= onsets[5]]
    other = full[full < onsets[25]] - 10.0

    match = burst.match_bursts(reference, other)
    assert match.offset_s == pytest.approx(10.0, abs=1e-6)
    assert match.n_matched == 20  # bursts 5..24
    assert match.n_reference == 25
    assert match.n_other == 25
    assert match.runner_up_matches == 0


@pytest.mark.parametrize("ppm", [5.0, 20.0, 50.0])
def test_match_bursts_holds_the_right_cycle_under_hours_of_drift(ppm):
    # Three hours at 50 ppm is 540 ms -- no single offset lines the bursts up to
    # a few ms, which used to let a wrong cycle win on chance coincidences.
    # Pairing by code involves no offset, and the line through the pairs carries
    # the drift.
    reference = coded_burst_times(770, pulses_per_burst=8, start_s=105.0)
    other = (reference - 12.3456) * (1 + ppm * 1e-6)

    match = burst.match_bursts(reference, other)

    assert match.n_matched == 770
    assert (match.slope - 1.0) * 1e6 == pytest.approx(-ppm, rel=1e-3)
    assert match.max_residual_s < 1e-6
    assert np.allclose(match.apply(other), reference, atol=1e-6)


def test_a_code_repeated_hourly_is_resolved_by_agreement():
    # The real burst code repeats every hour, so each burst also matches the
    # one 3600 s away. Those false pairs disagree with the true offset by an
    # hour, and lose the vote.
    hour = coded_burst_times(257, pulses_per_burst=8)[: 257 * 8]
    reference = np.concatenate([hour, hour + 257 * 14.0])     # two hours, same codes
    other = reference[reference < 4000.0] - 10.0

    match = burst.match_bursts(reference, other)

    assert match.offset_s == pytest.approx(10.0, abs=1e-6)
    assert match.code_matches > match.runner_up_matches > 0


def test_codes_that_never_agree_are_refused_not_guessed():
    reference = coded_burst_times(10, pulses_per_burst=8)
    other = coded_burst_times(10, pulses_per_burst=5)          # different code entirely
    with pytest.raises(ValueError, match="carries the code"):
        burst.match_bursts(reference, other)


def test_match_bursts_rejects_empty_input():
    with pytest.raises(ValueError):
        burst.match_bursts(np.empty(0), coded_burst_times(3))
