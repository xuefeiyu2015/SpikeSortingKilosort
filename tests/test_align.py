"""Overlap trimming, time mapping, and step-9 validation."""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._sync import align


def test_overlap_window_uses_the_later_start_and_earlier_end():
    reference = np.arange(0.0, 100.0)
    other = np.arange(20.0, 150.0)
    assert align.overlap_window(reference, other) == (20.0, 99.0)


def test_overlap_window_applies_the_offset():
    reference = np.arange(0.0, 100.0)
    other = np.arange(0.0, 100.0) - 30.0
    assert align.overlap_window(reference, other, offset_s=30.0) == (0.0, 99.0)


def test_overlap_window_rejects_empty():
    with pytest.raises(ValueError):
        align.overlap_window(np.empty(0), np.arange(3.0))


def test_trim_pair_to_overlap_gives_matching_counts():
    # Blackrock started 20 s earlier and stopped 30 s later than SpikeGLX.
    reference = np.arange(0.0, 200.0)
    other = np.arange(20.0, 170.0) - 5.0  # SpikeGLX clock runs 5 s behind

    ref_trim, other_trim = align.trim_pair_to_overlap(reference, other, offset_s=5.0)
    assert ref_trim.size == other_trim.size
    assert np.allclose(ref_trim, other_trim + 5.0)


def test_trim_margin_drops_boundary_edges():
    reference = np.arange(0.0, 10.0)
    other = np.arange(0.0, 10.0)
    no_margin, _ = align.trim_pair_to_overlap(reference, other)
    with_margin, _ = align.trim_pair_to_overlap(reference, other, margin_s=0.25)
    assert with_margin.size == no_margin.size - 2


def test_fit_linear_map_recovers_slope_and_intercept():
    other = np.arange(0.0, 1000.0)
    slope, intercept = 1.000025, -12.5  # 25 ppm faster clock
    reference = slope * other + intercept

    mapping = align.fit_linear_map(other, reference)
    assert mapping.slope == pytest.approx(slope, rel=1e-12)
    assert mapping.intercept == pytest.approx(intercept, abs=1e-6)
    assert mapping.drift_ppm == pytest.approx(25.0, abs=1e-3)
    assert np.abs(mapping.residuals_s).max() < 1e-9


def test_fit_linear_map_requires_matched_lengths():
    with pytest.raises(ValueError, match="same length"):
        align.fit_linear_map(np.arange(5.0), np.arange(6.0))


def test_fit_linear_map_requires_two_points():
    with pytest.raises(ValueError, match="at least two"):
        align.fit_linear_map(np.array([1.0]), np.array([2.0]))


def test_apply_linear_map_round_trips_through_the_fit():
    other = np.arange(0.0, 100.0)
    reference = 1.00001 * other + 3.0
    mapping = align.fit_linear_map(other, reference)
    assert np.allclose(mapping.apply(other), reference, atol=1e-9)


def test_validate_alignment_passes_within_tolerance():
    reference = np.arange(0.0, 500.0, 14.0)
    mapped = reference + 2e-4  # 0.2 ms off
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert report.passed
    assert report.n_checked == reference.size
    assert report.max_residual_s == pytest.approx(2e-4)
    assert "PASS" in report.summary()


def test_validate_alignment_fails_outside_tolerance():
    reference = np.arange(0.0, 500.0, 14.0)
    mapped = reference + 5e-3  # 5 ms off
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert not report.passed
    assert "FAIL" in report.summary()


def test_validate_alignment_reports_nothing_matched():
    # A gross misalignment must read as "not validated", never as a pass.
    reference = np.arange(0.0, 100.0, 14.0)
    mapped = reference + 1000.0
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert report.n_checked == 0
    assert not report.passed
    assert "NOT validated" in report.summary()


def test_validate_alignment_fails_when_part_of_the_session_is_off():
    # The second half lands 80 ms off: those bursts find no partner within
    # 50 ms, so the residuals alone (first half, all exact) would read PASS.
    reference = np.arange(0.0, 7000.0, 14.0)
    mapped = reference.copy()
    mapped[reference.size // 2:] += 0.08
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert report.max_residual_s < 1e-9
    assert report.n_expected >= reference.size - 1     # the shifted last one may fall outside
    assert report.coverage == pytest.approx(0.5, abs=0.01)
    assert not report.passed
    assert "50% of the bursts matched" in report.summary()


def test_validate_alignment_tolerates_a_few_missing_bursts():
    # A burst one system missed is not a mismatch: it lowers both counts.
    reference = np.arange(0.0, 7000.0, 14.0)
    mapped = np.delete(reference, [10, 200, 300]) + 2e-4
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert report.passed
    assert report.n_checked == report.n_expected == reference.size - 3


def test_a_burst_channel_that_stops_early_is_measured_against_the_shared_interval():
    # 100 bursts on Blackrock, only the first three on the other side. Judged by
    # where the detected bursts overlap that is 3/3 over 28 s; judged by the
    # stretch both systems recorded (the 1 Hz pairs) it is 3 of 100.
    reference = np.arange(100) * 14.0 + 50.0
    mapped = reference[:3] + 1e-4
    alone = align.validate_alignment(mapped, reference, tolerance_s=1e-3)
    assert alone.passed and alone.n_expected == 3            # the hole

    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3,
                                      shared_interval_s=(40.0, 1440.0))
    assert not report.passed
    assert (report.n_checked, report.n_expected) == (3, 100)
    assert report.shared_s == pytest.approx(1400.0)
    assert "over 0.5 min of 23.3" in report.summary()


def test_the_shared_interval_still_forgives_a_missed_burst_or_two():
    reference = np.arange(100) * 14.0 + 50.0
    mapped = np.delete(reference, [40, 41]) + 1e-4
    report = align.validate_alignment(mapped, reference, tolerance_s=1e-3,
                                      shared_interval_s=(40.0, 1440.0))
    assert report.passed and (report.n_checked, report.n_expected) == (98, 100)


def test_validate_alignment_needs_more_than_two_bursts():
    reference = np.array([0.0, 14.0])
    report = align.validate_alignment(reference + 1e-4, reference, tolerance_s=1e-3)
    assert not report.passed
    assert "fewer than 3" in report.fail_reason


def test_interval_offset_map_is_tprimes_rule():
    # One offset per interval, the interval chosen by the reference ("to")
    # edges. As TPrime is fed here, the two lists differ only by a residual.
    frm = np.array([0.0, 1.0, 2.0, 3.0])
    to = frm + np.array([0.0, 1e-3, 3e-3, 2e-3])
    # before the first edge; mid-interval; just past to[1] (1.001) but before
    # it on frm's side -- the reference decides; and past the last edge.
    t = np.array([-0.5, 0.5, 1.0005, 1.0015, 9.0])
    offsets = np.array([0.0, 0.0, 0.0, 1e-3, 2e-3])
    assert np.allclose(align.interval_offset_map(t, frm, to), t + offsets)
    with pytest.raises(ValueError, match="matched edge pairs"):
        align.interval_offset_map(t, frm, to[:-1])


def test_validate_alignment_handles_empty_input():
    report = align.validate_alignment(np.empty(0), np.arange(3.0))
    assert report.n_checked == 0
    assert not report.passed


def test_uncorrected_drift_is_caught_by_validation():
    # A map fit without slope correction leaves a residual that grows with time;
    # over an hour a 20 ppm error is 72 ms, far outside the 1 ms tolerance.
    bursts = np.arange(0.0, 3600.0, 14.0)
    drifting = bursts * (1 + 20e-6)
    report = align.validate_alignment(drifting, bursts, tolerance_s=1e-3, match_tolerance_s=1.0)
    assert not report.passed
    assert report.max_residual_s > 0.05


def test_fit_same_run_map_takes_the_onebox_clock_onto_the_probe_clock():
    # Two streams of one run: they start together, so the edges pair with no
    # coarse offset, and only the crystals differ.
    probe = np.arange(0.0, 3600.0, 1.0) + 0.25
    onebox = (probe - 0.002) / (1.0 + 30e-6)

    mapping = align.fit_same_run_map(onebox, probe)

    assert mapping.drift_ppm == pytest.approx(30.0, abs=1e-3)
    assert mapping.intercept == pytest.approx(0.002, abs=1e-9)
    assert np.allclose(mapping.apply(onebox), probe, atol=1e-9)


def test_compose_linear_maps_is_inner_then_outer():
    # OneBox -> probe, then probe -> Blackrock: one line for a OneBox analog trace.
    inner = align.LinearMap(1.0 + 30e-6, 0.002, 3600, np.empty(0))
    outer = align.LinearMap(1.0 + 50e-6, 12.3456, 10800, np.empty(0))
    t = np.linspace(0.0, 10800.0, 7)

    composed = align.compose_linear_maps(outer, inner)

    assert np.allclose(composed.apply(t), outer.apply(inner.apply(t)), rtol=0, atol=1e-9)
    assert composed.drift_ppm == pytest.approx(80.0015, abs=1e-3)
    assert composed.n_points == 3600


def test_fit_same_run_map_refuses_trains_that_do_not_pair():
    with pytest.raises(ValueError, match="cannot fit"):
        align.fit_same_run_map(np.array([0.5]), np.array([0.0, 1.0]))
