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
