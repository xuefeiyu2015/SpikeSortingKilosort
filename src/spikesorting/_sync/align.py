"""Trimming, mapping and validation for cross-system alignment (steps 8.2 and 9).

Pure computation. Blackrock is the reference timebase throughout: Neuropixels
times are mapped *onto* Blackrock time, never the other way round.

Two mappings live here:

* :func:`fit_linear_map` -- a least-squares fit of the matched 1 Hz edges. Enough
  on its own for most purposes, and the fallback when TPrime is unavailable
  (the HPC, macOS).
* TPrime -- the authoritative fine alignment, run by :mod:`spikesorting._sync.tprime`.

Both are validated the same way: map the 14 s burst onsets and check the
residuals, which is pipeline step 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LinearMap",
    "AlignmentReport",
    "overlap_window",
    "trim_to_window",
    "trim_pair_to_overlap",
    "fit_linear_map",
    "apply_linear_map",
    "validate_alignment",
]


@dataclass(frozen=True)
class LinearMap:
    """``reference_time = slope * other_time + intercept``."""

    slope: float
    intercept: float
    n_points: int
    #: Residuals of the fit itself, in seconds.
    residuals_s: np.ndarray

    @property
    def drift_ppm(self) -> float:
        """Clock-rate mismatch in parts per million.

        Two free-running crystals typically differ by tens of ppm; that is normal
        and is exactly what the fine alignment corrects.
        """
        return (self.slope - 1.0) * 1e6

    def apply(self, times_s: np.ndarray) -> np.ndarray:
        return apply_linear_map(times_s, self)


@dataclass(frozen=True)
class AlignmentReport:
    """Step 9: does the alignment actually hold on the 14 s bursts?"""

    n_checked: int
    max_residual_s: float
    median_residual_s: float
    tolerance_s: float
    passed: bool
    #: Residuals over time, for plotting.
    times_s: np.ndarray
    residuals_s: np.ndarray

    def summary(self) -> str:
        if self.n_checked == 0:
            return "alignment NOT validated: no burst onsets matched"
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"alignment {verdict}: {self.n_checked} bursts checked, "
            f"max |residual| {self.max_residual_s * 1e3:.3f} ms, "
            f"median {self.median_residual_s * 1e3:.3f} ms "
            f"(tolerance {self.tolerance_s * 1e3:.3f} ms)"
        )


def overlap_window(
    reference_s: np.ndarray, other_s: np.ndarray, offset_s: float = 0.0
) -> tuple[float, float]:
    """Common time span of two pulse trains, in the reference timebase.

    ``offset_s`` is added to ``other`` first (the coarse offset from the burst
    match), so both are compared on the same clock.
    """
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    other = np.asarray(other_s, dtype=np.float64).reshape(-1) + offset_s
    if reference.size == 0 or other.size == 0:
        raise ValueError("cannot compute an overlap window from an empty pulse train")
    return (float(max(reference[0], other[0])), float(min(reference[-1], other[-1])))


def trim_to_window(times_s: np.ndarray, start_s: float, stop_s: float) -> np.ndarray:
    """Keep only the times inside ``[start_s, stop_s]``."""
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    return times[(times >= start_s) & (times <= stop_s)]


def trim_pair_to_overlap(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    offset_s: float = 0.0,
    margin_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim both trains to their common window.

    Required before TPrime: it aligns edge *sequences*, so leading or trailing
    edges present in only one recording would shift the correspondence by whole
    cycles. ``margin_s`` shrinks the window slightly at both ends, which avoids
    keeping a boundary edge in one train but not the other.
    """
    start, stop = overlap_window(reference_s, other_s, offset_s)
    start += margin_s
    stop -= margin_s
    reference = trim_to_window(reference_s, start, stop)
    other = trim_to_window(np.asarray(other_s, dtype=np.float64) + offset_s, start, stop) - offset_s
    return reference, other


def fit_linear_map(other_s: np.ndarray, reference_s: np.ndarray) -> LinearMap:
    """Least-squares fit taking ``other`` times onto ``reference`` times.

    The two arrays must already be matched pairwise (same length, same pulses).
    """
    other = np.asarray(other_s, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    if other.size != reference.size:
        raise ValueError(
            f"matched pulse trains must be the same length, got {other.size} and {reference.size}"
        )
    if other.size < 2:
        raise ValueError("need at least two matched pulses to fit a time map")

    slope, intercept = np.polyfit(other, reference, 1)
    residuals = reference - (slope * other + intercept)
    return LinearMap(
        slope=float(slope),
        intercept=float(intercept),
        n_points=int(other.size),
        residuals_s=residuals,
    )


def apply_linear_map(times_s: np.ndarray, mapping: LinearMap) -> np.ndarray:
    """Map times onto the reference timebase."""
    return mapping.slope * np.asarray(times_s, dtype=np.float64) + mapping.intercept


def validate_alignment(
    mapped_s: np.ndarray,
    reference_s: np.ndarray,
    tolerance_s: float = 1e-3,
    match_tolerance_s: float = 0.05,
) -> AlignmentReport:
    """Step 9: compare already-mapped burst onsets against the reference ones.

    ``match_tolerance_s`` is the window for deciding which onsets correspond; it
    is deliberately far looser than ``tolerance_s``, which is the accuracy the
    alignment must then achieve. Using one tight window for both would silently
    discard the very mismatches this check exists to catch.
    """
    from .burst import match_times  # local import keeps the two modules independent

    mapped = np.asarray(mapped_s, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)

    if mapped.size == 0 or reference.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return AlignmentReport(0, float("nan"), float("nan"), tolerance_s, False, empty, empty)

    ref_idx, other_idx = match_times(reference, mapped, 0.0, match_tolerance_s)
    if ref_idx.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return AlignmentReport(0, float("nan"), float("nan"), tolerance_s, False, empty, empty)

    matched_ref = reference[ref_idx]
    residuals = matched_ref - mapped[other_idx]
    max_res = float(np.abs(residuals).max())

    return AlignmentReport(
        n_checked=int(ref_idx.size),
        max_residual_s=max_res,
        median_residual_s=float(np.median(np.abs(residuals))),
        tolerance_s=tolerance_s,
        passed=bool(max_res <= tolerance_s),
        times_s=matched_ref,
        residuals_s=residuals,
    )
