"""Trimming, mapping and validation for cross-system alignment (steps 8.2 and 9).

Pure computation. Blackrock is the reference timebase throughout: Neuropixels
times are mapped *onto* Blackrock time, never the other way round.

Two mappings live here:

* :func:`fit_linear_map` -- a least-squares fit of the matched 1 Hz edges. Enough
  on its own for most purposes, and the fallback when TPrime is unavailable
  (the HPC, macOS).
* TPrime -- the authoritative fine alignment, run by :mod:`spikesorting._sync.tprime`.

:func:`fit_same_run_map` is a third, inside one SpikeGLX run: it takes the OneBox
clock onto the probe's, so the burst can be compared on the probe clock.

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
    "trim_pair_under_map",
    "fit_linear_map",
    "fit_same_run_map",
    "apply_linear_map",
    "validate_alignment",
    "interval_offset_map",
    "MIN_BURST_COVERAGE",
    "MIN_BURSTS_MATCHED",
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
    #: Bursts both systems hold inside their common window -- what *should* match.
    n_expected: int = 0
    #: Time from the first matched burst to the last.
    span_s: float = 0.0
    #: Length of the stretch both systems recorded, when it was given.
    shared_s: float | None = None
    #: Why it failed, when it did; empty on a pass.
    fail_reason: str = ""

    @property
    def coverage(self) -> float:
        """Matched bursts as a fraction of those both systems hold."""
        return self.n_checked / self.n_expected if self.n_expected else 0.0

    def summary(self) -> str:
        if self.n_checked == 0:
            return "alignment FAIL, NOT validated: no burst onsets matched"
        verdict = "PASS" if self.passed else "FAIL"
        text = (
            f"alignment {verdict}: {self.n_checked} of {self.n_expected} bursts matched "
            f"({100 * self.coverage:.0f}%, over {self.span_s / 60:.1f} min"
            + (f" of {self.shared_s / 60:.1f}" if self.shared_s else "") + "), "
            f"max |residual| {self.max_residual_s * 1e3:.3f} ms, "
            f"median {self.median_residual_s * 1e3:.3f} ms "
            f"(tolerance {self.tolerance_s * 1e3:.3f} ms)"
        )
        return f"{text} -- {self.fail_reason}" if self.fail_reason else text


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


def trim_pair_under_map(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    other_mapped_s: np.ndarray,
    margin_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim both trains to their common window, judged through a coarse map.

    ``other_mapped_s`` is ``other_s`` already on the reference clock (the burst
    map, drift included). Returns ``(reference, other, other_mapped)`` trimmed,
    with ``other`` still on its own clock -- that is what the fine fit needs.
    """
    other = np.asarray(other_s, dtype=np.float64).reshape(-1)
    mapped = np.asarray(other_mapped_s, dtype=np.float64).reshape(-1)
    if other.size != mapped.size:
        raise ValueError(f"other has {other.size} times but its mapping has {mapped.size}")
    start, stop = overlap_window(reference_s, mapped)
    start += margin_s
    stop -= margin_s
    keep = (mapped >= start) & (mapped <= stop)
    return trim_to_window(reference_s, start, stop), other[keep], mapped[keep]


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


def fit_same_run_map(
    other_s: np.ndarray, reference_s: np.ndarray, period_s: float = 1.0
) -> LinearMap:
    """Fit one SpikeGLX stream's 1 Hz edges onto another's from the same run.

    The streams of one run start together, so the edges pair with no offset; only
    the rates differ. A gap wider than a quarter period is a missing edge, not a
    match, which keeps a dropped pulse from pairing with its neighbour.
    """
    from .burst import match_times  # local import keeps the two modules independent

    reference, other = trim_pair_to_overlap(reference_s, other_s, 0.0, margin_s=period_s / 4)
    ref_idx, other_idx = match_times(reference, other, 0.0, tolerance_s=period_s / 4)
    if ref_idx.size < 2:
        raise ValueError(
            f"only {ref_idx.size} 1 Hz edges pair between the two streams; "
            "cannot fit a map between them"
        )
    return fit_linear_map(other[other_idx], reference[ref_idx])


def compose_linear_maps(outer: LinearMap, inner: LinearMap) -> LinearMap:
    """``outer(inner(t))`` as one map: inner's clock straight onto outer's reference.

    Two lines compose to a line, so the OneBox -> probe fit followed by the
    probe -> Blackrock fit is ``blackrock = (s_o * s_i) * t + (s_o * i_i + i_o)``.
    There is no fit of its own, so ``residuals_s`` is empty; ``n_points`` is the
    smaller of the two, the fewer edges either step rests on.
    """
    return LinearMap(
        slope=float(outer.slope * inner.slope),
        intercept=float(outer.slope * inner.intercept + outer.intercept),
        n_points=int(min(outer.n_points, inner.n_points)),
        residuals_s=np.empty(0),
    )


def apply_linear_map(times_s: np.ndarray, mapping: LinearMap) -> np.ndarray:
    """Map times onto the reference timebase."""
    return mapping.slope * np.asarray(times_s, dtype=np.float64) + mapping.intercept


#: A pass needs at least this fraction of the bursts both systems hold to match.
#: A burst the map puts more than ``match_tolerance_s`` off has no partner and
#: so no residual -- without a coverage floor, a map wrong over half the session
#: would pass on the half it got right.
MIN_BURST_COVERAGE = 0.9

#: ...and at least this many, so a check over two bursts is not a pass.
MIN_BURSTS_MATCHED = 3


def validate_alignment(
    mapped_s: np.ndarray,
    reference_s: np.ndarray,
    tolerance_s: float = 1e-3,
    match_tolerance_s: float = 0.05,
    min_coverage: float = MIN_BURST_COVERAGE,
    min_matched: int = MIN_BURSTS_MATCHED,
    shared_interval_s: tuple[float, float] | None = None,
) -> AlignmentReport:
    """Step 9: compare already-mapped burst onsets against the reference ones.

    ``match_tolerance_s`` is the window for deciding which onsets correspond; it
    is deliberately far looser than ``tolerance_s``, which is the accuracy the
    alignment must then achieve. Using one tight window for both would silently
    discard the very mismatches this check exists to catch.

    Passing takes three things: every matched residual within ``tolerance_s``,
    at least ``min_matched`` matches, and at least ``min_coverage`` of the
    bursts both systems hold inside their common window matched. The last is
    what catches a map that is wrong over part of the session: those bursts
    land outside ``match_tolerance_s``, find no partner, and would otherwise
    simply be left out.

    ``shared_interval_s`` is the stretch both systems recorded, on the
    reference clock (the matched 1 Hz edges). Given it, a burst *either* system
    holds inside it is expected -- so a burst channel that stops partway on one
    side is a shortfall, not a short session -- and only bursts inside it are
    matched and judged, so matches outside cannot make up for misses inside.
    Without it the window is where the
    detected bursts overlap, and the expected count is the smaller side's.
    """
    from .burst import match_times  # local import keeps the two modules independent

    mapped = np.asarray(mapped_s, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)

    if shared_interval_s is not None:
        # One window for everything: matching, residuals and coverage. Matches
        # outside it would otherwise count toward a coverage measured inside
        # it, and could hide every burst inside being wrong.
        start = float(shared_interval_s[0]) - match_tolerance_s
        stop = float(shared_interval_s[1]) + match_tolerance_s
        reference = reference[(reference >= start) & (reference <= stop)]
        mapped = mapped[(mapped >= start) & (mapped <= stop)]

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

    if shared_interval_s is not None:
        combine = max                      # start/stop: the window set above
    else:
        start = max(reference.min(), mapped.min()) - match_tolerance_s
        stop = min(reference.max(), mapped.max()) + match_tolerance_s
        combine = min
    n_expected = int(combine(
        np.count_nonzero((reference >= start) & (reference <= stop)),
        np.count_nonzero((mapped >= start) & (mapped <= stop)),
    ))
    n_checked = int(ref_idx.size)
    n_expected = max(n_expected, n_checked)
    coverage = n_checked / n_expected

    reasons = []
    if max_res > tolerance_s:
        reasons.append(f"max residual exceeds {tolerance_s * 1e3:g} ms")
    if n_checked < min_matched:
        reasons.append(f"fewer than {min_matched} bursts matched")
    if coverage < min_coverage:
        reasons.append(
            f"only {100 * coverage:.0f}% of the bursts matched (need "
            f"{100 * min_coverage:.0f}%); the rest are more than "
            f"{match_tolerance_s * 1e3:g} ms off"
        )

    return AlignmentReport(
        n_checked=n_checked,
        max_residual_s=max_res,
        median_residual_s=float(np.median(np.abs(residuals))),
        tolerance_s=tolerance_s,
        passed=not reasons,
        times_s=matched_ref,
        residuals_s=residuals,
        n_expected=n_expected,
        span_s=float(matched_ref.max() - matched_ref.min()),
        shared_s=(float(shared_interval_s[1] - shared_interval_s[0])
                  if shared_interval_s is not None else None),
        fail_reason="; ".join(reasons),
    )


def interval_offset_map(
    times_s: np.ndarray, from_edges_s: np.ndarray, to_edges_s: np.ndarray
) -> np.ndarray:
    """TPrime's own model: ``t + (to_k - from_k)``, ``k`` the last *to* edge <= ``t``.

    Measured on a real session (TPrime's output fits it to 0.1 us): one constant
    offset per sync interval, the interval chosen by comparing ``t`` against the
    reference edges. Before the first edge the first interval's offset applies.
    The two edge lists must be matched pairs.
    """
    t = np.asarray(times_s, dtype=np.float64)
    frm = np.asarray(from_edges_s, dtype=np.float64).reshape(-1)
    to = np.asarray(to_edges_s, dtype=np.float64).reshape(-1)
    if frm.size != to.size or not to.size:
        raise ValueError(
            f"interval map needs matched edge pairs, got {frm.size} and {to.size}"
        )
    k = np.clip(np.searchsorted(to, t, side="right") - 1, 0, to.size - 1)
    return t + (to[k] - frm[k])
