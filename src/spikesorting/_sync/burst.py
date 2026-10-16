"""14 s coded burst: coarse alignment between the two systems (step 8.1).

The burst is a dense, uniquely coded pattern that Blackrock DO1 emits every 14 s
while it is recording, captured by SpikeGLX on OneBox XA1. Because the code does
not repeat within an hour, matching bursts fixes *which* 1 Hz cycle corresponds to
which -- the ambiguity a 1 Hz square wave cannot resolve on its own.

Pure computation: arrays in, arrays out.

The strategy is deliberately not "assume the first burst in each file is the same
burst". The two systems start at different times, so one file routinely misses
bursts the other has.

**Bursts are paired by their code, never by one global offset.** The two clocks
drift apart -- 20 ppm is 216 ms over three hours -- so no single offset lines
every burst up to a few ms, and scoring offsets that way lets a wrong cycle win
on chance coincidences once the true one stops fitting. A burst's code is its
intra-burst intervals, which span ~200 ms: drift over that is nanoseconds. So
each burst is paired with the bursts carrying the same code, and the pairs that
agree on one offset (to within a few seconds -- drift, not cycles) are kept. That
also resolves the code's hourly repeat, whose false pairs sit 3600 s away. A line
through the kept onsets is the coarse map, drift included.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BurstMatch",
    "group_burst_onsets",
    "burst_patterns",
    "estimate_offset",
    "candidate_offsets",
    "match_times",
    "match_burst_codes",
    "match_bursts",
]


@dataclass(frozen=True)
class BurstMatch:
    """Result of matching two burst-onset sequences.

    ``reference = slope * other + intercept`` is the coarse map, drift included;
    use :meth:`apply` to put other times on the reference clock. ``offset_s`` is
    its constant summary, for logs -- over hours it is wrong by the drift.
    """

    #: Median of ``reference - other`` over the matched onsets.
    offset_s: float
    #: Onsets that matched, in each timebase.
    reference_matched: np.ndarray
    other_matched: np.ndarray
    n_reference: int
    n_other: int
    #: Residuals after applying the offset, in seconds.
    residuals_s: np.ndarray
    slope: float = 1.0
    intercept: float = 0.0
    #: Code-matched bursts agreeing with the winning offset, and with the next
    #: most popular one. A runner-up near the winner means the code did not
    #: actually disambiguate.
    code_matches: int = 0
    runner_up_matches: int = 0

    def apply(self, times_s: np.ndarray) -> np.ndarray:
        """Other times onto the reference clock, through the coarse map."""
        return self.slope * np.asarray(times_s, dtype=np.float64) + self.intercept

    @property
    def n_matched(self) -> int:
        return int(self.reference_matched.size)

    @property
    def max_residual_s(self) -> float:
        return float(np.abs(self.residuals_s).max()) if self.residuals_s.size else float("nan")

    @property
    def median_residual_s(self) -> float:
        return float(np.median(np.abs(self.residuals_s))) if self.residuals_s.size else float("nan")


def group_burst_onsets(times_s: np.ndarray, min_gap_s: float = 7.0) -> np.ndarray:
    """Collapse the pulses of each coded burst to a single onset time.

    A new burst starts wherever the gap since the previous pulse exceeds
    ``min_gap_s``. The default of half the 14 s interval separates bursts from the
    dense pulses inside one.
    """
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return np.empty(0, dtype=np.float64)
    starts = np.concatenate(([True], np.diff(times) > min_gap_s))
    return times[starts]


def burst_patterns(times_s: np.ndarray, min_gap_s: float = 7.0) -> list[np.ndarray]:
    """Intra-burst inter-pulse intervals, one array per burst.

    This is the actual "code" carried by each burst, and what
    :func:`match_burst_codes` compares.
    """
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(times) > min_gap_s) + 1
    return [np.diff(group) for group in np.split(times, boundaries)]


def estimate_offset(
    reference_s: np.ndarray, other_s: np.ndarray, tolerance_s: float = 5e-3
) -> float:
    """Constant offset that best aligns ``other`` onto ``reference``.

    Returns the offset to *add* to ``other``, found as the densest cluster of
    pairwise differences -- robust to pulses present in only one recording.

    .. warning::

       On a perfectly periodic train this is ambiguous by construction: every
       whole-period shift is equally valid, and the densest cluster is simply the
       one giving maximum overlap. That is why the 14 s signal is *coded*. Pass
       the full pulse trains (not just burst onsets) to :func:`match_bursts`, which
       pairs bursts by their intra-burst code and so picks the right cycle rather
       than the most overlapping one.
    """
    candidates = candidate_offsets(reference_s, other_s, tolerance_s, max_candidates=1)
    if candidates.size == 0:
        raise ValueError("cannot estimate an offset from an empty burst sequence")
    return float(candidates[0])


def candidate_offsets(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    tolerance_s: float = 5e-3,
    max_candidates: int = 16,
) -> np.ndarray:
    """Plausible offsets, densest cluster of pairwise differences first.

    For a periodic burst train these come out roughly one burst interval apart --
    they are precisely the whole-cycle ambiguities that the intra-burst code has
    to resolve.
    """
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    other = np.asarray(other_s, dtype=np.float64).reshape(-1)
    if reference.size == 0 or other.size == 0:
        return np.empty(0, dtype=np.float64)

    diffs = np.sort(np.subtract.outer(reference, other).ravel())
    alive = np.ones(diffs.size, dtype=bool)
    window = 2.0 * tolerance_s
    found: list[float] = []

    for _ in range(max_candidates):
        live_idx = np.flatnonzero(alive)
        if live_idx.size == 0:
            break
        live = diffs[live_idx]
        right = np.searchsorted(live, live + window, side="right")
        counts = right - np.arange(live.size)
        best = int(np.argmax(counts))
        center = float(live[best : right[best]].mean())
        found.append(center)
        # Suppress this cluster so the next iteration finds a different one.
        alive[live_idx[np.abs(live - center) <= window]] = False

    return np.asarray(found, dtype=np.float64)


def match_times(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    offset_s: float,
    tolerance_s: float = 5e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair up two time lists after applying ``offset_s`` to ``other``.

    Returns ``(reference_indices, other_indices)`` for pairs that are mutual
    nearest neighbours within ``tolerance_s`` -- so one reference pulse can never
    claim two others, or vice versa.
    """
    reference = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    other = np.asarray(other_s, dtype=np.float64).reshape(-1) + offset_s
    if reference.size == 0 or other.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    forward = _nearest_index(other, reference)  # for each reference, nearest other
    backward = _nearest_index(reference, other)  # for each other, nearest reference

    ref_idx = np.arange(reference.size)
    mutual = backward[forward] == ref_idx
    close = np.abs(reference - other[forward]) <= tolerance_s
    keep = mutual & close
    return ref_idx[keep], forward[keep]


def _nearest_index(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Index of the nearest ``haystack`` element for each ``needle``. Both sorted."""
    if haystack.size == 1:
        return np.zeros(needles.size, dtype=np.int64)
    pos = np.searchsorted(haystack, needles)
    pos = np.clip(pos, 1, haystack.size - 1)
    left, right = haystack[pos - 1], haystack[pos]
    take_left = np.abs(needles - left) <= np.abs(needles - right)
    return np.where(take_left, pos - 1, pos)


def match_burst_codes(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    min_gap_s: float = 7.0,
    code_tolerance_s: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every pair of bursts, one per recording, carrying the same code.

    Returns ``(reference_onsets, other_onsets, reference_idx, other_idx)``. Codes
    match when they have the same number of pulses and every interval agrees to
    ``code_tolerance_s``. A burst with fewer than two intervals carries no code --
    the partial burst at either end of a recording, typically -- and pairs with
    nothing. No offset is involved, so drift cannot affect it.
    """
    ref_onsets, ref_codes = _onsets_and_codes(reference_s, min_gap_s)
    other_onsets, other_codes = _onsets_and_codes(other_s, min_gap_s)

    ref_pairs: list[np.ndarray] = []
    other_pairs: list[np.ndarray] = []
    ref_lengths = np.array([c.size for c in ref_codes], dtype=np.int64)
    other_lengths = np.array([c.size for c in other_codes], dtype=np.int64)
    for n in np.intersect1d(ref_lengths, other_lengths):
        if n < 2:
            continue
        ref_i = np.flatnonzero(ref_lengths == n)
        other_i = np.flatnonzero(other_lengths == n)
        ref_block = np.stack([ref_codes[i] for i in ref_i])
        other_block = np.stack([other_codes[i] for i in other_i])
        distance = np.abs(ref_block[:, None, :] - other_block[None, :, :]).max(axis=2)
        a, b = np.nonzero(distance <= code_tolerance_s)
        ref_pairs.append(ref_i[a])
        other_pairs.append(other_i[b])

    empty = np.empty(0, dtype=np.int64)
    return (
        ref_onsets,
        other_onsets,
        np.concatenate(ref_pairs) if ref_pairs else empty,
        np.concatenate(other_pairs) if other_pairs else empty,
    )


def _onsets_and_codes(
    times_s: np.ndarray, min_gap_s: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    return group_burst_onsets(times_s, min_gap_s), burst_patterns(times_s, min_gap_s)


def _densest_window(values: np.ndarray, width: float) -> tuple[np.ndarray, int]:
    """Indices of ``values`` inside the most populated window of ``width``."""
    order = np.argsort(values)
    ordered = values[order]
    right = np.searchsorted(ordered, ordered + width, side="right")
    counts = right - np.arange(ordered.size)
    best = int(np.argmax(counts))
    return order[best : right[best]], int(counts[best])


def match_bursts(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    min_gap_s: float = 7.0,
    code_tolerance_s: float = 1e-3,
) -> BurstMatch:
    """Full coarse alignment: pair bursts by code, then fit a line through them.

    Pass the **full pulse trains**, not pre-grouped onsets: onsets are periodic
    and carry no code. Code-matched pairs vote with their onset difference, and
    the most popular offset -- within ``min_gap_s``, far wider than any drift and
    far narrower than the hourly repeat -- wins. The winners are fitted with a
    line, so the map holds across the whole recording rather than at its middle.

    ``reference_s`` is normally the Blackrock burst channel, since Blackrock is
    the reference timebase for the whole pipeline.
    """
    ref_pulses = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    other_pulses = np.asarray(other_s, dtype=np.float64).reshape(-1)
    if ref_pulses.size == 0 or other_pulses.size == 0:
        raise ValueError("cannot match bursts from an empty pulse train")

    ref_onsets, other_onsets, ref_idx, other_idx = match_burst_codes(
        ref_pulses, other_pulses, min_gap_s, code_tolerance_s
    )
    if ref_idx.size == 0:
        raise ValueError(
            f"no burst in one recording carries the code of any burst in the other "
            f"({ref_onsets.size} vs {other_onsets.size} bursts, intervals compared to "
            f"{code_tolerance_s * 1e3:g} ms). Check both burst thresholds: a missed or "
            "extra pulse changes a burst's code."
        )

    differences = ref_onsets[ref_idx] - other_onsets[other_idx]
    winners, n_win = _densest_window(differences, min_gap_s)
    centre = float(np.median(differences[winners]))
    losers = np.flatnonzero(np.abs(differences - centre) > min_gap_s)
    runner_up = _densest_window(differences[losers], min_gap_s)[1] if losers.size else 0

    # One partner per burst: a code repeated within the window would be a burst
    # matched twice, so keep the pair nearest the consensus.
    winners = winners[np.argsort(np.abs(differences[winners] - centre), kind="stable")]
    _, first_ref = np.unique(ref_idx[winners], return_index=True)
    winners = winners[first_ref]
    _, first_other = np.unique(other_idx[winners], return_index=True)
    winners = np.sort(winners[first_other])

    matched_ref = ref_onsets[ref_idx[winners]]
    matched_other = other_onsets[other_idx[winners]]
    order = np.argsort(matched_other)
    matched_ref, matched_other = matched_ref[order], matched_other[order]

    slope, intercept = 1.0, float(np.median(matched_ref - matched_other))
    if matched_other.size >= 2:
        # Centred first: Blackrock onsets are ~1.5e9 s PTP times, and fitting
        # those raw spends most of float64's digits on a constant.
        x0 = float(matched_other.mean())
        slope, b = np.polyfit(matched_other - x0, matched_ref, 1)
        slope, intercept = float(slope), float(b - slope * x0)

    match = BurstMatch(
        offset_s=float(np.median(matched_ref - matched_other)),
        reference_matched=matched_ref,
        other_matched=matched_other,
        n_reference=int(ref_onsets.size),
        n_other=int(other_onsets.size),
        residuals_s=np.empty(0),
        slope=slope,
        intercept=intercept,
        code_matches=int(n_win),
        runner_up_matches=int(runner_up),
    )
    residuals = matched_ref - match.apply(matched_other)
    return dataclasses.replace(match, residuals_s=residuals)
