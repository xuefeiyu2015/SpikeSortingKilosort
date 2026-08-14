"""14 s coded burst: coarse alignment between the two systems (step 8.1).

The burst is a dense, uniquely coded pattern that Blackrock DO1 emits every 14 s
while it is recording, captured by SpikeGLX on OneBox XA1. Because the code does
not repeat within an hour, matching bursts fixes *which* 1 Hz cycle corresponds to
which -- the ambiguity a 1 Hz square wave cannot resolve on its own.

Pure computation: arrays in, arrays out.

The strategy is deliberately not "assume the first burst in each file is the same
burst". The two systems start at different times, so one file routinely misses
bursts the other has. Instead the offset is found as the densest cluster of
pairwise onset differences, which tolerates missing bursts at either end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BurstMatch",
    "group_burst_onsets",
    "burst_patterns",
    "estimate_offset",
    "candidate_offsets",
    "score_offset",
    "match_times",
    "match_bursts",
]


@dataclass(frozen=True)
class BurstMatch:
    """Result of matching two burst-onset sequences."""

    #: Seconds to add to *other* times to bring them onto the reference timebase.
    offset_s: float
    #: Onsets that matched, in each timebase.
    reference_matched: np.ndarray
    other_matched: np.ndarray
    n_reference: int
    n_other: int
    #: Residuals after applying the offset, in seconds.
    residuals_s: np.ndarray
    #: How many individual pulses (not just onsets) the winning offset matched,
    #: and how many the runner-up did. A winner that barely beats the runner-up
    #: means the burst code did not actually disambiguate.
    pulse_score: int = 0
    runner_up_score: int = 0

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

    This is the actual "code" carried by each burst. Useful for verifying a match
    found by :func:`estimate_offset` when extra confidence is wanted.
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
       scores candidate offsets against the intra-burst code and so picks the
       right cycle rather than the most overlapping one.
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


def score_offset(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    offset_s: float,
    tolerance_s: float = 5e-3,
) -> int:
    """How many pulses line up under a given offset.

    Run on the *full* pulse trains this reads the burst code: a wrong-by-one-cycle
    offset still lines the onsets up, but the intra-burst patterns then disagree
    and the score collapses.
    """
    ref_idx, _ = match_times(reference_s, other_s, offset_s, tolerance_s)
    return int(ref_idx.size)


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


def match_bursts(
    reference_s: np.ndarray,
    other_s: np.ndarray,
    min_gap_s: float = 7.0,
    tolerance_s: float = 5e-3,
    max_candidates: int = 16,
    already_onsets: bool = False,
) -> BurstMatch:
    """Full coarse alignment, using the burst code to pick the right cycle.

    Pass the **full pulse trains**, not pre-grouped onsets. Onsets alone are
    periodic and cannot distinguish burst *k* from burst *k+1*; the intra-burst
    pattern can, so candidate offsets are proposed from the onsets and then scored
    against every pulse.

    ``reference_s`` is normally the Blackrock burst channel, since Blackrock is
    the reference timebase for the whole pipeline.
    """
    ref_pulses = np.asarray(reference_s, dtype=np.float64).reshape(-1)
    other_pulses = np.asarray(other_s, dtype=np.float64).reshape(-1)
    if ref_pulses.size == 0 or other_pulses.size == 0:
        raise ValueError("cannot match bursts from an empty pulse train")

    ref_onsets = ref_pulses if already_onsets else group_burst_onsets(ref_pulses, min_gap_s)
    other_onsets = other_pulses if already_onsets else group_burst_onsets(other_pulses, min_gap_s)

    candidates = candidate_offsets(ref_onsets, other_onsets, tolerance_s, max_candidates)
    if candidates.size == 0:
        raise ValueError("cannot estimate an offset from an empty burst sequence")

    scores = np.array(
        [score_offset(ref_pulses, other_pulses, c, tolerance_s) for c in candidates]
    )
    order = np.argsort(scores)[::-1]
    offset = float(candidates[order[0]])
    best_score = int(scores[order[0]])
    runner_up = int(scores[order[1]]) if scores.size > 1 else 0

    # Refine using every matched pulse, not just the onsets.
    pulse_ref_idx, pulse_other_idx = match_times(
        ref_pulses, other_pulses, offset, tolerance_s
    )
    if pulse_ref_idx.size:
        offset += float(
            np.mean(ref_pulses[pulse_ref_idx] - (other_pulses[pulse_other_idx] + offset))
        )

    ref_idx, other_idx = match_times(ref_onsets, other_onsets, offset, tolerance_s)
    matched_ref = ref_onsets[ref_idx]
    matched_other = other_onsets[other_idx]
    residuals = matched_ref - (matched_other + offset)

    return BurstMatch(
        offset_s=offset,
        reference_matched=matched_ref,
        other_matched=matched_other,
        n_reference=int(ref_onsets.size),
        n_other=int(other_onsets.size),
        residuals_s=residuals,
        pulse_score=best_score,
        runner_up_score=runner_up,
    )
