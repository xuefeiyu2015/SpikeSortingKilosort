"""Which 1 Hz cycle pairs with which, when there is no 14 s burst to say.

Pure: arrays in, a vote out. The pipeline reads the signals.

The 1 Hz square wave is periodic, so fitting it alone gets the *drift* right
and leaves the intercept ambiguous by whole periods: pair every edge with the
one a cycle later and the line fits just as well. The burst exists to resolve
that. When it is missing, anything else both systems recorded resolves it as
well, provided it is not periodic: the photodiode (task-driven flashes) or the
eye. Map the OneBox copy onto Blackrock time through the provisional 1 Hz fit,
and the only error left is a whole number of periods -- so try each, and keep
the one that lines the two copies up.

That is the *only* thing these signals decide. The fine map is still the 1 Hz
fit, redone on the corrected pairing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CycleVote", "cycle_shift_from_events", "cycle_shift_from_lags"]


@dataclass(frozen=True)
class CycleVote:
    """The whole-period correction to add to the provisional map, and its support."""

    #: Seconds to add to the provisional OneBox/probe -> Blackrock times.
    shift_s: float
    #: Whole periods that is.
    n_periods: int
    #: Events (or windows) agreeing with it, and with the next best.
    n_support: int
    n_runner_up: int
    #: Events (or windows) that took part.
    n_total: int

    def summary(self) -> str:
        return (
            f"shift {self.n_periods:+d} cycle(s) ({self.shift_s:+.3f} s), "
            f"{self.n_support} of {self.n_total} agree, runner-up {self.n_runner_up}"
        )


def _decide(
    counts: dict[int, int], period_s: float, n_total: int, min_support: int, margin: float
) -> CycleVote | None:
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    best_k, best = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    if best < min_support or best < margin * runner:
        return None
    return CycleVote(best_k * period_s, best_k, best, runner, n_total)


def cycle_shift_from_events(
    blackrock_s: np.ndarray,
    mapped_s: np.ndarray,
    period_s: float = 1.0,
    search_s: float = 60.0,
    tolerance_s: float = 0.03,
    min_support: int = 10,
    margin: float = 3.0,
) -> CycleVote | None:
    """Whole periods that best line up two copies of one irregular event train.

    ``mapped_s`` are the OneBox events through the provisional map. For each
    whole-period shift within ``search_s``, count the mapped events that land
    within ``tolerance_s`` of a Blackrock event. ``tolerance_s`` covers
    Blackrock's filter delay on its 1 kHz stream (a few ms) and a 1 kHz sample.

    ``None`` unless the winner has ``min_support`` events and ``margin`` times
    the runner-up -- a periodic or sparse train must not pick a cycle by chance.
    """
    br = np.sort(np.asarray(blackrock_s, dtype=np.float64).ravel())
    obx = np.asarray(mapped_s, dtype=np.float64).ravel()
    if br.size == 0 or obx.size == 0:
        return None
    n_max = int(np.floor(search_s / period_s))
    counts: dict[int, int] = {}
    for k in range(-n_max, n_max + 1):
        counts[k] = int(np.count_nonzero(_nearest_gap(br, obx + k * period_s) <= tolerance_s))
    return _decide(counts, period_s, int(obx.size), min_support, margin)


def _nearest_gap(sorted_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Distance from each ``x`` to the nearest value of ``sorted_ref``."""
    idx = np.searchsorted(sorted_ref, x)
    lo = np.clip(idx - 1, 0, sorted_ref.size - 1)
    hi = np.clip(idx, 0, sorted_ref.size - 1)
    return np.minimum(np.abs(sorted_ref[lo] - x), np.abs(sorted_ref[hi] - x))


def cycle_shift_from_lags(
    lags_s: np.ndarray,
    period_s: float = 1.0,
    max_off_s: float = 0.1,
    min_support: int = 3,
    margin: float = 2.0,
) -> CycleVote | None:
    """Whole periods from windowed cross-correlation lags of a continuous signal.

    ``lags_s`` are how much later the mapped OneBox copy is than Blackrock's
    (``compare.xcorr_lag``'s sign), one per window. Each window votes for the
    nearest whole period; a lag further than ``max_off_s`` from any is a
    window that locked onto nothing and does not vote. The correction is minus
    the lag: a OneBox copy that is late is moved earlier.
    """
    lags = np.asarray(lags_s, dtype=np.float64).ravel()
    lags = lags[np.isfinite(lags)]
    counts: dict[int, int] = {}
    for lag in lags:
        k = int(np.round(-lag / period_s))
        if abs(-lag - k * period_s) <= max_off_s:
            counts[k] = counts.get(k, 0) + 1
    return _decide(counts, period_s, int(lags.size), min_support, margin)
