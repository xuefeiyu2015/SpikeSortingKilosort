"""How well the OneBox and Blackrock agree, before and after synchronization.

Pure: arrays in, arrays out. The readers that produce the arrays live in
``_io/analog.py`` and the figures in ``_plots/sync.py``.

**"Before" is what you would have without the fitted map**: OneBox seconds
(``sample / obSampRate``) plus *one* constant offset, chosen so the first few
events line up. That is a start-aligned recording, and what remains is the
clock drift between the two crystals -- exactly what ``onebox_to_blackrock``
exists to remove. **"After"** is that map. Both are measured on the same event
pairs, so the two numbers differ only by the synchronization.

Events are paired under the fitted map, never under the offset alone: drift of
tens of ppm moves a late event by hundreds of ms, past the gap between two
burst pulses, and pairing under the offset would quietly pair the wrong ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .burst import match_times

__all__ = [
    "SyncComparison",
    "compare_event_times",
    "summarize_asynchrony",
    "largest_asynchrony",
    "onebox_sample_range",
    "blackrock_sample_range",
    "normalize_trace",
    "display_stride",
    "display_envelope",
    "auto_threshold",
    "offset_and_drift",
    "xcorr_lag",
]


@dataclass(frozen=True)
class SyncComparison:
    """One set of events seen by both systems, paired."""

    label: str
    #: Paired event times on the NSP clock (the reference).
    blackrock_s: np.ndarray
    #: The same events on the OneBox clock (``sample / obSampRate``).
    onebox_s: np.ndarray
    #: OneBox + one constant offset, minus Blackrock: the unsynchronized error.
    before_s: np.ndarray
    #: Mapped OneBox minus Blackrock: what synchronization leaves.
    after_s: np.ndarray
    #: The constant offset "before" uses.
    offset_s: float
    #: Events seen on each system, paired or not -- unpaired ones are listed
    #: so a detection problem on one side is visible.
    n_blackrock: int
    n_onebox: int

    @property
    def n_pairs(self) -> int:
        return int(self.blackrock_s.size)


def compare_event_times(
    blackrock_s: np.ndarray,
    onebox_s: np.ndarray,
    slope: float,
    intercept: float,
    label: str = "",
    tolerance_s: float = 5e-3,
    n_anchor: int = 10,
) -> SyncComparison:
    """Pair events across the two clocks and measure the error before and after.

    ``slope``/``intercept`` are ``onebox_to_blackrock``. ``tolerance_s`` is the
    pairing window *after* mapping, so it can be tight: keep it under half the
    shortest gap between events (burst pulses are tens of ms apart).
    ``n_anchor`` events at the start set the "before" offset.
    """
    blackrock_s = np.sort(np.asarray(blackrock_s, dtype=np.float64).ravel())
    onebox_s = np.sort(np.asarray(onebox_s, dtype=np.float64).ravel())
    mapped = slope * onebox_s + intercept

    ref_idx, other_idx = match_times(blackrock_s, mapped, 0.0, tolerance_s=tolerance_s)
    br, obx = blackrock_s[ref_idx], onebox_s[other_idx]
    if br.size == 0:
        raise ValueError(
            f"{label or 'events'}: none of {blackrock_s.size} Blackrock and "
            f"{onebox_s.size} OneBox events pair within {tolerance_s * 1e3:g} ms "
            "under the map -- check thresholds and that the map is stamped"
        )

    offset = float(np.median(br[:n_anchor] - obx[:n_anchor]))
    return SyncComparison(
        label=label,
        blackrock_s=br,
        onebox_s=obx,
        before_s=(obx + offset) - br,
        after_s=(slope * obx + intercept) - br,
        offset_s=offset,
        n_blackrock=int(blackrock_s.size),
        n_onebox=int(onebox_s.size),
    )


def summarize_asynchrony(comparison: SyncComparison) -> dict[str, float]:
    """Median, 99th percentile and maximum of |error|, before and after, in ms."""
    out: dict[str, float] = {"n_pairs": float(comparison.n_pairs)}
    for which in ("before", "after"):
        err = np.abs(getattr(comparison, f"{which}_s")) * 1e3
        out[f"{which}_median_ms"] = float(np.median(err))
        out[f"{which}_p99_ms"] = float(np.percentile(err, 99))
        out[f"{which}_max_ms"] = float(err.max())
    out["improvement_x"] = (
        out["before_max_ms"] / out["after_max_ms"] if out["after_max_ms"] > 0 else np.inf
    )
    return out


def largest_asynchrony(comparison: SyncComparison, which: str = "after") -> int:
    """Index of the pair with the largest |error|, ``before`` or ``after`` sync."""
    if which not in ("before", "after"):
        raise ValueError(f"which must be 'before' or 'after', got {which!r}")
    return int(np.argmax(np.abs(getattr(comparison, f"{which}_s"))))


def onebox_sample_range(
    t0_s: float, t1_s: float, fs: float, slope: float, intercept: float, n_samples: int
) -> tuple[int, int]:
    """OneBox samples covering ``[t0_s, t1_s)`` on Blackrock time, clipped to the file.

    Inverts ``blackrock = slope * (k / fs) + intercept``.
    """
    k0 = int(np.floor((t0_s - intercept) / slope * fs))
    k1 = int(np.ceil((t1_s - intercept) / slope * fs))
    return max(k0, 0), min(max(k1, 0), n_samples)


def blackrock_sample_range(
    t0_s: float, t1_s: float, segments: Sequence[tuple[int, int, float, float]]
) -> tuple[int, int]:
    """Blackrock samples covering ``[t0_s, t1_s)`` on the NSP clock.

    ``segments`` are ``(first_sample, n_samples, seconds_per_sample, intercept)``
    -- an :class:`~spikesorting._io.blackrock.NspTimeMap`'s segments. The window
    is clipped to the first segment it starts in; a window across a pause would
    mix two runs of samples, so it ends where that segment does.
    """
    for first, n, slope, intercept in segments:
        end = intercept + slope * n
        if t0_s < end:
            s0 = first + int(np.floor((t0_s - intercept) / slope))
            s1 = first + int(np.ceil((t1_s - intercept) / slope))
            return max(s0, first), min(max(s1, first), first + n)
    first, n, _, _ = segments[-1]
    return first + n, first + n


def normalize_trace(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 between the 1st and 99th percentile, for overlaying two systems.

    The two systems record the same wire with different gains and units, so
    shape and timing are what can be compared, not amplitude.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = _level_range(values)
    span = hi - lo
    return (values - lo) / span if span > 0 else values - lo


def _level_range(values: np.ndarray) -> tuple[float, float]:
    """The 1st..99th percentile, or min..max when that range is empty.

    A sparse pulse train -- the burst is high ~0.3% of the time -- has both
    percentiles at the baseline, which would leave it unscaled and put a
    threshold on the floor. Min..max is what still sees the pulses.
    """
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    return float(lo), float(hi)


def display_stride(n_samples: int, max_points: int) -> int:
    """Stride that keeps a trace under ``max_points`` for drawing. Never below 1."""
    return max(1, int(np.ceil(n_samples / max(max_points, 1))))


def display_envelope(
    times: np.ndarray, values: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Min and max of each block, interleaved, so a long window draws in few points.

    A plain stride drops anything narrower than the stride -- a 10 ms burst
    pulse vanishes from a 5-minute window. The envelope keeps every excursion.
    Short windows come back unchanged.
    """
    times = np.asarray(times)
    values = np.asarray(values)
    step = display_stride(values.size, max(max_points // 2, 1))
    if step == 1:
        return times, values
    n = values.size // step * step
    blocks = values[:n].reshape(-1, step)
    lo, hi = blocks.min(axis=1), blocks.max(axis=1)
    t = times[:n:step]
    return np.repeat(t, 2), np.column_stack([lo, hi]).ravel()


def auto_threshold(values: np.ndarray) -> float:
    """Midway between the 1st and 99th percentile: a two-level signal's crossing."""
    lo, hi = _level_range(np.asarray(values, dtype=np.float64))
    return (lo + hi) / 2.0


def offset_and_drift(times_s: np.ndarray, residuals_s: np.ndarray) -> dict[str, float]:
    """Split residuals into a constant offset, a drift across the session, and spread.

    The distinction is the whole point of checking against a second copy of a
    signal. A **constant** offset is the recording hardware -- Blackrock filters
    its 1 kHz stream before storing it, which delays it by a fixed amount -- and
    says nothing about synchronization. A residual that **changes** over the
    session is a rate error, which is what synchronization exists to remove. So
    the drift (a line through residual vs time, end minus start) is the number
    to judge; the offset is reported so a large one can be explained.
    """
    t = np.asarray(times_s, dtype=np.float64).ravel()
    r = np.asarray(residuals_s, dtype=np.float64).ravel()
    keep = np.isfinite(t) & np.isfinite(r)
    t, r = t[keep], r[keep]
    if r.size < 3:
        raise ValueError(f"only {r.size} measurements; need at least 3 for a trend")
    centre = t.mean()
    slope, intercept = np.polyfit(t - centre, r, 1)
    span = float(t.max() - t.min())
    spread = np.abs(r - (slope * (t - centre) + intercept))
    return {
        "n": float(r.size),
        "span_min": span / 60.0,
        "median_offset_ms": float(np.median(r)) * 1e3,
        "drift_ms": float(slope * span) * 1e3,
        "drift_ppm": float(slope) * 1e6,
        "spread_p95_ms": float(np.percentile(spread, 95)) * 1e3,
    }


def xcorr_lag(
    reference: np.ndarray, other: np.ndarray, fs: float, max_lag_s: float
) -> tuple[float, float]:
    """Lag of ``other`` behind ``reference``, in seconds, and the peak correlation.

    Both on one sample grid at ``fs``. Positive means ``other`` happens later --
    the same sign as an event residual (OneBox minus Blackrock). Each trace is
    standardized, so the peak is a correlation coefficient: a flat stretch with
    no eye movement gives a low peak, and the caller drops it. The peak is
    refined between samples with a parabola, since a 1 kHz grid alone would
    round every lag to a millisecond.
    """
    from scipy import signal

    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    sa, sb = a.std(), b.std()
    if n < 3 or sa == 0 or sb == 0:
        return float("nan"), 0.0
    a, b = (a - a.mean()) / sa, (b - b.mean()) / sb

    corr = signal.correlate(b, a, mode="full", method="fft") / n
    lags = signal.correlation_lags(n, n, mode="full")
    window = np.abs(lags) <= int(round(max_lag_s * fs))
    corr, lags = corr[window], lags[window]
    i = int(np.argmax(corr))
    return _refine_lag(a, b, int(lags[i])) / fs, float(corr[i])


def _refine_lag(a: np.ndarray, b: np.ndarray, whole: int) -> float:
    """The fractional lag, in samples, within one sample of ``whole``.

    Finds the shift that best overlays ``b`` on ``a`` by interpolating ``b``
    between samples. A parabola through the three correlation values around the
    peak is the usual shortcut, but it is biased whenever the peak is not
    parabolic -- for a sharp-cornered one, by most of a sample.
    """
    from scipy.optimize import minimize_scalar

    idx = np.arange(b.size, dtype=np.float64)
    edge = abs(whole) + 2
    inner = slice(edge, a.size - edge)
    if inner.stop - inner.start < 3:
        return float(whole)

    def misfit(lag: float) -> float:
        shifted = np.interp(idx[inner] + lag, idx, b)
        return float(np.mean((shifted - a[inner]) ** 2))

    best = minimize_scalar(misfit, bounds=(whole - 1.0, whole + 1.0), method="bounded",
                           options={"xatol": 1e-4})
    return float(best.x)
