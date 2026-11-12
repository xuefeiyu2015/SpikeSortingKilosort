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
    "EventLevels",
    "event_levels",
    "stream_event_onsets",
    "offset_and_drift",
    "xcorr_lag",
    "clipped_diff",
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


@dataclass(frozen=True)
class EventLevels:
    """Where an event channel rests, where it goes on an event, and its noise.

    An event is the signal *leaving* its baseline, in whichever direction that
    is: a photodiode can rest high and dip when the target appears. Each system
    gets its own levels, so a copy recorded inverted still yields the same
    moments.
    """

    baseline: float
    active: float
    noise: float

    @property
    def polarity(self) -> int:
        """+1 when an event goes up from the baseline, -1 when it goes down."""
        return 1 if self.active >= self.baseline else -1

    @property
    def amplitude(self) -> float:
        return abs(self.active - self.baseline)

    def summary(self) -> str:
        return (f"baseline {self.baseline:.3g}, events go "
                f"{'up' if self.polarity > 0 else 'down'} to {self.active:.3g} "
                f"(noise {self.noise:.2g})")


#: Hysteresis: an event starts past this fraction of the way to ``active``...
ENTER_FRACTION = 0.6
#: ...and ends only back below this one, so noise at one level cannot chatter.
LEAVE_FRACTION = 0.4


def event_levels(samples: np.ndarray, min_snr: float = 8.0) -> EventLevels:
    """Baseline, event level and noise of a two-level channel, from samples of it.

    ``samples`` should span the recording (excerpts are fine), not its first
    minutes: a photodiode that sits still until the task starts has only noise
    there, and a threshold set from noise detects noise.

    The noise comes from sample-to-sample differences, which a level change
    barely touches whatever the duty cycle. Samples more than ``min_snr`` noise
    widths from the median are departures; the side with more of them is where
    events go, so a flash lit 0.2% of the time is still seen (percentiles would
    put both levels on the baseline). The two levels are then the medians either
    side of the midpoint, and the baseline is the one the signal spends more
    time at. Raises when nothing departs: there are no events to find.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("no samples to set event levels from")
    noise = float(1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2.0))
    centre = float(np.median(x))
    far = x[np.abs(x - centre) > min_snr * noise]
    if far.size == 0 or np.ptp(x) == 0:
        raise ValueError(
            f"the signal never leaves its baseline ({centre:.3g}, range {np.ptp(x):.2g}, "
            f"noise {noise:.2g}): no events to detect"
        )
    up = far > centre
    side = far[up] if np.count_nonzero(up) >= np.count_nonzero(~up) else far[~up]
    midpoint = (centre + float(np.median(side))) / 2.0
    above = x >= midpoint
    high, low = float(np.median(x[above])), float(np.median(x[~above]))
    baseline, active = (low, high) if np.count_nonzero(~above) >= np.count_nonzero(above) \
        else (high, low)
    return EventLevels(baseline, active, noise)


def stream_event_onsets(
    chunks,
    levels: EventLevels,
    fs: float,
    min_gap_s: float = 0.05,
    max_rate_hz: float = 20.0,
) -> np.ndarray:
    """Seconds at which the signal leaves its baseline, over ``(start, samples)`` chunks.

    Hysteresis between ``ENTER_FRACTION`` and ``LEAVE_FRACTION`` of the way to
    the event level; an event that follows the previous one's end by less than
    ``min_gap_s`` is the same event (a monitor's refresh flicker inside one
    flash), and is merged into it. More than ``max_rate_hz`` events a second is
    not a photodiode, and raises rather than vote with noise.
    """
    sign = levels.polarity
    enter = ENTER_FRACTION * levels.amplitude
    leave = LEAVE_FRACTION * levels.amplitude
    onsets, offsets = [], []
    state: bool | None = None
    n_total = 0
    for start, block in chunks:
        s = (np.asarray(block, dtype=np.float64) - levels.baseline) * sign
        if s.size == 0:
            continue
        n_total = start + s.size
        code = np.zeros(s.size, dtype=np.int8)
        code[s >= enter] = 1
        code[s <= leave] = -1
        if state is None:                        # the recording's first sample
            state = bool(s[0] >= (enter + leave) / 2)
        # Forward-fill: between the two levels the signal keeps its last state.
        last = np.where(code != 0, np.arange(s.size), -1)
        np.maximum.accumulate(last, out=last)
        now = np.where(last >= 0, code[np.maximum(last, 0)] > 0, state)
        before = np.concatenate([[state], now[:-1]])
        onsets.append(np.flatnonzero(now & ~before) + start)
        offsets.append(np.flatnonzero(~now & before) + start)
        state = bool(now[-1])
    on = np.concatenate(onsets) if onsets else np.empty(0, dtype=np.int64)
    off = np.concatenate(offsets) if offsets else np.empty(0, dtype=np.int64)
    if on.size:
        # The end of the event before each onset; keep onsets after a real gap.
        prev = np.searchsorted(off, on) - 1
        gap = np.where(prev >= 0, on - off[np.maximum(prev, 0)], np.iinfo(np.int64).max)
        on = on[gap >= min_gap_s * fs]
    duration_s = n_total / fs if fs else 0.0
    if duration_s and on.size / duration_s > max_rate_hz:
        raise ValueError(
            f"{on.size} events in {duration_s:.0f} s ({on.size / duration_s:.0f}/s): "
            f"noise crossing the threshold, not events ({levels.summary()})"
        )
    return on / fs


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


def clipped_diff(values: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """Sample-to-sample change, clipped at ``n_mad`` median absolute deviations.

    For correlating an eye trace by its movements: a blink, or a value that
    saturated, is one enormous step that would otherwise own the correlation.
    """
    d = np.diff(np.asarray(values, dtype=np.float64))
    if d.size == 0:
        return d
    mad = float(np.median(np.abs(d - np.median(d))))
    if mad == 0:
        return d
    limit = n_mad * mad
    return np.clip(d, -limit, limit)


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
