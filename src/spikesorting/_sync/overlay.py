"""What ``sync_overview.png`` draws, computed: pure NumPy, no files.

The figure shows one window of every channel both systems recorded -- each
system on its own clock ("before"), then the OneBox put on Blackrock time by
the fitted map ("after"), then a few milliseconds around one 1 Hz edge -- and
every 1 Hz edge's residual across the session. The reading is done by
``pipeline._sync_figure_data``; the drawing by ``_plots/sync.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SYNC_WINDOW_S",
    "SYNC_ZOOM_S",
    "EDGE_PAIRING_S",
    "MAX_POINTS",
    "centre_window",
    "square_wave",
    "nearest_event",
    "nearest_crossing",
    "normalise",
    "thin",
]

#: Width of the before/after window, centred in the recordings' overlap. Short
#: enough that each 1 Hz pulse and each flash is its own mark, so a shift of a
#: fraction of a second -- or of whole seconds -- is plain to see.
SYNC_WINDOW_S = 20.0
#: Half-width of each channel's zoom around one of its edges.
SYNC_ZOOM_S = 0.02
#: 1 Hz edges further apart than this after mapping are not the same pulse.
EDGE_PAIRING_S = 0.25
#: Points per trace drawn; a window of OneBox samples is thinned to this.
MAX_POINTS = 20_000


def centre_window(start_s: float, stop_s: float, width_s: float) -> tuple[float, float]:
    """``width_s`` seconds centred in ``[start_s, stop_s]``; all of it when shorter."""
    if stop_s <= start_s:
        raise ValueError(f"empty interval {start_s}..{stop_s}")
    if stop_s - start_s <= width_s:
        return float(start_s), float(stop_s)
    middle = (start_s + stop_s) / 2.0
    return middle - width_s / 2.0, middle + width_s / 2.0


def square_wave(rising_s: np.ndarray, t_s: np.ndarray, high_s: float) -> np.ndarray:
    """The 1 Hz pulse train rebuilt from its rising edges: 1 for ``high_s`` after each.

    The edge files hold rising edges only; the pulse width is the session's
    ``sync_pulse_ms``. Before the first edge the train reads 0.
    """
    rising = np.sort(np.asarray(rising_s, dtype=np.float64).ravel())
    t = np.asarray(t_s, dtype=np.float64)
    if rising.size == 0:
        return np.zeros(t.shape)
    last = np.searchsorted(rising, t, side="right") - 1
    since = t - rising[np.clip(last, 0, None)]
    return ((last >= 0) & (since < high_s)).astype(np.float64)


def nearest_event(events_s: np.ndarray, lo_s: float, hi_s: float, target_s: float) -> float | None:
    """The event in ``[lo_s, hi_s]`` closest to ``target_s``, or None."""
    events = np.asarray(events_s, dtype=np.float64).ravel()
    inside = events[(events >= lo_s) & (events <= hi_s)]
    if inside.size == 0:
        return None
    return float(inside[np.argmin(np.abs(inside - target_s))])


def nearest_crossing(t_s: np.ndarray, y: np.ndarray, level: float, target_s: float,
                     margin_s: float = 0.0) -> float | None:
    """When ``y`` crosses ``level`` (either way) closest to ``target_s``, or None.

    Crossings within ``margin_s`` of either end are skipped, so a zoom around the
    result has samples on both sides.
    """
    t, y = np.asarray(t_s, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if t.size < 2:
        return None
    above = y >= level
    k = np.flatnonzero(above[1:] != above[:-1]) + 1
    times = t[k]
    times = times[(times >= t[0] + margin_s) & (times <= t[-1] - margin_s)]
    if times.size == 0:
        return None
    return float(times[np.argmin(np.abs(times - target_s))])


def normalise(y: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """``y`` scaled so ``reference``'s 1st..99th percentile spans 0..1.

    The two copies of a channel differ in units and gain (OneBox volts, Blackrock
    millivolts, filtered); drawn on one axis only their timing should differ.
    ``reference`` defaults to ``y``; a flat reference gives zeros.
    """
    y = np.asarray(y, dtype=np.float64)
    ref = y if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.size == 0:
        return np.zeros(y.shape)
    lo, hi = np.percentile(ref, [1.0, 99.0])
    if hi - lo <= 0:
        return np.zeros(y.shape)
    return (y - lo) / (hi - lo)


def thin(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Every k-th point, at most ``max_points`` of them -- what a trace is drawn from."""
    x, y = np.asarray(x), np.asarray(y)
    step = max(1, int(np.ceil(x.size / max_points)))
    return x[::step], y[::step]
