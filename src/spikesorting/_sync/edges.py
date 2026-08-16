"""Rising-edge detection for synchronization pulses.

Pure computation: every function here takes arrays and returns arrays, touches no
files, and imports nothing beyond NumPy. This is the fallback used wherever CatGT
is unavailable (the HPC, macOS) and the reference implementation the CatGT output
is checked against.

Edge *indices* are sample offsets; :func:`indices_to_seconds` converts them to the
seconds-from-stream-start convention that CatGT edge files and TPrime both use.

Detection is chunk-friendly: pass the last sample's state as ``previous_above`` so
a long recording can be streamed without ever holding a full channel in memory.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "rising_edge_indices",
    "falling_edge_indices",
    "digital_bit_signal",
    "pulse_widths",
    "filter_pulses_by_width",
    "indices_to_seconds",
    "detect_pulse_times",
    "stream_pulse_times",
    "compare_edge_sets",
]


def _above(signal: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(signal) >= threshold


def rising_edge_indices(
    signal: np.ndarray, threshold: float, previous_above: bool = False
) -> np.ndarray:
    """Indices where ``signal`` crosses ``threshold`` upward.

    An edge at index ``i`` means ``signal[i] >= threshold`` and the preceding
    sample was below. ``previous_above`` supplies that preceding sample for
    ``i == 0`` when streaming chunks.
    """
    above = _above(signal, threshold)
    if above.size == 0:
        return np.empty(0, dtype=np.int64)
    prev = np.empty_like(above)
    prev[0] = previous_above
    prev[1:] = above[:-1]
    return np.flatnonzero(above & ~prev).astype(np.int64)


def falling_edge_indices(
    signal: np.ndarray, threshold: float, previous_above: bool = False
) -> np.ndarray:
    """Indices where ``signal`` crosses ``threshold`` downward."""
    above = _above(signal, threshold)
    if above.size == 0:
        return np.empty(0, dtype=np.int64)
    prev = np.empty_like(above)
    prev[0] = previous_above
    prev[1:] = above[:-1]
    return np.flatnonzero(~above & prev).astype(np.int64)


def digital_bit_signal(words: np.ndarray, bit: int) -> np.ndarray:
    """Extract one bit from a digital word stream as a 0/1 array.

    SpikeGLX stores the SMA1 sync square wave in bit 6 of each stream's SY word.
    """
    if bit < 0:
        raise ValueError(f"bit must be >= 0, got {bit}")
    return ((np.asarray(words).astype(np.uint64) >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)


def pulse_widths(rising: np.ndarray, falling: np.ndarray) -> np.ndarray:
    """Width in samples of each pulse started by ``rising``.

    Each rising edge is paired with the first falling edge after it. A trailing
    rising edge with no matching fall (recording stopped mid-pulse) gets width
    ``-1`` so callers can decide whether to keep it.
    """
    rising = np.asarray(rising, dtype=np.int64)
    falling = np.asarray(falling, dtype=np.int64)
    if rising.size == 0:
        return np.empty(0, dtype=np.int64)
    if falling.size == 0:
        return np.full(rising.size, -1, dtype=np.int64)

    pos = np.searchsorted(falling, rising, side="right")
    widths = np.full(rising.size, -1, dtype=np.int64)
    valid = pos < falling.size
    widths[valid] = falling[pos[valid]] - rising[valid]
    return widths


def filter_pulses_by_width(
    rising: np.ndarray,
    widths: np.ndarray,
    fs: float,
    duration_ms: float = 0.0,
    tolerance_ms: float | None = None,
) -> np.ndarray:
    """Keep the rising edges whose pulse width matches ``duration_ms``.

    Mirrors CatGT's extractor semantics: ``duration_ms == 0`` accepts every pulse
    (including one truncated by the end of the recording), and the default
    tolerance is +/-20% of the requested duration.
    """
    rising = np.asarray(rising, dtype=np.int64)
    widths = np.asarray(widths, dtype=np.int64)
    if duration_ms <= 0:
        return rising

    tol_ms = 0.2 * duration_ms if tolerance_ms is None else tolerance_ms
    width_ms = widths / fs * 1000.0
    keep = (widths >= 0) & (np.abs(width_ms - duration_ms) <= tol_ms)
    return rising[keep]


def indices_to_seconds(indices: np.ndarray, fs: float, start_sample: int = 0) -> np.ndarray:
    """Convert sample indices to seconds from stream start.

    This is the unit CatGT edge files and TPrime use.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return (np.asarray(indices, dtype=np.int64) + int(start_sample)) / float(fs)


def detect_pulse_times(
    signal: np.ndarray,
    threshold: float,
    fs: float,
    duration_ms: float = 0.0,
    tolerance_ms: float | None = None,
    start_sample: int = 0,
) -> np.ndarray:
    """One-shot detection: signal in, leading-edge times (s) out.

    Convenience wrapper for in-memory channels. Long recordings should stream with
    :func:`stream_pulse_times` instead.
    """
    rising = rising_edge_indices(signal, threshold)
    falling = falling_edge_indices(signal, threshold)
    widths = pulse_widths(rising, falling)
    kept = filter_pulses_by_width(rising, widths, fs, duration_ms, tolerance_ms)
    return indices_to_seconds(kept, fs, start_sample)


def stream_pulse_times(
    chunks: Iterable[tuple[int, np.ndarray]],
    threshold: float,
    fs: float,
    duration_ms: float = 0.0,
    tolerance_ms: float | None = None,
) -> np.ndarray:
    """Same result as :func:`detect_pulse_times`, over an iterator of chunks.

    ``chunks`` yields ``(start_sample, samples)``. State carries across chunk
    boundaries, so an edge falling exactly on a boundary is detected once and only
    once. Memory stays proportional to the number of *edges*, not samples, which
    is what makes hour-long 30 kHz channels tractable.
    """
    rising_parts: list[np.ndarray] = []
    falling_parts: list[np.ndarray] = []
    previous_above = False

    for start, block in chunks:
        block = np.asarray(block)
        if block.size == 0:
            continue
        rising_parts.append(rising_edge_indices(block, threshold, previous_above) + start)
        falling_parts.append(falling_edge_indices(block, threshold, previous_above) + start)
        previous_above = bool(block[-1] >= threshold)

    empty = np.empty(0, dtype=np.int64)
    rising = np.concatenate(rising_parts) if rising_parts else empty
    falling = np.concatenate(falling_parts) if falling_parts else empty

    widths = pulse_widths(rising, falling)
    kept = filter_pulses_by_width(rising, widths, fs, duration_ms, tolerance_ms)
    return indices_to_seconds(kept, fs)


def compare_edge_sets(
    a: np.ndarray, b: np.ndarray, tolerance_s: float = 1e-4
) -> dict[str, float | int | bool]:
    """Compare two edge-time lists, e.g. CatGT's output against the NumPy fallback.

    Matches each edge in ``a`` to its nearest neighbour in ``b`` rather than
    assuming equal counts, so a single missed pulse does not shift every
    subsequent comparison.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.size == 0 or b.size == 0:
        return {
            "n_a": int(a.size),
            "n_b": int(b.size),
            "n_matched": 0,
            "max_abs_diff_s": float("nan"),
            "median_abs_diff_s": float("nan"),
            "agree": bool(a.size == b.size == 0),
        }

    pos = np.clip(np.searchsorted(b, a), 1, b.size - 1) if b.size > 1 else np.zeros(a.size, int)
    if b.size > 1:
        left, right = b[pos - 1], b[pos]
        nearest = np.where(np.abs(a - left) <= np.abs(a - right), left, right)
    else:
        nearest = np.full(a.size, b[0])

    diffs = np.abs(a - nearest)
    matched = int(np.count_nonzero(diffs <= tolerance_s))
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_matched": matched,
        "max_abs_diff_s": float(diffs.max()),
        "median_abs_diff_s": float(np.median(diffs)),
        "agree": bool(a.size == b.size and matched == a.size),
    }
