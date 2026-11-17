"""What the overview's waveform page draws, computed: pure NumPy, no files.

Every exported unit's measured mean waveform, drawn as a small glyph where the
unit sits on the probe, against the probe's depth from its tip and the banks its
recorded sites belong to. The reading is in ``pipeline._write_overview``, the
drawing in ``_plots.summary.plot_probe_waveforms``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GLYPH_WIDTH_UM",
    "GLYPH_HEIGHT_UM",
    "compute_unit_positions",
    "compute_bank_spans",
    "compute_glyph_scale",
    "compute_waveform_glyphs",
]

#: How wide each waveform is drawn on the probe, in um: about one site column.
GLYPH_WIDTH_UM = 30.0
#: How tall the *median* unit's trough-to-peak is drawn, in um. One scale for
#: every unit, so a large unit is drawn larger.
GLYPH_HEIGHT_UM = 40.0


def compute_unit_positions(
    xy: np.ndarray, clusters: np.ndarray, unit_ids: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Each unit's median spike position on the probe, from per-spike ``xy``.

    The median, not the mean: a few spikes localised far away (a collision, a
    noisy channel) must not move the unit. Units with no spike in ``xy`` are
    left out.
    """
    xy = np.asarray(xy, dtype=np.float64)
    clusters = np.asarray(clusters)
    out: dict[int, tuple[float, float]] = {}
    for unit in np.asarray(unit_ids):
        mine = xy[clusters == unit]
        if mine.size:
            out[int(unit)] = (float(np.median(mine[:, 0])), float(np.median(mine[:, 1])))
    return out


def compute_bank_spans(site_y: np.ndarray, site_bank: np.ndarray) -> list[tuple[int, float, float]]:
    """``(bank, y_low, y_high)`` for each run of recorded sites in one bank, in um.

    Sites are sorted by depth and grouped while the bank stays the same; each
    span reaches half a site pitch past its outermost sites, so neighbouring
    spans meet. Sites with an unknown bank (negative) are skipped.
    """
    y = np.asarray(site_y, dtype=np.float64)
    bank = np.asarray(site_bank, dtype=np.int64)
    keep = bank >= 0
    y, bank = y[keep], bank[keep]
    if y.size == 0:
        return []
    order = np.argsort(y, kind="stable")
    y, bank = y[order], bank[order]
    rows = np.unique(y)
    half = float(np.min(np.diff(rows))) / 2 if rows.size > 1 else 10.0
    spans: list[tuple[int, float, float]] = []
    start = 0
    for i in range(1, y.size + 1):
        if i == y.size or bank[i] != bank[start]:
            spans.append((int(bank[start]), float(y[start] - half), float(y[i - 1] + half)))
            start = i
    return spans


def compute_glyph_scale(waveforms: np.ndarray, height_um: float = GLYPH_HEIGHT_UM) -> float:
    """um per waveform unit, so the median unit's trough-to-peak is ``height_um``.

    ``waveforms`` is ``(n_samples, n_units)``. One scale for all units keeps
    their sizes comparable; a flat set gives 1.0 rather than a division by zero.
    """
    w = np.asarray(waveforms, dtype=np.float64)
    if w.size == 0:
        return 1.0
    ptp = np.ptp(w, axis=0)
    ptp = ptp[np.isfinite(ptp) & (ptp > 0)]
    return float(height_um / np.median(ptp)) if ptp.size else 1.0


def compute_waveform_glyphs(
    waveforms: np.ndarray,
    unit_ids: np.ndarray,
    positions: dict[int, tuple[float, float]],
    scale: float,
    width_um: float = GLYPH_WIDTH_UM,
) -> list[dict]:
    """One glyph per unit that has both a waveform and a position.

    Each is ``{"unit_id", "x", "y"}``: the waveform's samples spread over
    ``width_um`` centred on the unit's x, and its values times ``scale`` added
    to the unit's depth -- a trough below the unit, a peak above it.
    """
    w = np.asarray(waveforms, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != len(unit_ids):
        raise ValueError(f"waveforms {w.shape} do not have one column per unit ({len(unit_ids)})")
    offsets = np.linspace(-width_um / 2, width_um / 2, w.shape[0])
    glyphs = []
    for column, unit in enumerate(np.asarray(unit_ids)):
        unit = int(unit)
        trace = w[:, column]
        if unit not in positions or not np.all(np.isfinite(trace)):
            continue
        x0, y0 = positions[unit]
        glyphs.append({"unit_id": unit, "x": x0 + offsets, "y": y0 + scale * trace})
    return glyphs
