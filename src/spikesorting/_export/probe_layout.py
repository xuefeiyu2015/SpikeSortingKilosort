"""What the overview's waveform page draws, computed: pure NumPy, no files.

Every exported unit's measured mean waveform, drawn as a small glyph where the
unit sits on the probe, against the probe's depth from its tip. The reading is in ``pipeline._write_overview``, the
drawing in ``_plots.summary.plot_probe_waveforms``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GLYPH_WIDTH_UM",
    "GLYPH_HEIGHT_UM",
    "compute_unit_positions",
    "place_units",
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


def place_units(
    medians: dict[int, tuple[float, float]],
    unit_ids: np.ndarray,
    peak_rows: dict[int, int | None],
    site_xy: np.ndarray | None,
    channel_map: np.ndarray | None,
) -> dict[int, tuple[float, float]]:
    """Where each unit sits: its spikes' median position, else its peak channel's site.

    ``medians`` are :func:`compute_unit_positions`'s; ``peak_rows`` each unit's
    peak channel as a binary row, found in ``channel_map`` to pick its row of
    ``site_xy``. A unit with neither is left out. The one rule both
    ``units.csv``'s ``x_um``/``y_um`` and ``overview.pdf``'s waveform centres use.
    """
    placed = {int(u): medians[int(u)] for u in unit_ids if int(u) in medians}
    if site_xy is None or channel_map is None:
        return placed
    rows = np.asarray(channel_map).ravel()
    for unit in unit_ids:
        unit = int(unit)
        row = peak_rows.get(unit)
        if unit in placed or row is None:
            continue
        hit = np.flatnonzero(rows == row)
        if hit.size:
            placed[unit] = (float(site_xy[hit[0], 0]), float(site_xy[hit[0], 1]))
    return placed


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
