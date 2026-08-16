"""Blackrock Utah array geometry.

.. warning::

   **The default grid here is a placeholder and is almost certainly not this
   array's wiring.** A Utah array's electrode-to-amplifier-channel mapping is
   array-specific and lives in the ``.cmp`` map file shipped with it. Sorting
   against the wrong map still produces clusters -- they are just attributed to
   the wrong cortical location. Supply the real ``.cmp`` before trusting any
   Blackrock sorting result.

At 400 um pitch, no spike is visible on two electrodes, so the array is a set of
independent single channels rather than a dense probe. :func:`independent_kcoords`
puts each channel in its own group, which stops Kilosort from trying to template
across sites that cannot share a unit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["utah_grid_probe", "probe_from_cmp", "independent_kcoords"]

#: Standard Utah array electrode spacing.
UTAH_PITCH_UM = 400.0


def independent_kcoords(n_channels: int) -> np.ndarray:
    """One group per channel -- the right choice for a 400 um grid."""
    return np.arange(n_channels, dtype=np.float32)


def utah_grid_probe(
    n_channels: int = 96,
    pitch_um: float = UTAH_PITCH_UM,
    n_cols: int = 10,
    independent: bool = True,
) -> dict:
    """Placeholder geometry: channels laid out row-major on a square grid.

    Only correct if the array happens to be wired in channel order, which is
    unusual. Use :func:`probe_from_cmp` with the real map file instead.
    """
    index = np.arange(n_channels)
    xc = (index % n_cols) * pitch_um
    yc = (index // n_cols) * pitch_um
    return {
        "chanMap": index.astype(np.int32),
        "xc": xc.astype(np.float32),
        "yc": yc.astype(np.float32),
        "kcoords": independent_kcoords(n_channels)
        if independent
        else np.zeros(n_channels, dtype=np.float32),
        "n_chan": int(n_channels),
        "_placeholder": True,
    }


def probe_from_cmp(
    path: str | Path,
    pitch_um: float = UTAH_PITCH_UM,
    independent: bool = True,
) -> dict:
    """Build a probe dict from a Blackrock ``.cmp`` channel map file.

    ``.cmp`` files are tab-separated with ``//`` comment lines and columns
    ``col  row  bank  elec  label``. Column and row are grid indices, which this
    multiplies by ``pitch_um``. The channel order of the returned map follows the
    amplifier channel number implied by ``bank``/``elec``.

    Raises a descriptive error rather than guessing if the layout does not match --
    ``.cmp`` has had several variants, and a silently misparsed map is worse than
    a hard failure.
    """
    path = Path(path)
    rows: list[tuple[int, int, str, int, str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            col, row = int(fields[0]), int(fields[1])
            bank, elec = fields[2], int(fields[3])
        except ValueError:
            continue
        label = fields[4] if len(fields) > 4 else f"{bank}{elec}"
        rows.append((col, row, bank, elec, label))

    if not rows:
        raise ValueError(
            f"no usable rows parsed from {path}. Expected tab-separated "
            "'col row bank elec label' lines with // comments. If this array's .cmp "
            "uses a different layout, the parser needs updating for it."
        )

    banks = sorted({r[2] for r in rows})
    bank_offset = {bank: i * 32 for i, bank in enumerate(banks)}
    channels = [bank_offset[bank] + elec - 1 for _, _, bank, elec, _ in rows]

    order = np.argsort(channels)
    cols = np.array([rows[i][0] for i in order], dtype=np.float64)
    grid_rows = np.array([rows[i][1] for i in order], dtype=np.float64)
    n_channels = len(order)

    return {
        "chanMap": np.arange(n_channels, dtype=np.int32),
        "xc": (cols * pitch_um).astype(np.float32),
        "yc": (grid_rows * pitch_um).astype(np.float32),
        "kcoords": independent_kcoords(n_channels)
        if independent
        else np.zeros(n_channels, dtype=np.float32),
        "n_chan": int(n_channels),
        "labels": [rows[i][4] for i in order],
    }
