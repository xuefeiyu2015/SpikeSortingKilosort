"""Neuropixels channel maps, built from the run's own ``.meta``.

Preferring the meta over a stored ``.mat`` matters: the NHP long probe has far
more sites than recordable channels, so which sites are active is an imro-table
choice made per recording. A hardcoded map that happens to be wrong produces a
sorting that looks fine and is spatially nonsense.

A Kilosort probe dict has:

``chanMap``  0-based row index into the binary for each probe channel
``xc``/``yc`` site coordinates in micrometres
``kcoords``  shank/group index per channel
``n_chan``   number of probe channels

``n_chan`` here is the probe's channel count, *not* the binary's channel count --
the binary also carries the SY word, and that count goes in Kilosort's
``settings['n_chan_bin']`` instead.
"""

from __future__ import annotations

import numpy as np

from .._io import spikeglx

__all__ = ["probe_from_geom", "probe_from_meta"]


def probe_from_geom(geom: np.ndarray, shank_pitch_um: float = 0.0) -> dict:
    """Build a Kilosort probe dict from a parsed ``~snsGeomMap`` array.

    ``geom`` is ``(n, 4)`` of ``[shank, x, z, used]`` as returned by
    :func:`spikesorting._io.spikeglx.parse_geom_map`.

    ``shank_pitch_um`` shifts each shank's x by ``shank * pitch``. SpikeGLX
    reports x relative to each shank's own origin, so without this every shank of
    a multi-shank probe lands on top of the others and Kilosort sees one
    impossibly dense column. Irrelevant for single-shank probes such as the
    NP 1.0 NHP long, where the shank index is always 0.

    All rows are kept, including sites flagged unused: Kilosort has its own bad
    channel detection, and dropping rows here would make ``chanMap`` sparse, whose
    interaction with ``n_chan`` is version-dependent. Sites that really must go
    belong in Kilosort's ``bad_channels`` instead.
    """
    geom = np.asarray(geom, dtype=np.float64)
    if geom.ndim != 2 or geom.shape[1] < 4:
        raise ValueError(f"geom must be (n, 4), got {geom.shape}")

    n = geom.shape[0]
    shank = geom[:, 0]
    xc = geom[:, 1] + shank * float(shank_pitch_um)
    return {
        "chanMap": np.arange(n, dtype=np.int32),
        "xc": xc.astype(np.float32),
        "yc": geom[:, 2].astype(np.float32),
        "kcoords": shank.astype(np.float32),
        "n_chan": int(n),
    }


def probe_from_meta(meta: dict[str, str]) -> dict:
    """Build a Kilosort probe dict from a SpikeGLX ``.meta`` dict.

    Raises if the meta has no ``~snsGeomMap`` -- older SpikeGLX versions wrote
    ``~snsShankMap`` instead, which carries column/row indices rather than
    micrometres and needs the probe's pitch to convert. Rather than guess a pitch,
    point ``neuropixels.probe_name`` at an explicit Kilosort probe file for those
    runs.
    """
    geom = spikeglx.parse_geom_map(meta)
    if geom is None:
        raise ValueError(
            "meta has no ~snsGeomMap (older SpikeGLX writes ~snsShankMap). "
            "Set neuropixels.probe_name in the session config to an explicit "
            "Kilosort probe file for this run."
        )
    header = spikeglx.parse_geom_header(meta)
    pitch = float(header["shank_pitch_um"]) if header else 0.0
    return probe_from_geom(geom, shank_pitch_um=pitch)
