"""Conversions between the two probe representations in play.

Kilosort's native API takes a plain dict (``chanMap``/``xc``/``yc``/``kcoords``).
SpikeInterface instead expects a ``probeinterface.Probe`` attached to the
recording, and its Kilosort4 wrapper reads geometry from there. Blackrock data
goes through SpikeInterface, so it needs the second form.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["to_probeinterface", "from_probeinterface", "probe_summary"]


def to_probeinterface(probe: dict, contact_radius_um: float = 5.0) -> Any:
    """Convert a Kilosort probe dict to a ``probeinterface.Probe``.

    ``set_device_channel_indices`` is required: it tells SpikeInterface which
    recording channel each contact corresponds to. Without it the probe attaches
    but the channel order is undefined.
    """
    from probeinterface import Probe  # lazy

    positions = np.column_stack(
        [np.asarray(probe["xc"], dtype=np.float64), np.asarray(probe["yc"], dtype=np.float64)]
    )
    result = Probe(ndim=2, si_units="um")
    result.set_contacts(
        positions=positions,
        shapes="circle",
        shape_params={"radius": contact_radius_um},
    )
    result.set_device_channel_indices(np.asarray(probe["chanMap"], dtype=int))
    return result


def from_probeinterface(probe: Any) -> dict:
    """Convert a ``probeinterface.Probe`` to a Kilosort probe dict.

    The inverse of :func:`to_probeinterface`, and what lets probeinterface's own
    readers -- ``read_spikeglx`` on a ``.meta``, ``read_prb``, the probe library
    -- be used as a source of geometry. ``device_channel_indices`` is the
    contact-to-binary-row mapping and becomes ``chanMap``; contacts marked ``-1``
    are not wired to anything and are dropped, since a negative row index is not
    something Kilosort can read.
    """
    positions = np.asarray(probe.contact_positions, dtype=np.float64)
    indices = probe.device_channel_indices
    if indices is None:
        raise ValueError(
            "probe has no device_channel_indices, so which recording channel each "
            "contact corresponds to is unknown"
        )
    indices = np.asarray(indices, dtype=int)
    wired = indices >= 0

    shank_ids = getattr(probe, "shank_ids", None)
    if shank_ids is None or len(shank_ids) != positions.shape[0]:
        kcoords = np.zeros(positions.shape[0])
    else:
        # probeinterface names shanks with strings; Kilosort wants a number per
        # channel, and only the grouping matters.
        _, kcoords = np.unique(np.asarray(shank_ids, dtype=str), return_inverse=True)

    return {
        "chanMap": indices[wired].astype(np.int32),
        "xc": positions[wired, 0].astype(np.float32),
        "yc": positions[wired, 1].astype(np.float32),
        "kcoords": np.asarray(kcoords)[wired].astype(np.float32),
        "n_chan": int(wired.sum()),
    }


def probe_summary(probe: dict) -> dict[str, Any]:
    """Small, loggable description of a probe dict. Pure."""
    xc = np.asarray(probe["xc"], dtype=np.float64)
    yc = np.asarray(probe["yc"], dtype=np.float64)
    kcoords = np.asarray(probe.get("kcoords", np.zeros_like(xc)))
    return {
        "n_chan": int(probe.get("n_chan", xc.size)),
        "x_range_um": [float(xc.min()), float(xc.max())] if xc.size else [0.0, 0.0],
        "y_range_um": [float(yc.min()), float(yc.max())] if yc.size else [0.0, 0.0],
        "n_groups": int(np.unique(kcoords).size),
        "placeholder_geometry": bool(probe.get("_placeholder", False)),
    }
