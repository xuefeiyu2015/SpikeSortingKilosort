"""Reading and writing probe files.

Kilosort's probe JSON is just a dict of lists, so this writes it directly rather
than going through ``kilosort.io.save_probe``. That keeps probe construction --
which is desk work, done once per array or per run -- possible on a laptop with
no Kilosort, torch or GPU installed.

Three formats are in play:

``.json``  Kilosort4's native probe file. What ``run_kilosort`` loads.
``.mat``   the older Kilosort channel map (``NeuroPix1_default.mat`` and friends),
           read-only here.
``.prb``   probeinterface's format, for the SpikeInterface route used by
           Blackrock data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "save_probe_json",
    "load_probe_json",
    "probe_from_mat",
    "save_probe_prb",
    "validate_probe",
]

#: Keys Kilosort expects in a probe dict.
REQUIRED_KEYS = ("chanMap", "xc", "yc", "kcoords", "n_chan")


def validate_probe(probe: dict) -> list[str]:
    """Return a list of problems with a probe dict; empty means it is usable.

    Returns findings instead of raising so a builder can report everything wrong
    at once. A probe that passes here can still be *wrong* -- correct geometry is
    not something code can check -- but it will at least load.
    """
    problems: list[str] = []

    missing = [key for key in REQUIRED_KEYS if key not in probe]
    if missing:
        return [f"missing required key(s): {', '.join(missing)}"]

    chan_map = np.asarray(probe["chanMap"])
    xc = np.asarray(probe["xc"])
    yc = np.asarray(probe["yc"])
    kcoords = np.asarray(probe["kcoords"])

    sizes = {"chanMap": chan_map.size, "xc": xc.size, "yc": yc.size, "kcoords": kcoords.size}
    if len(set(sizes.values())) != 1:
        # Every check below assumes the arrays line up element-wise, so stop here
        # rather than reporting a cascade of consequences.
        return [f"array lengths disagree: {sizes}"]

    if int(probe["n_chan"]) != chan_map.size:
        problems.append(f"n_chan ({probe['n_chan']}) != len(chanMap) ({chan_map.size})")

    if chan_map.size and chan_map.min() < 0:
        problems.append("chanMap contains negative indices")

    if np.unique(chan_map).size != chan_map.size:
        problems.append("chanMap contains duplicate channel indices")

    # Uniqueness is per (shank, x, y): distinct shanks legitimately reuse
    # coordinates when those coordinates are shank-relative.
    if xc.size:
        positions = np.column_stack([kcoords, xc, yc])
        if np.unique(positions, axis=0).shape[0] != positions.shape[0]:
            problems.append("two or more contacts share the same (kcoords, xc, yc) position")

    return problems


def save_probe_json(probe: dict, path: str | Path, indent: int | None = None) -> Path:
    """Write a Kilosort probe ``.json``.

    Byte-compatible with ``kilosort.io.save_probe``: NumPy arrays become plain
    lists and nothing else is added. Extra bookkeeping keys (``labels``,
    ``_placeholder``) are dropped, since Kilosort's loader does not expect them.
    """
    problems = validate_probe(probe)
    if problems:
        raise ValueError("refusing to save an invalid probe:\n  - " + "\n  - ".join(problems))

    payload: dict[str, Any] = {}
    for key in REQUIRED_KEYS:
        value = probe[key]
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            payload[key] = value.item()
        else:
            payload[key] = value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def load_probe_json(path: str | Path) -> dict:
    """Read a Kilosort probe ``.json`` back into arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"probe file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "chanMap": np.asarray(raw["chanMap"], dtype=np.int32),
        "xc": np.asarray(raw["xc"], dtype=np.float32),
        "yc": np.asarray(raw["yc"], dtype=np.float32),
        "kcoords": np.asarray(raw["kcoords"], dtype=np.float32),
        "n_chan": int(raw["n_chan"]),
    }


def probe_from_mat(path: str | Path) -> dict:
    """Read an older Kilosort ``.mat`` channel map.

    Handles both index conventions: ``chanMap`` is 1-based MATLAB indexing while
    ``chanMap0ind`` is 0-based, and Kilosort4 wants 0-based. Getting this wrong
    shifts every channel by one, which looks like a plausible sorting.
    """
    from scipy.io import loadmat  # scipy is a base dependency

    data = loadmat(str(path))

    def column(key: str) -> np.ndarray | None:
        return np.asarray(data[key]).reshape(-1) if key in data else None

    chan_map = column("chanMap0ind")
    if chan_map is None:
        one_based = column("chanMap")
        if one_based is None:
            raise ValueError(f"{path} has neither chanMap0ind nor chanMap")
        chan_map = one_based - 1

    xc, yc = column("xcoords"), column("ycoords")
    if xc is None or yc is None:
        raise ValueError(f"{path} has no xcoords/ycoords")

    kcoords = column("kcoords")
    if kcoords is None:
        kcoords = column("shankInd")
    if kcoords is None:
        kcoords = np.zeros(chan_map.size)

    return {
        "chanMap": chan_map.astype(np.int32),
        "xc": xc.astype(np.float32),
        "yc": yc.astype(np.float32),
        "kcoords": kcoords.astype(np.float32),
        "n_chan": int(chan_map.size),
    }


def save_probe_prb(probe: dict, path: str | Path, contact_radius_um: float = 5.0) -> Path:
    """Write a probeinterface ``.prb``, for the SpikeInterface route.

    Needs ``probeinterface`` installed; the JSON writer above does not.
    """
    from probeinterface import ProbeGroup, write_prb  # lazy

    from .common import to_probeinterface

    group = ProbeGroup()
    group.add_probe(to_probeinterface(probe, contact_radius_um))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prb(str(path), group)
    return path
