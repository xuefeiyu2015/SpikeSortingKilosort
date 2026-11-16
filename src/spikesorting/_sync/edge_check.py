"""Were the sync pulses there? Counts against what the recording's length implies.

The 1 Hz square wave should give one rising edge per second of recording, and
the 14 s coded burst one burst per ``burst_interval_s``. A train with no edges
at all is *missing* -- a cable not connected, a generator off, a threshold that
never fires -- and one with far fewer than expected is *too few*. Neither stops
anything here; they are reported, so the problem is known at sorting time
rather than discovered when synchronization fails.

Pure: edge times in, one entry per train out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import burst

__all__ = ["ONE_HZ", "BURST", "MIN_1HZ_FRACTION", "MIN_BURST_FRACTION",
           "check_sync_edges", "sync_problems"]

ONE_HZ = "1 Hz"
BURST = "14 s burst"
#: A 1 Hz train below this fraction of one edge per second is "too few".
MIN_1HZ_FRACTION = 0.9
#: Bursts below this fraction of the expected count are "too few". Looser than
#: the 1 Hz: the burst generator is often started a little after the recording.
MIN_BURST_FRACTION = 0.5


def check_sync_edges(
    trains: dict[str, tuple[str, np.ndarray | None, str]],
    duration_s: float,
    sync_period_s: float = 1.0,
    burst_interval_s: float = 14.0,
) -> list[dict[str, Any]]:
    """One entry per train: what was found, what was expected, and a status.

    ``trains`` maps an edge-file name to ``(kind, times_s, where)``: ``kind`` is
    :data:`ONE_HZ` or :data:`BURST`, ``times_s`` the edges (None when the train
    was not extracted at all) and ``where`` the stream it comes from. Status is
    ``ok``, ``too few``, ``missing`` (no edges) or ``not extracted``.
    """
    out = []
    for name, (kind, times, where) in trains.items():
        entry: dict[str, Any] = {"name": name, "kind": kind, "stream": where}
        if kind == BURST:
            entry["unit"], expected = "bursts", duration_s / burst_interval_s
            floor = MIN_BURST_FRACTION
        else:
            entry["unit"], expected = "pulses", duration_s / sync_period_s
            floor = MIN_1HZ_FRACTION
        entry["expected"] = int(round(expected))
        if times is None:
            entry.update(edges=None, found=None, status="not extracted")
            out.append(entry)
            continue
        times = np.sort(np.asarray(times, dtype=np.float64).ravel())
        found = (burst.group_burst_onsets(times, burst_interval_s / 2).size
                 if kind == BURST and times.size else times.size)
        entry.update(edges=int(times.size), found=int(found))
        if times.size == 0:
            entry["status"] = "missing"
        elif found < floor * expected:
            entry["status"] = "too few"
        else:
            entry["status"] = "ok"
        out.append(entry)
    return out


def sync_problems(checks: list[dict[str, Any]]) -> list[str]:
    """A sentence for every train that is not ``ok``."""
    lines = []
    for c in checks:
        if c["status"] == "ok":
            continue
        if c["status"] == "missing":
            what = f"no edges at all (expected ~{c['expected']} {c['unit']})"
        elif c["status"] == "too few":
            what = f"only {c['found']} {c['unit']} (expected ~{c['expected']})"
        else:
            what = "not extracted"
        lines.append(f"{c['kind']} {c['name']} ({c['stream']}): {what}")
    return lines
