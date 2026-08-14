"""Blackrock ``.nsX`` access.

Two distinct jobs, deliberately kept apart:

* **Sync channels** (``NSP-*.ns5`` channels 1 and 2) are read with ``neo.rawio``
  one channel at a time, in chunks. A full 30 kHz ``.nsX`` is far too large to
  load, and only two channels are ever needed.
* **Spike data** (``HUB-*.ns6``) is read with SpikeInterface, which hands a
  ``BaseRecording`` straight to the preprocessing and sorting stages.

``neo`` and ``spikeinterface`` are imported inside the functions so that importing
this module costs nothing on a machine without them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = [
    "BlackrockStream",
    "nsx_number",
    "open_reader",
    "stream_for_channel",
    "read_channel",
    "iter_channel",
    "read_recording",
]

_NSX_RE = re.compile(r"\.ns(\d)$", re.IGNORECASE)


def nsx_number(path: str | Path) -> int:
    """``NSP-Athos_001.ns5`` -> ``5``, ``HUB-Athos_001.ns6`` -> ``6``."""
    match = _NSX_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"not a Blackrock .nsX file: {path}")
    return int(match.group(1))


def open_reader(path: str | Path, load_nev: bool = False) -> Any:
    """Parsed ``BlackrockRawIO`` for ``path``.

    ``load_nev`` defaults to False: the sync file's ``.nev`` is irrelevant here and
    may be missing, and parsing it is slow.
    """
    from neo.rawio import BlackrockRawIO  # imported lazily; see module docstring

    path = Path(path)
    reader = BlackrockRawIO(
        filename=str(path.with_suffix("")),
        nsx_to_load=nsx_number(path),
        load_nev=load_nev,
    )
    reader.parse_header()
    return reader


@dataclass(frozen=True)
class BlackrockStream:
    """Where one channel lives inside a Blackrock file."""

    channel_id: str
    channel_name: str
    stream_index: int
    fs: float
    n_samples: int
    n_segments: int


def _channel_table(reader: Any) -> np.ndarray:
    return reader.header["signal_channels"]


def stream_for_channel(reader: Any, channel: int | str, seg_index: int = 0) -> BlackrockStream:
    """Resolve a Blackrock channel number to its stream and shape.

    Matches on the channel id first (Blackrock's electrode number, which is what
    ``SynchronizePulse_setup.md`` means by "channel 1" / "channel 2"), then on the
    channel name, then falls back to positional index.
    """
    channels = _channel_table(reader)
    wanted = str(channel)

    index = None
    for i, row in enumerate(channels):
        if str(row["id"]) == wanted:
            index = i
            break
    if index is None:
        for i, row in enumerate(channels):
            if str(row["name"]) == wanted:
                index = i
                break
    if index is None and isinstance(channel, int) and 0 <= channel < len(channels):
        index = channel
    if index is None:
        available = ", ".join(str(row["id"]) for row in channels[:16])
        raise KeyError(f"channel {channel!r} not found; available ids include: {available} ...")

    row = channels[index]
    stream_ids = list(reader.header["signal_streams"]["id"])
    stream_index = stream_ids.index(row["stream_id"]) if row["stream_id"] in stream_ids else 0

    n_segments = reader.segment_count(block_index=0)
    return BlackrockStream(
        channel_id=str(row["id"]),
        channel_name=str(row["name"]),
        stream_index=stream_index,
        fs=float(reader.get_signal_sampling_rate(stream_index=stream_index)),
        n_samples=int(reader.get_signal_size(block_index=0, seg_index=seg_index, stream_index=stream_index)),
        n_segments=int(n_segments),
    )


def _read_chunk(
    reader: Any,
    stream: BlackrockStream,
    start: int,
    stop: int,
    seg_index: int,
    scaled: bool,
) -> np.ndarray:
    raw = reader.get_analogsignal_chunk(
        block_index=0,
        seg_index=seg_index,
        i_start=start,
        i_stop=stop,
        stream_index=stream.stream_index,
        channel_ids=[stream.channel_id],
    )
    if scaled:
        raw = reader.rescale_signal_raw_to_float(
            raw,
            dtype="float64",
            stream_index=stream.stream_index,
            channel_ids=[stream.channel_id],
        )
    return np.asarray(raw).reshape(-1)


def iter_channel(
    reader: Any,
    stream: BlackrockStream,
    chunk_samples: int = 30_000_000,
    seg_index: int = 0,
    scaled: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream one channel in chunks, yielding ``(start_sample, samples)``.

    ``scaled=True`` returns the reader's physical units (microvolts for neural
    channels, volts-scaled for analog inputs, per the file's gain/offset), which
    is what the configured thresholds are expressed in.
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    for start in range(0, stream.n_samples, chunk_samples):
        stop = min(start + chunk_samples, stream.n_samples)
        yield start, _read_chunk(reader, stream, start, stop, seg_index, scaled)


def read_channel(
    reader: Any,
    stream: BlackrockStream,
    start: int = 0,
    stop: int | None = None,
    seg_index: int = 0,
    scaled: bool = True,
) -> np.ndarray:
    """Read one channel in full (or a slice of it)."""
    stop = stream.n_samples if stop is None else min(stop, stream.n_samples)
    return _read_chunk(reader, stream, start, stop, seg_index, scaled)


def read_recording(
    path: str | Path,
    stream_id: str | None = None,
    exclude_channels: tuple[str, ...] = (),
) -> Any:
    """SpikeInterface recording for the Utah array spike file (``HUB-*.ns6``).

    Non-neural channels sharing the file (sync, analog inputs) must be dropped
    before sorting -- pass them as ``exclude_channels``.
    """
    from spikeinterface.extractors import read_blackrock  # lazy; heavy import

    recording = read_blackrock(Path(path), stream_id=stream_id) if stream_id else read_blackrock(Path(path))
    if exclude_channels:
        keep = [c for c in recording.channel_ids if str(c) not in set(exclude_channels)]
        recording = recording.select_channels(keep)
    return recording


def list_streams(path: str | Path) -> tuple[list[str], list[str]]:
    """``(stream_names, stream_ids)`` available in a Blackrock file.

    Useful before choosing ``blackrock.stream_id`` in a session config.
    """
    from spikeinterface.extractors import get_neo_streams  # lazy

    names, ids = get_neo_streams("blackrock", Path(path))
    return list(names), list(ids)
