"""Blackrock ``.nsX`` access.

Two distinct jobs, deliberately kept apart:

* **Sync channels** (``NSP-*.ns5`` channels 1 and 2) are read with ``neo.rawio``
  one channel at a time, in chunks. A full 30 kHz ``.nsX`` is far too large to
  load, and only two channels are ever needed.
* **Spike data** (``HUB-*.ns6``) is read with SpikeInterface's
  ``read_blackrock``, called in :mod:`spikesorting.pipeline` next to the
  transform that turns it into the binary Kilosort reads. :func:`list_streams`
  here is the companion for choosing ``blackrock.stream_id`` beforehand.

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
    "NspSegment",
    "NspTimeMap",
    "fit_nsp_segment",
    "nsp_time_map",
    "nsx_number",
    "open_reader",
    "stream_for_channel",
    "read_channel",
    "iter_channel",
    "list_streams",
]

_NSX_RE = re.compile(r"\.ns(\d)$", re.IGNORECASE)


def nsx_number(path: str | Path) -> int:
    """``NSP-Athos_001.ns5`` -> ``5``, ``HUB-Athos_001.ns6`` -> ``6``."""
    match = _NSX_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"not a Blackrock .nsX file: {path}")
    return int(match.group(1))


def open_reader(
    path: str | Path, load_nev: bool = False, gap_tolerance_ms: float | None = None
) -> Any:
    """Parsed ``BlackrockRawIO`` for ``path``.

    ``load_nev`` defaults to False: the sync file's ``.nev`` is irrelevant here and
    may be missing, and parsing it is slow.

    ``gap_tolerance_ms`` is neo's, and on a PTP recording it is usually *required*
    to open the file at all: neo raises on any jump larger than two sampling
    periods, and these files carry occasional corrupted packet timestamps. Below
    the tolerance a jump is absorbed; above it the recording is split into
    segments. See :func:`nsp_time_map` for why absorbing them loses no time.
    """
    from neo.rawio import BlackrockRawIO  # imported lazily; see module docstring

    path = Path(path)
    reader = BlackrockRawIO(
        filename=str(path.with_suffix("")),
        nsx_to_load=nsx_number(path),
        load_nev=load_nev,
        gap_tolerance_ms=gap_tolerance_ms,
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
    #: Segment start on the NSP clock, in seconds. Nonzero on every PTP file.
    t_start: float = 0.0


# ---------------------------------------------------------------------------
# Sample index -> NSP clock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NspSegment:
    """One uninterrupted run of samples, and the line through its timestamps."""

    first_sample: int
    n_samples: int
    #: Seconds per sample -- the *measured* rate, not the nominal one.
    slope: float
    #: NSP seconds at ``first_sample``.
    intercept: float
    residual_max_s: float

    def apply(self, samples: np.ndarray) -> np.ndarray:
        return self.slope * (np.asarray(samples, dtype=np.float64) - self.first_sample) + self.intercept


@dataclass(frozen=True)
class NspTimeMap:
    """Sample index in a Blackrock stream -> seconds on the NSP (PTP) clock.

    This is the axis everything else in the lab uses: ``.nev`` event markers, eye
    traces and online spikes are all timestamped on it. Kilosort, by contrast,
    writes plain sample indices, so something has to carry the origin and the
    rate -- and neither is what you would guess:

    * the origin is not zero (~1.5e9 s on a PTP clock), and
    * the rate is not the nominal one. A measured ``.ns6`` ran at 29999.865 Hz
      rather than 30000, which is 51 ms of error by the end of a three-hour
      session.

    Both come from the file's own per-sample timestamps, by least squares rather
    than by reading individual values: PTP files carry occasional corrupted
    timestamps -- jumps of a millisecond, sometimes *backwards* -- which a fit
    rides over and a lookup would return verbatim. They cost real accuracy
    nothing, because they cancel within a few dozen samples: measured across one
    such disturbance, 98 ns of real time was unaccounted for.
    """

    segments: tuple[NspSegment, ...]
    fs_nominal: float
    #: neo's file-spec string, e.g. "3.0".
    spec: str
    #: False when the file carries one timestamp per data block rather than per
    #: sample, in which case the rate could not be measured and is the nominal one.
    per_sample_timestamps: bool
    #: Sizes, in ms, of the timestamp jumps that were absorbed rather than split on.
    gaps_absorbed: tuple[float, ...] = ()

    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Sample indices (into the concatenated stream) -> NSP seconds."""
        samples = np.asarray(samples, dtype=np.int64)
        out = np.empty(samples.shape, dtype=np.float64)
        remaining = np.ones(samples.shape, dtype=bool)
        for segment in self.segments:
            in_seg = remaining & (samples >= segment.first_sample) & (
                samples < segment.first_sample + segment.n_samples
            )
            out[in_seg] = segment.apply(samples[in_seg])
            remaining &= ~in_seg
        if remaining.any():
            # Past the end of the last segment: extrapolate from it rather than
            # emit a NaN, but say so -- this is a spike the sorter placed beyond
            # the samples the file claims to hold.
            last = self.segments[-1]
            out[remaining] = last.apply(samples[remaining])
        return out

    @property
    def drift_ppm(self) -> float:
        """How far the measured rate sits from the nominal one, in ppm."""
        measured = 1.0 / self.segments[0].slope
        return (measured / self.fs_nominal - 1.0) * 1e6

    @property
    def residual_max_s(self) -> float:
        return max(segment.residual_max_s for segment in self.segments)

    def summary(self) -> str:
        rate = 1.0 / self.segments[0].slope
        return (
            f"{len(self.segments)} segment(s), t0 {self.segments[0].intercept:.6f} s, "
            f"{rate:.4f} Hz ({self.drift_ppm:+.1f} ppm), "
            f"residual {self.residual_max_s * 1e6:.0f} us"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": "nsp_seconds = slope * (sample - first_sample) + intercept",
            "fs_nominal": self.fs_nominal,
            "measured_rate_hz": 1.0 / self.segments[0].slope,
            "drift_ppm": self.drift_ppm,
            "residual_max_s": self.residual_max_s,
            "spec": self.spec,
            "per_sample_timestamps": self.per_sample_timestamps,
            "gaps_absorbed_ms": list(self.gaps_absorbed),
            "segments": [
                {
                    "first_sample": s.first_sample,
                    "n_samples": s.n_samples,
                    "slope": s.slope,
                    "intercept": s.intercept,
                    "residual_max_s": s.residual_max_s,
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NspTimeMap":
        return cls(
            segments=tuple(
                NspSegment(
                    first_sample=int(s["first_sample"]),
                    n_samples=int(s["n_samples"]),
                    slope=float(s["slope"]),
                    intercept=float(s["intercept"]),
                    residual_max_s=float(s["residual_max_s"]),
                )
                for s in payload["segments"]
            ),
            fs_nominal=float(payload["fs_nominal"]),
            spec=str(payload.get("spec", "")),
            per_sample_timestamps=bool(payload.get("per_sample_timestamps", True)),
            gaps_absorbed=tuple(payload.get("gaps_absorbed_ms", ())),
        )


def fit_nsp_segment(
    samples: np.ndarray,
    seconds: np.ndarray,
    first_sample: int = 0,
    n_samples: int | None = None,
) -> NspSegment:
    """Least-squares line through ``(sample, second)`` pairs. Pure.

    Separated from the reader so the fit is testable without neo or a recording.

    ``n_samples`` is the segment's real length, which is *not* recoverable from
    ``samples``: those are a subsample, so the last of them is short of the
    segment's end and every sample past it would fall through to the next
    segment's line.
    """
    samples = np.asarray(samples, dtype=np.float64)
    seconds = np.asarray(seconds, dtype=np.float64)
    if samples.size < 2:
        raise ValueError("need at least two timestamps to measure a rate")

    # Fit against an origin at the segment's own start rather than against the
    # PTP epoch. A NSP timestamp is ~1.5e9 s and a session spans a few thousand,
    # so fitting the raw values spends eleven of float64's sixteen digits on a
    # constant and measures the rate with what is left -- enough to shift it by
    # milli-Hz, which is the very drift being measured.
    x = samples - first_sample
    origin = seconds[0]
    slope, offset = np.polyfit(x, seconds - origin, 1)
    intercept = offset + origin
    residual = (seconds - origin) - (slope * x + offset)
    return NspSegment(
        first_sample=int(first_sample),
        n_samples=int(n_samples if n_samples is not None else samples[-1] - first_sample + 1),
        slope=float(slope),
        intercept=float(intercept),
        residual_max_s=float(np.abs(residual).max()),
    )


def nsp_time_map(
    reader: Any,
    nsx_nb: int,
    stream_index: int = 0,
    probes: int = 20_000,
    tolerance_s: float | None = 1e-3,
) -> NspTimeMap:
    """Build the sample -> NSP seconds map for one ``.nsX`` stream.

    ``probes`` evenly spaced timestamps per segment are read rather than all of
    them: the array is memory-mapped and hundreds of millions of entries long,
    and a line needs far fewer. Subsampling also averages the corrupted
    timestamps described on :class:`NspTimeMap`.

    ``tolerance_s`` fails the fit when the residual exceeds it -- a stream that
    is not linear in time must not be exported as though it were. Pass None to
    measure without judging.
    """
    fs = float(reader.get_signal_sampling_rate(stream_index=stream_index))
    spec = str(getattr(reader, "_nsx_spec", {}).get(nsx_nb, ""))
    n_segments = int(reader.segment_count(0))

    segments: list[NspSegment] = []
    per_sample = True
    first_sample = 0
    for seg in range(n_segments):
        n = int(reader.get_signal_size(0, seg, stream_index))
        stamps = np.atleast_1d(reader._nsx_data_header[nsx_nb][seg]["timestamp"])
        t_start = float(reader.get_signal_t_start(0, seg, stream_index))

        if stamps.size == n and n >= 2:
            resolution = float(_timestamp_resolution(reader, nsx_nb))
            step = max(1, n // max(1, probes))
            index = np.arange(0, n, step, dtype=np.int64)
            seconds = stamps[index].astype(np.float64) / resolution
            segments.append(
                fit_nsp_segment(index + first_sample, seconds, first_sample, n_samples=n)
            )
        else:
            # One timestamp for the whole block: the rate cannot be measured from
            # the file, so the nominal one stands and the residual is unknown.
            per_sample = False
            segments.append(
                NspSegment(
                    first_sample=first_sample,
                    n_samples=n,
                    slope=1.0 / fs,
                    intercept=t_start,
                    residual_max_s=float("nan"),
                )
            )
        first_sample += n

    result = NspTimeMap(
        segments=tuple(segments),
        fs_nominal=fs,
        spec=spec,
        per_sample_timestamps=per_sample,
    )
    if tolerance_s is not None and per_sample and result.residual_max_s > tolerance_s:
        raise ValueError(
            f"sample times in this stream are not linear: residual "
            f"{result.residual_max_s * 1e3:.3f} ms exceeds {tolerance_s * 1e3:.3f} ms. "
            "Exporting spike times against this map would be wrong; check the file "
            "for a paused recording (a larger gap_tolerance_ms hides one)."
        )
    return result


def _timestamp_resolution(reader: Any, nsx_nb: int) -> float:
    """Ticks per second for this file's timestamps (1e9 on a PTP recording)."""
    header = reader._nsx_basic_header[nsx_nb]
    if header.dtype.names and "timestamp_resolution" in header.dtype.names:
        return float(header["timestamp_resolution"])
    return 30_000.0  # spec 2.1/2.2/2.3 count in 30 kHz ticks


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
        t_start=float(
            reader.get_signal_t_start(
                block_index=0, seg_index=seg_index, stream_index=stream_index
            )
        ),
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


def list_streams(path: str | Path) -> tuple[list[str], list[str]]:
    """``(stream_names, stream_ids)`` available in a Blackrock file.

    Useful before choosing ``blackrock.stream_id`` in a session config.
    """
    from spikeinterface.extractors import get_neo_streams  # lazy

    names, ids = get_neo_streams("blackrock", Path(path))
    return list(names), list(ids)
