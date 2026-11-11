"""Read the OneBox analog export and the matching Blackrock channels, on one clock.

The OneBox side is ``<run>.obx0.obx.analog.mat`` (``extract_analog``), already
stamped with ``onebox_to_blackrock``. The Blackrock side is read straight from
the ``.nsX`` with its own NSP time map, so both land on NSP seconds -- the axis
the ``.nev`` markers use.

Reading only; the comparisons are in ``_sync/compare.py`` and the figures in
``_plots/sync.py``. ``h5py`` and ``neo`` are imported inside the functions, as
everywhere else in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .._sync import compare

__all__ = [
    "OneBoxAnalog",
    "BlackrockAnalog",
    "WindowTrace",
    "find_onebox_analog",
    "load_onebox_analog",
    "read_onebox",
    "iter_onebox",
    "open_blackrock_analog",
    "read_blackrock",
    "iter_blackrock",
    "read_edge_file",
    "get_recording_overlap",
    "detect_onebox_events",
    "detect_blackrock_events",
    "get_window_traces",
    "measure_lags",
    "source_for",
]


@dataclass(frozen=True)
class OneBoxAnalog:
    """What the ``.analog.mat`` says about itself. Samples are read on demand."""

    path: Path
    names: tuple[str, ...]
    #: label -> 0-based row of ``data``
    index: dict[str, int]
    fs: float
    n_samples: int
    v_per_digit: np.ndarray
    timebase: str
    #: ``onebox_to_blackrock``: blackrock = slope * (k / fs) + intercept
    slope: float
    intercept: float

    def blackrock_time(self, k0: int, k1: int) -> np.ndarray:
        return self.slope * (np.arange(k0, k1) / self.fs) + self.intercept


@dataclass
class BlackrockAnalog:
    """An open ``.nsX`` plus the channels the caller named and its NSP clock."""

    path: Path
    reader: Any
    #: label -> :class:`~spikesorting._io.blackrock.BlackrockStream`
    streams: dict[str, Any]
    time_map: Any

    @property
    def segments(self) -> list[tuple[int, int, float, float]]:
        return [
            (s.first_sample, s.n_samples, s.slope, s.intercept)
            for s in self.time_map.segments
        ]


@dataclass
class WindowTrace:
    """One channel over one window, both systems on Blackrock time, ready to draw.

    Normalized to 0..1 (the two systems' units differ) and strided for display;
    ``None`` on the side that does not record this channel.
    """

    name: str
    t_blackrock: np.ndarray | None = None
    y_blackrock: np.ndarray | None = None
    t_onebox: np.ndarray | None = None
    y_onebox: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def find_onebox_analog(export_dir: str | Path) -> Path:
    """The one ``*.analog.mat`` in an export folder."""
    found = sorted(Path(export_dir).glob("*.analog.mat"))
    if not found:
        raise FileNotFoundError(
            f"no *.analog.mat in {export_dir}; run run_exporting_pipeline.py with "
            "export_analog: true in the session"
        )
    if len(found) > 1:
        raise ValueError(f"several *.analog.mat in {export_dir}: {[p.name for p in found]}")
    return found[0]


def _scalar(group: Any, key: str) -> float:
    return float(np.asarray(group[key][()]).ravel()[0])


def _text(group: Any, key: str) -> str:
    return "".join(chr(c) for c in np.asarray(group[key][()]).ravel())


def load_onebox_analog(path: str | Path, require_blackrock: bool = True) -> OneBoxAnalog:
    """The export's labels, rate and map. Raises if it is not on Blackrock time yet."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        a = handle["analog"]
        names = tuple(_text(a, "names").split(","))
        info = OneBoxAnalog(
            path=path,
            names=names,
            index={name: int(_scalar(a["index"], name)) - 1 for name in names},
            fs=_scalar(a, "fs"),
            n_samples=int(_scalar(a, "n_samples")),
            v_per_digit=np.asarray(a["v_per_digit"][()]).ravel(),
            timebase=_text(a, "timebase"),
            slope=_scalar(a, "blackrock_slope"),
            intercept=_scalar(a, "blackrock_intercept"),
        )
    if require_blackrock and info.timebase != "blackrock":
        raise ValueError(
            f"{path.name} is still on the {info.timebase} clock: run time_remapping "
            "(run_exporting_pipeline.py) so onebox_to_blackrock is stamped in"
        )
    return info


def read_onebox(analog: OneBoxAnalog, name: str, k0: int, k1: int) -> np.ndarray:
    """Volts of one channel, samples ``k0:k1``."""
    import h5py

    row = analog.index[name]
    with h5py.File(analog.path, "r") as handle:
        raw = handle["analog/data"][k0:k1, row]
    return raw.astype(np.float64) * analog.v_per_digit[row]


def iter_onebox(
    analog: OneBoxAnalog, name: str, chunk_samples: int = 10_000_000
) -> Iterator[tuple[int, np.ndarray]]:
    """``(start_sample, volts)`` for one channel, a chunk at a time."""
    for start in range(0, analog.n_samples, chunk_samples):
        stop = min(start + chunk_samples, analog.n_samples)
        yield start, read_onebox(analog, name, start, stop)


def open_blackrock_analog(
    path: str | Path,
    channels: dict[str, int],
    gap_tolerance_ms: float | None = None,
    tolerance_s: float | None = 1e-3,
    by: str = "id",
) -> BlackrockAnalog:
    """Open a ``.nsX`` and resolve the named channels (``{label: channel}``).

    ``by="id"`` for a channel id (the ``.ns5`` sync channels), ``by="position"``
    for a 0-based place in the file (``blackrock.analog_channels``). What each
    position resolved to is logged, so a wrong guess shows on the first run.

    The NSP time map is measured from this file, as ``extract_sync`` does for the
    sync file: each ``.nsX`` is its own crystal.
    """
    import logging

    from . import blackrock

    if by not in ("id", "position"):
        raise ValueError(f"by must be 'id' or 'position', got {by!r}")
    path = Path(path)
    reader = blackrock.open_reader(path, gap_tolerance_ms=gap_tolerance_ms)
    lookup = blackrock.stream_for_channel if by == "id" else blackrock.stream_for_position
    streams = {name: lookup(reader, ch) for name, ch in channels.items()}
    for name, stream in streams.items():
        logging.getLogger(__name__).info(
            "%s: %s -> channel id %s (%s), %g Hz",
            path.name, name, stream.channel_id, stream.channel_name, stream.fs,
        )
    stream_index = next(iter(streams.values())).stream_index if streams else 0
    time_map = blackrock.nsp_time_map(
        reader, blackrock.nsx_number(path), stream_index=stream_index, tolerance_s=tolerance_s
    )
    return BlackrockAnalog(path=path, reader=reader, streams=streams, time_map=time_map)


def read_blackrock(br: BlackrockAnalog, name: str, s0: int, s1: int) -> np.ndarray:
    """One channel, samples ``s0:s1``, in the file's physical units."""
    from . import blackrock

    return blackrock.read_channel(br.reader, br.streams[name], s0, s1)


def iter_blackrock(
    br: BlackrockAnalog, name: str, chunk_samples: int = 30_000_000
) -> Iterator[tuple[int, np.ndarray]]:
    from . import blackrock

    return blackrock.iter_channel(br.reader, br.streams[name], chunk_samples)


def read_edge_file(path: str | Path) -> np.ndarray:
    """An edge file: seconds, one per line (empty array for an empty file)."""
    return np.loadtxt(path, ndmin=1, dtype=np.float64)


def get_recording_overlap(
    onebox: OneBoxAnalog, blackrock: BlackrockAnalog | list[BlackrockAnalog] | None
) -> tuple[float, float]:
    """``(start, stop)`` of Blackrock time every given file was recording."""
    start = onebox.intercept
    stop = onebox.slope * (onebox.n_samples / onebox.fs) + onebox.intercept
    sources = blackrock if isinstance(blackrock, list) else [blackrock]
    for source in sources:
        if source is None:
            continue
        _, _, _, intercept = source.segments[0]
        _, n, slope, last_intercept = source.segments[-1]
        start = max(start, intercept)
        stop = min(stop, last_intercept + slope * n)
    return start, stop


def detect_onebox_events(
    onebox: OneBoxAnalog, name: str, threshold: float | None = None, probe_s: float = 300.0
) -> tuple[np.ndarray, float]:
    """Rising edges of one OneBox channel in OneBox seconds, and the threshold used.

    With no threshold, it is set midway between the 1st and 99th percentile of
    the first ``probe_s`` seconds -- fine for a two-level signal like a
    photodiode, provided it changes in that stretch.
    """
    from .._sync import edges

    if threshold is None:
        k1 = min(onebox.n_samples, int(probe_s * onebox.fs))
        threshold = compare.auto_threshold(read_onebox(onebox, name, 0, k1))
    return edges.stream_pulse_times(iter_onebox(onebox, name), threshold, onebox.fs), threshold


def detect_blackrock_events(
    br: BlackrockAnalog, name: str, threshold: float | None = None, probe_s: float = 300.0
) -> tuple[np.ndarray, float]:
    """Rising edges of one Blackrock channel on the NSP clock, and the threshold used."""
    from .._sync import edges

    stream = br.streams[name]
    if threshold is None:
        s1 = min(stream.n_samples, int(probe_s * stream.fs))
        threshold = compare.auto_threshold(read_blackrock(br, name, 0, s1))
    seconds = edges.stream_pulse_times(iter_blackrock(br, name), threshold, stream.fs)
    # Detection counts at the nominal rate; the NSP map supplies the real one.
    samples = np.rint(seconds * stream.fs).astype(np.int64)
    return br.time_map.apply(samples), threshold


def source_for(
    blackrock: BlackrockAnalog | list[BlackrockAnalog] | None, name: str
) -> BlackrockAnalog | None:
    """The Blackrock file that records ``name``, among one or several."""
    sources = blackrock if isinstance(blackrock, list) else [blackrock]
    return next((b for b in sources if b is not None and name in b.streams), None)


def get_window_traces(
    onebox: OneBoxAnalog,
    blackrock: BlackrockAnalog | list[BlackrockAnalog] | None,
    t0_s: float,
    t1_s: float,
    names: list[str] | None = None,
    max_points: int = 20_000,
    onebox_offset_s: float | None = None,
) -> list[WindowTrace]:
    """Every channel over ``[t0_s, t1_s)`` of Blackrock time, ready to overlay.

    ``onebox_offset_s`` puts the OneBox on the *unsynchronized* axis instead --
    ``k / fs + offset`` -- for drawing what the recording looks like before the
    map is applied.
    """
    names = list(names) if names is not None else list(onebox.names)
    traces = []
    for name in names:
        trace = WindowTrace(name=name)
        if name in onebox.index:
            if onebox_offset_s is None:
                k0, k1 = compare.onebox_sample_range(
                    t0_s, t1_s, onebox.fs, onebox.slope, onebox.intercept, onebox.n_samples
                )
                t = onebox.blackrock_time(k0, k1)
            else:
                k0 = max(int(np.floor((t0_s - onebox_offset_s) * onebox.fs)), 0)
                k1 = min(max(int(np.ceil((t1_s - onebox_offset_s) * onebox.fs)), 0), onebox.n_samples)
                t = np.arange(k0, k1) / onebox.fs + onebox_offset_s
            trace.t_onebox, trace.y_onebox = compare.display_envelope(
                t, compare.normalize_trace(read_onebox(onebox, name, k0, k1)), max_points
            )
        source = source_for(blackrock, name)
        if source is not None:
            s0, s1 = compare.blackrock_sample_range(t0_s, t1_s, source.segments)
            trace.t_blackrock, trace.y_blackrock = compare.display_envelope(
                source.time_map.apply(np.arange(s0, s1)),
                compare.normalize_trace(read_blackrock(source, name, s0, s1)),
                max_points,
            )
        traces.append(trace)
    return traces


def measure_lags(
    onebox: OneBoxAnalog,
    blackrock: BlackrockAnalog,
    name: str,
    n_windows: int = 60,
    window_s: float = 10.0,
    max_lag_s: float = 0.02,
    min_corr: float = 0.8,
    differentiate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(window centres, lags, peak correlations)`` of one continuous channel.

    ``n_windows`` windows spread evenly over the overlap -- enough to see a
    trend across the session without reading all of it. In each, the OneBox
    trace (on Blackrock time, through the stamped map) is interpolated onto
    Blackrock's own samples and cross-correlated with them. Windows whose peak
    correlation is below ``min_corr`` -- no movement to lock onto -- are dropped.
    A positive lag means the OneBox copy is later.

    ``differentiate`` correlates the sample-to-sample change instead -- for a
    search over seconds: an eye trace wanders slowly, so its own correlation
    peak is broad, while its changes (saccades) peak sharply. Clipped,
    so one blink cannot own the correlation.
    """
    start, stop = get_recording_overlap(onebox, blackrock)
    stream = blackrock.streams[name]
    span = stop - start - window_s - 2 * max_lag_s
    if span <= 0:
        raise ValueError(f"{name}: the overlap is shorter than one {window_s:g} s window")
    centres = start + max_lag_s + window_s / 2 + np.linspace(0.0, span, n_windows)

    kept_t, kept_lag, kept_corr = [], [], []
    for centre in centres:
        t0, t1 = centre - window_s / 2, centre + window_s / 2
        s0, s1 = compare.blackrock_sample_range(t0, t1, blackrock.segments)
        if s1 - s0 < 3:
            continue
        t_br = blackrock.time_map.apply(np.arange(s0, s1))
        y_br = read_blackrock(blackrock, name, s0, s1)
        k0, k1 = compare.onebox_sample_range(
            t0 - 2 * max_lag_s, t1 + 2 * max_lag_s,
            onebox.fs, onebox.slope, onebox.intercept, onebox.n_samples,
        )
        if k1 - k0 < 3:
            continue
        y_obx = np.interp(t_br, onebox.blackrock_time(k0, k1), read_onebox(onebox, name, k0, k1))
        if differentiate:
            y_br, y_obx = compare.clipped_diff(y_br), compare.clipped_diff(y_obx)
        lag, corr = compare.xcorr_lag(y_br, y_obx, stream.fs, max_lag_s)
        if np.isfinite(lag) and corr >= min_corr:
            kept_t.append(centre)
            kept_lag.append(lag)
            kept_corr.append(corr)
    return np.asarray(kept_t), np.asarray(kept_lag), np.asarray(kept_corr)
