"""SpikeGLX ``.bin`` / ``.meta`` access.

NumPy only -- no SpikeInterface, no neo, no torch -- so this module works on any
machine, which is what lets the sync-extraction fallback run on the HPC.

SpikeGLX writes sample-interleaved int16: sample 0 channels 0..N-1, then sample 1,
and so on. The last channel of an imec or obx stream is the SY word, whose bit 6
carries the SMA1 1 Hz square wave used for fine alignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Re-exported: the probe is not SpikeGLX-specific (a Blackrock .ns6 stalls the
# same way) but every caller here reaches for it through this module.
from .reachable import ReadProbe, check_reachable, probe_read  # noqa: F401

__all__ = [
    "StreamInfo",
    "read_meta",
    "meta_path_for",
    "stream_info",
    "ReadProbe",
    "probe_read",
    "check_reachable",
    "memmap_stream",
    "DEFAULT_CHUNK_BYTES",
    "read_channel",
    "iter_channel",
    "read_sy_word",
    "raw_to_volts",
    "RunLayout",
    "run_layout",
    "parse_geom_map",
    "parse_geom_header",
    "export_lfp",
]

#: Sample-rate meta key per stream type.
_RATE_KEYS = {"imec": "imSampRate", "nidq": "niSampRate", "obx": "obSampRate"}
#: (range_max_key, max_int_key) per stream type, for raw -> volts conversion.
_SCALE_KEYS = {
    "imec": ("imAiRangeMax", "imMaxInt"),
    "nidq": ("niAiRangeMax", "niMaxInt"),
    "obx": ("obAiRangeMax", "obMaxInt"),
}


def meta_path_for(bin_path: str | Path) -> Path:
    """``foo.imec0.ap.bin`` -> ``foo.imec0.ap.meta``."""
    return Path(bin_path).with_suffix(".meta")


def read_meta(path: str | Path) -> dict[str, str]:
    """Parse a SpikeGLX ``.meta`` file into a plain dict.

    Accepts either the ``.meta`` path or the matching ``.bin`` path. Keys keep
    their leading ``~`` where SpikeGLX uses one (``~snsGeomMap`` etc.).
    """
    path = Path(path)
    if path.suffix == ".bin":
        path = meta_path_for(path)
    if not path.exists():
        raise FileNotFoundError(f"SpikeGLX meta file not found: {path}")

    meta: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


@dataclass(frozen=True)
class StreamInfo:
    """Everything needed to read one SpikeGLX binary."""

    path: Path
    meta: dict[str, str]
    stream_type: str  # "imec" | "nidq" | "obx"
    n_chan: int
    fs: float
    n_samples: int
    #: Index of the SY (sync) word, or None if the stream has none.
    sy_index: int | None

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def is_lfp(self) -> bool:
        return ".lf." in self.path.name


def _stream_type(meta: dict[str, str], path: Path) -> str:
    declared = meta.get("typeThis", "").strip().lower()
    if declared in _RATE_KEYS:
        return declared
    name = path.name.lower()
    if ".obx" in name:
        return "obx"
    if ".nidq" in name:
        return "nidq"
    return "imec"


def _sy_index(meta: dict[str, str], stream_type: str, n_chan: int) -> int | None:
    """Index of the SY word.

    ``snsApLfSy``/``snsXaDwSy`` give the per-type channel counts; when present the
    SY word is the trailing one. imec and obx streams always carry a SY word;
    nidq streams may not.
    """
    for key in ("snsApLfSy", "snsLfSy", "snsXaDwSy"):
        if key in meta:
            try:
                counts = [int(x) for x in meta[key].split(",")]
            except ValueError:
                continue
            if counts and counts[-1] > 0:
                return n_chan - 1
            return None
    return n_chan - 1 if stream_type in ("imec", "obx") else None


def stream_info(
    bin_path: str | Path, n_chan: int | None = None, fs: float | None = None
) -> StreamInfo:
    """Read the meta beside ``bin_path`` and derive the stream's shape.

    ``n_chan`` and ``fs`` override the meta, and stand in for it entirely when no
    ``.meta`` exists beside the binary.
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"SpikeGLX binary not found: {bin_path}")

    meta_file = meta_path_for(bin_path)
    if meta_file.exists():
        meta = read_meta(meta_file)
    elif n_chan is not None and fs is not None:
        meta = {}
    else:
        raise FileNotFoundError(
            f"no meta file at {meta_file}; pass n_chan and fs explicitly for a bare binary"
        )

    stype = _stream_type(meta, bin_path)

    if n_chan is None:
        if "nSavedChans" not in meta:
            raise ValueError(f"{meta_file} has no nSavedChans key")
        n_chan = int(meta["nSavedChans"])

    if fs is None:
        rate_key = _RATE_KEYS[stype]
        if rate_key not in meta:
            raise ValueError(f"{meta_file} has no {rate_key} key")
        fs = float(meta[rate_key])

    # Trust the file on disk over fileSizeBytes: the meta can lag a truncated file.
    n_samples = bin_path.stat().st_size // (2 * n_chan)

    return StreamInfo(
        path=bin_path,
        meta=meta,
        stream_type=stype,
        n_chan=n_chan,
        fs=float(fs),
        n_samples=n_samples,
        sy_index=_sy_index(meta, stype, n_chan),
    )


def memmap_stream(info: StreamInfo) -> np.memmap:
    """Read-only ``(n_samples, n_chan)`` int16 view of the binary."""
    return np.memmap(info.path, dtype=np.int16, mode="r", shape=(info.n_samples, info.n_chan))


def _resolve_channel(info: StreamInfo, channel: int) -> int:
    """Allow negative indexing, so ``-1`` means the SY word like CatGT's word=-1."""
    index = channel if channel >= 0 else info.n_chan + channel
    if not 0 <= index < info.n_chan:
        raise IndexError(f"channel {channel} out of range for {info.n_chan}-channel stream")
    return index


#: Bytes of *file* pulled per read. Sized for the network share, not for the
#: channel: one row is n_chan samples, so reading any single channel costs the
#: whole file whatever we do, and the only choice is how it is fetched.
DEFAULT_CHUNK_BYTES = 64 << 20


def read_channel(
    info: StreamInfo, channel: int, start: int = 0, stop: int | None = None
) -> np.ndarray:
    """Raw int16 samples of one channel. ``channel=-1`` is the last (SY) word.

    Reads sequentially rather than slicing a memmap. The bytes transferred are the
    same -- the channels are interleaved, so one column costs every row -- but a
    memmap column slice fetches them as scattered page faults, which over SMB is
    minutes for a file a sequential pass reads in seconds. It also fails
    differently: a page fault on a dropped share raises SIGBUS and kills the
    process, where a failed ``read`` is an ``OSError`` a caller can catch.
    """
    index = _resolve_channel(info, channel)
    stop = info.n_samples if stop is None else min(stop, info.n_samples)
    if stop <= start:
        return np.empty(0, dtype=np.int16)

    row_bytes = info.n_chan * 2
    with open(info.path, "rb") as handle:
        handle.seek(start * row_bytes)
        raw = handle.read((stop - start) * row_bytes)
    return _column(raw, info.n_chan, index)


def _column(raw: bytes, n_chan: int, index: int) -> np.ndarray:
    """One channel out of a block of interleaved rows. Pure."""
    rows = len(raw) // (n_chan * 2)
    block = np.frombuffer(raw, dtype=np.int16, count=rows * n_chan)
    return np.ascontiguousarray(block.reshape(rows, n_chan)[:, index])


def iter_channel(
    info: StreamInfo, channel: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream one channel, yielding ``(start_sample, samples)``.

    ``chunk_bytes`` is measured in bytes *of the file*, which is what governs both
    the time each step takes and how often the loop comes back up into Python --
    so a long read reports progress and answers Ctrl-C between chunks. Sized in
    samples of the channel instead, the default used to be one 23 GB step for a
    385-channel stream: no progress, and uninterruptible.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    index = _resolve_channel(info, channel)
    row_bytes = info.n_chan * 2
    rows = max(1, chunk_bytes // row_bytes)

    # One open for the whole pass, and no seeks: on a share an open costs
    # latency of its own -- seconds, on a bad one -- and reopening per chunk
    # pays it once per chunk for nothing.
    with open(info.path, "rb") as handle:
        start = 0
        while start < info.n_samples:
            raw = handle.read(min(rows, info.n_samples - start) * row_bytes)
            got = len(raw) // row_bytes
            if got == 0:
                break
            yield start, _column(raw, info.n_chan, index)
            start += got


def read_sy_word(info: StreamInfo, start: int = 0, stop: int | None = None) -> np.ndarray:
    """The SY word as uint16, ready for :func:`~spikesorting._sync.edges.digital_bit_signal`."""
    if info.sy_index is None:
        raise ValueError(f"{info.path.name} has no SY word")
    return read_channel(info, info.sy_index, start, stop).view(np.uint16)


def raw_to_volts(info: StreamInfo, raw: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Convert raw int16 to volts using the stream's range/max-int meta keys.

    ``gain`` is 1 for auxiliary analog inputs such as the OneBox XA channels that
    carry the 14 s coded burst. Neural channels need their per-channel gain from
    ``~imroTbl``, which this function does not read.
    """
    range_key, max_key = _SCALE_KEYS[info.stream_type]
    if range_key not in info.meta or max_key not in info.meta:
        raise ValueError(f"{info.path.name} meta lacks {range_key}/{max_key}; cannot scale to volts")
    ai_range = float(info.meta[range_key])
    max_int = float(info.meta[max_key])
    return np.asarray(raw, dtype=np.float64) * (ai_range / max_int) / gain


#: A SpikeGLX AP/LF filename: ``<run>_g<gate>_t<trigger>.imec<probe>.ap.bin``.
#: ``<run>`` is greedy so a run name that itself contains ``_t1229`` still gives
#: up only the *last* ``_g<n>_t<n>`` -- which is the one SpikeGLX appended.
#: The trigger is ``cat`` in CatGT output, so it is not restricted to digits.
_RUN_FILENAME = re.compile(
    r"^(?P<run>.+)_g(?P<gate>\d+)_t(?P<trigger>\w+)\.imec(?P<probe>\d+)\.(ap|lf)\.bin$"
)


@dataclass(frozen=True)
class RunLayout:
    """The SpikeGLX run a binary belongs to, recovered from its own path.

    A session names one binary and nothing else, but CatGT is driven by
    ``-dir``/``-run``/``-g``/``-t``/``-prb`` and the OneBox burst lives in a
    separate ``.obx`` file. Both are recoverable, because SpikeGLX's layout is
    fixed::

        <dir>/<run>_g<gate>/<run>_g<gate>_imec<probe>/<run>_g<gate>_t<trig>.imec<probe>.ap.bin

    so nothing has to be restated in the session file. A binary that does not
    follow the convention -- the Kilosort demo file, a hand-made extract -- has no
    layout, :func:`run_layout` returns ``None``, and the callers degrade to what
    the binary alone supports.
    """

    #: CatGT's ``-dir``: the directory *containing* the gate folder.
    directory: Path
    run_name: str
    gate: int
    #: ``0`` for raw SpikeGLX output, ``"cat"`` for CatGT's own.
    trigger: int | str
    probe: int
    #: The ``<run>_g<gate>`` folder itself, which holds any ``.obx``.
    gate_dir: Path

    def sibling(self, suffix: str) -> Path | None:
        """The matching ``lf``/``ap`` binary for this run, if it is on disk.

        Looks beside the AP file first, then in the gate folder, since SpikeGLX
        writes the per-probe subfolder only when asked to.
        """
        name = f"{self.run_name}_g{self.gate}_t{self.trigger}.imec{self.probe}.{suffix}.bin"
        for directory in (self.gate_dir / f"{self.run_name}_g{self.gate}_imec{self.probe}",
                          self.gate_dir):
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    @property
    def obx(self) -> Path | None:
        """The OneBox binary carrying the 14 s coded burst, if the run has one.

        One per *run*, not per probe: the burst comes from the OneBox, so both
        probes of a run read the same file.
        """
        matches = sorted(self.gate_dir.glob(f"*_t{self.trigger}.obx*.bin"))
        return matches[0] if matches else None


def run_layout(bin_path: str | Path) -> RunLayout | None:
    """Recover the SpikeGLX run around ``bin_path``, or None if it is not one.

    Read off the *filename*, not the directory tree, so it works whether or not
    the per-probe subfolder is present and whether or not the file has been moved
    -- the ``-dir`` it reports is simply wrong in the latter case, and CatGT says
    so rather than silently extracting the wrong run.
    """
    bin_path = Path(bin_path)
    match = _RUN_FILENAME.match(bin_path.name)
    if match is None:
        return None

    run_name = match["run"]
    gate = int(match["gate"])
    trigger: int | str = match["trigger"]
    if isinstance(trigger, str) and trigger.isdigit():
        trigger = int(trigger)

    # The gate folder is whichever ancestor is named <run>_g<gate>. Directly the
    # parent when SpikeGLX wrote no per-probe subfolder, its parent when it did.
    gate_name = f"{run_name}_g{gate}"
    gate_dir = bin_path.parent
    if gate_dir.name != gate_name and gate_dir.parent.name == gate_name:
        gate_dir = gate_dir.parent

    return RunLayout(
        directory=gate_dir.parent,
        run_name=run_name,
        gate=gate,
        trigger=trigger,
        probe=int(match["probe"]),
        gate_dir=gate_dir,
    )


def parse_geom_header(meta: dict[str, str]) -> dict[str, float | str] | None:
    """Parse the header entry of ``~snsGeomMap``.

    The header is ``(part_number, n_shank, shank_pitch_um, shank_width_um)``.
    ``shank_pitch`` is what turns per-shank coordinates into absolute ones on a
    multi-shank probe.
    """
    raw = meta.get("~snsGeomMap")
    if not raw:
        return None
    head = raw.split(")", 1)[0].lstrip("(")
    fields = head.split(",")
    if len(fields) < 4:
        return None
    try:
        return {
            "part_number": fields[0],
            "n_shank": int(fields[1]),
            "shank_pitch_um": float(fields[2]),
            "shank_width_um": float(fields[3]),
        }
    except ValueError:
        return None


def parse_geom_map(meta: dict[str, str]) -> np.ndarray | None:
    """Parse ``~snsGeomMap`` into an ``(n_chan, 4)`` array of ``[shank, x, z, used]``.

    Returns None when the run predates ``~snsGeomMap`` (older SpikeGLX wrote
    ``~snsShankMap`` instead); callers should fall back to that or to an explicit
    probe file.

    Coordinates are micrometres **relative to each shank's own origin**, so on a
    multi-shank probe two shanks report identical (x, z). Use
    :func:`parse_geom_header` for the shank pitch needed to make them absolute.
    """
    raw = meta.get("~snsGeomMap")
    if not raw:
        return None
    entries = [chunk for chunk in raw.split(")") if "(" in chunk]
    rows: list[list[float]] = []
    for entry in entries[1:]:  # entries[0] is the header, e.g. "(NP1000,1,0,70"
        body = entry.split("(", 1)[1]
        parts = body.split(":")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float64) if rows else None


def export_lfp(
    lf_bin: str | Path,
    out_dir: str | Path,
    decimate: int = 1,
    chunk_samples: int = 10_000_000,
) -> Path:
    """Write the LF band to ``out_dir`` as one MATLAB v7.3 ``.mat``.

    The lab reads these from MATLAB and from ``jlab_loader``, so the product is a
    struct in their convention rather than a ``.npy`` plus a sidecar to
    reassemble by hand. **MATLAB sees ``data`` as nChan x nSamples**, which is
    openNSx's orientation; on disk that is an HDF5 dataset of ``(n_samples,
    n_chan)``, because MATLAB reverses dimensions -- see :mod:`.matlab`.

    Streamed, never held: an hour of 385-channel LF band is ~7 GB, so the file is
    filled a chunk at a time and gzipped on the way in.

    Samples stay **int16**, as recorded. Converting to microvolts needs a
    per-channel gain from ``~imroTbl`` that nothing here parses, so
    ``uv_per_digit`` is NaN rather than invented: a NaN propagates loudly through
    any scaling, where a fabricated 1.0 would quietly produce plausible, wrong
    numbers. ``ai_range_max`` and ``max_int`` are carried so the conversion can be
    finished downstream.

    This is **extraction**, so it runs beside the other extraction and needs
    nothing but the recording -- on the rig, straight after the session. The axis
    it writes is therefore this stream's own, and ``timebase`` says ``stream``.
    :func:`spikesorting.pipeline.stamp_lfp_timebase` fills in the Blackrock axis
    later, once ``time_remapping`` has fitted the map, by rewriting a handful of
    small fields rather than these gigabytes.

    Blackrock LFPs are already saved separately by Central, so this is for
    Neuropixels ``.lf.bin`` streams.
    """
    from .matlab import Chunked, write_mat

    info = stream_info(lf_bin)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{info.path.stem}.lfp.mat"

    if decimate < 1:
        raise ValueError("decimate must be >= 1")

    n_out = (info.n_samples + decimate - 1) // decimate
    n_chan = info.n_chan
    source = memmap_stream(info)

    def fill(dataset) -> None:
        written = 0
        for start in range(0, info.n_samples, chunk_samples):
            stop = min(start + chunk_samples, info.n_samples)
            # Keep the decimation grid anchored to sample 0 across chunk boundaries.
            first = start if start % decimate == 0 else start + (decimate - start % decimate)
            if first >= stop:
                continue
            block = source[first:stop:decimate, :]
            dataset[written : written + block.shape[0], :] = block
            written += block.shape[0]
        if written != n_out:
            raise RuntimeError(
                f"wrote {written} samples but the stream implies {n_out}; the "
                "decimation grid and the file length disagree"
            )

    range_key, max_key = _SCALE_KEYS[info.stream_type]

    fs_out = info.fs / decimate
    nan = float("nan")

    write_mat(
        out_path,
        {
            "lfp": {
                "data": Chunked(
                    shape=(n_out, n_chan),
                    dtype=np.int16,
                    fill=fill,
                    chunks=(min(65536, n_out), n_chan),
                ),
                "fs": fs_out,
                "t0": 0.0,
                "timebase": "stream",
                "fs_blackrock": nan,
                "t0_blackrock": nan,
                "blackrock_slope": nan,
                "blackrock_intercept": nan,
                "decimate": float(decimate),
                "n_samples": float(n_out),
                "n_chan": float(n_chan),
                "channel_ids": np.arange(n_chan, dtype=np.float64),
                "uv_per_digit": np.full(n_chan, np.nan),
                "ai_range_max": float(info.meta.get(range_key, "nan")),
                "max_int": float(info.meta.get(max_key, "nan")),
                "sy_index": float(info.sy_index if info.sy_index is not None else -1),
                "source": str(info.path),
                "note": (
                    "data is int16, nChan x nSamples in MATLAB; channel_ids are "
                    "0-based rows of the source binary. Sample k is at "
                    "t0 + k/fs on this stream's own clock, and at "
                    "t0_blackrock + k/fs_blackrock on Blackrock's when timebase "
                    "is 'blackrock' (NaN when no time map had been fitted yet). "
                    "volts = value * ai_range_max / max_int / gain, and the "
                    "per-channel gain lives in the meta's ~imroTbl"
                ),
            }
        },
    )
    return out_path
