"""SpikeGLX ``.bin`` / ``.meta`` access.

NumPy only -- no SpikeInterface, no neo, no torch -- so this module works on any
machine, which is what lets the sync-extraction fallback run on the HPC.

SpikeGLX writes sample-interleaved int16: sample 0 channels 0..N-1, then sample 1,
and so on. The last channel of an imec or obx stream is the SY word, whose bit 6
carries the SMA1 1 Hz square wave used for fine alignment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "StreamInfo",
    "read_meta",
    "meta_path_for",
    "stream_info",
    "ReadProbe",
    "probe_read",
    "memmap_stream",
    "DEFAULT_CHUNK_BYTES",
    "read_channel",
    "iter_channel",
    "read_sy_word",
    "raw_to_volts",
    "find_run_files",
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


@dataclass(frozen=True)
class ReadProbe:
    """How fast this file actually reads, measured on a small piece of it.

    Pure once constructed: the arithmetic below is testable without pretending to
    know a disk speed, and nothing here prints.
    """

    path: Path
    #: Size of the whole file.
    total_bytes: int
    #: How much of it was read to time the probe.
    sample_bytes: int
    #: Time for that read alone -- the throughput measurement.
    seconds: float
    #: Time to stat and open, which on a share is latency rather than throughput
    #: and would otherwise be invisible in the rate.
    latency_s: float = 0.0

    @property
    def bytes_per_s(self) -> float:
        return self.sample_bytes / max(self.seconds, 1e-6)

    def estimated_seconds(self, n_bytes: int | None = None) -> float:
        """How long reading ``n_bytes`` (default: the whole file) should take."""
        want = self.total_bytes if n_bytes is None else n_bytes
        return want / self.bytes_per_s

    def summary(self) -> str:
        estimate = self.estimated_seconds()
        span = f"{estimate:.0f} s" if estimate < 120 else f"{estimate / 60:.1f} min"
        latency = f", open took {self.latency_s:.1f} s" if self.latency_s > 0.5 else ""
        return (
            f"{self.total_bytes / 1e9:.2f} GB at "
            f"{self.bytes_per_s / 1e6:.0f} MB/s -- about {span}{latency}"
        )


def probe_read(
    path: str | Path, sample_bytes: int = 4 << 20, time_budget_s: float = 2.0
) -> ReadProbe:
    """Time a small read of ``path``, to say up front what the whole one costs.

    Reading one channel of an AP binary pulls the entire file, which on a network
    share can be many minutes with nothing to show for it. This is the cheap check
    that happens first: ``stat`` fails immediately and by name when the share is
    not mounted, and the timed sample turns "it is still going" into an estimate
    made in a second.

    The sample comes from the **end** of the file, which checks reachability of
    the part a truncated or half-synced copy would be missing, and avoids timing
    a head that something else has already pulled into the page cache. Even so the
    estimate is a hint: a warm cache makes it optimistic, and it assumes the rest
    of the file reads like this piece.

    The sample is read in small blocks and stops at ``time_budget_s``, so the
    check stays cheap on exactly the share it exists to warn about: a slow one
    answers in about two seconds rather than taking a minute to measure how slow
    it is.
    """
    path = Path(path)
    opened = time.perf_counter()
    total = path.stat().st_size          # unmounted share fails here, not in the loop
    want = max(1, min(int(sample_bytes), total))
    handle = open(path, "rb")
    latency = time.perf_counter() - opened

    block = 1 << 18
    read = 0
    start = time.perf_counter()
    try:
        handle.seek(max(0, total - want))
        while read < want:
            piece = handle.read(min(block, want - read))
            if not piece:
                break
            read += len(piece)
            if time.perf_counter() - start >= time_budget_s:
                break
    finally:
        handle.close()
    seconds = time.perf_counter() - start

    return ReadProbe(
        path=path,
        total_bytes=total,
        sample_bytes=read,
        seconds=seconds,
        latency_s=latency,
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


def find_run_files(
    run_dir: str | Path,
    run_name: str,
    gate: int = 0,
    trigger: int | str = 0,
    probe: int = 0,
) -> dict[str, Path | None]:
    """Locate the AP / LF / OBX binaries of a SpikeGLX run.

    Globs rather than assembling exact names, so both the raw layout
    (``run_g0_t0.imec0.ap.bin``) and CatGT output (``run_g0_tcat.imec0.ap.bin``)
    are found, with or without the per-probe subfolder.

    Returns a dict with keys ``ap``, ``lf``, ``obx``; values are None when absent.
    """
    run_dir = Path(run_dir)
    gate_dir = run_dir / f"{run_name}_g{gate}"
    if not gate_dir.exists():
        gate_dir = run_dir  # already pointed at the gate folder

    probe_dir = gate_dir / f"{run_name}_g{gate}_imec{probe}"
    search_dirs = [d for d in (probe_dir, gate_dir) if d.exists()]
    if not search_dirs:
        raise FileNotFoundError(f"no SpikeGLX run folder found under {run_dir} for {run_name}_g{gate}")

    def first_match(pattern: str) -> Path | None:
        for directory in search_dirs:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    tag = f"t{trigger}"
    return {
        "ap": first_match(f"*_{tag}.imec{probe}.ap.bin") or first_match(f"*.imec{probe}.ap.bin"),
        "lf": first_match(f"*_{tag}.imec{probe}.lf.bin") or first_match(f"*.imec{probe}.lf.bin"),
        "obx": first_match(f"*_{tag}.obx*.bin") or first_match("*.obx*.bin"),
    }


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
    """Write the LF band to ``out_dir`` as one int16 ``.npy`` plus a JSON sidecar.

    Channels are columns, so ``lfp[:, k]`` is channel ``k`` -- per-channel LFP
    without paying for hundreds of separate files. The sidecar records sampling
    rate, channel count and the scale factor needed to recover volts.

    Blackrock LFPs are already saved separately by Central, so this is for
    Neuropixels ``.lf.bin`` streams.
    """
    info = stream_info(lf_bin)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{info.path.stem}.lfp.npy"

    if decimate < 1:
        raise ValueError("decimate must be >= 1")

    n_out = (info.n_samples + decimate - 1) // decimate
    n_chan = info.n_chan
    source = memmap_stream(info)
    result = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.int16, shape=(n_out, n_chan)
    )

    written = 0
    for start in range(0, info.n_samples, chunk_samples):
        stop = min(start + chunk_samples, info.n_samples)
        # Keep the decimation grid anchored to sample 0 across chunk boundaries.
        first = start if start % decimate == 0 else start + (decimate - start % decimate)
        if first >= stop:
            continue
        block = source[first:stop:decimate, :]
        result[written : written + block.shape[0], :] = block
        written += block.shape[0]
    result.flush()

    range_key, max_key = _SCALE_KEYS[info.stream_type]
    sidecar = {
        "source": str(info.path),
        "fs": info.fs / decimate,
        "decimate": decimate,
        "n_samples": int(written),
        "n_chan": n_chan,
        "sy_index": info.sy_index,
        "ai_range_max": float(info.meta.get(range_key, "nan")),
        "max_int": float(info.meta.get(max_key, "nan")),
        "note": "int16, channels in columns; volts = value * ai_range_max / max_int / gain",
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)

    return out_path
