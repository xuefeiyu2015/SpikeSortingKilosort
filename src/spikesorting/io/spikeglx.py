"""SpikeGLX ``.bin`` / ``.meta`` access.

NumPy only -- no SpikeInterface, no neo, no torch -- so this module works on any
machine, which is what lets the sync-extraction fallback run on the HPC.

SpikeGLX writes sample-interleaved int16: sample 0 channels 0..N-1, then sample 1,
and so on. The last channel of an imec or obx stream is the SY word, whose bit 6
carries the SMA1 1 Hz square wave used for fine alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "StreamInfo",
    "read_meta",
    "meta_path_for",
    "stream_info",
    "memmap_stream",
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


def memmap_stream(info: StreamInfo) -> np.memmap:
    """Read-only ``(n_samples, n_chan)`` int16 view of the binary."""
    return np.memmap(info.path, dtype=np.int16, mode="r", shape=(info.n_samples, info.n_chan))


def _resolve_channel(info: StreamInfo, channel: int) -> int:
    """Allow negative indexing, so ``-1`` means the SY word like CatGT's word=-1."""
    index = channel if channel >= 0 else info.n_chan + channel
    if not 0 <= index < info.n_chan:
        raise IndexError(f"channel {channel} out of range for {info.n_chan}-channel stream")
    return index


def read_channel(
    info: StreamInfo, channel: int, start: int = 0, stop: int | None = None
) -> np.ndarray:
    """Raw int16 samples of one channel. ``channel=-1`` is the last (SY) word."""
    index = _resolve_channel(info, channel)
    stop = info.n_samples if stop is None else min(stop, info.n_samples)
    data = memmap_stream(info)
    return np.ascontiguousarray(data[start:stop, index])


def iter_channel(
    info: StreamInfo, channel: int, chunk_samples: int = 30_000_000
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream one channel in chunks, yielding ``(start_sample, samples)``.

    Keeps memory bounded on hour-long recordings. The default chunk is ~1000 s of
    a 30 kHz channel (60 MB as int16).
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    for start in range(0, info.n_samples, chunk_samples):
        stop = min(start + chunk_samples, info.n_samples)
        yield start, read_channel(info, channel, start, stop)


def read_sy_word(info: StreamInfo, start: int = 0, stop: int | None = None) -> np.ndarray:
    """The SY word as uint16, ready for :func:`~spikesorting.sync.edges.digital_bit_signal`."""
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
