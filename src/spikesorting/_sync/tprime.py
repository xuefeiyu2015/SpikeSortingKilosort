"""TPrime wrapper: fine alignment onto the Blackrock timebase (step 8.3).

TPrime maps event times from one stream onto another using the two streams' 1 Hz
sync edge files. Here the ``-tostream`` reference is *Blackrock*, whose edges were
produced by the NumPy detector rather than CatGT -- TPrime cannot tell, since both
write the same seconds-per-line format.

Usage mirrors the SpikeGLX documentation::

    TPrime -syncperiod=1.0 \\
           -tostream=blackrock_1hz.txt \\
           -fromstream=1,npx_1hz.txt \\
           -events=1,spike_seconds.npy,spike_seconds_blackrock.npy

Note the unit change: Kilosort writes spike times as **sample indices**, and
TPrime requires **seconds**. :func:`spike_times_to_seconds` does that conversion;
skipping it is the single easiest way to produce a confidently wrong alignment.

Reference: https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TPrimeEvent",
    "tprime_executable",
    "build_tprime_args",
    "run_tprime",
    "spike_times_to_seconds",
    "seconds_to_spike_times",
]


@dataclass(frozen=True)
class TPrimeEvent:
    """One ``-events=stream_id,input,output`` mapping request."""

    stream_id: int
    input_path: Path
    output_path: Path

    def to_flag(self) -> str:
        return f"-events={self.stream_id},{self.input_path},{self.output_path}"


def tprime_executable(tprime_dir: str | Path) -> Path:
    """Locate the TPrime binary or its launcher script inside ``tprime_dir``."""
    tprime_dir = Path(tprime_dir)
    for name in ("TPrime.exe", "TPrime", "runit.bat", "runit.sh"):
        candidate = tprime_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no TPrime executable or runit script in {tprime_dir}")


def build_tprime_args(
    to_stream: str | Path,
    from_streams: dict[int, str | Path],
    events: list[TPrimeEvent],
    sync_period_s: float = 1.0,
) -> list[str]:
    """Assemble the TPrime argument list.

    ``to_stream`` is the reference edge file (Blackrock). ``from_streams`` maps a
    stream id to that stream's edge file; the same ids are used by ``events``.
    """
    if not events:
        raise ValueError("no events requested; TPrime would do nothing")
    unknown = {event.stream_id for event in events} - set(from_streams)
    if unknown:
        raise ValueError(f"events reference stream ids with no -fromstream: {sorted(unknown)}")

    args = [f"-syncperiod={sync_period_s}", f"-tostream={Path(to_stream)}"]
    for stream_id, path in sorted(from_streams.items()):
        args.append(f"-fromstream={stream_id},{Path(path)}")
    args.extend(event.to_flag() for event in events)
    return args


def run_tprime(
    tprime_dir: str | Path,
    args: list[str],
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess:
    """Run TPrime, raising with its own output on failure."""
    executable = tprime_executable(tprime_dir)
    command = [str(executable), *args]
    result = subprocess.run(
        command,
        cwd=str(Path(tprime_dir)),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TPrime failed (exit {code})\ncommand: {cmd}\nstdout:\n{out}\nstderr:\n{err}".format(
                code=result.returncode,
                cmd=" ".join(command),
                out=result.stdout,
                err=result.stderr,
            )
        )
    return result


def spike_times_to_seconds(spike_times: np.ndarray, fs: float) -> np.ndarray:
    """Kilosort sample indices -> seconds. Pure.

    Required before handing spike times to TPrime.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return np.asarray(spike_times, dtype=np.float64) / float(fs)


def seconds_to_spike_times(times_s: np.ndarray, fs: float) -> np.ndarray:
    """Seconds -> sample indices, for writing aligned times back in sample units."""
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return np.rint(np.asarray(times_s, dtype=np.float64) * float(fs)).astype(np.int64)
