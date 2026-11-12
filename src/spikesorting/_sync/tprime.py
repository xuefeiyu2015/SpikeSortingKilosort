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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TPrimeEvent",
    "tprime_executable",
    "build_tprime_args",
    "run_tprime",
    "TPrimeTimeout",
    "tprime_timeout_s",
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


class TPrimeTimeout(RuntimeError):
    """TPrime ran past its deadline and was stopped."""


def tprime_timeout_s(n_events: int) -> float:
    """A deadline no healthy run comes near: 10 min plus 1 min per million events.

    TPrime maps millions of events in seconds. One that is still computing
    after this is stuck, as it was for hours on edges 1.8e9 s apart.
    """
    return 600.0 + 60.0 * n_events / 1e6


def run_tprime(
    tprime_dir: str | Path,
    args: list[str],
    timeout_s: float | None = None,
    poll_s: float = 0.5,
) -> subprocess.CompletedProcess:
    """Run TPrime, raising with its own output on failure.

    Waits in short polls rather than one blocking wait, so Ctrl+C reaches
    Python on Windows; whatever ends the wait -- a deadline, Ctrl+C, an error --
    kills TPrime, so no orphan is left holding the output file open. Its
    output goes to a file, not a pipe, so a chatty run cannot fill a pipe and
    stall.
    """
    executable = tprime_executable(tprime_dir)
    command = [str(executable), *args]
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(
            command,
            cwd=str(Path(tprime_dir)),
            # Nothing to read from: TPrime can never sit waiting on a keyboard.
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                try:
                    proc.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    if deadline is not None and time.monotonic() > deadline:
                        raise TPrimeTimeout(
                            "TPrime still running after {t:.0f} s; stopped it\n"
                            "command: {cmd}\n{log}".format(
                                t=timeout_s, cmd=subprocess.list2cmdline(command),
                                log=_log_tail(Path(tprime_dir) / "TPrime.log"),
                            )
                        ) from None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        out.seek(0)
        text = out.read().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            "TPrime failed (exit {code})\ncommand: {cmd}\noutput:\n{out}\n{log}".format(
                code=proc.returncode,
                cmd=subprocess.list2cmdline(command),
                out=text,
                log=_log_tail(Path(tprime_dir) / "TPrime.log"),
            )
        )
    return subprocess.CompletedProcess(command, proc.returncode, text, "")


def _log_tail(path: Path, lines: int = 20) -> str:
    """The last ``lines`` of TPrime's log, labelled -- or say there is none."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"{path.name}: not found at {path}"
    return f"{path} (last {lines} lines):\n" + "\n".join(text[-lines:])


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
