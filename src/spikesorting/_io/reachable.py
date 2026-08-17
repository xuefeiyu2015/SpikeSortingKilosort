"""Is this file actually readable, and how fast?

Not SpikeGLX-specific: a Blackrock ``.ns6`` sits on the same share and stalls the
same way. Kept out of ``spikeglx.py`` for that reason, and because it depends on
nothing but the standard library.

Two questions, because they fail differently:

* **how fast** -- :func:`probe_read` times a small read and scales it up, so a
  caller can say what a whole-file pass will cost before starting one;
* **whether at all** -- :func:`check_reachable` puts a deadline on that, because
  a ``stat`` on a hung mount blocks in the kernel and no timeout inside the
  process can interrupt it.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR

__all__ = ["ReadProbe", "probe_read", "check_reachable", "REACHABLE_TIMEOUT_S"]

#: How long a filesystem gets to answer at all before we stop waiting. Generous:
#: it bounds "is anything there", not the read.
REACHABLE_TIMEOUT_S = 10.0


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
        if not self.total_bytes:
            # A directory: it answered, and there is no size to put a rate on.
            return f"answered in {(self.latency_s + self.seconds) * 1e3:.0f} ms"
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
    stat = path.stat()                   # unmounted share fails here, not in the loop

    # A SpikeGLX run is named by its directory, and the globbing that finds the
    # binaries happens before anything has a filename -- so a directory has to be
    # answerable too. Listing one entry is the equivalent question there.
    if S_ISDIR(stat.st_mode):
        start = time.perf_counter()
        with os.scandir(path) as entries:
            next(entries, None)
        return ReadProbe(
            path=path,
            total_bytes=0,
            sample_bytes=0,
            seconds=time.perf_counter() - start,
            latency_s=start - opened,
        )

    total = stat.st_size
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


def check_reachable(
    path: str | Path, timeout_s: float = REACHABLE_TIMEOUT_S, **probe: object
) -> ReadProbe:
    """:func:`probe_read` with a deadline. Raises ``TimeoutError`` on silence.

    A share that has gone away does not fail, it *stops answering*: the ``stat``
    blocks in the kernel, uninterruptibly, so the process has no way to cancel it.
    What it can do is stop waiting -- the probe runs in a daemon thread and the
    caller gets control back with a message naming the path.

    The abandoned thread stays blocked until the mount recovers. That is the
    honest cost of the approach, and it is why this only ever runs on the handful
    of files a verb is about to open, not in a loop.

    Errors raised by the probe itself are re-raised unchanged: a missing file
    answered immediately, and reporting that as a timeout would be a lie.
    ``TimeoutError`` is an ``OSError``, so callers already guarding these verbs
    with ``except OSError`` catch this too.
    """
    path = Path(path)
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["probe"] = probe_read(path, **probe)  # type: ignore[arg-type]
        except BaseException as error:  # noqa: BLE001 -- re-raised in the caller
            result["error"] = error

    worker = threading.Thread(target=run, name=f"probe-{path.name}", daemon=True)
    worker.start()
    worker.join(timeout_s)

    if worker.is_alive():
        raise TimeoutError(
            f"{path} did not respond within {timeout_s:g} s. The share is mounted "
            "but not answering, or was disconnected -- check the mount before "
            "starting a job that reads from it."
        )
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["probe"]  # type: ignore[return-value]
