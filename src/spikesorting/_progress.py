"""Progress of a long read, reported through logging and drawn by whoever listens.

A whole-file pass over an AP binary on a share is minutes, and a silent loop
looks exactly like a hang. The pipeline does not draw anything: each update is
one ``logging`` record carrying a ``progress`` attribute, so the drivers can
redraw a bar in place while a notebook, or the saved run log, sees only a
line every ``step`` of the way and one at the end.

- every update: a DEBUG record. Nothing shows it unless a handler asks -- the
  drivers' terminal handler does, to move the bar.
- each ``step`` crossed, and the end: an INFO record whose message is a whole
  sentence (percent, amount, rate, time left), for any handler.

The cost is one function call per update -- per 64 MB block on a binary read.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

__all__ = ["ProgressState", "Progress", "format_amount", "format_done", "format_duration"]


@dataclass(frozen=True)
class ProgressState:
    """One update: what is being read, how far, how fast. Attached to a record."""

    label: str
    done: float
    total: float
    unit: str            # "bytes" or a plural noun ("reads")
    elapsed_s: float
    final: bool = False

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total) if self.total else 1.0

    @property
    def rate(self) -> float | None:
        """Units per second, or None before any time has passed."""
        return self.done / self.elapsed_s if self.elapsed_s > 0 else None

    @property
    def left_s(self) -> float | None:
        rate = self.rate
        if not rate or self.final:
            return None
        return max(0.0, (self.total - self.done) / rate)

    def sentence(self) -> str:
        """The INFO line: ``read 30% of x.bin (45.0/150.0 GB, 412 MB/s, ~4.3 min left)``."""
        amount = format_done(self.done, self.total, self.unit)
        rate = f"{format_amount(self.rate, self.unit)}/s" if self.rate else ""
        if self.final:
            return (f"{self.label}: done, {format_amount(self.total, self.unit)} "
                    f"in {format_duration(self.elapsed_s)}" + (f", {rate}" if rate else ""))
        left = f"~{format_duration(self.left_s)} left" if self.left_s is not None else ""
        details = ", ".join(x for x in (amount, rate, left) if x)
        return f"{self.label}: {100 * self.fraction:.0f}% ({details})"


class Progress:
    """Report progress through ``total`` units to ``log``.

    ``update(done)`` with the running total; ``finish()`` once at the end (also
    called by ``update`` when ``done`` reaches ``total``). ``step`` is how often,
    as a fraction, an INFO line is logged.
    """

    def __init__(self, log: logging.Logger, label: str, total: float,
                 unit: str = "bytes", step: float = 0.1):
        self.log, self.label, self.total, self.unit = log, label, float(total), unit
        self.step = step
        self._started = time.perf_counter()
        self._next = step
        self._finished = False

    def _state(self, done: float, final: bool = False) -> ProgressState:
        return ProgressState(self.label, float(done), self.total, self.unit,
                             time.perf_counter() - self._started, final)

    def update(self, done: float) -> None:
        if self._finished:
            return
        if self.total and done >= self.total:
            self.finish()
            return
        state = self._state(done)
        level = logging.DEBUG
        if self.total and state.fraction >= self._next:
            level = logging.INFO
            while self._next <= state.fraction:
                self._next += self.step
        self.log.log(level, "%s", state.sentence(), extra={"progress": state})

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        state = self._state(self.total, final=True)
        self.log.info("%s", state.sentence(), extra={"progress": state})


def format_amount(value: float | None, unit: str) -> str:
    """``412 MB``, ``45.0 GB`` for bytes; ``1,204 reads`` otherwise."""
    if value is None:
        return "--"
    if unit != "bytes":
        return f"{value:,.0f} {unit}"
    for size, name in ((1 << 40, "TB"), (1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if value >= size:
            return f"{value / size:.1f} {name}" if name in ("GB", "TB") else f"{value / size:.0f} {name}"
    return f"{value:.0f} B"


def format_done(done: float, total: float, unit: str) -> str:
    """``45.0 GB/150.0 GB`` for bytes; ``50/100 reads`` otherwise."""
    if unit == "bytes":
        return f"{format_amount(done, unit)}/{format_amount(total, unit)}"
    return f"{done:,.0f}/{total:,.0f} {unit}"


def format_duration(seconds: float | None) -> str:
    """``42 s``, ``4.3 min``, ``1.52 h``."""
    if seconds is None:
        return "--"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"
