"""Progress of a long read: logged as records, drawn as a bar by the drivers."""

from __future__ import annotations

import io
import logging

from _cli import RunLog, TerminalLog, render_bar
from spikesorting._progress import Progress, ProgressState, format_amount, format_duration

GB = 1 << 30


class _Screen(io.StringIO):
    encoding = "utf-8"


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _logger(name, *handlers):
    log = logging.getLogger(f"spikesorting.test.{name}")
    log.handlers[:] = list(handlers)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log


def test_every_update_is_a_record_and_every_tenth_an_info_line():
    sink = _Records()
    progress = Progress(_logger("p", sink), "read x.bin", 100 * GB)
    for done in range(1, 101):
        progress.update(done * GB)
    progress.finish()                                   # already finished: no-op

    assert len(sink.records) == 100
    assert all(hasattr(r, "progress") for r in sink.records)
    info = [r for r in sink.records if r.levelno == logging.INFO]
    assert len(info) == 10                              # 10%, 20%, ... 90%, then done
    assert info[0].getMessage().startswith("read x.bin: 10% (10.0 GB/100.0 GB")
    assert info[-1].progress.final
    assert info[-1].getMessage().startswith("read x.bin: done, 100.0 GB in")


def test_a_large_block_crossing_several_milestones_logs_one_line():
    sink = _Records()
    progress = Progress(_logger("q", sink), "read", 10.0, unit="reads")
    progress.update(3.5)
    progress.update(9.0)
    info = [r.getMessage() for r in sink.records if r.levelno == logging.INFO]
    assert [m.split("%")[0] for m in info] == ["read: 35", "read: 90"]


def test_state_reports_rate_and_time_left():
    state = ProgressState("read x.bin", 30 * GB, 120 * GB, "bytes", elapsed_s=60.0)
    assert state.fraction == 0.25
    assert state.rate == 0.5 * GB
    assert state.left_s == 180.0
    assert format_amount(state.rate, "bytes") == "512 MB"
    assert format_duration(state.left_s) == "3.0 min"
    assert format_amount(1204, "reads") == "1,204 reads"


def test_the_bar_line_fits_the_terminal_and_shows_where_the_read_is():
    state = ProgressState("read Athos_20261110_t1310_d6700_J4_exp_g0_t0.imec0.ap.bin",
                          77.6 * GB, 150 * GB, "bytes", elapsed_s=180.0)
    line = render_bar(state, 100)
    assert len(line) <= 100
    assert "51.7%" in line and "77.6 GB/150.0 GB" in line and "left" in line
    assert line.count("█") + line.count("░") >= 20
    assert "imec0.ap.bin" in line                       # the file's end survives
    quarter = render_bar(ProgressState("read x.bin", 1, 4, "reads", 1.0), 120)
    assert "read x.bin [" in quarter and " 25.0%" in quarter and "1/4 reads" in quarter
    assert render_bar(ProgressState("r", 1, 4, "reads", 1.0), 80, "#", "-").count("#") >= 2


def test_on_a_terminal_the_bar_redraws_in_place_and_ends_with_a_newline():
    out = _Screen()
    handler = TerminalLog(out, interval_s=0.0, live=True)
    handler.setFormatter(logging.Formatter("       %(message)s"))
    log = _logger("live", handler)
    progress = Progress(log, "read x.bin", 4 * GB)
    for done in range(1, 5):
        progress.update(done * GB)
    log.info("next stage")

    text = out.getvalue()
    assert text.count("\r") == 4                         # one line, redrawn
    assert text.count("\n") == 2                         # the bar's end, then the line
    assert text.endswith("       next stage\n")
    assert "done in" in text


def test_redirected_output_gets_only_the_milestone_lines():
    out = io.StringIO()
    handler = TerminalLog(out, live=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    progress = Progress(_logger("batch", handler), "read x.bin", 1000.0, unit="reads")
    for done in range(1, 1001):
        progress.update(done)

    lines = out.getvalue().splitlines()
    assert "\r" not in out.getvalue()
    assert len(lines) == 10
    assert lines[-1].startswith("read x.bin: done")


def test_the_run_log_keeps_the_milestones_not_the_bar(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["run_exporting_pipeline.py"])
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    run_log = RunLog("exporting")
    try:
        log = logging.getLogger("spikesorting")
        previous = log.level
        log.setLevel(logging.DEBUG)
        progress = Progress(log, "read x.bin", 100.0, unit="reads")
        for done in range(1, 101):
            progress.update(done)
        log.setLevel(previous)
    finally:
        run_log.close()
    kept = [line for line in run_log.buffer.getvalue().splitlines() if "read x.bin" in line]
    assert len(kept) == 10
