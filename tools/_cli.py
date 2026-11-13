"""Shared argument parsing and reporting for the pipeline scripts.

Also puts ``src/`` on ``sys.path`` so the scripts run straight from a checkout
without ``pip install -e .`` -- useful on the HPC, where installing into the
shared conda env is not always possible.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spikesorting import (  # noqa: E402
    SessionConfig,
    load_session_config,
    skip_reason,
)
from spikesorting._config import with_overrides  # noqa: E402


def default_machine() -> str:
    """Guess the machine profile from the platform, overridable by env var."""
    override = os.environ.get("SPIKESORTING_MACHINE")
    if override:
        return override
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("win"):
        return "windows_rig"
    return "hpc"


def shell_arg(path: str | Path) -> str:
    """``path`` ready to paste into a shell: double-quoted when it has a space.

    The shares are named like ``Z:/Monkey Athos/...``, and an unquoted space
    splits one path into two arguments. Double quotes are the one form that
    cmd, PowerShell and bash all read as a single argument, so every path a
    driver prints as a command to copy goes through here. A path with no
    whitespace is printed as it is, so ``--config session_x`` reads back the
    way it was typed.
    """
    text = str(path)
    return f'"{text}"' if any(c.isspace() for c in text) else text


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        help="session YAML (path, or a name resolved inside configs/)",
    )
    parser.add_argument(
        "--machine",
        default=default_machine(),
        help=f"machine profile in configs/machines/ (default: {default_machine()})",
    )
    return parser



def flag_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Session settings this run replaces, from the flags actually passed.

    Every one of these is a session key first -- ``export_lfp``,
    ``waveforms.export_snippets``, ``kilosort_on_<system>`` -- so a
    setting is never reachable through the command line alone. A flag changes it
    for one run; the file keeps saying what the session actually wants.
    """
    overrides: dict[str, object] = {}

    system = getattr(args, "system", None)
    if system is not None:
        for name in ("neuropixels", "blackrock"):
            overrides[f"kilosort_on_{name}"] = name == system
    if getattr(args, "export_lfp", False):
        overrides["export_lfp"] = True
    if getattr(args, "export_waveforms", False):
        # Nested: the waveform settings are per system, and a flag applies to
        # this run rather than to one system's band. with_overrides fans it out.
        overrides["waveforms"] = {"export_snippets": True}

    return overrides


def _select_probes(config: SessionConfig, wanted: list[int] | None) -> SessionConfig:
    """Narrow ``neuropixels.bin_files`` to the probes ``--probe`` asked for.

    Deliberately not in :func:`flag_overrides`: every flag there replaces a
    session key for one run, while this one runs a *subset* of what the session
    already says. So it can only ever remove binaries -- naming a probe the
    session does not cover is an error rather than a way to add it, since nothing
    would then say where that probe's binary is.
    """
    if wanted is None:
        return config

    missing = sorted(set(wanted) - set(config.probes))
    if missing:
        raise SystemExit(
            f"--probe {missing} : session '{config.session}' covers probes "
            f"{list(config.probes)}. Name the missing binary in "
            "neuropixels.bin_files to run it."
        )
    if config.neuropixels.bin_files is None:
        # One binary, and it is one of the probes asked for. Narrowing would mean
        # writing a bin_files list, which flips the session into the multi-probe
        # output layout on the strength of a flag -- moving a plain one-probe
        # session's outputs. So a request it already satisfies is a no-op.
        return config

    keep = config.neuropixels.binaries_by_probe()
    return replace(
        config,
        neuropixels=replace(
            config.neuropixels,
            bin_files=tuple(keep[probe] for probe in sorted(set(wanted))),
        ),
    )


def load(args: argparse.Namespace, require_inputs: bool = True) -> SessionConfig:
    """Load the session config, apply any flag overrides, create its output folders."""
    config = load_session_config(args.config, args.machine)

    overrides = flag_overrides(args)
    if overrides:
        config = with_overrides(config, **overrides)
    config = _select_probes(config, getattr(args, "probe", None))

    # Check before creating anything: a mistyped directory must not be built on
    # disk. mkdirs refuses one that does not exist even when the inputs are not
    # required (the exporting driver), since every output goes inside it.
    if require_inputs:
        config.require_inputs()
    for _, probe_config in config.per_probe():
        probe_config.paths.mkdirs()

    print(f"session '{config.session}' on machine '{config.machine.name}'")
    if overrides:
        print(
            "  overridden for this run: "
            + ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        )
    # Each system writes beside its own recording, so there are up to two trees --
    # and a multi-probe session has one Neuropixels tree per probe.
    directory = config.paths.dir_for("blackrock")
    if directory is not None:
        print(f"  blackrock: {directory}")
    for tag, probe_config in config.per_probe():
        directory = probe_config.paths.dir_for("neuropixels")
        if directory is None:
            continue
        label = "neuropixels" if tag is None else f"neuropixels {tag}"
        print(f"  {label}: {directory}")
    return config


class Runner:
    """Calls pipeline verbs and reports what happened.

    The verbs return plain values and say nothing; every ``[ok] / [--] / [!!]``
    in this project is printed here. That is the whole reason the verbs are
    usable from a notebook -- a cell shows the recording, not a status wrapper.
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self.failures = 0
        self.stopped = False
        #: True when the last call raised, so a caller can skip what depended on it.
        self.last_failed = False
        #: Every stage that ran, in order: ``{"label", "stage", "status",
        #: "seconds", "started", "steps"}``. What ``record_timing`` is given.
        self.stages: list[dict] = []

    def __call__(self, verb, *args, system: str | None = None, label: str | None = None,
                 config: SessionConfig | None = None, **kwargs):
        """Run one verb. Returns its value, or None if it was skipped or failed.

        ``label`` names the stream in the printed line when that is not simply the
        system -- one probe of a multi-probe run. ``config`` is the config to ask
        for the skip reason, which for a per-probe stage is that probe's own
        projection rather than the session as a whole.
        """
        if self.stopped:
            return None
        self.last_failed = False
        shown = label or system
        label = verb.__name__ + (f"({shown})" if shown else "")

        why = skip_reason(config or self.config, verb, system)
        if why:
            print(f"[--] {label}\n       {why}")
            return None

        from datetime import datetime

        wall = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.perf_counter()
        try:
            value = verb(*args, **kwargs)
        except Exception as error:  # a stage that blows up must not hide the rest
            self.failures += 1
            self.last_failed = True
            self._record(shown, verb, "failed", time.perf_counter() - started, wall, None)
            took = _elapsed(time.perf_counter() - started)
            print(f"[!!] {label}  ({took})\n       {type(error).__name__}: {error}")
            self.stopped = True
            return None

        seconds = time.perf_counter() - started
        took = _elapsed(seconds)
        self._record(shown, verb, "nothing to do" if value is None else "ok",
                     seconds, wall, value)
        if value is None:
            print(f"[--] {label}  ({took})\n       nothing to do")
        else:
            print(f"[ok] {label}  ({took})\n       {_describe(value)}")
        return value

    def _record(self, shown, verb, status, seconds, wall, value) -> None:
        steps = value.get("timing") if isinstance(value, dict) else None
        self.stages.append({
            "label": shown, "stage": verb.__name__, "status": status,
            "seconds": round(seconds, 2), "started": wall,
            "steps": steps if isinstance(steps, dict) else None,
        })

    def stages_for(self, label: str) -> list[dict]:
        """The stages that ran for one stream (``label`` as printed), in order."""
        return [s for s in self.stages if s["label"] == label]

    def stage_table(self) -> str:
        """Every stage of this run, one line each: stream, stage, outcome, time."""
        if not self.stages:
            return ""
        width = max(len(s["label"] or "") for s in self.stages) + 2
        lines = ["stage times"]
        for s in self.stages:
            lines.append(
                f"  {s['label'] or '':<{width}}{s['stage']:<22}{s['status']:<15}"
                f"{_elapsed(s['seconds']):>10}"
            )
        total = sum(s["seconds"] for s in self.stages)
        lines.append(f"  {'':<{width}}{'total':<37}{_elapsed(total):>10}")
        return "\n".join(lines)

    def finish(self, name: str) -> int:
        table = self.stage_table()
        if table:
            print()
            print(table)
        print()
        if self.failures:
            print(f"{name} finished with {self.failures} failure(s)")
            return 1
        print(f"{name} finished")
        return 0


class _Tee(io.TextIOBase):
    """A stream that writes to the terminal and to the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class RunLog:
    """Everything a driver prints, kept so it can be saved beside the outputs.

    The terminal shows the run as it happens and is gone afterwards; this keeps
    the same text -- the ``[ok]/[--]/[!!]`` lines, the pipeline's log messages,
    the stage times and timing tables -- and :meth:`save` writes it, named by
    script, start time and machine, so runs accumulate rather than overwrite
    and a run from the share can be read beside one from a local disk.

    Kilosort's own log is not in it (Kilosort writes ``kilosort4.log`` beside
    the sorting), nor are progress bars, which go to stderr.
    """

    def __init__(self, script: str):
        self.script = script
        self.started = datetime.now().astimezone()
        self.buffer = io.StringIO()
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self.buffer)
        self._handler = logging.StreamHandler(self.buffer)
        self._handler.setFormatter(logging.Formatter("       %(message)s"))
        logging.getLogger("spikesorting").addHandler(self._handler)
        print(f"{script} started {self.started.isoformat(timespec='seconds')}, "
              f"command: {' '.join(sys.argv)}")

    def close(self) -> None:
        """Stop capturing. Safe to call twice."""
        sys.stdout = self._stdout
        logging.getLogger("spikesorting").removeHandler(self._handler)

    def filename(self, machine: str) -> str:
        return f"{self.script}_{self.started:%Y%m%d-%H%M%S}_{machine}.log"

    def save(self, directories, machine: str) -> list[Path]:
        """Write the log into ``<dir>/logs/`` for each directory. Returns the paths."""
        text = self.buffer.getvalue()
        saved = []
        for directory in dict.fromkeys(Path(d) for d in directories):
            target = directory / "logs" / self.filename(machine)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            except OSError as error:  # a log must never turn a finished run into a failure
                print(f"could not write {target}: {error}")
                continue
            saved.append(target)
        return saved


def log_directories(config: SessionConfig) -> list[Path]:
    """Where a run's log goes: each stream's sorting folder the session declares.

    Blackrock only when its array is sorted: a sync-only Blackrock's outputs
    are in the Neuropixels folder, and so is the log.

    Beside ``run_info.json`` and the sorting it describes; one copy per stream,
    so each folder holds the record of every run that touched it.
    """
    dirs = []
    for _, probe_config in config.per_probe():
        if probe_config.has_data("neuropixels"):
            dirs.append(probe_config.paths.sorted_for("neuropixels"))
    if config.paths.writes_to("blackrock"):     # only a sorted Utah array
        dirs.append(config.paths.sorted_for("blackrock"))
    return dirs


def _elapsed(seconds: float) -> str:
    """``0.4 s``, ``12.3 s``, ``4.2 min``, ``1.53 h`` -- how long a stage took."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _describe(value) -> str:
    """One line about whatever a verb returned."""
    for attr in ("summary", "render"):
        if callable(getattr(value, attr, None)):
            return getattr(value, attr)()
    if isinstance(value, dict):
        # A probe dict is mostly a 384-element chanMap; printing it verbatim buries
        # the run. probe_summary is the loggable form the repo already has.
        if {"chanMap", "xc", "yc"} <= set(value):
            from spikesorting._probes.common import probe_summary

            value = probe_summary(value)
        # Scalars only: a verb's dict may carry the arrays it computed (the
        # waveform export returns its SnippetResult) so a notebook gets them, and
        # printing those verbatim floods the run log with numbers.
        return ", ".join(
            f"{k}={v}" for k, v in value.items()
            if not k.startswith("_") and _is_scalar(v)
        )
    return str(value)


def _is_scalar(value) -> bool:
    import numpy as np

    return value is None or isinstance(value, (str, int, float, bool, Path, np.generic))
