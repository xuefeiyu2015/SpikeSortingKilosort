"""Shared argument parsing and reporting for the pipeline scripts.

Also puts ``src/`` on ``sys.path`` so the scripts run straight from a checkout
without ``pip install -e .`` -- useful on the HPC, where installing into the
shared conda env is not always possible.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
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
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="do not verify that input files exist before starting",
    )
    parser.add_argument(
        "--probe",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="run only these probes of the run the session names (default: all "
        "of neuropixels.probes). Selects a subset; it cannot add a probe the "
        "session does not cover",
    )
    return parser


def flag_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Session settings this run replaces, from the flags actually passed.

    Every one of these is a session key first -- ``export_lfp``, ``lfp_decimate``,
    ``export_groups``, ``export_figures``, ``waveforms.export_snippets``,
    ``kilosort_on_<system>`` -- so a
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
    if getattr(args, "lfp_decimate", None) is not None:
        overrides["export_lfp"] = True     # asking for a stride is asking for the export
        overrides["lfp_decimate"] = int(args.lfp_decimate)
    if getattr(args, "groups", None) is not None:
        overrides["export_groups"] = tuple(args.groups)
    if getattr(args, "export_waveforms", False):
        # Nested: the waveform settings are per system, and a flag applies to
        # this run rather than to one system's band. with_overrides fans it out.
        overrides["waveforms"] = {"export_snippets": True}
    if getattr(args, "no_figures", False):
        overrides["export_figures"] = False

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

    for _, probe_config in config.per_probe():
        probe_config.paths.mkdirs()
    if require_inputs and not getattr(args, "skip_checks", False):
        config.require_inputs()

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

    def __init__(self, config: SessionConfig, keep_going: bool = False):
        self.config = config
        self.keep_going = keep_going
        self.failures = 0
        self.stopped = False
        #: True when the last call raised, so a caller can skip what depended on it.
        self.last_failed = False

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

        try:
            value = verb(*args, **kwargs)
        except Exception as error:  # a stage that blows up must not hide the rest
            self.failures += 1
            self.last_failed = True
            print(f"[!!] {label}\n       {type(error).__name__}: {error}")
            if not self.keep_going:
                self.stopped = True
            return None

        if value is None:
            print(f"[--] {label}\n       nothing to do")
        else:
            print(f"[ok] {label}\n       {_describe(value)}")
        return value

    def finish(self, name: str) -> int:
        print()
        if self.failures:
            print(f"{name} finished with {self.failures} failure(s)")
            return 1
        print(f"{name} finished")
        return 0


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
        return ", ".join(f"{k}={v}" for k, v in value.items() if not k.startswith("_"))
    return str(value)
