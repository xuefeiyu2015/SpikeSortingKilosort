"""Shared argument parsing and reporting for the pipeline scripts.

Also puts ``src/`` on ``sys.path`` so the scripts run straight from a checkout
without ``pip install -e .`` -- useful on the HPC, where installing into the
shared conda env is not always possible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spikesorting import (  # noqa: E402
    SessionConfig,
    load_session_config,
    skip_reason,
    stream_label,
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
        "--probe",
        type=int,
        default=None,
        help="run one Neuropixels probe instead of every probe the session declares",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="do not verify that input files exist before starting",
    )
    return parser


def flag_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Session settings this run replaces, from the flags actually passed.

    Every one of these is a session key first -- ``export_lfp``, ``lfp_decimate``,
    ``export_groups``, ``export_figures``, ``kilosort_on_<system>`` -- so a
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
    if getattr(args, "no_figures", False):
        overrides["export_figures"] = False

    return overrides


def load(args: argparse.Namespace, require_inputs: bool = True) -> SessionConfig:
    """Load the session config, apply any flag overrides, create its output folders."""
    config = load_session_config(args.config, args.machine)

    overrides = flag_overrides(args)
    if overrides:
        config = with_overrides(config, **overrides)

    config.paths.mkdirs()
    if require_inputs and not getattr(args, "skip_checks", False):
        config.require_inputs()

    only = getattr(args, "probe", None)
    declared = config.probe_indices("neuropixels")
    if only is not None and only not in declared:
        raise SystemExit(
            f"--probe {only}: this session declares probes {list(declared)}. "
            "Add it to neuropixels.probes, or drop the flag to run them all."
        )

    print(f"session '{config.session}' on machine '{config.machine.name}'")
    if overrides:
        print(
            "  overridden for this run: "
            + ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        )
    # Each system writes beside its own recording, so there are up to two trees.
    for system in ("blackrock", "neuropixels"):
        directory = config.paths.dir_for(system)
        if directory is None:
            continue
        streams = ""
        if system == "neuropixels":
            streams = " (%s)" % ", ".join(
                stream_label(system, p) for p in run_probes(config, system, only)
            )
        print(f"  {system}: {directory}{streams}")
    return config


def run_probes(
    config: SessionConfig, system: str, only: int | None = None
) -> tuple[int, ...]:
    """Which streams of ``system`` this run covers.

    Blackrock has one, so the drivers' inner loop runs once for it however many
    probes the run holds -- the NSP file is not re-extracted per probe. ``only``
    is ``--probe``, already checked against the declared list by :func:`load`.
    """
    declared = config.probe_indices(system)
    if only is None or system != "neuropixels":
        return declared
    return (only,)


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

    def __call__(self, verb, *args, system: str | None = None, probe: int | None = None, **kwargs):
        """Run one verb. Returns its value, or None if it was skipped or failed.

        ``probe`` only names the line: a run holding imec0 and imec1 prints two
        of most stages, and "which probe" has to be readable at a glance.
        """
        if self.stopped:
            return None
        self.last_failed = False
        label = verb.__name__ + (f"({_scope(system, probe)})" if system else "")

        why = skip_reason(self.config, verb, system)
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


def _scope(system: str | None, probe: int | None) -> str:
    """``blackrock`` / ``neuropixels imec1`` -- what a status line is about."""
    if system is None:
        return ""
    stream = stream_label(system, probe or 0)
    return system if stream == system or probe is None else f"{system} {stream}"


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
