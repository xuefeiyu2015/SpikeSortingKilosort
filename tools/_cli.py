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

from spikesorting import SessionConfig, load_session_config, skip_reason  # noqa: E402


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
        "--probe", type=int, default=0, help="Neuropixels probe index (default: 0)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="do not verify that input files exist before starting",
    )
    return parser


def load(args: argparse.Namespace, require_inputs: bool = True) -> SessionConfig:
    """Load the session config and create its output folders."""
    config = load_session_config(args.config, args.machine)
    config.paths.mkdirs()
    if require_inputs and not getattr(args, "skip_checks", False):
        config.require_inputs()
    print(f"session '{config.session}' on machine '{config.machine.name}'")
    # Each system writes beside its own recording, so there are up to two trees.
    for system in ("blackrock", "neuropixels"):
        directory = config.paths.dir_for(system)
        if directory is not None:
            print(f"  {system}: {directory}")
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

    def __call__(self, verb, *args, system: str | None = None, **kwargs):
        """Run one verb. Returns its value, or None if it was skipped or failed."""
        if self.stopped:
            return None
        self.last_failed = False
        label = verb.__name__ + (f"({system})" if system else "")

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
