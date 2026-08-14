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

from spikesorting.config import SessionConfig, load_session  # noqa: E402
from spikesorting.pipeline import StepResult  # noqa: E402


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
    config = load_session(args.config, args.machine)
    config.paths.mkdirs()
    if require_inputs and not getattr(args, "skip_checks", False):
        config.require_inputs()
    print(f"session '{config.session}' on machine '{config.machine.name}'")
    print(f"  outputs: {config.output_root}")
    return config


def report(result: StepResult) -> int:
    """Print a step result. Returns the process exit code."""
    print(result.render())
    return 0 if result.status in ("ok", "skipped") else 1
