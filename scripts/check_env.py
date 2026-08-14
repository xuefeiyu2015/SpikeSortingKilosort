#!/usr/bin/env python
"""Report whether this machine is set up to run the pipeline.

**Not a pipeline stage** -- deliberately outside the numbered ``NN_`` sequence,
like ``make_probe.py``. Run it straight after cloning onto a new PC:

    python scripts/check_env.py
    python scripts/check_env.py --machine windows_rig
    python scripts/check_env.py --config configs/Athos.yaml   # also checks inputs

It inspects the ``kilosort4`` and ``phy`` conda environments, the CUDA GPU, the
CatGT/TPrime command-line tools, the PowerShell execution policy on Windows, and
the paths the machine profile declares. Everything it finds missing is reported
with the pipeline stages that gap disables and a pointer to the official setup
instructions.

It installs nothing. Setting up Kilosort and phy stays a human task done from
upstream's own docs -- they change (the right torch wheel depends on the GPU,
and phy has already moved off conda once), and a stale recipe baked in here
would be worse than none.

Read-only, and it never runs ``conda activate``, so it works on a fresh Windows
box whose execution policy still blocks that -- which is exactly when the answer
matters. Exits non-zero when something required is missing, so it can gate a
batch job.
"""

from __future__ import annotations

import argparse

from _cli import default_machine

from spikesorting import doctor
from spikesorting.config import load_machine, load_session

MARKERS = {doctor.OK: "[ok]", doctor.WARN: "[--]", doctor.MISSING: "[!!]"}

SECTION_ORDER = (
    doctor.SECTION_ENVS,
    doctor.SECTION_GPU,
    doctor.SECTION_TOOLS,
    doctor.SECTION_SHELL,
    doctor.SECTION_PATHS,
    doctor.SECTION_SESSION,
)


def render_report(checks: list[doctor.Check]) -> str:
    """Group findings by section. Rendering only -- takes precomputed checks."""
    lines: list[str] = []
    for section in SECTION_ORDER:
        in_section = [check for check in checks if check.section == section]
        if not in_section:
            continue
        lines.append("")
        lines.append(section)
        lines.append("-" * len(section))
        for check in in_section:
            marker = MARKERS.get(check.status, "[??]")
            lines.append(f"{marker} {check.name:<28} {check.detail}")
            if check.disables and check.status != doctor.OK:
                lines.append(f"     {'':<28} disables: {', '.join(check.disables)}")
    return "\n".join(lines)


def render_actions(checks: list[doctor.Check]) -> str:
    """What to do about everything that is not ok, de-duplicated in place."""
    actions: list[str] = []
    for check in checks:
        if check.status == doctor.OK or not check.fix:
            continue
        entry = f"  - {check.name}: {check.fix}"
        if entry not in actions:
            actions.append(entry)
    if not actions:
        return ""
    return "\n".join(["", "what to do", "----------", *actions])


def render_summary(checks: list[doctor.Check]) -> str:
    """One-line verdict, plus the stages that are currently unavailable."""
    missing = [check for check in checks if check.status == doctor.MISSING]
    warned = [check for check in checks if check.status == doctor.WARN]
    verdict = doctor.worst_status(checks)
    if verdict == doctor.OK:
        head = "ready: everything checked is in place"
    elif verdict == doctor.WARN:
        head = f"ready ({len(warned)} warning(s), none blocking)"
    else:
        head = f"{len(missing)} blocking problem(s), {len(warned)} warning(s)"

    blocked: list[str] = []
    for check in missing:
        for stage in check.disables:
            if stage not in blocked:
                blocked.append(stage)
    lines = ["", head]
    if blocked:
        lines.append(f"unavailable: {', '.join(blocked)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--machine",
        default=default_machine(),
        help=f"machine profile in configs/machines/ (default: {default_machine()})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional session YAML; adds a check of its input files",
    )
    args = parser.parse_args()

    machine = load_machine(args.machine)
    config = load_session(args.config, machine) if args.config else None

    print(f"machine profile '{machine.name}' (device: {machine.device})")
    if config is not None:
        print(f"session '{config.session}'")

    checks = doctor.run_all(machine, config)
    print(render_report(checks))
    actions = render_actions(checks)
    if actions:
        print(actions)
    print(render_summary(checks))
    return doctor.exit_code(checks)


if __name__ == "__main__":
    raise SystemExit(main())
