#!/usr/bin/env python
"""Run the pipeline stages in order, honouring the session's skip flags.

    python scripts/run_pipeline.py --config configs/demo.yaml
    python scripts/run_pipeline.py --config configs/Athos.yaml --machine windows_rig
    python scripts/run_pipeline.py --config configs/Athos.yaml --steps align validate

Manual curation in Phy sits between sorting and export, so a real session is
normally run in two passes: ``--steps extract_sync sort_neuropixels``, then curate,
then ``--steps align validate export``. The default runs every stage in order.

Exits non-zero if any stage fails. Stages that are skipped -- no Blackrock data,
no alignment, no GPU -- are reported and do not fail the run.
"""

from __future__ import annotations

import traceback

from _cli import build_parser, load

from spikesorting.pipeline import (
    step_align,
    step_export,
    step_extract_sync,
    step_sort_blackrock,
    step_sort_neuropixels,
    step_validate,
)

DEFAULT_ORDER = [
    "extract_sync",
    "sort_neuropixels",
    "sort_blackrock",
    "align",
    "validate",
    "export",
]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=DEFAULT_ORDER,
        choices=DEFAULT_ORDER,
        help="stages to run, in the given order (default: all)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    config = load(args)

    runners = {
        "extract_sync": lambda: step_extract_sync(config, probe=args.probe),
        "sort_neuropixels": lambda: step_sort_neuropixels(config, probe_index=args.probe),
        "sort_blackrock": lambda: step_sort_blackrock(config),
        "align": lambda: step_align(config),
        "validate": lambda: step_validate(config, figures=not args.no_figures),
        "export": lambda: step_export(config, figures=not args.no_figures),
    }

    print()
    failures = 0
    for name in args.steps:
        try:
            result = runners[name]()
        except Exception as error:  # a stage that blows up should not hide the rest
            failures += 1
            print(f"[!!] {name}\n       {type(error).__name__}: {error}")
            traceback.print_exc()
            if not args.keep_going:
                break
            continue

        print(result.render())
        if result.status == "failed":
            failures += 1
            if not args.keep_going:
                break

    print()
    print("pipeline finished" if not failures else f"pipeline finished with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
