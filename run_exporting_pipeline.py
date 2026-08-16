#!/usr/bin/env python
"""Pipeline part 2 of 2: everything after manual curation.

    python run_exporting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

Run this once Phy has written ``cluster_group.tsv``: its labels override
Kilosort's own ``cluster_KSLabel.tsv``, and every stage here reads the human
labels when they exist. Running it before curating is not an error -- you get
Kilosort's labels instead -- but it is rarely what you want.

    time_remapping       coarse offset from the 14 s bursts, then a fine fit on
                         the 1 Hz train, onto Blackrock time. Blackrock is the
                         reference timebase; this maps onto it, never the reverse.
    validate_remapping   the same map checked against the burst onsets, which
                         were held out of the fit. Non-zero past tolerance, so it
                         can gate a batch job.
    export_results       aligned spike times, metrics, waveforms, figures.

**Needs no GPU and no sorter** -- only the edge files and the sorted output. It
does use CatGT/TPrime where the machine has them, so this is the half that wants
to run on the rig even when sorting went to a cluster.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# The shared CLI helpers live in tools/; _cli adds src/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from _cli import Runner, build_parser, load  # noqa: E402

import spikesorting as ss  # noqa: E402

STAGES = ["time_remapping", "validate_remapping", "export_results"]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=STAGES,
        choices=STAGES,
        help="stages to run (default: all)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["good", "mua"],
        help="Phy cluster_group labels to export (default: good mua)",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="       %(message)s")

    config = load(args, require_inputs=False)
    run = Runner(config, keep_going=args.keep_going)
    figures = not args.no_figures
    print()

    # The pipeline, in order. Blackrock is the reference timebase, so it is what
    # the other system is mapped *onto* rather than a system to remap.
    if "time_remapping" in args.steps:
        for system in ss.SYSTEMS:
            run(ss.time_remapping, config, system, system=system)

    if "validate_remapping" in args.steps:
        run(ss.validate_remapping, config, figures)

    if "export_results" in args.steps:
        for system in ss.SYSTEMS:
            run(
                ss.export_results,
                config,
                system,
                tuple(args.groups),
                figures,
                system=system,
            )

    return run.finish("exporting pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
