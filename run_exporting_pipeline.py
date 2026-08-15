#!/usr/bin/env python
"""Pipeline part 2 of 2: everything after manual curation.

    python run_exporting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

Run this once Phy has written ``cluster_group.tsv``: its labels override
Kilosort's own ``cluster_KSLabel.tsv``, and every stage here reads the human
labels when they exist. Running it before curating is not an error -- you get
Kilosort's labels instead -- but it is rarely what you want.

    align       coarse offset from the 14 s bursts, then a fine fit on the 1 Hz
                train, onto Blackrock time. Blackrock is the reference timebase;
                this maps onto it, never the reverse.
    validate    the same map checked against the burst onsets, which were held
                out of the fit. Exits non-zero past the tolerance, so it can gate
                a batch job.
    export      aligned spike times, metrics, waveforms and summary figures.

**Needs no GPU and no sorter** -- only the edge files and the sorted output. It
does use CatGT/TPrime when the machine has them, so this is the half that wants
to run on the rig even when sorting went to a cluster.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# The shared CLI helpers live with the numbered stages; _cli adds src/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from _cli import build_parser, load  # noqa: E402

from spikesorting.pipeline import step_align, step_export, step_validate  # noqa: E402

ORDER = ["align", "validate", "export"]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=ORDER,
        choices=ORDER,
        help="stages to run, in the given order (default: all)",
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
    config = load(args)

    systems = [s for s in ("neuropixels", "blackrock") if getattr(config, f"has_{s}_data")]
    figures = not args.no_figures

    runners = {
        "align": lambda: [step_align(config, system=s) for s in systems],
        "validate": lambda: [step_validate(config, figures=figures)],
        "export": lambda: [
            step_export(config, system=s, groups=tuple(args.groups), figures=figures)
            for s in systems
        ],
    }

    print()
    failures = 0
    for name in args.steps:
        try:
            results = runners[name]()
        except Exception as error:  # a stage that blows up should not hide the rest
            failures += 1
            print(f"[!!] {name}\n       {type(error).__name__}: {error}")
            traceback.print_exc()
            if not args.keep_going:
                break
            continue

        stop = False
        for result in results:
            print(result.render())
            if result.status == "failed":
                failures += 1
                stop = not args.keep_going
        if stop:
            break

    print()
    print(
        "exporting pipeline finished"
        if not failures
        else f"exporting pipeline finished with {failures} failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
